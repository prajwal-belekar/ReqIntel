#!/usr/bin/env python3
"""
End-to-end integration tests for the API <-> PostgreSQL persistence contract.

These tests deliberately drive the real FastAPI app, the real SQLAlchemy ORM
models (database/models.py), the real AnalysisRepository and the real response
serialisation. Only two things are substituted:

  * the RAG pipeline, because calling qwen3:8b is neither fast nor the subject
    of these tests, and
  * the database engine, which points at an in-process SQLite database instead
    of a PostgreSQL server.

That substitution is not a mock of our own code: every query, every ORM
mapping, every commit and every JSON payload goes through the same code paths
it does against PostgreSQL. It is what makes these integration tests rather
than unit tests, and it is why they run without a live database while still
catching real wiring bugs (wrong column, dropped field, lost enum value,
missing commit).

The contracts pinned here are the ones the frontend depends on:

  * a successful analysis is persisted,
  * GET /api/v1/history returns it,
  * GET /api/v1/history/{id} returns that exact record,
  * GET /api/v1/stats counts it,
  * an unknown id is a 404, not a 500,
  * an unavailable database is a controlled 503 rather than a traceback,
  * a failing database is a controlled 500 rather than a traceback.

completeness_score stays a 0.0-1.0 fraction all the way through: it is asserted
in the response, in the stored row and in the aggregate statistics.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database.connection as db_connection
from database.models import AnalysisHistory, Base
from rag.pipeline import RequirementAnalysis

from api.main import app


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def session_maker():
    """
    A real, isolated database for one test.

    StaticPool keeps every session on the same in-memory connection, so the
    rows written by the API are visible to the assertions. Tables are created
    from the real metadata, which means a schema mistake fails the test.
    """
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    maker = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    yield maker

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def client(session_maker):
    """
    TestClient wired to the test database.

    database.connection keeps its engine and session factory in module globals,
    which is exactly what the routers read at request time, so patching those
    two globals is what puts the API on the test database.
    """
    with patch.object(db_connection, "_engine", session_maker.kw["bind"]), \
         patch.object(db_connection, "_session_maker", session_maker):
        # Deliberately not used as a context manager: the lifespan would call
        # init_db() and try to reach a real PostgreSQL server.
        yield TestClient(app)


@pytest.fixture
def client_without_db():
    """TestClient for the API running with no database configured at all."""
    with patch.object(db_connection, "_engine", None), \
         patch.object(db_connection, "_session_maker", None):
        yield TestClient(app)


def make_analysis(requirement="The system must authenticate all users.", **overrides):
    """A valid pipeline result. Only the RAG/LLM step is substituted."""
    data = dict(
        requirement=requirement,
        category="Security",
        ml_confidence=0.82,
        risk_level="Medium",
        risk_factors=["Authentication method is unspecified"],
        completeness_score=0.65,
        missing_requirements=["Authentication mechanism"],
        recommendations=["Specify the authentication mechanism"],
        similar_guidelines=[
            {
                "id": "SEC-001",
                "title": "Authentication Requirements",
                "similarity": 0.88,
                "category": "Security",
                "text": "All access must be authenticated before granting resources.",
            }
        ],
        explanation="The requirement does not state how users are authenticated.",
    )
    data.update(overrides)
    return RequirementAnalysis(**data)


def post_analysis(client, **overrides):
    """POST one analysis with the pipeline stubbed out."""
    analysis = make_analysis(**overrides)
    with patch("api.routers.analysis.analyze_requirement", return_value=analysis):
        response = client.post("/api/v1/analyze", json={"requirement": analysis.requirement})
    assert response.status_code == 200, response.text
    return response.json()


def count_rows(session_maker):
    with session_maker() as session:
        return session.execute(
            select(func.count()).select_from(AnalysisHistory)
        ).scalar_one()


# --------------------------------------------------------------------------
# Analysis is persisted
# --------------------------------------------------------------------------

def test_successful_analysis_is_persisted(client, session_maker):
    """A 200 from /analyze must leave exactly one committed row behind."""
    body = post_analysis(client)

    assert count_rows(session_maker) == 1

    with session_maker() as session:
        record = session.execute(select(AnalysisHistory)).scalar_one()

    assert record.requirement == body["requirement"]
    assert record.category == "Security"
    assert record.ml_confidence == body["ml_confidence"]
    assert record.risk_level == "Medium"
    assert record.risk_factors == body["risk_factors"]
    assert record.missing_requirements == body["missing_requirements"]
    assert record.recommendations == body["recommendations"]
    assert record.explanation == body["explanation"]
    assert record.created_at is not None


def test_persisted_guidelines_survive_the_json_round_trip(client, session_maker):
    """similar_guidelines is nested JSON; it must not be flattened or dropped."""
    post_analysis(client)

    with session_maker() as session:
        record = session.execute(select(AnalysisHistory)).scalar_one()
        guidelines = record.similar_guidelines

    assert isinstance(guidelines, list)
    assert len(guidelines) == 1
    assert guidelines[0]["id"] == "SEC-001"
    assert guidelines[0]["title"] == "Authentication Requirements"
    assert guidelines[0]["similarity"] == 0.88
    assert guidelines[0]["category"] == "Security"
    assert guidelines[0]["text"].startswith("All access must be authenticated")


def test_completeness_score_is_stored_as_a_fraction(client, session_maker):
    """
    The 0.0-1.0 contract must survive persistence.

    0.65 is 65%. If anything in the persistence path multiplied or divided by
    100, this is where it would show up.
    """
    body = post_analysis(client, completeness_score=0.65)

    assert body["completeness_score"] == 0.65

    with session_maker() as session:
        record = session.execute(select(AnalysisHistory)).scalar_one()
        assert record.completeness_score == 0.65


def test_every_analysis_gets_its_own_row(client, session_maker):
    """Two analyses must not overwrite each other."""
    post_analysis(client, requirement="First requirement about authentication.")
    post_analysis(client, requirement="Second requirement about encryption.")

    assert count_rows(session_maker) == 2

    history = client.get("/api/v1/history").json()
    assert history["total"] == 2


# --------------------------------------------------------------------------
# History
# --------------------------------------------------------------------------

def test_history_returns_the_persisted_analysis(client, session_maker):
    """GET /history must surface the row that /analyze wrote."""
    body = post_analysis(client)

    response = client.get("/api/v1/history")
    assert response.status_code == 200

    data = response.json()
    assert data["total"] == 1
    assert data["limit"] == 20
    assert data["offset"] == 0
    assert len(data["items"]) == 1

    item = data["items"][0]
    assert item["requirement"] == body["requirement"]
    assert item["category"] == body["category"]
    assert item["ml_confidence"] == body["ml_confidence"]
    assert item["risk_level"] == body["risk_level"]
    assert item["completeness_score"] == body["completeness_score"]
    assert item["created_at"] is not None


def test_history_is_empty_before_any_analysis(client):
    """A fresh database yields an empty, well-formed page rather than an error."""
    response = client.get("/api/v1/history")

    assert response.status_code == 200
    data = response.json()
    assert data == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_history_orders_newest_first(client):
    """list_recent orders by created_at desc; the frontend relies on it."""
    post_analysis(client, requirement="Older requirement about authentication.")
    post_analysis(client, requirement="Newer requirement about encryption.")

    items = client.get("/api/v1/history").json()["items"]

    assert len(items) == 2
    assert items[0]["requirement"] == "Newer requirement about encryption."
    assert items[1]["requirement"] == "Older requirement about authentication."


def test_history_respects_limit_and_offset(client):
    """Pagination is part of the response contract the frontend parses."""
    post_analysis(client, requirement="Requirement one about authentication.")
    post_analysis(client, requirement="Requirement two about encryption.")
    post_analysis(client, requirement="Requirement three about availability.")

    first_page = client.get("/api/v1/history", params={"limit": 2}).json()
    assert len(first_page["items"]) == 2
    assert first_page["limit"] == 2
    # total counts every row, not just the returned page.
    assert first_page["total"] == 3

    second_page = client.get(
        "/api/v1/history", params={"limit": 2, "offset": 2}
    ).json()
    assert len(second_page["items"]) == 1
    assert second_page["offset"] == 2
    assert second_page["total"] == 3

    first_ids = {item["id"] for item in first_page["items"]}
    second_ids = {item["id"] for item in second_page["items"]}
    assert not (first_ids & second_ids)


def test_history_rejects_out_of_range_pagination(client):
    """limit/offset bounds are declared in the route signature and enforced."""
    assert client.get("/api/v1/history", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/history", params={"limit": 101}).status_code == 422
    assert client.get("/api/v1/history", params={"offset": -1}).status_code == 422


# --------------------------------------------------------------------------
# History detail
# --------------------------------------------------------------------------

def test_history_detail_returns_the_exact_record(client, session_maker):
    """GET /history/{id} must return the row identified by id, not the newest."""
    post_analysis(client, requirement="First requirement about authentication.")
    post_analysis(client, requirement="Second requirement about encryption.")

    items = client.get("/api/v1/history").json()["items"]
    first = next(i for i in items if i["requirement"].startswith("First"))
    second = next(i for i in items if i["requirement"].startswith("Second"))
    assert first["id"] != second["id"]

    response = client.get(f"/api/v1/history/{first['id']}")
    assert response.status_code == 200

    detail = response.json()
    assert detail["id"] == first["id"]
    assert detail["requirement"] == first["requirement"]
    assert detail["category"] == first["category"]
    assert detail["risk_level"] == first["risk_level"]
    assert detail["ml_confidence"] == first["ml_confidence"]
    assert detail["completeness_score"] == first["completeness_score"]
    assert detail["created_at"] == first["created_at"]

    # Fields absent from the list view must be present in the detail view.
    assert detail["risk_factors"] == ["Authentication method is unspecified"]
    assert detail["missing_requirements"] == ["Authentication mechanism"]
    assert detail["recommendations"] == ["Specify the authentication mechanism"]
    assert detail["explanation"].startswith("The requirement does not state")
    assert len(detail["similar_guidelines"]) == 1
    assert detail["similar_guidelines"][0]["id"] == "SEC-001"


def test_history_detail_returns_404_for_unknown_id(client):
    """A missing id is a client error, not a server error."""
    response = client.get("/api/v1/history/424242")

    assert response.status_code == 404
    assert "424242" in response.json()["detail"]


def test_history_detail_rejects_a_non_numeric_id(client):
    """The route declares analysis_id as int, so a string id cannot reach the DB."""
    response = client.get("/api/v1/history/not-a-number")

    assert response.status_code == 422
    assert "detail" in response.json()


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------

def test_stats_are_empty_before_any_analysis(client):
    """The dashboard renders from this payload, so its empty shape is a contract."""
    response = client.get("/api/v1/stats")

    assert response.status_code == 200
    assert response.json() == {
        "total_analyses": 0,
        "average_completeness": 0.0,
        "average_confidence": 0.0,
        "category_distribution": {},
        "risk_distribution": {},
        "recent_count_7d": 0,
    }


def test_stats_update_after_an_analysis(client):
    """The dashboard counters must reflect the analysis that was just saved."""
    post_analysis(client, completeness_score=0.65, ml_confidence=0.82)

    response = client.get("/api/v1/stats")
    assert response.status_code == 200

    stats = response.json()
    assert stats["total_analyses"] == 1
    assert stats["average_completeness"] == 0.65
    assert stats["average_confidence"] == 0.82
    assert stats["category_distribution"] == {"Security": 1}
    assert stats["risk_distribution"] == {"Medium": 1}
    assert stats["recent_count_7d"] == 1


def test_stats_average_over_all_saved_analyses(client):
    """Averages and distributions must aggregate, not just report the last row."""
    post_analysis(
        client,
        requirement="First requirement about authentication.",
        category="Security",
        risk_level="High",
        completeness_score=0.5,
        ml_confidence=0.9,
    )
    post_analysis(
        client,
        requirement="Second requirement about encryption.",
        category="Performance",
        risk_level="Low",
        completeness_score=0.9,
        ml_confidence=0.7,
    )

    stats = client.get("/api/v1/stats").json()

    assert stats["total_analyses"] == 2
    assert stats["average_completeness"] == pytest.approx(0.7)
    assert stats["average_confidence"] == pytest.approx(0.8)
    assert stats["category_distribution"] == {"Security": 1, "Performance": 1}
    assert stats["risk_distribution"] == {"High": 1, "Low": 1}
    assert stats["recent_count_7d"] == 2


def test_stats_match_the_history_total(client):
    """The dashboard and the history page must not disagree about the row count."""
    post_analysis(client, requirement="First requirement about authentication.")
    post_analysis(client, requirement="Second requirement about encryption.")

    history = client.get("/api/v1/history").json()
    stats = client.get("/api/v1/stats").json()

    assert stats["total_analyses"] == history["total"] == 2


# --------------------------------------------------------------------------
# Database failures are controlled
# --------------------------------------------------------------------------

def test_history_returns_503_when_database_is_unavailable(client_without_db):
    """No database configured is a 503 with a readable message, not a traceback."""
    for path in ("/api/v1/history", "/api/v1/stats"):
        response = client_without_db.get(path)
        assert response.status_code == 503, path
        assert response.json()["detail"] == "Database is not available"


def test_history_detail_returns_503_when_database_is_unavailable(client_without_db):
    response = client_without_db.get("/api/v1/history/1")

    assert response.status_code == 503
    assert response.json()["detail"] == "Database is not available"


def test_history_returns_500_when_the_query_fails(client):
    """A broken query must surface as a controlled 500 with a generic message."""
    with patch(
        "database.repository.AnalysisRepository.list_recent",
        side_effect=RuntimeError("connection reset"),
    ):
        response = client.get("/api/v1/history")

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error while retrieving history"


def test_stats_returns_500_when_the_query_fails(client):
    with patch(
        "database.repository.AnalysisRepository.get_stats",
        side_effect=RuntimeError("connection reset"),
    ):
        response = client.get("/api/v1/stats")

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error while retrieving stats"


def test_history_detail_returns_500_when_the_query_fails(client):
    with patch(
        "database.repository.AnalysisRepository.get_by_id",
        side_effect=RuntimeError("connection reset"),
    ):
        response = client.get("/api/v1/history/1")

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error while retrieving analysis"


def test_analysis_survives_a_persistence_failure(client, session_maker):
    """
    A dead database must not lose the analysis.

    The pipeline already produced a valid result, so the endpoint degrades to
    "analysed but not stored" instead of throwing the work away. That is the
    documented behaviour in api/routers/analysis.py and in the health endpoint's
    "PostgreSQL being down is degraded" rule.
    """
    with patch(
        "database.repository.AnalysisRepository.save",
        side_effect=RuntimeError("Failed to save analysis: connection reset"),
    ):
        body = post_analysis(client)

    assert body["requirement"] == "The system must authenticate all users."
    assert body["completeness_score"] == 0.65
    # Nothing half-written must be left behind.
    assert count_rows(session_maker) == 0


def test_analysis_succeeds_without_a_database(client_without_db):
    """With no database at all, /analyze still returns the full analysis."""
    body = post_analysis(client_without_db)

    assert body["category"] == "Security"
    assert body["risk_level"] == "Medium"
    assert body["completeness_score"] == 0.65
    assert len(body["similar_guidelines"]) == 1


def test_analysis_rejects_an_empty_requirement_before_the_pipeline(client, session_maker):
    """Empty input is a 422 and must never reach the expensive pipeline."""
    with patch("api.routers.analysis.analyze_requirement") as mock_pipeline:
        response = client.post("/api/v1/analyze", json={"requirement": "   "})

    assert response.status_code == 422
    mock_pipeline.assert_not_called()
    assert count_rows(session_maker) == 0


def test_analysis_rejects_an_over_long_requirement(client, session_maker):
    with patch("api.routers.analysis.analyze_requirement") as mock_pipeline:
        response = client.post("/api/v1/analyze", json={"requirement": "x" * 5001})

    assert response.status_code == 422
    mock_pipeline.assert_not_called()
    assert count_rows(session_maker) == 0