"""
Tests for RequirementLens database functionality.

Tests the database layer created for Mission 7.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from sqlalchemy import text
from database.connection import check_db_connection, init_db, get_engine, get_session_maker
from database.models import AnalysisHistory
from database.repository import AnalysisRepository


def test_database_connection_check():
    """Test that the database connection check works."""
    result = check_db_connection()

    # If database is available, should return healthy status
    # If not available, should return unhealthy status
    assert "status" in result
    assert result["status"] in ["healthy", "unhealthy"]
    assert "error" in result or "details" in result


def test_database_init_on_import():
    """Test that database initialization happens when the module is imported."""
# This tests that the connection can be established when imported
    from database.connection import get_engine
    engine = get_engine()
    session_maker = get_session_maker()

    # Engine and session factory must agree: both unavailable, or both ready.
    # Previously every exception was swallowed here, so a genuine failure in
    # the connection layer reported as a passing test.
    assert (engine is None) == (session_maker is None), (
        "engine and session_maker must both be None or both be initialised"
    )

    if engine is None:
        # No DATABASE_URL configured; the API is expected to degrade gracefully.
        print("Database not configured, which is acceptable")
        return

    print(f"Database engine initialized: {type(engine)}")

    # Test that we can create a session and execute a simple query
    with session_maker() as session:
        result = session.execute(text("SELECT 1")).fetchall()
        assert len(result) == 1
        assert result[0][0] == 1


def test_analysis_history_model():
    """Test that the AnalysisHistory model can be instantiated."""
    # Create an analysis record
    analysis = AnalysisHistory(
        requirement="Test requirement",
        category="Security",
        ml_confidence=0.85,
        risk_level="Medium",
        risk_factors=["Potential vulnerability"],
        completeness_score=0.75,
        missing_requirements=["Authentication"],
        recommendations=["Add authentication"],
        similar_guidelines=[
            {
                "id": "SEC-001",
                "title": "Authentication",
                "similarity": 0.9,
                "category": "Security",
                "text": "All user input must be validated"
            }
        ],
        explanation="Test explanation"
    )

    # Verify attributes
    assert analysis.requirement == "Test requirement"
    assert analysis.category == "Security"
    assert analysis.ml_confidence == 0.85
    assert analysis.risk_level == "Medium"
    assert analysis.risk_factors == ["Potential vulnerability"]
    assert analysis.completeness_score == 0.75
    assert len(analysis.missing_requirements) == 1
    assert len(analysis.recommendations) == 1
    assert len(analysis.similar_guidelines) == 1
    assert analysis.explanation == "Test explanation"


def test_analysis_history_to_dict():
    """Test that the AnalysisHistory model can be converted to a dictionary."""
    # Create an analysis record
    analysis = AnalysisHistory(
        id=1,
        requirement="Test requirement",
        category="Security",
        ml_confidence=0.85,
        risk_level="Medium",
        risk_factors=["Potential vulnerability"],
        completeness_score=0.75,
        missing_requirements=["Authentication"],
        recommendations=["Add authentication"],
        similar_guidelines=[
            {
                "id": "SEC-001",
                "title": "Authentication",
                "similarity": 0.9,
                "category": "Security",
                "text": "All user input must be validated"
            }
        ],
        explanation="Test explanation"
    )

    # Convert to dictionary
    data = analysis.to_dict()

    # Verify keys
    expected_keys = [
        "id", "requirement", "category", "ml_confidence", "risk_level",
        "risk_factors", "completeness_score", "missing_requirements",
        "recommendations", "similar_guidelines", "explanation", "created_at"
    ]
    
    for key in expected_keys:
        assert key in data, f"Missing key: {key}"

    # Verify values
    assert data["id"] == 1
    assert data["requirement"] == "Test requirement"
    assert data["category"] == "Security"
    assert data["ml_confidence"] == 0.85
    assert data["risk_level"] == "Medium"
    assert data["risk_factors"] == ["Potential vulnerability"]
    assert data["completeness_score"] == 0.75
    assert data["missing_requirements"] == ["Authentication"]
    assert data["recommendations"] == ["Add authentication"]
    assert len(data["similar_guidelines"]) == 1
    assert data["explanation"] == "Test explanation"


def test_database_environment_variable():
    """Test that the DATABASE_URL environment variable is required."""
    # This test verifies that the database connection code properly checks
    # for the DATABASE_URL environment variable
    import database.connection as db_connection
    
    # Save the original environment
    original_db_url = os.getenv("DATABASE_URL")
    
    try:
        # Test 1: When DATABASE_URL is not set
        if "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]
        
        # Try to get the database URL
        try:
            db_url = db_connection.get_database_url()
            print("ERROR: Should have raised ValueError for missing DATABASE_URL")
            assert False, "Should have raised ValueError for missing DATABASE_URL"
        except ValueError as e:
            assert "DATABASE_URL" in str(e)
            print(f"Correctly raised error: {e}")
        
        # Test 2: When DATABASE_URL is set
        os.environ["DATABASE_URL"] = "postgresql://test:test@localhost:5432/test"
        db_url = db_connection.get_database_url()
        assert db_url == "postgresql://test:test@localhost:5432/test"
        print(f"Correctly retrieved DATABASE_URL: {db_url}")
        
    finally:
        # Restore the original environment
        if original_db_url is not None:
            os.environ["DATABASE_URL"] = original_db_url
        elif "DATABASE_URL" in os.environ:
            del os.environ["DATABASE_URL"]


def test_repository_get_stats_empty():
    """Test get_stats returns empty stats when no data."""
    from unittest.mock import MagicMock
    session = MagicMock()
    repo = AnalysisRepository(session)
    
    # Mock count to return 0
    repo.count = MagicMock(return_value=0)
    
    stats = repo.get_stats()
    
    assert stats["total_analyses"] == 0
    assert stats["average_completeness"] == 0.0
    assert stats["average_confidence"] == 0.0
    assert stats["category_distribution"] == {}
    assert stats["risk_distribution"] == {}
    assert stats["recent_count_7d"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

