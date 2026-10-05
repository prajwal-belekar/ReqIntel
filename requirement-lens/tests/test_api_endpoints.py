"""
Test for the RequirementLens API endpoints using TestClient.
"""

import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_root_endpoint():
    """Test the root endpoint."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "RequirementLens API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "online"


def test_health_endpoint():
    """Test the health endpoint with mocked dependencies."""
    from fastapi.testclient import TestClient
    from api.main import app

    # Mock every component check so the aggregate status is deterministic.
    # check_postgresql must be mocked too: DATABASE_URL is unset in this
    # environment, so the real check reports unhealthy and downgrades the
    # overall status to "degraded".
    with patch('api.routers.health.check_ml_model') as mock_ml, \
         patch('api.routers.health.check_faiss') as mock_faiss, \
         patch('api.routers.health.check_knowledge_base') as mock_kb, \
         patch('api.routers.health.check_ollama') as mock_ollama, \
         patch('api.routers.health.check_postgresql') as mock_pg:

        # Configure mocks to return healthy status
        mock_ml.return_value = {"status": "healthy", "details": {}}
        mock_faiss.return_value = {"status": "healthy", "details": {}}
        mock_kb.return_value = {"status": "healthy", "details": {}}
        mock_ollama.return_value = {"status": "healthy", "details": {}}
        mock_pg.return_value = {"status": "healthy", "details": {}}

        client = TestClient(app)
        response = client.get("/api/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["components"]["api"] == "healthy"
        assert data["components"]["ml_model"] == "healthy"
        assert data["components"]["faiss"] == "healthy"
        assert data["components"]["knowledge_base"] == "healthy"
        assert data["components"]["ollama"] == "healthy"
        assert data["components"]["postgresql"] == "healthy"


def test_analyze_endpoint_valid_request():
    """Test the analyze endpoint with a valid request."""
    from fastapi.testclient import TestClient
    from api.main import app
    from rag.pipeline import RequirementAnalysis

    # Mock the analyze_requirement function to return a valid response
    mock_result = RequirementAnalysis(
        requirement="Test requirement",
        category="Security",
        ml_confidence=0.9,
        risk_level="Medium",
        risk_factors=["No authentication specified"],
        completeness_score=0.7,
        missing_requirements=["Authentication method"],
        recommendations=["Add authentication mechanism"],
        similar_guidelines=[{
            "id": "SEC-001",
            "title": "Input Validation",
            "similarity": 0.8,
            "category": "Security",
            "text": "All user input must be validated"
        }],
        explanation="Test explanation"
    )

    with patch('api.routers.analysis.analyze_requirement', return_value=mock_result):
        client = TestClient(app)
        response = client.post("/api/v1/analyze", json={"requirement": "Test requirement"})

        assert response.status_code == 200
        data = response.json()
        assert data["requirement"] == "Test requirement"
        assert data["category"] == "Security"
        assert data["ml_confidence"] == 0.9
        assert data["risk_level"] == "Medium"
        assert "No authentication specified" in data["risk_factors"]
        assert data["completeness_score"] == 0.7
        assert "Authentication method" in data["missing_requirements"]
        assert "Add authentication mechanism" in data["recommendations"]
        assert len(data["similar_guidelines"]) == 1
        assert data["explanation"] == "Test explanation"


def test_analyze_endpoint_empty_request():
    """Test the analyze endpoint with an empty request."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    response = client.post("/api/v1/analyze", json={"requirement": ""})

    assert response.status_code == 422  # Validation error
    data = response.json()
    assert "detail" in data


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))