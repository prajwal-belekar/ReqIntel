"""
Basic test for the RequirementLens API.
"""

import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test that all API modules can be imported."""
    from api.main import app
    from api.routers import health, analysis
    from api.schemas import __all__ as schemas_all
    from api.dependencies import get_rag_pipeline

    assert app is not None
    assert callable(get_rag_pipeline)
    assert isinstance(schemas_all, (list, tuple))


def test_health_router():
    """Test the health router functions."""
    from api.routers.health import check_ml_model, check_faiss, check_knowledge_base, check_ollama

    # We won't actually run these checks as they require external services
    # but we can verify the functions exist
    assert callable(check_ml_model)
    assert callable(check_faiss)
    assert callable(check_knowledge_base)
    assert callable(check_ollama)


def test_analysis_router():
    """Test the analysis router functions."""
    from api.routers.analysis import AnalysisRequest, AnalysisResponse

    # Test that we can create instances
    req = AnalysisRequest(requirement="Test requirement")
    assert req.requirement == "Test requirement"


def test_app_creation():
    """Test that the FastAPI app is created correctly."""
    from api.main import app
    from fastapi import FastAPI

    assert isinstance(app, FastAPI)
    assert app.title == "RequirementLens API"
    assert app.version == "1.0.0"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))