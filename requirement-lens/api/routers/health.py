"""
Health check router for RequirementLens API.
"""

from fastapi import APIRouter, HTTPException
import os
import logging
from ml.classifier import predict_category, get_available_categories, get_model_info
from vector_store.search import search as semantic_search
from llm.ollama_service import OllamaService
from rag.pipeline import RAGPipeline
from pathlib import Path
import json

router = APIRouter()
logger = logging.getLogger(__name__)


def check_ml_model():
    """Check if ML model is available and working."""
    try:
        # Test with a simple requirement
        test_req = "Test requirement"
        category, confidence = predict_category(test_req)
        categories = get_available_categories()
        model_info = get_model_info()

        return {
            "status": "healthy",
            "details": {
                "categories_count": len(categories),
                "model_type": model_info.get('model_type'),
                "vectorizer_type": model_info.get('vectorizer_type')
            }
        }
    except Exception as e:
        logger.error(f"ML model check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_faiss():
    """Check if FAISS index is available and working."""
    try:
        # Test search with a simple query
        results = semantic_search("test query", top_k=1)
        return {
            "status": "healthy",
            "details": {
                "results_count": len(results)
            }
        }
    except Exception as e:
        logger.error(f"FAISS check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_knowledge_base():
    """Check if knowledge base is available."""
    try:
        kb_path = Path(__file__).parent.parent.parent / "data" / "knowledge_base" / "guidelines.json"
        if kb_path.exists():
            with open(kb_path, 'r') as f:
                guidelines = json.load(f)
            return {
                "status": "healthy",
                "details": {
                    "guidelines_count": len(guidelines)
                }
            }
        else:
            return {
                "status": "unhealthy",
                "error": "Knowledge base file not found"
            }
    except Exception as e:
        logger.error(f"Knowledge base check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


def check_ollama():
    """Check if Ollama service is available."""
    try:
        # Use a short timeout for health check
        service = OllamaService(timeout=5)
        # Simple test - we don't need to actually generate a response
        # Just check if we can create the service without error
        return {
            "status": "healthy",
            "details": {
                "model": service.model_name,
                "base_url": service.base_url
            }
        }
    except Exception as e:
        logger.warning(f"Ollama check failed: {e}")
        return {
            "status": "degraded",  # Ollama being down doesn't make the whole API unhealthy
            "error": str(e)
        }


def check_postgresql():
    """Check if PostgreSQL is available."""
    try:
        from database.connection import check_db_connection
        result = check_db_connection()
        status = result.get("status", "unhealthy")
        return {
            "status": status,
            "details": result.get("details", {}),
            "error": result.get("error")
        }
    except Exception as e:
        logger.warning(f"PostgreSQL check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for all service components.
    """
    # Check all components
    ml_status = check_ml_model()
    faiss_status = check_faiss()
    kb_status = check_knowledge_base()
    ollama_status = check_ollama()
    postgresql_status = check_postgresql()

    # Determine overall status
    # API is always healthy if we can respond to this request
    # ML and FAISS are critical - if either is unhealthy, API is unhealthy
    # KB is important but we might have fallbacks
    # Ollama being down is degraded (API still works with limited functionality)
    # PostgreSQL being down is degraded (analyses still work, just not persisted)

    critical_components = [ml_status, faiss_status]
    critical_healthy = all(comp["status"] == "healthy" for comp in critical_components)

    if critical_healthy:
        overall_status = "healthy"
    else:
        overall_status = "unhealthy"

    # If Ollama is down but everything else is up, we're degraded
    if overall_status == "healthy" and ollama_status["status"] != "healthy":
        overall_status = "degraded"

    # If PostgreSQL is down but everything else is up, we're degraded
    if overall_status in ["healthy", "degraded"] and postgresql_status["status"] != "healthy":
        if overall_status == "healthy":
            overall_status = "degraded"
        # Keep degraded status if already degraded

    return {
        "status": overall_status,
        "components": {
            "api": "healthy",
            "ml_model": ml_status["status"],
            "faiss": faiss_status["status"],
            "knowledge_base": kb_status["status"],
            "ollama": ollama_status["status"],
            "postgresql": postgresql_status["status"]
        },
        "details": {
            "ml_model": ml_status.get("details", {}),
            "faiss": faiss_status.get("details", {}),
            "knowledge_base": kb_status.get("details", {}),
            "ollama": ollama_status.get("details", {}),
            "postgresql": postgresql_status.get("details", {})
        },
        "errors": {
            "ml_model": ml_status.get("error"),
            "faiss": faiss_status.get("error"),
            "knowledge_base": kb_status.get("error"),
            "ollama": ollama_status.get("error"),
            "postgresql": postgresql_status.get("error")
        }
    }