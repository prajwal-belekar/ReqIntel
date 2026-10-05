"""
Analysis router for RequirementLens API.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator
import logging
from rag.pipeline import analyze_requirement, RequirementAnalysis
from database.connection import get_engine, get_session_maker
from database.repository import AnalysisRepository
from sqlalchemy.orm import Session

router = APIRouter()
logger = logging.getLogger(__name__)


# Request model
class AnalysisRequest(BaseModel):
    requirement: str = Field(..., min_length=1, max_length=5000, description="The software requirement to analyze")

    @validator('requirement')
    def requirement_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('requirement must not be empty or whitespace only')
        return v.strip()


# Response model - we'll reuse the existing RequirementAnalysis from the pipeline
# but we need to make sure it's compatible with FastAPI's JSON serialization
class AnalysisResponse(BaseModel):
    requirement: str
    category: str
    ml_confidence: float
    risk_level: str
    risk_factors: list[str]
    completeness_score: float
    missing_requirements: list[str]
    recommendations: list[str]
    similar_guidelines: list[dict]
    explanation: str

    class Config:
        # Allow population by field name or alias
        allow_population_by_field_name = True


def get_analysis_repository():
    """
    Dependency to get an AnalysisRepository instance.
    Returns None if database is not available.
    """
    session_maker = get_session_maker()
    if session_maker is None:
        return None
    session = session_maker()
    return AnalysisRepository(session), session


@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_requirement_endpoint(request: AnalysisRequest):
    """
    Analyze a software requirement using the RequirementLens RAG pipeline.

    This endpoint combines:
    1. ML classification (TF-IDF + Logistic Regression)
    2. Semantic retrieval (FAISS with all-MiniLM-L6-v2 embeddings)
    3. LLM reasoning (Ollama)

    Returns a structured analysis including:
    - ML predicted category and confidence
    - Risk assessment
    - Completeness score
    - Missing specifications
    - Recommendations
    - Similar guidelines from the knowledge base

    The analysis is persisted to PostgreSQL for history tracking.
    """
    try:
        logger.info(f"Received analysis request: {request.requirement[:100]}...")

        # Call the existing RAG pipeline
        result: RequirementAnalysis = analyze_requirement(request.requirement)

        # Convert the RequirementAnalysis object to a dictionary for the response
        # We need to handle the similar_guidelines which are GuidelineReference objects
        response_data = {
            "requirement": result.requirement,
            "category": result.category,
            "ml_confidence": result.ml_confidence,
            "risk_level": result.risk_level.value if hasattr(result.risk_level, 'value') else str(result.risk_level),
            "risk_factors": result.risk_factors,
            "completeness_score": result.completeness_score,
            "missing_requirements": result.missing_requirements,
            "recommendations": result.recommendations,
            "similar_guidelines": [
                {
                    "id": guideline.id,
                    "title": guideline.title,
                    "similarity": guideline.similarity,
                    "category": guideline.category,
                    "text": guideline.text
                }
                for guideline in result.similar_guidelines
            ],
            "explanation": result.explanation
        }

        # Save to database (don't fail the request if DB is unavailable)
        session_maker = get_session_maker()
        if session_maker is not None:
            session = session_maker()
            try:
                repo = AnalysisRepository(session)
                saved_record = repo.save(response_data)
                logger.info(f"Analysis saved to database with id={saved_record.id}")
            except Exception as db_e:
                logger.warning(f"Failed to save analysis to database: {db_e}")
                session.rollback()
                # Don't fail the request - analysis is still valid
            finally:
                session.close()
        else:
            logger.debug("Database not available, skipping persistence")

        logger.info(f"Analysis completed successfully. Category: {response_data['category']}, Risk: {response_data['risk_level']}")

        return AnalysisResponse(**response_data)

    except ValueError as e:
        # Handle validation errors from the pipeline
        logger.warning(f"Validation error in analysis request: {e}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        # Handle any other errors
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")