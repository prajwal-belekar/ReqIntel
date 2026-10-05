""" 
History router for RequirementLens API.

Provides endpoints for retrieving past analyses from PostgreSQL.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime

from database.connection import get_session_maker, get_engine
from database.repository import AnalysisRepository

router = APIRouter()
logger = logging.getLogger(__name__)


# Response models
class HistoryItemResponse(BaseModel):
    id: int
    requirement: str
    category: str
    ml_confidence: float
    risk_level: str
    completeness_score: float
    created_at: str


class AnalysisHistoryResponse(BaseModel):
    id: int
    requirement: str
    category: str
    ml_confidence: float
    risk_level: str
    risk_factors: List[str]
    completeness_score: float
    missing_requirements: List[str]
    recommendations: List[str]
    similar_guidelines: List[dict]
    explanation: str
    created_at: str


def _get_repository():
    """
    Get a repository instance with session.

    Returns:
        Tuple of (repository, session) or (None, None) if DB unavailable
    """
    session_maker = get_session_maker()
    if session_maker is None:
        return None, None
    session = session_maker()
    return AnalysisRepository(session), session


@router.get("/history", response_model=dict, tags=["History"])
async def get_history(
    limit: int = Query(20, ge=1, le=100, description="Number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip")
):
    """
    Retrieve recent analyses from history.

    Returns a paginated list of recent analyses with basic information.

    Query parameters:
    - limit: Maximum number of records to return (1-100, default 20)
    - offset: Number of records to skip for pagination (default 0)
    """
    try:
        repo, session = _get_repository()
        if repo is None:
            raise HTTPException(
                status_code=503,
                detail="Database is not available"
            )

        try:
            records = repo.list_recent(limit=limit, offset=offset)
            total = repo.count()

            items = [
                {
                    "id": record.id,
                    "requirement": record.requirement,
                    "category": record.category,
                    "ml_confidence": record.ml_confidence,
                    "risk_level": record.risk_level,
                    "completeness_score": record.completeness_score,
                    "created_at": record.created_at.isoformat() if record.created_at else None
                }
                for record in records
            ]

            return {
                "items": items,
                "total": total,
                "limit": limit,
                "offset": offset
            }
        finally:
            session.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving history"
        )


@router.get("/history/{analysis_id}", response_model=AnalysisHistoryResponse, tags=["History"])
async def get_analysis(analysis_id: int):
    """
    Retrieve a specific analysis by its ID.

    Returns the complete analysis result including all details.
    """
    try:
        repo, session = _get_repository()
        if repo is None:
            raise HTTPException(
                status_code=503,
                detail="Database is not available"
            )

        try:
            record = repo.get_by_id(analysis_id)
            if record is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Analysis with id {analysis_id} not found"
                )

            return {
                "id": record.id,
                "requirement": record.requirement,
                "category": record.category,
                "ml_confidence": record.ml_confidence,
                "risk_level": record.risk_level,
                "risk_factors": record.risk_factors or [],
                "completeness_score": record.completeness_score,
                "missing_requirements": record.missing_requirements or [],
                "recommendations": record.recommendations or [],
                "similar_guidelines": record.similar_guidelines or [],
                "explanation": record.explanation,
                "created_at": record.created_at.isoformat() if record.created_at else None
            }
        finally:
            session.close()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis {analysis_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving analysis"
        )


@router.get("/stats", response_model=dict, tags=["History"])
async def get_stats():
    """
    Retrieve aggregate statistics for analysis history.

    Returns total analyses, average scores, and distributions by category and risk.
    """
    try:
        repo, session = _get_repository()
        if repo is None:
            raise HTTPException(
                status_code=503,
                detail="Database is not available"
            )

        try:
            stats = repo.get_stats()
            return stats
        finally:
            session.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while retrieving stats"
        )

