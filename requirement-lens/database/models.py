"""
SQLAlchemy ORM models for RequirementLens database.

Defines the analysis_history table for persisting analysis results.
"""

import json
from datetime import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    JSON,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AnalysisHistory(Base):
    """
    ORM model for the analysis_history table.

    Stores complete analysis results from the RAG pipeline.
    Nested arrays/objects are stored as JSONB columns.
    """

    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    requirement = Column(Text, nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    ml_confidence = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False, index=True)
    risk_factors = Column(JSON, nullable=False, default=list)
    completeness_score = Column(Float, nullable=False)
    missing_requirements = Column(JSON, nullable=False, default=list)
    recommendations = Column(JSON, nullable=False, default=list)
    similar_guidelines = Column(JSON, nullable=False, default=list)
    explanation = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    def to_dict(self) -> dict:
        """
        Convert the model instance to a dictionary suitable for JSON serialization.

        Returns:
            Dictionary representation of this analysis record
        """
        return {
            "id": self.id,
            "requirement": self.requirement,
            "category": self.category,
            "ml_confidence": self.ml_confidence,
            "risk_level": self.risk_level,
            "risk_factors": self.risk_factors or [],
            "completeness_score": self.completeness_score,
            "missing_requirements": self.missing_requirements or [],
            "recommendations": self.recommendations or [],
            "similar_guidelines": self.similar_guidelines or [],
            "explanation": self.explanation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


def create_analysis_record(data: dict) -> AnalysisHistory:
    """
    Create an AnalysisHistory ORM instance from a dictionary of analysis results.

    Args:
        data: Dictionary containing analysis result fields

    Returns:
        AnalysisHistory instance ready to be added to a session
    """
    return AnalysisHistory(
        requirement=data.get("requirement", ""),
        category=data.get("category", ""),
        ml_confidence=float(data.get("ml_confidence", 0.0)),
        risk_level=str(data.get("risk_level", "Low")),
        risk_factors=list(data.get("risk_factors", [])),
        completeness_score=float(data.get("completeness_score", 0.0)),
        missing_requirements=list(data.get("missing_requirements", [])),
        recommendations=list(data.get("recommendations", [])),
        similar_guidelines=list(data.get("similar_guidelines", [])),
        explanation=str(data.get("explanation", "")),
    )