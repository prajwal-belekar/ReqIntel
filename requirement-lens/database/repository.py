""" 
Repository layer for analysis history persistence.

Provides clean data-access methods for the AnalysisHistory model.
Does NOT contain any ML/RAG logic.
"""

import logging
from typing import Optional, List

from sqlalchemy import select, func
from sqlalchemy.orm import Session

from database.models import AnalysisHistory, create_analysis_record

logger = logging.getLogger(__name__)


class AnalysisRepository:
    """
    Repository for analysis history data access.

    All methods are stateless and operate on a provided Session.
    """

    def __init__(self, session: Session):
        """
        Initialize repository with a database session.

        Args:
            session: SQLAlchemy Session instance
        """
        self.session = session

    def save(self, analysis_data: dict) -> AnalysisHistory:
        """
        Save a new analysis result to the database.

        Args:
            analysis_data: Dictionary with analysis result fields

        Returns:
            The created AnalysisHistory instance (with id populated)

        Raises:
            RuntimeError: If save fails
        """
        try:
            record = create_analysis_record(analysis_data)
            self.session.add(record)
            self.session.commit()
            self.session.refresh(record)
            logger.debug(f"Saved analysis record with id={record.id}")
            return record
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to save analysis record: {e}")
            raise RuntimeError(f"Failed to save analysis: {str(e)}")

    def get_by_id(self, analysis_id: int) -> Optional[AnalysisHistory]:
        """
        Retrieve a single analysis by its ID.

        Args:
            analysis_id: The analysis record ID

        Returns:
            AnalysisHistory instance or None if not found
        """
        return self.session.get(AnalysisHistory, analysis_id)

    def list_recent(self, limit: int = 20, offset: int = 0) -> List[AnalysisHistory]:
        """
        List recent analyses with pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of AnalysisHistory instances ordered by created_at descending
        """
        stmt = (
            select(AnalysisHistory)
            .order_by(AnalysisHistory.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.session.execute(stmt).scalars().all())

    def count(self) -> int:
        """
        Get the total count of analysis records.

        Returns:
            Total number of records
        """
        stmt = select(func.count()).select_from(AnalysisHistory)
        return self.session.execute(stmt).scalar_one()

    def delete(self, analysis_id: int) -> bool:
        """
        Delete an analysis record by ID.

        Args:
            analysis_id: The analysis record ID

        Returns:
            True if deleted, False if not found
        """
        record = self.session.get(AnalysisHistory, analysis_id)
        if record is None:
            return False
        self.session.delete(record)
        self.session.commit()
        return True

    def get_stats(self) -> dict:
        """
        Get aggregate statistics for analysis history.

        Returns:
            Dictionary with total count, average scores, and distributions
        """
        total = self.count()
        if total == 0:
            return {
                "total_analyses": 0,
                "average_completeness": 0.0,
                "average_confidence": 0.0,
                "category_distribution": {},
                "risk_distribution": {},
                "recent_count_7d": 0,
            }

        avg_completeness = self.session.execute(
            select(func.avg(AnalysisHistory.completeness_score))
        ).scalar_one() or 0.0

        avg_confidence = self.session.execute(
            select(func.avg(AnalysisHistory.ml_confidence))
        ).scalar_one() or 0.0

        category_counts = self.session.execute(
            select(AnalysisHistory.category, func.count(AnalysisHistory.id))
            .group_by(AnalysisHistory.category)
        ).all()
        category_distribution = {cat: int(cnt) for cat, cnt in category_counts}

        risk_counts = self.session.execute(
            select(AnalysisHistory.risk_level, func.count(AnalysisHistory.id))
            .group_by(AnalysisHistory.risk_level)
        ).all()
        risk_distribution = {risk: int(cnt) for risk, cnt in risk_counts}

        from datetime import datetime, timedelta
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_count = self.session.execute(
            select(func.count(AnalysisHistory.id)).where(
                AnalysisHistory.created_at >= seven_days_ago
            )
        ).scalar_one()

        return {
            "total_analyses": total,
            "average_completeness": round(float(avg_completeness), 4),
            "average_confidence": round(float(avg_confidence), 4),
            "category_distribution": category_distribution,
            "risk_distribution": risk_distribution,
            "recent_count_7d": int(recent_count),
        }

