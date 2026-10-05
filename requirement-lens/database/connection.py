"""
Database connection management for RequirementLens.

Uses SQLAlchemy with psycopg2 driver for PostgreSQL.
Connection is managed via environment variable DATABASE_URL.
"""

import os
import logging
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[Engine] = None
_session_maker: Optional[sessionmaker] = None


def get_database_url() -> str:
    """
    Get the database URL from environment.

    Returns:
        Database connection string

    Raises:
        ValueError: If DATABASE_URL is not set
    """
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError(
            "DATABASE_URL environment variable is not set. "
            "See .env.example for configuration."
        )
    return url


def init_db() -> Engine:
    """
    Initialize the database engine and session factory.

    Creates the engine, tests the connection, and ensures
    the schema is created.

    Returns:
        SQLAlchemy Engine instance

    Raises:
        RuntimeError: If connection fails
    """
    global _engine, _session_maker

    if _engine is not None:
        return _engine

    try:
        database_url = get_database_url()

        # Create engine with appropriate pool settings
        _engine = create_engine(
            database_url,
            pool_pre_ping=True,  # Check connection before using
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600,  # Recycle connections after 1 hour
            echo=False,
        )

        # Test connection
        with _engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        # Create session factory
        _session_maker = sessionmaker(bind=_engine, autocommit=False, autoflush=False)

        logger.info("Database connection established successfully")

        # Ensure schema exists
        from database.models import Base
        Base.metadata.create_all(bind=_engine)
        logger.info("Database schema ensured")

        return _engine

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        _engine = None
        _session_maker = None
        raise RuntimeError(f"Database initialization failed: {str(e)}")


def get_engine() -> Optional[Engine]:
    """
    Get the current database engine.

    Returns:
        Engine instance or None if not initialized
    """
    return _engine


def get_session_maker() -> Optional[sessionmaker]:
    """
    Get the current session factory.

    Returns:
        sessionmaker instance or None if not initialized
    """
    return _session_maker


def get_db_session():
    """
    Get a database session.

    Yields:
        SQLAlchemy Session instance

    Raises:
        RuntimeError: If database is not initialized
    """
    if _session_maker is None:
        raise RuntimeError(
            "Database is not initialized. Call init_db() first."
        )
    session = _session_maker()
    try:
        yield session
    finally:
        session.close()


def check_db_connection() -> dict:
    """
    Check if the database is reachable.

    Returns:
        Dictionary with status and details
    """
    if _engine is None:
        return {"status": "unhealthy", "error": "Database engine not initialized"}

    try:
        with _engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        # Get connection pool info
        pool = _engine.pool
        return {
            "status": "healthy",
            "details": {
                "pool_size": pool.size(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
            }
        }
    except Exception as e:
        logger.warning(f"Database health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


def close_db():
    """
    Close the database connection and clean up resources.
    """
    global _engine, _session_maker
    if _engine is not None:
        _engine.dispose()
        _engine = None
        _session_maker = None
        logger.info("Database connections closed")