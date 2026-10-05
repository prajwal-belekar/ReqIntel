"""
FastAPI application for RequirementLens.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env before anything reads os.getenv(...). Without this the file is
# inert: DATABASE_URL, OLLAMA_* and FRONTEND_ORIGIN would only ever come from
# the real process environment, so a .env copied from .env.example would
# silently not take effect. Real environment variables still win.
load_dotenv()

# Import routers
from api.routers import health, analysis, history

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Application metadata
SERVICE_NAME = "RequirementLens API"
SERVICE_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    logger.info(f"Starting {SERVICE_NAME} v{SERVICE_VERSION}")

    # Initialize database on startup (optional, will fail gracefully if unavailable)
    try:
        from database.connection import init_db
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed (API will continue without persistence): {e}")

    yield

    # Cleanup on shutdown
    try:
        from database.connection import close_db
        close_db()
    except Exception as e:
        logger.warning(f"Database cleanup failed: {e}")

    logger.info(f"Shutting down {SERVICE_NAME}")


# Create FastAPI application
app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description="API for software requirement analysis using ML classification and semantic retrieval",
    lifespan=lifespan
)

# Configure CORS
frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint returning basic API information.
    """
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "status": "online"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)