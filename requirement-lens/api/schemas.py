"""
Shared Pydantic schemas for the API.
"""

# Note: We're reusing existing schemas from the RAG pipeline where possible
# to avoid duplication. This file can be extended if API-specific schemas are needed.

from rag.pipeline import RequirementAnalysis

# Re-export for convenience
__all__ = ["RequirementAnalysis"]