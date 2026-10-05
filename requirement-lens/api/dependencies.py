"""
Dependencies for the API.
"""

# This file can be used for shared dependencies like database connections,
# authentication, etc. For now, it's empty as we don't have those components yet.

def get_rag_pipeline():
    """
    Dependency to get the RAG pipeline instance.
    In a more complex application, this might handle connection pooling,
    but for our stateless API, we can just create a new instance each time.
    """
    from rag.pipeline import RAGPipeline
    return RAGPipeline()