#!/usr/bin/env python3
"""
Embedding module for RequirementLens ML pipeline.

This module provides access to semantic embeddings using Sentence Transformers.
It re-exports the embedding pipeline from semantic.embedding for consistency.
"""

from semantic.embedding import EmbeddingPipeline, get_embedding_info

__all__ = ['EmbeddingPipeline', 'get_embedding_info']


def main():
    """Quick test of embedding pipeline."""
    pipeline = EmbeddingPipeline()
    
    test_texts = [
        "Users should be locked out after multiple failed passwords.",
        "The system must respond within 2 seconds.",
        "The interface should be intuitive for new users."
    ]
    
    print("Testing embedding pipeline:")
    for text in test_texts:
        embedding = pipeline.embed_text(text)
        print(f"\nText: {text}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Norm: {embedding.norm():.4f}")
    
    get_embedding_info()


if __name__ == "__main__":
    main()
