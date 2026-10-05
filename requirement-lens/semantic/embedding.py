#!/usr/bin/env python3
"""
Semantic Embedding Pipeline for RequirementLens

This module provides a reusable embedding pipeline using Sentence Transformers.
"""

import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer


class EmbeddingPipeline:
    """
    Reusable embedding pipeline using all-MiniLM-L6-v2.
    Generates normalized embeddings suitable for cosine-similarity retrieval.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding pipeline.
        
        Args:
            model_name: Sentence Transformer model name
        """
        print(f"Loading model {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate normalized embedding for a single text string.
        
        Args:
            text: Input requirement text
            
        Returns:
            Normalized embedding vector (numpy array)
        """
        if not text or not isinstance(text, str) or not text.strip():
            print("Warning: Empty or invalid input received. Returning zero vector.")
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.astype(np.float32)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate normalized embeddings for a batch of texts.
        
        Args:
            texts: List of input requirement texts
            
        Returns:
            Normalized embedding matrix (numpy array)
        """
        if not texts:
            print("Warning: Empty list received. Returning empty array.")
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            print("Warning: No valid texts found. Returning zero matrix.")
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        try:
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )
            
            result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            for idx, emb_idx in zip(valid_indices, range(len(valid_texts))):
                result[idx] = embeddings[emb_idx]
            
            return result
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)


def get_embedding_info():
    """Return information about the embedding configuration."""
    info = """
    Embedding Configuration
    ======================
    
    What is an Embedding?
    ---------------------
    An embedding is a dense vector representation of text in a continuous vector space.
    Unlike traditional bag-of-words or TF-IDF, embeddings capture semantic meaning and
    relationships between concepts. Words with similar meanings are positioned close together
    in the vector space.
    
    Why Sentence Transformers?
    --------------------------
    Sentence Transformers are designed specifically for encoding sentences/documents
    into dense vectors. They use siamese/triplet networks to optimize for semantic similarity,
    making them ideal for tasks like semantic search and retrieval.
    
    Why all-MiniLM-L6-v2?
    ----------------------
    - Lightweight: Only 22MB model size, CPU-friendly
    - Fast: ~300 words/second on CPU
    - Good balance of quality vs. speed for semantic retrieval
    - Pre-trained on large corpus, captures semantic relationships well
    - 384-dimensional output
    - L6 (6 layers) provides good accuracy with minimal computational cost
    
    Embedding Dimensionality
    -----------------------
    - all-MiniLM-L6-v2: 384 dimensions
    - Each dimension captures a different semantic aspect of the text
    - Higher dimensionality = more expressive, but slower and more memory
    
    Why Normalization?
    ------------------
    Normalization (L2 normalization) is critical for cosine similarity:
    
    Cosine similarity = dot product of normalized vectors
    = cos(theta) where theta is angle between vectors
    
    Benefits:
    - Enables fast similarity search using L2 distance
    - Makes magnitude irrelevant (only direction matters)
    - FAISS FlatIP index works optimally with normalized vectors
    - Equivalent to cosine similarity when using Inner Product
    """
    print(info)
    return info


if __name__ == "__main__":
    # Quick test
    pipeline = EmbeddingPipeline()
    
    test_texts = [
        "The system shall authenticate users before accessing sensitive data",
        "Users should be able to log in with username and password",
        "",
        "   ",
        None
    ]
    
    print("\nTesting single text embedding:")
    emb = pipeline.embed_text(test_texts[0])
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding norm: {np.linalg.norm(emb):.4f}")
    
    print("\nTesting batch embedding:")
    embeddings = pipeline.embed_batch(test_texts)
    print(f"Batch shape: {embeddings.shape}")
    print(f"All norms close to 1: {np.allclose(np.linalg.norm(embeddings[0]), 1.0)}")
    
    get_embedding_info()
