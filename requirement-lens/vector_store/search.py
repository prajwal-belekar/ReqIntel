#!/usr/bin/env python3
"""
Semantic search module for RequirementLens.

Provides search functionality over the FAISS index for finding
semantically similar software engineering guidelines.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from semantic.embedding import EmbeddingPipeline
from typing import List, Dict, Any

class SemanticSearch:
    """
    Semantic search over software engineering guidelines using FAISS.
    """
    
    def __init__(self, index_path=None, metadata_path=None):
        """
        Initialize semantic search.
        
        Args:
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSON file
        """
        if index_path is None:
            index_path = Path(__file__).parent / "index.faiss"
        if metadata_path is None:
            metadata_path = Path(__file__).parent / "metadata.json"
        
        # Load FAISS index
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize embedding pipeline
        print("Initializing embedding pipeline...")
        self.embedding_pipeline = EmbeddingPipeline()
        
        print(f"Semantic search ready with {self.index.ntotal} guidelines")
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for semantically similar guidelines.
        
        Args:
            query: Requirement text to search for
            top_k: Number of results to return
            
        Returns:
            List of matching guidelines with similarity scores
        """
        if not query or not isinstance(query, str) or not query.strip():
            print("Warning: Empty or invalid query received")
            return []
        
        # Generate embedding for query
        # EmbeddingPipeline.embed_text() returns L2-normalized embeddings
        query_embedding = self.embedding_pipeline.embed_text(query)
        
        # Search FAISS index
        # Using inner product search (cosine similarity for normalized vectors)
        # Because embeddings are L2-normalized, inner product equals cosine similarity
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            top_k
        )
        
        # Format results
        # Note: Similarity scores indicate relative semantic closeness between
        # the query and retrieved vectors. They should not be interpreted as
        # probability or confidence values without experimental calibration.
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                guideline = self.metadata[idx]
                results.append({
                    'guideline_id': guideline['guideline_id'],
                    'title': guideline['title'],
                    'category': guideline['category'],
                    'similarity_score': float(score),
                    'guideline_text': guideline['guideline_text'],
                    'tags': guideline['tags'],
                    'source': guideline['source']
                })
        
        return results
    
    def search_with_details(self, query: str, top_k: int = 3) -> str:
        """
        Search and return formatted results as string.
        
        Args:
            query: Requirement text to search for
            top_k: Number of results to return
            
        Returns:
            Formatted string with search results
        """
        results = self.search(query, top_k)
        
        if not results:
            return "No results found."
        
        output = f"\nQuery: \"{query}\"\n"
        output += "=" * 80 + "\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"   Category: {result['category']}\n"
            output += f"   Similarity: {result['similarity_score']:.3f}\n"
            output += f"   Text: {result['guideline_text'][:150]}...\n"
            output += f"   Tags: {', '.join(result['tags'])}\n\n"
        
        return output


def search(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Convenience function for semantic search.
    
    Args:
        query: Requirement text to search for
        top_k: Number of results to return
        
    Returns:
        List of matching guidelines
    """
    searcher = SemanticSearch()
    return searcher.search(query, top_k)


def main():
    """Demo semantic search."""
    print("=== Semantic Search Demo ===\n")
    
    searcher = SemanticSearch()
    
    # Test queries
    test_queries = [
        "Users should be locked out after multiple failed passwords.",
        "The system must respond within 2 seconds to user requests.",
        "The application should be easy to use for new users.",
        "System should be available 99.9% of the time during business hours.",
        "The system should handle increased load by adding more servers.",
        "",
        "Some random unrelated text about cooking recipes."
    ]
    
    for query in test_queries:
        print(searcher.search_with_details(query, top_k=3))
        print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
