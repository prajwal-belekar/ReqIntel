#!/usr/bin/env python3
"""
Build FAISS index for semantic search.

This script loads the knowledge base, generates embeddings, and builds
a FAISS index for efficient semantic retrieval.
"""

import json
import numpy as np
import faiss
from pathlib import Path
from semantic.embedding import EmbeddingPipeline

def load_knowledge_base(knowledge_base_path):
    """Load guidelines from JSON file."""
    with open(knowledge_base_path, 'r') as f:
        guidelines = json.load(f)
    return guidelines

def build_faiss_index(guidelines, embedding_pipeline):
    """
    Build FAISS index from guidelines.
    
    Uses IndexFlatIP for inner product search with normalized vectors
    (equivalent to cosine similarity).
    
    Note: Embeddings are L2-normalized by EmbeddingPipeline.embed_batch()
    before being added to FAISS IndexFlatIP. This is critical for
    cosine similarity to equal inner product.
    """
    print(f"Generating embeddings for {len(guidelines)} guidelines...")
    
    # Generate text for embedding (combine title and guideline text)
    texts = []
    for guideline in guidelines:
        text = f"{guideline['title']}. {guideline['guideline_text']}"
        texts.append(text)
    
    # Generate embeddings
    embeddings = embedding_pipeline.embed_batch(texts)
    
    # Build FAISS index
    dimension = embeddings.shape[1]
    print(f"Embedding dimension: {dimension}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Use IndexFlatIP for inner product search with normalized vectors
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index, embeddings

def create_metadata(guidelines):
    """Create metadata mapping FAISS positions to guideline IDs."""
    metadata = []
    for idx, guideline in enumerate(guidelines):
        metadata.append({
            'faiss_id': idx,
            'guideline_id': guideline['id'],
            'title': guideline['title'],
            'category': guideline['category'],
            'guideline_text': guideline['guideline_text'],
            'tags': guideline['tags'],
            'source': guideline['source']
        })
    return metadata

def main():
    # Paths
    base_dir = Path(__file__).parent.parent
    knowledge_base_path = base_dir / "data" / "knowledge_base" / "guidelines.json"
    index_path = Path(__file__).parent / "index.faiss"
    metadata_path = Path(__file__).parent / "metadata.json"
    
    # Load knowledge base
    print("Loading knowledge base...")
    guidelines = load_knowledge_base(knowledge_base_path)
    print(f"Loaded {len(guidelines)} guidelines")
    
    # Initialize embedding pipeline
    print("\nInitializing embedding pipeline...")
    embedding_pipeline = EmbeddingPipeline()
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    index, embeddings = build_faiss_index(guidelines, embedding_pipeline)
    
    # Create metadata
    print("\nCreating metadata...")
    metadata = create_metadata(guidelines)
    
    # Save index
    print(f"\nSaving index to {index_path}...")
    faiss.write_index(index, str(index_path))
    
    # Save metadata
    print(f"Saving metadata to {metadata_path}...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n=== Index Build Complete ===")
    print(f"Guidelines indexed: {len(guidelines)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Index type: IndexFlatIP (inner product)")
    print(f"Index saved: {index_path}")
    print(f"Metadata saved: {metadata_path}")

if __name__ == "__main__":
    main()
