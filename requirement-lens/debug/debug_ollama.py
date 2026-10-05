#!/usr/bin/env python3
"""
Debug script to test Ollama interaction with the actual pipeline prompt.
"""
import os
import sys
sys.path.insert(0, '.')

from rag.pipeline import RAGPipeline
from llm.ollama_service import OllamaService
import json

def main():
    print("=== Debugging Ollama Interaction ===")
    
    # Create pipeline and get to the point where we call Ollama
    pipeline = RAGPipeline()
    
    # Simulate the steps up to LLM generation
    requirement = "The system should lock a user account after five failed password attempts."
    
    print(f"Requirement: {requirement}")
    
    # Step 1: ML Classification
    from ml.classifier import predict_category
    category, confidence = predict_category(requirement)
    print(f"ML Classification - Category: {category}, Confidence: {confidence}")
    
    # Step 2: Generate embedding
    from ml.embeddings import EmbeddingPipeline
    embedding_pipe = EmbeddingPipeline()
    embedding = embedding_pipe.embed_text(requirement)
    print(f"Embedding generated, shape: {len(embedding)}")
    
    # Step 3: FAISS search
    from vector_store.search import search as semantic_search
    guidelines = semantic_search(requirement, top_k=3)
    print(f"Retrieved {len(guidelines)} guidelines")
    
    # Step 4: Build context
    from llm.context_builder import ContextBuilder
    context_builder = ContextBuilder()
    context = context_builder.build_context(requirement, category, confidence)
    print(f"Context built with {len(context['top_guidelines'])} guidelines")
    
    # Print the prompt that will be sent
    prompt = pipeline._build_prompt(context)
    print("\n=== PROMPT SENT TO OLLAMA ===")
    print(prompt)
    print("=== END PROMPT ===\n")
    
    # Test Ollama directly
    ollama_service = OllamaService(timeout=60)
    
    print("=== CALLING OLLAMA ===")
    try:
        # Use the system prompt from the pipeline
        raw_response = ollama_service._call_ollama(prompt, system_prompt=pipeline.prompt_template)
        print("Raw response from Ollama:")
        print(json.dumps(raw_response.dict(), indent=2))
        print("\nContent field:")
        print(repr(raw_response.content))
    except Exception as e:
        print(f"Ollama error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()