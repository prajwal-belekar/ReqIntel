#!/usr/bin/env python3
"""
Debug script to see raw Ollama response.
"""
import os
import sys
sys.path.insert(0, '.')

from rag.pipeline import RAGPipeline
import json
import requests

def main():
    print("=== Debugging Raw Ollama Response ===")
    
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
    
    # Call Ollama directly and capture raw response
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        'model': 'qwen3:8b',
        'prompt': prompt,
        'system': pipeline.prompt_template,
        'stream': False,
        'options': {'temperature': 0.1, 'top_p': 0.9, 'num_predict': 512}
    }
    
    print("\n=== Calling Ollama ===")
    response = requests.post(ollama_url, json=payload, timeout=120)
    
    if response.status_code == 200:
        result = response.json()
        raw_response = result.get('response', '')
        print(f"Raw response length: {len(raw_response)}")
        print(f"Raw response (first 500 chars):")
        print(repr(raw_response[:500]))
        print(f"\nRaw response (next 500 chars):")
        if len(raw_response) > 500:
            print(repr(raw_response[500:1000]))
        
        # Try to parse as JSON
        try:
            parsed = json.loads(raw_response)
            print("\n✅ Successfully parsed JSON!")
            print("Keys:", list(parsed.keys()))
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON parse error: {e}")
            
            # Check if it's because of newlines in the string
            lines = raw_response.split('\n')
            print(f"\nNumber of lines in response: {len(lines)}")
            
            # Print first few lines
            for i, line in enumerate(lines[:10]):
                print(f"Line {i}: {repr(line[:200])}")
                
    else:
        print(f"Error: {response.status_code}")
        print(f"Response: {response.text}")

if __name__ == "__main__":
    main()