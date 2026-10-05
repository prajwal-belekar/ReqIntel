#!/usr/bin/env python3
"""
Context builder for RequirementLens RAG pipeline.

This module combines:
1. Original requirement
2. ML predicted category
3. ML confidence
4. Top 3 FAISS guidelines

The context must explicitly state:
- ML prediction is evidence, not ground truth
- FAISS results are contextual information
- Retrieved guidelines are not facts about the user's system
"""

from typing import List, Dict, Any
from vector_store.search import search as semantic_search
from ml.embeddings import EmbeddingPipeline
from pathlib import Path
import json

class ContextBuilder:
    """
    Builds RAG context from requirement and existing components.
    """
    
    def __init__(self):
        """Initialize context builder."""
        self.embedding_pipeline = EmbeddingPipeline()
        
        # Load metadata for guideline references
        metadata_path = Path(__file__).parent.parent / "vector_store" / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
    
    def build_context(self, requirement: str, ml_category: str, ml_confidence: float) -> Dict[str, Any]:
        """
        Build RAG context from requirement and existing components.
        
        Args:
            requirement: Original requirement text
            ml_category: ML predicted category
            ml_confidence: ML confidence score
            
        Returns:
            Dictionary containing:
            - requirement
            - ml_category
            - ml_confidence
            - top_guidelines
            - context
        """
        # Step 1: Search FAISS for top 3 semantically similar guidelines
        top_guidelines = semantic_search(requirement, top_k=3)
        
        # Step 2: Build context
        context = {
            'requirement': requirement,
            'ml_category': ml_category,
            'ml_confidence': ml_confidence,
            'top_guidelines': [],
            'context': ''
        }
        
        # Format guidelines with similarity scores
        for guideline in top_guidelines:
            guideline_ref = {
                'id': guideline['guideline_id'],
                'title': guideline['title'],
                'similarity': guideline['similarity_score'],
                'category': guideline['category'],
                'text': guideline['guideline_text'][:500] + "..."
            }
            context['top_guidelines'].append(guideline_ref)
        
        # Build the context string
        context_text = self._build_context_text(requirement, ml_category, ml_confidence, top_guidelines)
        context['context'] = context_text
        
        return context
    
    def _build_context_text(self, requirement: str, ml_category: str, ml_confidence: float, top_guidelines: List[Dict[str, Any]]) -> str:
        """
        Builds the context string with explicit disclaimers.
        """
        # Start with requirement and ML prediction
        context = f"""
        ### RAG Context for Requirement Analysis
        
        Original Requirement:
        {requirement}
        
        ML Classification:
        - Category: {ml_category}
        - Confidence: {ml_confidence:.2f}
        
        **Important Note:**
        The ML classification is evidence-based but not absolute truth.
        The FAISS results are contextual information about software engineering principles.
        These guidelines are NOT facts about the user's specific system.
        """
        
        # Add guidelines with disclaimers
        for i, guideline in enumerate(top_guidelines, 1):
            context += f"\n### Guideline {i} ({guideline['category']})"
            context += f"\nTitle: {guideline['title']}"
            context += f"\nSimilarity: {guideline['similarity_score']:.2f}"
            context += f"\nText: {guideline['guideline_text']}"
            
            # Explicit disclaimer for each guideline
            context += f"\n**Important:** This guideline is a general software engineering principle, not a fact about your system."
        
# End with clear instructions for LLM
        context += f"\n**LLM Instructions:**\nAnalyze the requirement and context above.\n\n1. Identify missing specifications (not explicitly stated in requirement)\n2. Consider the ML classification as evidence\n3. Consider the guidelines as potential considerations\n4. Provide recommendations based on the analysis\n5. Do NOT assume anything not explicitly stated\n6. If information is not available, explicitly state it's not specified"
        
        return context


def main():
    """Test context builder."""
    builder = ContextBuilder()
    
    # Test requirement
    requirement = "The system shall be available 99.9% of the time during business hours."
    
    # Mock ML results
    ml_category = "Availability"
    ml_confidence = 0.85
    
    context = builder.build_context(requirement, ml_category, ml_confidence)
    
    print("=== Context Builder Output ===")
    print(f"Requirement: {context['requirement']}")
    print(f"ML Category: {context['ml_category']}")
    print(f"ML Confidence: {context['ml_confidence']}")
    print(f"Top Guidelines: {len(context['top_guidelines'])}")
    print("\nContext:")
    print(context['context'][:1000] + "..." if len(context['context']) > 1000 else context['context'])


if __name__ == "__main__":
    main()