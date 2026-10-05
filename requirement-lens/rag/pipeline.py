#!/usr/bin/env python3
"""
End-to-end RAG pipeline for RequirementLens.

This module implements the complete RAG pipeline:
1. ML classification
2. Embedding generation
3. FAISS search
4. Context building
5. Ollama generation
6. Structured output validation
"""

import os
import json
from typing import Optional
from pathlib import Path
from ml.embeddings import EmbeddingPipeline
from ml.classifier import predict_category
from vector_store.search import search as semantic_search
from llm.ollama_service import OllamaService
from llm.context_builder import ContextBuilder
from llm.schemas import RequirementAnalysis, LLMError, validate_llm_json, repair_llm_json

class RAGPipeline:
    """
    End-to-end RAG pipeline for requirement analysis.
    """
    
    def __init__(self):
        """Initialize pipeline components."""
        self.embedding_pipeline = EmbeddingPipeline()
        self.context_builder = ContextBuilder()
        self.ollama_service = OllamaService()
        
        # Load prompt template
        prompt_path = Path(__file__).parent.parent / "llm" / "rag_prompt_template.md"
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()
        
        # Verify Ollama availability
        self._verify_ollama()
        
    def _verify_ollama(self):
        """Verify Ollama is available and model exists."""
        try:
            # Check Ollama availability
            self.ollama_service.generate_response(
                "Test",
                system_prompt="Respond with 'OK'"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama verification failed: {str(e)}")
        
    def analyze_requirement(self, requirement: str) -> RequirementAnalysis:
        """
        Analyze a requirement through the complete RAG pipeline.
        
        Args:
            requirement: Requirement text to analyze
            
        Returns:
            RequirementAnalysis with complete results
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If pipeline fails
        """
        if not requirement or not isinstance(requirement, str) or not requirement.strip():
            raise ValueError("Requirement must be a non-empty string")
        
        try:
            # Step 1: ML Classification
            category, confidence = predict_category(requirement)
            
            # Step 2: Generate embedding
            embedding = self.embedding_pipeline.embed_text(requirement)
            
            # Step 3: FAISS search
            guidelines = semantic_search(requirement, top_k=3)
            
            # Step 4: Build context
            context = self.context_builder.build_context(
                requirement,
                category,
                confidence
            )
            
            # Step 5: Generate LLM response
            llm_response = self._generate_llm_response(context)
            
            # Step 6: Validate and combine results
            analysis = self._combine_results(
                requirement,
                category,
                confidence,
                guidelines,
                llm_response
            )
            
            return analysis
        
        except Exception as e:
            raise RuntimeError(f"Pipeline failed: {str(e)}")
        
    def _generate_llm_response(self, context: dict) -> dict:
        """
        Generate LLM response with error handling.
        
        Args:
            context: Context dictionary
            
        Returns:
            Parsed and validated LLM response
            
        Raises:
            RuntimeError: If LLM generation fails
        """
        try:
            # Build prompt
            prompt = self._build_prompt(context)
            
            # Generate response using Ollama service
            response_content = self.ollama_service.generate_response(
                prompt,
                system_prompt=self.prompt_template
            )
            
            # Strategy 1: Try to validate the response field directly
            try:
                return validate_llm_json(response_content)
            except ValueError:
                pass  # Continue to next strategy
            
            # Strategy 2: Check if the thinking field contains valid JSON
            if (hasattr(self.ollama_service, 'last_response') and 
                self.ollama_service.last_response and 
                self.ollama_service.last_response.thinking):
                thinking = self.ollama_service.last_response.thinking
                
                # Try to parse thinking directly as JSON
                try:
                    return validate_llm_json(thinking)
                except ValueError:
                    pass  # Continue to extraction
                
                # Try to extract JSON object from thinking
                json_start = thinking.find('{')
                json_end = thinking.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = thinking[json_start:json_end]
                    try:
                        return validate_llm_json(json_str)
                    except ValueError:
                        pass  # Continue to repair
            
            # Strategy 3: Attempt to repair the response content
            try:
                return repair_llm_json(response_content)
            except ValueError as e:
                raise RuntimeError(f"LLM output validation failed. Repair attempt failed: {str(e)}")
        
        except Exception as e:
            raise RuntimeError(f"LLM generation failed: {str(e)}")
        
    def _build_prompt(self, context: dict) -> str:
        """
        Build prompt from context.
        
        Args:
            context: Context dictionary
            
        Returns:
            Formatted prompt string
        """
        # Format guidelines
        guidelines_text = "\n\n".join(
            f"[GUIDELINE {i+1}]\nTitle: {g['title']}\nSimilarity: {g['similarity']:.2f}\nText: {g['text']}"
            for i, g in enumerate(context['top_guidelines'])
        )
        
        # Build prompt
        prompt = f"""
        === REQUIREMENT ===
        {context['requirement']}
        
        === ML CLASSIFICATION ===
        Category: {context['ml_category']}
        Confidence: {context['ml_confidence']:.2f}
        
        === RETRIEVED GUIDELINES ===
        {guidelines_text}
        """

        # Add JSON format request to ensure proper response format
        prompt += """

You MUST provide your analysis in the following JSON format exactly:
{
  "risk_level": "Low" | "Medium" | "High",
  "risk_factors": ["string", ...],
  "completeness_score": <decimal fraction>,
  "missing_requirements": ["string", ...],
  "recommendations": ["string", ...],
  "explanation": "string"
}

completeness_score MUST be a decimal fraction between 0.0 and 1.0, NOT a percentage:
- 0.0 means 0% complete
- 0.5 means 50% complete
- 0.65 means 65% complete
- 1.0 means 100% complete
Never output a whole number such as 60 or 65 to express a percentage.

Do NOT include any text before or after the JSON. Do NOT use markdown formatting. Output ONLY valid JSON."""

        return prompt
    
    def _combine_results(self, requirement: str, category: str, confidence: float, guidelines: list, llm_response: dict) -> RequirementAnalysis:
        """
        Combine deterministic and LLM results.
        
        Args:
            requirement: Original requirement
            category: ML predicted category
            confidence: ML confidence
            guidelines: Retrieved guidelines
            llm_response: LLM response
            
        Returns:
            RequirementAnalysis with complete results
        """
        # Format guidelines for output
        formatted_guidelines = [
            {
                "id": g['guideline_id'],
                "title": g['title'],
                "similarity": g['similarity_score'],
                "category": g['category'],
                "text": g['guideline_text']
            }
            for g in guidelines
        ]
        
        # Create analysis
        analysis = RequirementAnalysis(
            requirement=requirement,
            category=category,
            ml_confidence=confidence,
            risk_level=llm_response['risk_level'],
            risk_factors=llm_response['risk_factors'],
            completeness_score=llm_response['completeness_score'],
            missing_requirements=llm_response['missing_requirements'],
            recommendations=llm_response['recommendations'],
            similar_guidelines=formatted_guidelines,
            explanation=llm_response['explanation']
        )
        
        return analysis


def analyze_requirement(requirement: str) -> RequirementAnalysis:
    """
    Convenience function for requirement analysis.
    
    Args:
        requirement: Requirement text to analyze
        
    Returns:
        RequirementAnalysis with complete results
        
    Raises:
        ValueError: If requirement is empty or whitespace only
    """
    # Validate before constructing RAGPipeline: __init__ loads the ML model,
    # the sentence-transformer and the FAISS index, so an invalid requirement
    # would otherwise pay the full initialisation cost (and network access)
    # only to be rejected.
    if not requirement or not isinstance(requirement, str) or not requirement.strip():
        raise ValueError("Requirement must be a non-empty string")
    
    pipeline = RAGPipeline()
    return pipeline.analyze_requirement(requirement)


def main():
    """Test the RAG pipeline."""
    try:
        # Test requirement
        requirement = "The system shall be available 99.9% of the time during business hours."
        
        print(f"Analyzing requirement: {requirement}")
        analysis = analyze_requirement(requirement)
        
        print("\n=== Requirement Analysis ===")
        print(f"Requirement: {analysis.requirement}")
        print(f"Category: {analysis.category}")
        print(f"ML Confidence: {analysis.ml_confidence:.2f}")
        print(f"Risk Level: {analysis.risk_level}")
        print(f"Completeness: {analysis.completeness_score:.2f}")
        print(f"\nRisk Factors: {analysis.risk_factors}")
        print(f"Missing Requirements: {analysis.missing_requirements}")
        print(f"Recommendations: {analysis.recommendations}")
        print(f"\nExplanation: {analysis.explanation}")
        
        print(f"\nSimilar Guidelines: {len(analysis.similar_guidelines)}")
        for i, guideline in enumerate(analysis.similar_guidelines, 1):
            print(f"\n{i}. {guideline['title']} (Similarity: {guideline['similarity']:.2f})")
            print(f"   Category: {guideline['category']}")
            print(f"   Text: {guideline['text'][:100]}...")
        
    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()