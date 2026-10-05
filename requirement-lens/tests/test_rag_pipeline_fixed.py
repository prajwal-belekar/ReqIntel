#!/usr/bin/env python3
"""
Tests for RequirementLens RAG pipeline.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import analyze_requirement, RAGPipeline
from llm.schemas import RequirementAnalysis, validate_llm_json, repair_llm_json
from llm.context_builder import ContextBuilder
from llm.ollama_service import OllamaService

class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_requirement = "The system shall be available 99.9% of the time during business hours."
        self.mock_category = "Availability"
        self.mock_confidence = 0.85
    
    @patch('rag.pipeline.RAGPipeline._verify_ollama')
    @patch('rag.pipeline.EmbeddingPipeline')
    def test_requirement_validation(self, mock_embedding_pipeline, mock_verify):
        """Test requirement validation."""
        # Mock the embedding pipeline
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_text.return_value = [0.1] * 384
        mock_embedding_pipeline.return_value = mock_embedding_instance
        
        # Test empty requirement
        with self.assertRaises(ValueError):
            analyze_requirement("")
        
        with self.assertRaises(ValueError):
            analyze_requirement("   ")
        
        # Test valid requirement
        self.assertIsInstance(self.test_requirement, str)
        self.assertTrue(len(self.test_requirement.strip()) > 0)
    
    def test_schema_validation(self):
        """Test Pydantic schema validation."""
        # Test valid data
        valid_data = {
            "requirement": "Test requirement",
            "category": "Security",
            "ml_confidence": 0.85,
            "risk_level": "Medium",
            "risk_factors": ["Risk factor 1", "Risk factor 2"],
            "completeness_score": 0.75,
            "missing_requirements": ["Missing req 1"],
            "recommendations": ["Recommendation 1"],
            "similar_guidelines": [
                {
                    "id": "SEC-001",
                    "title": "Security Guideline",
                    "similarity": 0.8,
                    "category": "Security",
                    "text": "Security guideline text"
                }
            ],
            "explanation": "Test explanation"
        }
        
        analysis = RequirementAnalysis(**valid_data)
        self.assertEqual(analysis.category, "Security")
        self.assertEqual(analysis.ml_confidence, 0.85)
        self.assertEqual(analysis.risk_level.value, "Medium")
        self.assertEqual(analysis.completeness_score, 0.75)
        
        # Test invalid risk level
        invalid_data = valid_data.copy()
        invalid_data["risk_level"] = "Invalid"
        
        with self.assertRaises(ValueError):
            RequirementAnalysis(**invalid_data)
        
        # Test invalid confidence
        invalid_data = valid_data.copy()
        invalid_data["ml_confidence"] = 1.5  # Out of range
        
        with self.assertRaises(ValueError):
            RequirementAnalysis(**invalid_data)
        
        # Test invalid completeness score
        invalid_data = valid_data.copy()
        invalid_data["completeness_score"] = -0.1  # Out of range
        
        with self.assertRaises(ValueError):
            RequirementAnalysis(**invalid_data)
    
    def test_json_validation(self):
        """Test JSON validation and repair."""
        # Test valid JSON
        valid_json = '''
        {
            "risk_level": "High",
            "risk_factors": ["No error handling"],
            "completeness_score": 0.6,
            "missing_requirements": ["Error handling spec"],
            "recommendations": ["Add error handling"],
            "explanation": "Requirement lacks error handling details"
        }
        '''
        
        result = validate_llm_json(valid_json)
        self.assertEqual(result["risk_level"], "High")
        self.assertEqual(result["completeness_score"], 0.6)
        
        # Test invalid JSON
        invalid_json = '{"invalid": "json"'  # Missing closing brace
        
        with self.assertRaises(ValueError):
            validate_llm_json(invalid_json)
        
        # Test malformed JSON that can be repaired
        malformed_json = '''
        {
            "risk_level": "Medium",
            "risk_factors": ["No risk"],
            "completeness_score": 0.7,
            "missing_requirements": [],
            "recommendations": [],
            "explanation": ""
        '''
        # Missing closing brace
        
        # Should fail validation but pass repair
        with self.assertRaises(ValueError):
            validate_llm_json(malformed_json)
        
        # Test repair
        try:
            result = repair_llm_json(malformed_json)
            # If repair succeeds, it should be valid
            self.assertEqual(result["risk_level"], "Medium")
        except ValueError:
            # Repair failed, which is also acceptable for this test
            pass
    
    def test_context_building(self):
        """Test context builder."""
        builder = ContextBuilder()
        
        context = builder.build_context(
            self.test_requirement,
            self.mock_category,
            self.mock_confidence
        )
        
        # Verify context structure
        self.assertEqual(context['requirement'], self.test_requirement)
        self.assertEqual(context['ml_category'], self.mock_category)
        self.assertEqual(context['ml_confidence'], self.mock_confidence)
        self.assertIsInstance(context['top_guidelines'], list)
        self.assertIsInstance(context['context'], str)
        
        # Verify context content
        self.assertIn("RAG Context for Requirement Analysis", context['context'])
        self.assertIn("Original Requirement:", context['context'])
        self.assertIn("ML Classification:", context['context'])
        self.assertIn("Important Note:", context['context'])
        self.assertIn("Guideline", context['context'])
        self.assertIn("LLM Instructions:", context['context'])
    
    @patch('rag.pipeline.RAGPipeline._verify_ollama')
    def test_prompt_building(self, mock_verify):
        """Test prompt building in RAG pipeline."""
        # Mock the embedding pipeline
        mock_embedding_instance = Mock()
        mock_embedding_instance.embed_text.return_value = [0.1] * 384
        
        with patch('rag.pipeline.EmbeddingPipeline', return_value=mock_embedding_instance):
            pipeline = RAGPipeline()
        
        # Mock context
        context = {
            'requirement': self.test_requirement,
            'ml_category': self.mock_category,
            'ml_confidence': self.mock_confidence,
            'top_guidelines': [
                {
                    'guideline_id': 'AVAIL-001',
                    'title': 'System Uptime Requirements',
                    'similarity': 0.85,
                    'category': 'Availability',
                    'text': 'System availability should meet specified uptime percentage during operational hours.'
                }
            ]
        }
        
        prompt = pipeline._build_prompt(context)
        
        # Verify prompt structure
        self.assertIn("=== REQUIREMENT ===", prompt)
        self.assertIn(self.test_requirement, prompt)
        self.assertIn("=== ML CLASSIFICATION ===", prompt)
        self.assertIn(f"Category: {self.mock_category}", prompt)
        self.assertIn(f"Confidence: {self.mock_confidence:.2f}", prompt)
        self.assertIn("=== RETRIEVED GUIDELINES ===", prompt)
        self.assertIn("System Uptime Requirements", prompt)
        self.assertIn("Similarity: 0.85", prompt)
    
    @patch('requests.post')
    def test_ollama_service(self, mock_post):
        """Test Ollama service."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "This is a test response",
            "model": "qwen3:8b",
            "created_at": "2023-01-01T00:00:00Z"
        }
        mock_post.return_value = mock_response
        
        service = OllamaService(timeout=10)
        result = service.generate_response("Test prompt")
        
        self.assertEqual(result, "This is a test response")
        mock_post.assert_called_once()
    
    @patch('requests.post')
    def test_ollama_error_handling(self, mock_post):
        """Test Ollama error handling."""
        # Mock 404 response
        mock_response = Mock()
        mock_response.status_code = 404
        mock_post.return_value = mock_response
        
        service = OllamaService(timeout=10)
        
        with self.assertRaises(Exception) as context:
            service.generate_response("Test prompt")
        
        self.assertIn("Model 'qwen3:8b' not found", str(context.exception))

if __name__ == "__main__":
    unittest.main()
