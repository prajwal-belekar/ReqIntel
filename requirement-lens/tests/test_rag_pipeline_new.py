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

from unittest.mock import patch

from rag.pipeline import analyze_requirement, RAGPipeline

class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_requirement = "The system shall be available 99.9% of the time during business hours."
        self.mock_category = "Availability"
        self.mock_confidence = 0.85
    
    @patch('rag.pipeline.RAGPipeline._verify_ollama')
    @patch('llm.context_builder.EmbeddingPipeline')
    @patch('rag.pipeline.EmbeddingPipeline')
    def test_requirement_validation(self, mock_pipeline_embedding, mock_builder_embedding, mock_verify):
        """Test requirement validation."""
        # Embedding pipelines are mocked so this validation test never loads
        # the sentence-transformer model or touches the HuggingFace Hub.
        mock_pipeline_embedding.return_value = Mock()
        mock_builder_embedding.return_value = Mock()

        # Test empty requirement
        with self.assertRaises(ValueError):
            analyze_requirement("")

        with self.assertRaises(ValueError):
            analyze_requirement("   ")

        # Test valid requirement
        self.assertIsInstance(self.test_requirement, str)
        self.assertTrue(len(self.test_requirement.strip()) > 0)

    @patch('rag.pipeline.RAGPipeline.__init__')
    def test_invalid_input_does_not_build_pipeline(self, mock_init):
        """Invalid input must be rejected before RAGPipeline is constructed.

        Regression test: analyze_requirement() used to build RAGPipeline()
        first, which loads the ML model, the sentence-transformer and the
        FAISS index before the empty-input check ever ran.
        """
        for bad_input in ("", "   ", "\n\t ", None, 123):
            with self.subTest(input=repr(bad_input)):
                with self.assertRaises(ValueError):
                    analyze_requirement(bad_input)

        # The expensive pipeline must never have been constructed.
        mock_init.assert_not_called()

    @patch('rag.pipeline.RAGPipeline._verify_ollama')
    @patch('llm.context_builder.EmbeddingPipeline')
    @patch('rag.pipeline.EmbeddingPipeline')
    @patch('rag.pipeline.RAGPipeline.__init__', return_value=None)
    def test_valid_input_still_builds_pipeline(self, mock_init, mock_pipe_emb, mock_builder_emb, mock_verify):
        """Valid input must still reach the pipeline unchanged."""
        mock_pipe_emb.return_value = Mock()
        mock_builder_emb.return_value = Mock()

        with patch.object(RAGPipeline, 'analyze_requirement') as mock_method:
            mock_method.return_value = "sentinel"
            result = analyze_requirement(self.test_requirement)

        mock_init.assert_called_once()
        mock_method.assert_called_once_with(self.test_requirement)
        self.assertEqual(result, "sentinel")

if __name__ == "__main__":
    unittest.main()