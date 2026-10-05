#!/usr/bin/env python3
"""
Contract tests for LLM score fields.

completeness_score is a decimal FRACTION, not a percentage:
    0.0 -> 0%, 0.5 -> 50%, 0.65 -> 65%, 1.0 -> 100%

qwen3:8b sometimes emits a percentage (e.g. 60) instead. The schema rejects
that on purpose -- silently coercing it would hide a contract violation and
is ambiguous (is 1 one percent or one hundred percent?).

These tests pin the intended representation on both sides of the contract:
the validators that enforce it, and the prompt that must communicate it.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from llm.schemas import (
    validate_llm_json,
    repair_llm_json,
    RequirementAnalysis,
)
from rag.pipeline import RAGPipeline


def _payload(score):
    return json.dumps({
        "risk_level": "Medium",
        "risk_factors": ["No error handling"],
        "completeness_score": score,
        "missing_requirements": [],
        "recommendations": [],
        "explanation": "test",
    })


class TestScoreContract(unittest.TestCase):
    """completeness_score must be a fraction in [0.0, 1.0]."""

    def test_fraction_values_are_accepted(self):
        """0.0, 0.5, 0.65 and 1.0 are all valid fractions."""
        for score in (0.0, 0.5, 0.65, 1.0):
            with self.subTest(score=score):
                result = validate_llm_json(_payload(score))
                self.assertEqual(result["completeness_score"], score)

    def test_percentage_values_are_rejected(self):
        """A whole number like 60 is a percentage, not a fraction."""
        for score in (60, 65, 100, 1.5, -0.1):
            with self.subTest(score=score):
                with self.assertRaises(ValueError):
                    validate_llm_json(_payload(score))

    def test_repair_does_not_coerce_percentages(self):
        """Repair must not silently rescale a percentage into a fraction.

        This locks in the decision taken during the fix: a percentage is a
        contract violation and stays a hard error, so it surfaces as a failed
        analysis instead of a quietly wrong score.
        """
        with self.assertRaises(ValueError):
            repair_llm_json(_payload(60))

    def test_schema_model_enforces_fraction(self):
        """RequirementAnalysis itself must reject out-of-range scores."""

        def build(score):
            return RequirementAnalysis(
                requirement="r",
                category="Security",
                ml_confidence=0.5,
                risk_level="Medium",
                risk_factors=[],
                completeness_score=score,
                missing_requirements=[],
                recommendations=[],
                similar_guidelines=[],
                explanation="x",
            )

        self.assertEqual(build(0.65).completeness_score, 0.65)
        for bad in (60, 1.5, -0.1):
            with self.subTest(score=bad):
                with self.assertRaises(Exception):
                    build(bad)


class TestScorePromptContract(unittest.TestCase):
    """The prompt must state the scale unambiguously."""

    def setUp(self):
        # Bypass __init__ so no model, FAISS index or network call is needed.
        self.pipeline = RAGPipeline.__new__(RAGPipeline)

    def _prompt(self):
        return self.pipeline._build_prompt({
            'requirement': 'The system shall be available 99.9% of the time.',
            'ml_category': 'Availability',
            'ml_confidence': 0.85,
            'top_guidelines': [
                {
                    'id': 'AVAIL-001',
                    'title': 'System Uptime Requirements',
                    'similarity': 0.85,
                    'category': 'Availability',
                    'text': 'System availability should meet the uptime target.',
                }
            ],
        })

    def test_prompt_defines_fraction_scale(self):
        """The prompt must map 0.0/0.5/1.0 onto percentages."""
        prompt = self._prompt()
        self.assertIn('"completeness_score"', prompt)
        self.assertIn('0.0 means 0% complete', prompt)
        self.assertIn('0.5 means 50% complete', prompt)
        self.assertIn('1.0 means 100% complete', prompt)

    def test_prompt_forbids_whole_number_percentages(self):
        """The prompt must explicitly reject the 60-style output."""
        prompt = self._prompt()
        self.assertIn('Never output a whole number', prompt)

    def test_prompt_still_requests_the_same_json_keys(self):
        """Clarifying the scale must not change the required schema."""
        prompt = self._prompt()
        for field in (
            'risk_level', 'risk_factors', 'completeness_score',
            'missing_requirements', 'recommendations', 'explanation',
        ):
            with self.subTest(field=field):
                self.assertIn(f'"{field}"', prompt)


if __name__ == "__main__":
    unittest.main()