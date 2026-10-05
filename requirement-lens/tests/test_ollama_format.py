#!/usr/bin/env python3
"""
Test Ollama response format.

Verifies that qwen3:8b honours the JSON contract requested by the system
prompt. Previously both tests printed diagnostics and returned None, so a
model or service regression could not fail the suite.
"""

import json
import re

import requests

from llm.ollama_service import OllamaService

# Fields the system prompt demands from the model.
REQUIRED_JSON_FIELDS = [
    'risk_level',
    'risk_factors',
    'completeness_score',
    'missing_requirements',
    'recommendations',
    'explanation',
]

VALID_RISK_LEVELS = {'Low', 'Medium', 'High'}

SYSTEM_PROMPT = """You are an expert software requirements analyzer. You MUST output ONLY a valid JSON object with exactly these fields:
- risk_level: string (must be "Low", "Medium", or "High")
- risk_factors: array of strings
- completeness_score: number between 0.0 and 1.0
- missing_requirements: array of strings
- recommendations: array of strings
- explanation: string

Do NOT include any text before or after the JSON. Do NOT use markdown formatting. Output ONLY the JSON."""

PROMPT = """
=== REQUIREMENT ===
The system should lock a user account after five failed password attempts.

=== ML CLASSIFICATION ===
Category: SE
Confidence: 0.32

=== RETRIEVED GUIDELINES ===
[GUIDELINE 1]
Title: Account Lockout After Failed Attempts
Similarity: 0.75
Text: User accounts should be temporarily locked after multiple consecutive failed login attempts to prevent brute force attacks....

[GUIDELINE 2]
Title: Password Reset Security
Similarity: 0.54
Text: Password reset mechanisms must verify user identity through secure channels before allowing password changes....

[GUIDELINE 3]
Title: Multi-Factor Authentication
Similarity: 0.53
Text: Critical systems should implement multi-factor authentication combining password with additional verification methods....
"""


def test_direct_ollama():
    """Raw /api/generate call must return a model response honouring the schema."""
    # num_predict must match llm/ollama_service.py. At 512 the whole budget was
    # consumed by qwen3's "thinking" tokens and the response came back empty.
    payload = {
        'model': 'qwen3:8b',
        'prompt': PROMPT,
        'system': SYSTEM_PROMPT,
        'stream': False,
        'options': {'temperature': 0.1, 'top_p': 0.9, 'num_predict': 1024}
    }

    response = requests.post('http://localhost:11434/api/generate', json=payload, timeout=120)

    assert response.status_code == 200, (
        f"Ollama returned {response.status_code}: {response.text}"
    )

    result = response.json()
    assert result.get('model') == 'qwen3:8b', (
        f"expected model qwen3:8b, got {result.get('model')!r}"
    )
    assert result.get('done') is True, "generation did not complete"

    raw_response = result.get('response', '')
    assert raw_response.strip(), "Ollama returned an empty response"

    # num_predict is shared with qwen3's "thinking" budget, so this call can be
    # truncated mid-object. Assert schema adherence rather than completeness.
    assert raw_response.lstrip().startswith('{'), (
        "response does not start with a JSON object"
    )
    assert not raw_response.lstrip().startswith('```'), (
        "response is wrapped in a markdown code fence"
    )

    emitted = set(re.findall(r'"([A-Za-z_]+)"\s*:', raw_response))
    assert emitted, "response contains no JSON fields"
    assert emitted <= set(REQUIRED_JSON_FIELDS), (
        f"model emitted unexpected fields {sorted(emitted - set(REQUIRED_JSON_FIELDS))}"
    )

    # risk_level is emitted first, so it must always survive truncation.
    match = re.search(r'"risk_level"\s*:\s*"([^"]+)"', raw_response)
    assert match, "response does not contain a risk_level value"
    assert match.group(1) in VALID_RISK_LEVELS, (
        f"invalid risk_level {match.group(1)!r}"
    )

    truncated = not raw_response.rstrip().endswith('}')
    if not truncated:
        parsed = json.loads(raw_response)
        missing = [f for f in REQUIRED_JSON_FIELDS if f not in parsed]
        assert not missing, f"complete response is missing fields {missing}"
        for field in ('risk_factors', 'missing_requirements', 'recommendations'):
            assert isinstance(parsed[field], list), f"{field} must be a list"

    print(
        f"\ndirect ollama: {len(raw_response)} chars, "
        f"fields={sorted(emitted)}, truncated={truncated}"
    )


def test_ollama_service():
    """OllamaService must return a complete, schema-conforming JSON object."""
    service = OllamaService(timeout=120)

    result = service.generate_response(PROMPT, system_prompt=SYSTEM_PROMPT)

    assert isinstance(result, str), f"expected str, got {type(result).__name__}"
    assert result.strip(), "OllamaService returned an empty response"

    parsed = json.loads(result)

    missing = [f for f in REQUIRED_JSON_FIELDS if f not in parsed]
    assert not missing, f"OllamaService response is missing fields {missing}"

    assert parsed['risk_level'] in VALID_RISK_LEVELS, (
        f"invalid risk_level {parsed['risk_level']!r}"
    )
    for field in ('risk_factors', 'missing_requirements', 'recommendations'):
        assert isinstance(parsed[field], list), f"{field} must be a list"
    assert isinstance(parsed['explanation'], str)
    assert parsed['explanation'].strip(), "explanation must not be empty"

    print(f"\nollama service: valid JSON, risk_level={parsed['risk_level']}")


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))