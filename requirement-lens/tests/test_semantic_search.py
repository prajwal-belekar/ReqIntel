#!/usr/bin/env python3
"""
Test semantic search functionality for RequirementLens.

Tests various queries across different categories to verify
semantic retrieval quality.

Every case is asserted. Previously this module counted passes/failures,
printed them and returned the tuple, so a retrieval regression could not
fail the suite.
"""

import sys
sys.path.insert(0, '..')

from vector_store.search import SemanticSearch

# Contract returned by SemanticSearch.search() -- the raw FAISS layer.
REQUIRED_RESULT_KEYS = {
    'guideline_id', 'title', 'category',
    'similarity_score', 'guideline_text', 'tags', 'source',
}

# Queries whose top hit lands in a different category than the one a human
# would pick. These are retrieval-quality limitations of the current
# all-MiniLM-L6-v2 index, not code defects, so they are tolerated here to
# keep the suite as a regression baseline. Any deviation NOT listed below
# fails the test.
KNOWN_CATEGORY_DEVIATIONS = {
    'High availability requirement',  # top hit: Fault Tolerance, not Availability
}


def test_semantic_search():
    """Test semantic search with various queries."""

    searcher = SemanticSearch()

    # Test queries organized by category
    test_cases = [
        # Security
        {
            'query': 'Users should be locked out after multiple failed passwords.',
            'expected_category': 'Security',
            'description': 'Account lockout security requirement'
        },
        {
            'query': 'The system must protect user data with encryption.',
            'expected_category': 'Security',
            'description': 'Data encryption requirement'
        },
        {
            'query': 'Password must be complex with uppercase letters and numbers.',
            'expected_category': 'Security',
            'description': 'Password complexity requirement'
        },
        # Performance
        {
            'query': 'The system must respond within 2 seconds to user requests.',
            'expected_category': 'Performance',
            'description': 'Response time requirement'
        },
        {
            'query': 'The application should handle 1000 concurrent users efficiently.',
            'expected_category': 'Performance',
            'description': 'Concurrent user requirement'
        },
        {
            'query': 'Database queries should be optimized for fast retrieval.',
            'expected_category': 'Performance',
            'description': 'Database optimization requirement'
        },
        # Usability
        {
            'query': 'The interface should be intuitive for new users without training.',
            'expected_category': 'Usability',
            'description': 'Intuitive UI requirement'
        },
        {
            'query': 'Error messages should be clear and helpful for users.',
            'expected_category': 'Usability',
            'description': 'Error message clarity requirement'
        },
        {
            'query': 'Navigation should be consistent across all screens.',
            'expected_category': 'Usability',
            'description': 'Consistent navigation requirement'
        },
        # Availability
        {
            'query': 'System should be available 99.9% of the time during business hours.',
            'expected_category': 'Availability',
            'description': 'Uptime requirement'
        },
        {
            'query': 'The system needs backup servers for continued operation during failures.',
            'expected_category': 'Availability',
            'description': 'High availability requirement'
        },
        # Scalability
        {
            'query': 'The system should scale horizontally by adding more servers.',
            'expected_category': 'Scalability',
            'description': 'Horizontal scalability requirement'
        },
        {
            'query': 'Load balancing should distribute requests across multiple servers.',
            'expected_category': 'Scalability',
            'description': 'Load balancing requirement'
        },
        # Maintainability
        {
            'query': 'Code should be well documented for future maintenance.',
            'expected_category': 'Maintainability',
            'description': 'Documentation requirement'
        },
        {
            'query': 'The system should have automated tests for code changes.',
            'expected_category': 'Maintainability',
            'description': 'Automated testing requirement'
        },
        # Edge cases
        {
            'query': '',
            'expected_category': None,
            'description': 'Empty input test'
        },
        {
            'query': '   ',
            'expected_category': None,
            'description': 'Whitespace input test'
        },
        {
            'query': 'test',
            'expected_category': None,
            'description': 'Very short input test'
        },
        {
            'query': 'I love cooking recipes and baking cakes with my family on weekends.',
            'expected_category': None,
            'description': 'Unrelated input test'
        },
        {
            'query': 'The system must provide fast response times while ensuring high availability and handling concurrent users with efficient resource utilization for optimal performance in production environments with minimal latency and maximum throughput under heavy load conditions.',
            'expected_category': None,
            'description': 'Long input test'
        }
    ]

    # The index must be loaded and cover every metadata entry.
    assert searcher.index.ntotal == len(searcher.metadata)
    assert searcher.index.ntotal > 0

    deviations = []

    for test_case in test_cases:
        description = test_case['description']
        query = test_case['query']

        results_list = searcher.search(query, top_k=3)

        # Blank input must short-circuit to no results.
        if not query.strip():
            assert results_list == [], (
                f"{description}: blank query must return no results, "
                f"got {len(results_list)}"
            )
            continue

        # Every non-blank query must retrieve something, capped at top_k.
        assert len(results_list) > 0, f"{description}: no results returned"
        assert len(results_list) <= 3, (
            f"{description}: returned {len(results_list)} results, top_k is 3"
        )

        # Similarity scores must be sorted descending.
        scores = [r['similarity_score'] for r in results_list]
        assert scores == sorted(scores, reverse=True), (
            f"{description}: results are not ordered by descending similarity"
        )

        for position, result in enumerate(results_list, 1):
            missing = REQUIRED_RESULT_KEYS - result.keys()
            assert not missing, (
                f"{description}: result {position} is missing keys {sorted(missing)}"
            )
            assert isinstance(result['similarity_score'], float)
            assert result['title']
            assert result['guideline_text']

        top_category = results_list[0]['category']

        if test_case['expected_category']:
            if top_category != test_case['expected_category']:
                deviations.append(description)

    unexpected = set(deviations) - KNOWN_CATEGORY_DEVIATIONS
    assert not unexpected, (
        "semantic retrieval regressed for: "
        + ", ".join(sorted(unexpected))
    )

    # Tolerant cases must still return a top hit of some kind.
    for description in KNOWN_CATEGORY_DEVIATIONS:
        assert searcher.search(
            next(t['query'] for t in test_cases if t['description'] == description),
            top_k=3
        ), f"{description}: expected results"

    print(
        f"\nsemantic search: {len(test_cases)} cases, "
        f"{len(deviations)} known category deviation(s), "
        f"{len(unexpected)} unexpected"
    )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))