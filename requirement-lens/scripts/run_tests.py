#!/usr/bin/env python3
"""
Simple test runner for RequirementLens.
Runs all tests and reports results.
"""

import sys
import subprocess
import os

def run_tests():
    """Run all tests and return the exit code."""
    print("=" * 80)
    print("RUNNING REQUIREMENTLENS TESTS")
    print("=" * 80)
    
    # Test database
    print("\n1. Testing database layer...")
    try:
         result = subprocess.run(
             [sys.executable, "-m", "pytest", "tests/test_database.py", "-v"],
             capture_output=True,
             text=True,
             cwd=os.path.join(os.path.dirname(__file__), '..')
         )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✓ Database tests PASSED")
        else:
            print("✗ Database tests FAILED")
            return result.returncode
    except Exception as e:
        print(f"ERROR running database tests: {e}")
        return 1
    
    # Test RAG pipeline
    print("\n2. Testing RAG pipeline...")
    try:
         result = subprocess.run(
             [sys.executable, "-m", "pytest", "tests/test_rag_pipeline.py", "-v"],
             capture_output=True,
             text=True,
             cwd=os.path.join(os.path.dirname(__file__), '..')
         )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✓ RAG pipeline tests PASSED")
        else:
            print("✗ RAG pipeline tests FAILED")
            return result.returncode
    except Exception as e:
        print(f"ERROR running RAG pipeline tests: {e}")
        return 1
    
    # Test semantic search
    print("\n3. Testing semantic search...")
    try:
         result = subprocess.run(
             [sys.executable, "-m", "pytest", "tests/test_semantic_search.py", "-v"],
             capture_output=True,
             text=True,
             cwd=os.path.join(os.path.dirname(__file__), '..')
         )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✓ Semantic search tests PASSED")
        else:
            print("✗ Semantic search tests FAILED")
            return result.returncode
    except Exception as e:
        print(f"ERROR running semantic search tests: {e}")
        return 1
    
    # Test API endpoints
    print("\n4. Testing API endpoints...")
    try:
         result = subprocess.run(
             [sys.executable, "-m", "pytest", "tests/test_api_endpoints.py", "-v"],
             capture_output=True,
             text=True,
             cwd=os.path.join(os.path.dirname(__file__), '..')
         )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print("✓ API endpoints tests PASSED")
        else:
            print("✗ API endpoints tests FAILED")
            return result.returncode
    except Exception as e:
        print(f"ERROR running API endpoints tests: {e}")
        return 1
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)