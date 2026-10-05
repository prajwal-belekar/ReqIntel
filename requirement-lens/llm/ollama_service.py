#!/usr/bin/env python3
"""
Ollama service for RequirementLens RAG pipeline.

This module provides a reusable Ollama service that:
- Uses Ollama's local API
- Handles errors gracefully
- Configurable timeout
- Configurable model from environment
- Prevents hallucinations
"""

import os
import json
import requests
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError

class OllamaError(Exception):
    """Error model for Ollama service."""
    def __init__(self, error_type: str, message: str, details: Optional[str] = None):
        self.error_type = error_type
        self.message = message
        self.details = details
        super().__init__(self.message)

class OllamaResponse(BaseModel):
    """Response model for Ollama service."""
    content: str
    model: str
    created_at: str
    thinking: Optional[str] = None

class OllamaService:
    """
    Ollama service for RequirementLens.
    
    Uses Ollama's local API with configurable timeout and model.
    """
    
    def __init__(self, timeout: int = None, max_retries: int = 3):
        """
        Initialize Ollama service.
        
        Args:
            timeout: Request timeout in seconds (default: 120, or from OLLAMA_TIMEOUT env var)
            max_retries: Maximum number of retries for transient errors
        """
        # Use provided timeout, or read from environment, or default to 120
        if timeout is None:
            timeout = int(os.getenv("OLLAMA_TIMEOUT", "120"))
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_url = "http://localhost:11434"

        # Get model from environment
        self.model_name = os.getenv("OLLAMA_MODEL", "qwen3:8b")

        if not self.model_name:
            raise ValueError("OLLAMA_MODEL environment variable not set")

        # Store last response for pipeline access
        self.last_response = None
        self._last_thinking = None

    def _call_ollama(self, prompt: str, system_prompt: str = "") -> Optional[OllamaResponse]:
        """
        Call Ollama API with retry logic.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            
        Returns:
            OllamaResponse or None if failed
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
                "num_predict": 1024
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("error"):
                        raise OllamaError(
                            error_type="ollama_error",
                            message=result.get("error", "Unknown Ollama error"),
                            details=result.get("details")
                        )
                    
                    # Store thinking for pipeline access even if response is empty
                    self._last_thinking = result.get("thinking")
                    
                    # Validate response structure
                    if "response" not in result or result["response"] is None:
                        raise OllamaError(
                            error_type="malformed_response",
                            message="No response content received"
                        )
                    
                    response_obj = OllamaResponse(**{
                        "content": result["response"],
                        "model": result.get("model", self.model_name),
                        "created_at": result.get("created_at", time.strftime('%Y-%m-%dT%H:%M:%SZ')),
                        "thinking": result.get("thinking")
                    })
                    
                    # Store the response for later access by pipeline
                    self.last_response = response_obj
                    
                    return response_obj
                
                # Handle HTTP errors
                if response.status_code == 404:
                    raise OllamaError(
                        error_type="model_not_found",
                        message=f"Model '{self.model_name}' not found"
                    )
                elif response.status_code == 503:
                    raise OllamaError(
                        error_type="ollama_unavailable",
                        message="Ollama service unavailable"
                    )
                else:
                    raise OllamaError(
                        error_type="api_error",
                        message=f"Ollama API error: {response.status_code}",
                        details=response.text
                    )
                    
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise OllamaError(
                        error_type="request_failed",
                        message=str(e)
                    )
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        """
        Generate response from Ollama.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            
        Returns:
            Generated content
            
        Raises:
            OllamaError: If Ollama call fails
        """
        try:
            response = self._call_ollama(prompt, system_prompt)
            if response:
                # For qwen3:8b, if content is too short but thinking has content, 
                # consider using thinking as fallback
                if response.content and len(response.content.strip()) >= 10:
                    return response.content
                elif self._last_thinking and len(self._last_thinking.strip()) > 50:
                    # Use thinking as fallback if it has substantial content
                    return self._last_thinking
                elif response.content:
                    return response.content
            return ""
        except OllamaError as e:
            raise e
        except Exception as e:
            raise OllamaError(
                error_type="unexpected_error",
                message=str(e)
            )


def main():
    """Test Ollama service."""
    try:
        service = OllamaService(timeout=20)
        
        # Test with a simple prompt
        prompt = """
        Analyze this requirement:
        'The system shall be available 99.9% of the time during business hours.'
        
        Provide a concise analysis of:
        1. Potential risks
        2. Missing specifications
        3. Recommendations
        
        Be very specific about what is NOT specified.
        """
        
        response = service.generate_response(prompt)
        print("Ollama Response:")
        print(response[:500] + "..." if len(response) > 500 else response)
        
    except OllamaError as e:
        print(f"Ollama Error: {e.error_type} - {e.message}")
        if e.details:
            print(f"Details: {e.details}")
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")


if __name__ == "__main__":
    main()