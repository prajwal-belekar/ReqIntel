#!/usr/bin/env python3
"""
Pydantic schemas for RequirementLens RAG pipeline.

This module defines structured output models for:
- Guideline references
- Requirement analysis
- Error handling
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator, confloat, conlist
from enum import Enum


class RiskLevel(str, Enum):
    """Possible risk levels."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class GuidelineReference(BaseModel):
    """Reference to a retrieved guideline."""
    id: str = Field(..., description="Guideline identifier")
    title: str = Field(..., description="Guideline title")
    similarity: confloat(ge=0.0, le=1.0) = Field(..., description="Similarity score between 0.0 and 1.0")
    category: str = Field(..., description="Guideline category")
    text: str = Field(..., description="Guideline text")


class RequirementAnalysis(BaseModel):
    """Structured analysis of a software requirement."""
    requirement: str = Field(..., description="Original requirement text")
    category: str = Field(..., description="ML predicted category")
    ml_confidence: confloat(ge=0.0, le=1.0) = Field(..., description="ML confidence score (0.0-1.0)")
    risk_level: RiskLevel = Field(..., description="Assessed risk level")
    risk_factors: List[str] = Field(..., description="List of identified risk factors")
    completeness_score: confloat(ge=0.0, le=1.0) = Field(..., description="Completeness score (0.0-1.0)")
    missing_requirements: List[str] = Field(..., description="List of missing specifications")
    recommendations: List[str] = Field(..., description="List of recommendations")
    similar_guidelines: List[GuidelineReference] = Field(..., description="Retrieved similar guidelines")
    explanation: str = Field(..., description="Explanation of the analysis")
    
    @validator('risk_level')
    def validate_risk_level(cls, v):
        """Ensure risk level is valid."""
        if v not in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]:
            raise ValueError(f"Invalid risk level: {v}. Must be one of: {list(RiskLevel)}")
        return v
    
    @validator('completeness_score', 'ml_confidence')
    def validate_scores(cls, v):
        """Ensure scores are between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v


class LLMError(BaseModel):
    """Error model for LLM processing."""
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional details")
    raw_output: Optional[str] = Field(None, description="Raw LLM output if available")


class LLMResponse(BaseModel):
    """Response model for LLM output."""
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used")
    created_at: str = Field(..., description="Creation timestamp")


def validate_llm_json(json_str: str) -> dict:
    """
    Validate JSON output from LLM.
    
    Args:
        json_str: Raw JSON string from LLM
        
    Returns:
        Parsed and validated dictionary
        
    Raises:
        ValueError: If JSON is invalid
    """
    try:
        import json
        data = json.loads(json_str)
        
        # Basic validation
        required_fields = ['risk_level', 'risk_factors', 'completeness_score', 
                          'missing_requirements', 'recommendations', 'explanation']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
        
        # Validate risk_level
        if data['risk_level'] not in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]:
            raise ValueError(f"Invalid risk_level: {data['risk_level']}")
        
        # Validate scores
        if not (0.0 <= data.get('completeness_score', 0) <= 1.0):
            raise ValueError(f"Invalid completeness_score: {data.get('completeness_score')}")
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise ValueError(f"Validation error: {str(e)}")


def repair_llm_json(json_str: str) -> dict:
    """
    Attempt to repair malformed JSON from LLM.
    
    Args:
        json_str: Raw JSON string from LLM
        
    Returns:
        Parsed and validated dictionary
        
    Raises:
        ValueError: If JSON cannot be repaired
    """
    try:
        import json
        
        # Try to parse
        try:
            data = json.loads(json_str)
            return validate_llm_json(json_str)
        except json.JSONDecodeError:
            pass
        
        # Try to fix common issues
        json_str = json_str.strip()
        
        # Fix missing brackets
        if json_str.startswith('{') and not json_str.endswith('}'):
            json_str += '}'
        if not json_str.startswith('{') and json_str.endswith('}'):
            json_str = '{' + json_str
        
        # Try again
        try:
            data = json.loads(json_str)
            return validate_llm_json(json_str)
        except json.JSONDecodeError:
            pass
        
        # If all else fails, raise error
        raise ValueError(f"Could not parse JSON: {json_str[:200]}...")
        
    except Exception as e:
        raise ValueError(f"Repair failed: {str(e)}")


def main():
    """Test schema validation."""
    
    # Test valid JSON
    valid_json = '''
    {
        "risk_level": "Medium",
        "risk_factors": ["Insufficient error handling", "No timeout specified"],
        "completeness_score": 0.65,
        "missing_requirements": ["Maximum file size", "Supported file formats"],
        "recommendations": ["Add error handling", "Specify file size limits"],
        "explanation": "The requirement specifies availability but lacks details about file handling."
    }
    '''
    
    try:
        result = validate_llm_json(valid_json)
        print("✅ Valid JSON parsed successfully")
        print(f"Risk level: {result['risk_level']}")
        print(f"Completeness score: {result['completeness_score']}")
    except Exception as e:
        print(f"❌ Valid JSON failed: {e}")
    
    # Test invalid JSON
    invalid_json = '''
    {
        "risk_level": "Very High",  # Invalid value
        "risk_factors": ["No risk"],
        "completeness_score": 1.2,  # Invalid value
        "missing_requirements": [],
        "recommendations": [],
        "explanation": ""
    }
    '''
    
    try:
        result = validate_llm_json(invalid_json)
        print("❌ Invalid JSON should have failed")
    except Exception as e:
        print(f"✅ Invalid JSON correctly rejected: {e}")
    
    # Test malformed JSON
    malformed_json = '''
    {
        "risk_level": "Medium",
        "risk_factors": ["No risk"],
        "completeness_score": 0.75,
        "missing_requirements": [],
        "recommendations": [],
        "explanation": ""
    '''
    
    try:
        result = validate_llm_json(malformed_json)
        print("❌ Malformed JSON should have failed")
    except Exception as e:
        print(f"✅ Malformed JSON correctly rejected: {e}")


if __name__ == "__main__":
    main()