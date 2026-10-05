#!/usr/bin/env python3
"""
Classifier module for RequirementLens ML pipeline.

This module provides a simple interface for making predictions using
the trained TF-IDF + Logistic Regression model from Experiment 2.
"""

import joblib
from pathlib import Path
import numpy as np

# Set up paths
base_path = Path(__file__).parent
models_path = base_path / 'models'

# Load the trained model and vectorizer (Experiment 2: best performing)
_model = None
_vectorizer = None
_class_labels = None

def _load_model():
    """Load the trained model and vectorizer."""
    global _model, _vectorizer, _class_labels
    
    if _model is None or _vectorizer is None:
        try:
            _model = joblib.load(models_path / 'experiment2_logistic_regression_balanced.pkl')
            _vectorizer = joblib.load(models_path / 'experiment2_tfidf_vectorizer.pkl')
            # Get class labels from the model
            _class_labels = _model.classes_.tolist()
        except Exception as e:
            raise RuntimeError(f"Failed to load ML model: {str(e)}")

def predict_category(requirement_text: str) -> tuple[str, float]:
    """
    Predict the category of a requirement text.
    
    Args:
        requirement_text: The requirement text to classify
        
    Returns:
        Tuple of (category, confidence)
        
    Raises:
        ValueError: If input is invalid
        RuntimeError: If model loading fails
    """
    if not requirement_text or not isinstance(requirement_text, str):
        raise ValueError("Requirement text must be a non-empty string")
    
    if not requirement_text.strip():
        raise ValueError("Requirement text cannot be empty or whitespace only")
    
    # Load model if not already loaded
    _load_model()
    
    try:
        # Vectorize the input text
        X = _vectorizer.transform([requirement_text.strip()])
        
        # Get prediction and probabilities
        prediction = _model.predict(X)[0]
        probabilities = _model.predict_proba(X)[0]
        
        # Get confidence (probability of predicted class)
        class_index = list(_model.classes_).index(prediction)
        confidence = float(probabilities[class_index])
        
        return (prediction, confidence)
        
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {str(e)}")

def get_available_categories() -> list[str]:
    """
    Get the list of available categories.
    
    Returns:
        List of category labels
    """
    _load_model()
    return _class_labels.copy()

def get_model_info() -> dict:
    """
    Get information about the loaded model.
    
    Returns:
        Dictionary with model information
    """
    _load_model()
    return {
        'model_type': 'LogisticRegression',
        'vectorizer_type': 'TfidfVectorizer',
        'classes': _class_labels,
        'num_classes': len(_class_labels)
    }


def main():
    """Test the classifier."""
    try:
        # Test with a sample requirement
        test_req = "The system shall be available 99.9% of the time during business hours."
        category, confidence = predict_category(test_req)
        
        print(f"Requirement: {test_req}")
        print(f"Predicted Category: {category}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show available categories
        categories = get_available_categories()
        print(f"\nAvailable Categories ({len(categories)}):")
        for cat in sorted(categories):
            print(f"  {cat}")
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()