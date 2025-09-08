"""
Core components for anomaly detection models.

This module contains the base template and factory for creating
anomaly detection models.
"""

from .template import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin, DummyTrainingHistory
from .factory import (
    create_model, 
    register_model, 
    list_available_models, 
    get_model_info,
    get_model_descriptions,
    validate_model_config
)

__all__ = [
    # Base classes
    'BaseAnomalyDetector',
    'ModelValidationMixin', 
    'ThresholdCalculatorMixin',
    'DummyTrainingHistory',
    
    # Factory functions
    'create_model',
    'register_model',
    'list_available_models',
    'get_model_info',
    'get_model_descriptions',
    'validate_model_config',
]
