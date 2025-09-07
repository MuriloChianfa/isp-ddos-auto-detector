"""
Anomaly Detection Models Package.

This package contains implementations of various anomaly detection models
that follow a consistent interface defined by the BaseAnomalyDetector class.

Available Models:
- AutoencoderAnomalyDetector: Neural network autoencoder for reconstruction-based anomaly detection
- IsolationForestAnomalyDetector: Ensemble method using isolation trees
- OneClassSVMAnomalyDetector: Support Vector Machine for one-class classification

Usage:
    from framework.models import create_model, list_available_models
    
    # Create a model
    model = create_model('autoencoder', latent_dim=42)
    
    # List available models
    models = list_available_models()
"""

from .base_model import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin, DummyTrainingHistory
from .model_factory import (
    create_model, 
    register_model, 
    list_available_models, 
    get_model_info,
    get_model_descriptions,
    validate_model_config
)

# Import specific models
from .autoencoder import AutoencoderAnomalyDetector

try:
    from .isolation_forest import IsolationForestAnomalyDetector
except ImportError:
    IsolationForestAnomalyDetector = None

try:
    from .one_class_svm import OneClassSVMAnomalyDetector
except ImportError:
    OneClassSVMAnomalyDetector = None

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
    
    # Model classes
    'AutoencoderAnomalyDetector',
    'IsolationForestAnomalyDetector',
    'OneClassSVMAnomalyDetector',
]
