"""
Anomaly Detection Models Package.

This package contains implementations of various anomaly detection models
that follow a consistent interface defined by the BaseAnomalyDetector class.

Available Models:
- AutoencoderAnomalyDetector: Neural network autoencoder for reconstruction-based anomaly detection
- LSTMAutoencoder: LSTM-based autoencoder for temporal anomaly detection
- TCNAutoencoder: Temporal Convolutional Network autoencoder for sequential data
- IsolationForestAnomalyDetector: Ensemble method using isolation trees
- OneClassSVMAnomalyDetector: Support Vector Machine for one-class classification

Usage:
    from framework.models import create_model, list_available_models
    
    # Create a model
    model = create_model('autoencoder', latent_dim=42)
    
    # Create temporal models
    lstm_model = create_model('lstm_autoencoder', sequence_length=60, latent_dim=32)
    tcn_model = create_model('tcn_autoencoder', sequence_length=60, num_blocks=4)
    
    # List available models
    models = list_available_models()
"""

from .core import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin, DummyTrainingHistory
from .temporal import (
    BaseTemporalAutoencoder, 
    SequenceProcessor, 
    TemporalBlock, 
    LSTMEncoder, 
    LSTMDecoder, 
    TCNEncoder, 
    TCNDecoder
)
from .core import (
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
    from .lstm_autoencoder import LSTMAutoencoder
except ImportError:
    LSTMAutoencoder = None

try:
    from .tcn_autoencoder import TCNAutoencoder
except ImportError:
    TCNAutoencoder = None

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
    'BaseTemporalAutoencoder',
    'ModelValidationMixin', 
    'ThresholdCalculatorMixin',
    'DummyTrainingHistory',
    
    # Temporal components
    'SequenceProcessor',
    'TemporalBlock',
    'LSTMEncoder',
    'LSTMDecoder', 
    'TCNEncoder',
    'TCNDecoder',
    
    # Factory functions
    'create_model',
    'register_model',
    'list_available_models',
    'get_model_info',
    'get_model_descriptions',
    'validate_model_config',
    
    # Model classes
    'AutoencoderAnomalyDetector',
    'LSTMAutoencoder',
    'TCNAutoencoder',
    'IsolationForestAnomalyDetector',
    'OneClassSVMAnomalyDetector',
]
