"""
Temporal Models Package.

This package contains temporal-based anomaly detection components for models
including base classes, temporal components, and training utilities.

Available Components:
- BaseTemporalAutoencoder: Base class for temporal autoencoder models
- SequenceProcessor: Utility for handling sequence creation and processing
- TemporalBlock, LSTMEncoder, LSTMDecoder: LSTM components
- TCNEncoder, TCNDecoder: TCN components

Training Utilities:
- WarmupLearningRateScheduler: Learning rate scheduler with warmup
- TemporalRegularizationCallback: Regularization callback for temporal models
- Temporal callbacks and data augmentation

Usage:
    from framework.models.temporal import BaseTemporalAutoencoder
    from framework.models.temporal import SequenceProcessor
    from framework.models import LSTMAutoencoder, TCNAutoencoder  # These are in main models package
    
    # Create temporal models
    lstm_model = LSTMAutoencoder(sequence_length=60, latent_dim=32)
    tcn_model = TCNAutoencoder(sequence_length=60, num_blocks=4)
"""

from .base_temporal_autoencoder import BaseTemporalAutoencoder
from .components import SequenceProcessor, TemporalBlock, LSTMEncoder, LSTMDecoder, TCNEncoder, TCNDecoder
from .training_utils import (
    get_enhanced_temporal_callbacks, 
    create_temporal_data_augmentation,
    WarmupLearningRateScheduler,
    TemporalRegularizationCallback
)

__all__ = [
    'BaseTemporalAutoencoder',
    'SequenceProcessor',
    'TemporalBlock',
    'LSTMEncoder',
    'LSTMDecoder',
    'TCNEncoder',
    'TCNDecoder',
    'get_enhanced_temporal_callbacks',
    'create_temporal_data_augmentation',
    'WarmupLearningRateScheduler',
    'TemporalRegularizationCallback',
]
