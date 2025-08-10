#!/usr/bin/env python3
"""
DDoS Detection System with Autoencoders

Enhanced system for detecting DDoS attacks using multiple autoencoder variants:
1. Standard Autoencoder
2. LSTM Autoencoder (for temporal patterns)

Features:
- Advanced feature engineering
- Real-time detection capabilities
- Multiple model comparison
- Attack timeline analysis
- GPU acceleration only during training
"""

from .config import *
from .utils import (
    configure_gpu_for_training,
    configure_cpu_for_inference,
    check_gpu_availability
)
from .feature_engineering import AdvancedFeatureEngineer
from .models import StandardAutoencoder, LSTMAutoencoder
from .detection_system import DDoSDetectionSystem
from .visualization import DDoSVisualizer
from .main import main

__version__ = "1.0.0"
__author__ = "Murilo Chianfa"
__description__ = "Enhanced DDoS Detection System with Autoencoders"

__all__ = [
    'main',
    'DDoSDetectionSystem',
    'DDoSVisualizer',
    'AdvancedFeatureEngineer',
    'StandardAutoencoder',
    'LSTMAutoencoder',
    'configure_gpu_for_training',
    'configure_cpu_for_inference',
    'check_gpu_availability'
]
