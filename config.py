#!/usr/bin/env python3
"""
Configuration settings for DDoS Detection System
"""

import os

# Directory Configuration
INPUT_DIR = "./datasets/ramfs"
ARTEFACTS_DIR = "./artefacts"
RESULTS_DIR = "./results"

# Data Processing Configuration
AGG_PERIOD = "1Min"
TRAIN_SPLIT = "2025-07-15"  # Train on July 14-15 (normal)

# Multiple Attack Periods Configuration
ATTACK_PERIODS = [
    {
        "name": "flow_attack",
        "start": "2025-07-16 17:27:00",
        "end": "2025-07-16 17:33:00",
        "type": "flow_based"
    },
    {
        "name": "volume_attack", 
        "start": "2025-07-16 19:20:00",
        "end": "2025-07-16 19:26:00",
        "type": "volume_based"
    },
    {
        "name": "volume_attack", 
        "start": "2025-07-16 20:50:00",
        "end": "2025-07-16 21:50:00",
        "type": "volume_based"
    }
]

# Model Training Configuration
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Model Architecture Configuration
STANDARD_AE_LAYERS = [64, 32, 16, 32, 64]
STANDARD_AE_ENCODING_DIM = 16
STANDARD_AE_DROPOUT = 0.2

LSTM_AE_LAYERS = [32, 16, 8, 16, 32]
LSTM_AE_SEQUENCE_LENGTH = 10

# Threshold calculation method
# Options: 'statistical' (MSE + stddev), 'percentile', 'enhanced'
THRESHOLD_METHOD = 'statistical'  # Change this to use MSE + stddev thresholds

# Detection sensitivity thresholds
# Higher percentiles = less sensitive (fewer detections)
# Lower percentiles = more sensitive (more detections)
THRESHOLD_PERCENTILES = {
    'standard_ae': {
        'standard': 99,  # Only flag top 1% of anomalies
        'flow': 99       # Only flag top 1% of flow anomalies
    },
    'lstm_ae': {
        'standard': 99,  # Only flag top 1% of anomalies  
        'flow': 99       # Only flag top 1% of flow anomalies
    }
}

# Flow-based attack specific thresholds
FLOW_MULTIPLIER_THRESHOLD = 10.0  # Alert only if flows > 10x baseline

# Visualization Configuration
FIGURE_SIZE_MAIN = (24, 20)
FIGURE_SIZE_TIMELINE = (15, 4)
FIGURE_SIZE_EVOLUTION = (16, 12)
FIGURE_SIZE_ARCHITECTURE = (18, 6)
FIGURE_SIZE_STATISTICAL = (18, 12)
DPI = 300

# Color Schemes
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
MARKERS = ['o', 's', '^', 'v']

# Create necessary directories
os.makedirs(ARTEFACTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Required model files for loading
REQUIRED_MODEL_FILES = [
    "standard_ae.keras",
    "lstm_ae.keras", 
    "scaler.pkl",
    "thresholds.pkl",
    "threshold_config.pkl"
]
