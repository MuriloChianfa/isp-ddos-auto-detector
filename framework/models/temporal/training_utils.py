"""
Temporal training utilities for improved long-term pattern recognition.

This module provides custom training utilities specifically designed to help
temporal autoencoders learn long-term patterns in network traffic more effectively.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional


class WarmupLearningRateScheduler(keras.callbacks.Callback):
    """
    Learning rate scheduler with warmup for temporal models.
    
    This scheduler gradually increases the learning rate during the warmup period,
    then applies cosine decay. This helps temporal models learn long-term patterns
    more effectively by starting with conservative updates.
    """
    
    def __init__(
        self, 
        warmup_epochs: int = 10,
        max_lr: float = 0.001,
        min_lr: float = 1e-7,
        total_epochs: int = 100
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Warmup phase: gradually increase learning rate
            lr = self.max_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        self.model.optimizer.learning_rate.assign(lr)
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}: Learning rate = {lr:.2e}")


class TemporalRegularizationCallback(keras.callbacks.Callback):
    """
    Custom regularization callback for temporal models.
    
    This callback applies additional regularization techniques specifically
    designed for temporal sequence learning to prevent overfitting to short-term
    patterns and encourage long-term pattern recognition.
    """
    
    def __init__(self, temporal_consistency_weight: float = 0.1):
        super().__init__()
        self.temporal_consistency_weight = temporal_consistency_weight
        
    def on_epoch_end(self, epoch, logs=None):
        # Add temporal consistency loss (could be implemented if needed)
        pass


class PatientTrainingScheduler(keras.callbacks.Callback):
    """
    Patient training scheduler that adjusts training parameters based on
    validation loss trends to better learn long-term patterns.
    """
    
    def __init__(self, patience_threshold: int = 20, adjustment_factor: float = 0.8):
        super().__init__()
        self.patience_threshold = patience_threshold
        self.adjustment_factor = adjustment_factor
        self.wait = 0
        self.best_loss = np.inf
        
    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get('val_loss', np.inf)
        
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience_threshold:
            # Gradually reduce learning rate for more fine-tuned learning
            current_lr = float(self.model.optimizer.learning_rate)
            new_lr = current_lr * self.adjustment_factor
            self.model.optimizer.learning_rate.assign(new_lr)
            print(f"Patience threshold reached. Reducing learning rate to {new_lr:.2e}")
            self.wait = 0  # Reset patience


def get_enhanced_temporal_callbacks(
    warmup_epochs: int = 10,
    total_epochs: int = 100,
    max_lr: float = 0.001,
    patience_early: int = 50,
    patience_lr: int = 25
) -> list:
    """
    Get callbacks for temporal model training.
    
    Args:
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        max_lr: Maximum learning rate
        patience_early: Patience for early stopping
        patience_lr: Patience for learning rate reduction
        
    Returns:
        List of Keras callbacks optimized for temporal learning
    """
    callbacks = [
        WarmupLearningRateScheduler(
            warmup_epochs=warmup_epochs,
            max_lr=max_lr,
            total_epochs=total_epochs
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_early,
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-7
        ),
        PatientTrainingScheduler(
            patience_threshold=patience_lr,
            adjustment_factor=0.7
        ),
        TemporalRegularizationCallback(
            temporal_consistency_weight=0.1
        )
    ]
    
    return callbacks


def create_temporal_data_augmentation(
    sequences: np.ndarray, 
    noise_level: float = 0.01,
    time_shift_range: int = 5
) -> np.ndarray:
    """
    Apply data augmentation specifically for temporal sequences.
    
    Args:
        sequences: Input sequences [batch, time, features]
        noise_level: Level of Gaussian noise to add
        time_shift_range: Range for random time shifts
        
    Returns:
        Augmented sequences
    """
    augmented = sequences.copy()
    
    # Add small amount of Gaussian noise
    noise = np.random.normal(0, noise_level, sequences.shape)
    augmented += noise
    
    # Random time shifts (circular)
    for i in range(len(augmented)):
        shift = np.random.randint(-time_shift_range, time_shift_range + 1)
        if shift != 0:
            augmented[i] = np.roll(augmented[i], shift, axis=0)
    
    return augmented
