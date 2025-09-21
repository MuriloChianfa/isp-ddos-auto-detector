"""
Neural network autoencoder for anomaly detection.

This module implements an autoencoder-based anomaly detection model that learns
to reconstruct normal network traffic patterns and identifies anomalies based
on reconstruction error.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, Any, Optional, List
import logging

import tensorflow as tf
from tensorflow import keras

from .core.template import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin

logger = logging.getLogger(__name__)


class AutoencoderAnomalyDetector(BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin):
    """
    Autoencoder-based anomaly detection model.
    
    This model uses a neural network autoencoder to learn normal traffic patterns
    and detect anomalies based on reconstruction error. The autoencoder is trained
    to reconstruct normal network traffic features, and samples with high
    reconstruction error are classified as anomalies.
    
    Attributes:
        latent_dim: Dimension of the latent (bottleneck) layer
        autoencoder: The Keras autoencoder model
        scaler: MinMax scaler for feature normalization
        history: Training history from model fitting
    """
    
    def __init__(self, latent_dim: int = 8):
        """
        Initialize the autoencoder anomaly detector.
        
        Args:
            latent_dim: Dimension of the latent (bottleneck) layer
        """
        super().__init__(model_name="autoencoder")
        self.latent_dim = latent_dim
        self.autoencoder = None
        self.scaler = MinMaxScaler()
        self.history = None
        
    def build_model(self, input_dim: int) -> None:
        """Build the autoencoder architecture with more capacity"""
        print(f"Building autoencoder for {input_dim} features...")
        
        # Validate input
        if input_dim <= 0:
            raise ValueError(f"Invalid input dimension: {input_dim}")
        
        hidden_dim1 = 24     # First hidden layer
        hidden_dim2 = 18     # Second hidden layer
        hidden_dim3 = 12     # Third hidden layer
        self.latent_dim = 8
        
        self.autoencoder = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            
            # Encoder
            keras.layers.Dense(hidden_dim1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(hidden_dim2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(hidden_dim3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            
            # Bottleneck (latent space)
            keras.layers.Dense(self.latent_dim, activation='relu'),
            
            # Decoder
            keras.layers.Dense(hidden_dim3, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(hidden_dim2, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            
            keras.layers.Dense(hidden_dim1, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            
            # Output layer
            keras.layers.Dense(input_dim, activation='linear')
        ])
        
        # Optimizer settings
        initial_learning_rate = 0.01
        optimizer = keras.optimizers.Adam(
            learning_rate=initial_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        self.autoencoder.compile(
            optimizer=optimizer, 
            loss='mse', 
            metrics=['mae']
        )
        
        logger.info(f"autoencoder architecture:")
        logger.info(f"  Input: {input_dim} features")
        logger.info(f"  Hidden layers: {hidden_dim1} -> {hidden_dim2} -> {hidden_dim3}")
        logger.info(f"  Latent dimension: {self.latent_dim}")
        
        print(f"autoencoder architecture:")
        print(f"  Input: {input_dim} features")
        print(f"  Hidden layers: {hidden_dim1} -> {hidden_dim2} -> {hidden_dim3}")
        print(f"  Latent dimension: {self.latent_dim}")
        
    def fit_scaler(self, training_features) -> None:
        """Fit the scaler on training data"""
        if training_features is None or len(training_features) == 0:
            raise ValueError("Training features cannot be empty")
        
        # Convert to numpy array if it's a DataFrame to avoid feature name warnings
        if hasattr(training_features, 'values'):
            training_data = training_features.values
        else:
            training_data = training_features
            
        self.scaler.fit(training_data)
        logger.info(f"Fitted scaler on {len(training_features)} training samples")
        
    def transform_data(self, features) -> np.ndarray:
        """Transform features using the fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        # Convert to numpy array if it's a DataFrame to avoid feature name warnings
        if hasattr(features, 'values'):
            feature_data = features.values
        else:
            feature_data = features
            
        return self.scaler.transform(feature_data)
        
    def train(self, scaled_train_data: np.ndarray, scaled_validation_data: Optional[np.ndarray] = None, 
              epochs: int = 100, batch_size: int = 64, **kwargs) -> Dict[str, Any]:
        """Train the autoencoder with improved training strategy"""
        
        # Validate training data
        self.validate_training_data(scaled_train_data, scaled_validation_data)
        self._log_training_start(scaled_train_data, scaled_validation_data)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1,
                min_delta=1e-6
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        print("Training autoencoder...")
        print(f"Training samples: {len(scaled_train_data)}")
        if scaled_validation_data is not None:
            print(f"Validation samples: {len(scaled_validation_data)}")
        print(f"Batch size: {batch_size}")
        
        validation_data = None
        if scaled_validation_data is not None:
            validation_data = (scaled_validation_data, scaled_validation_data)
        
        self.history = self.autoencoder.fit(
            scaled_train_data, scaled_train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        self._log_training_complete()
        final_epoch = len(self.history.history['loss'])
        print(f"Training completed after {final_epoch} epochs")
        return self.history
        
    def predict(self, scaled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get reconstructions and calculate MSE"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        self.validate_input_dimensions(scaled_data)
        
        reconstructions = self.autoencoder.predict(scaled_data, verbose=0)
        mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
        return reconstructions, mse
        
    def calculate_threshold(self, train_mse: np.ndarray, val_mse: np.ndarray, 
                          strategy: str = 'exponential_threshold') -> Tuple[float, Dict[str, float]]:
        """Calculate anomaly detection threshold"""
        self.validate_threshold_strategy(strategy)
        
        normal_mse_values = np.concatenate([train_mse, val_mse])
        all_strategies = self.calculate_threshold_strategies(normal_mse_values)
        
        self.threshold = all_strategies[strategy]
        
        # Log threshold information
        self.log_threshold_info(normal_mse_values, self.threshold, strategy)
        
        print(f"Normal Traffic Statistics:")
        print(f"  Mean MSE: {np.mean(normal_mse_values):.6f}")
        print(f"  Std MSE:  {np.std(normal_mse_values):.6f}")
        print(f"  Selected Threshold ({strategy}): {self.threshold:.6f}")
        
        return self.threshold, all_strategies
        
    def get_metrics(self, scaled_data: np.ndarray, reconstructions: np.ndarray, 
                   mse: np.ndarray) -> Dict[str, float]:
        """Calculate reconstruction metrics"""
        mae = np.mean(np.abs(scaled_data - reconstructions))
        mse_value = np.mean(mse)
        rmse = np.sqrt(mse_value)
        
        return {
            'mae': mae,
            'mse': mse_value,
            'rmse': rmse
        }
    
    def analyze_feature_importance(self, scaled_data: np.ndarray, 
                                 feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze feature importance based on reconstruction errors"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        reconstructions = self.autoencoder.predict(scaled_data, verbose=0)
        
        # Calculate feature-wise reconstruction errors
        feature_errors = np.mean(np.abs(scaled_data - reconstructions), axis=0)
        
        # Create importance ranking
        importance_indices = np.argsort(feature_errors)[::-1]  # Descending order
        
        print("\nFeature Importance Analysis (Top 15 features):")
        print("=" * 60)
        for i, idx in enumerate(importance_indices[:15]):
            print(f"{i+1:2d}. {feature_names[idx]:<30} | Error: {feature_errors[idx]:.6f}")
        
        logger.info("Feature importance analysis completed")
        
        return feature_errors, importance_indices
