"""
Base temporal autoencoder for sequence-based anomaly detection.

This module provides a common base class for temporal autoencoder models,
implementing shared functionality and ensuring consistent interfaces.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Tuple, Any, Optional, List
import logging

from tensorflow import keras

from ..core.template import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin
from .components import SequenceProcessor
from .training_utils import get_enhanced_temporal_callbacks, create_temporal_data_augmentation

logger = logging.getLogger(__name__)


class BaseTemporalAutoencoder(BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin):
    """
    Base class for temporal autoencoder models.
    
    This class provides common functionality for sequence-based autoencoder models,
    including sequence processing, training, and evaluation methods.
    
    Attributes:
        sequence_length: Length of input sequences
        latent_dim: Dimension of the latent representation
        model: The Keras model
        scaler: MinMax scaler for feature normalization
        sequence_processor: Utility for sequence operations
        history: Training history
    """
    
    def __init__(
        self, 
        model_name: str,
        sequence_length: int = 60, 
        latent_dim: int = 32
    ):
        """
        Initialize the temporal autoencoder.
        
        Args:
            model_name: Name identifier for the model
            sequence_length: Length of input sequences
            latent_dim: Dimension of the latent representation
        """
        super().__init__(model_name=model_name)
        self.sequence_length = sequence_length
        self.latent_dim = latent_dim
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_processor = SequenceProcessor(sequence_length)
        self.history = None
        self._feature_dim = None
        
    def fit_scaler(self, training_features: np.ndarray) -> None:
        """
        Fit the scaler on training data.
        
        Args:
            training_features: Training features to fit the scaler
            
        Raises:
            ValueError: If training features are empty
        """
        if training_features is None or len(training_features) == 0:
            raise ValueError("Training features cannot be empty")
        
        self.scaler.fit(training_features)
        logger.info(f"Fitted scaler on {len(training_features)} training samples")
        
    def transform_data(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        
        Args:
            features: Features to transform
            
        Returns:
            Scaled features
            
        Raises:
            ValueError: If scaler is not fitted
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        return self.scaler.transform(features)
        
    def _prepare_training_data(
        self, 
        scaled_train_data: np.ndarray, 
        scaled_validation_data: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare training data by creating sequences.
        
        Args:
            scaled_train_data: Scaled training data
            scaled_validation_data: Scaled validation data (optional)
            
        Returns:
            Tuple of (training_sequences, validation_sequences)
        """
        # Create training sequences
        train_sequences = self.sequence_processor.create_sequences(scaled_train_data)
        logger.info(f"Created {len(train_sequences)} training sequences")
        
        # Create validation sequences if provided
        val_sequences = None
        if scaled_validation_data is not None:
            val_sequences = self.sequence_processor.create_sequences(scaled_validation_data)
            logger.info(f"Created {len(val_sequences)} validation sequences")
        
        return train_sequences, val_sequences
        
    def _get_training_callbacks(self, 
                                   patience_early: int = 40, 
                                   patience_lr: int = 15,
                                   use_enhanced_callbacks: bool = False,  # Default to simple callbacks
                                   total_epochs: int = 80) -> List:
        """
        Get training callbacks optimized for temporal patterns.
        
        Args:
            patience_early: Patience for early stopping (conservative for temporal models)
            patience_lr: Patience for learning rate reduction
            use_enhanced_callbacks: Whether to use enhanced temporal callbacks
            total_epochs: Total training epochs for learning rate scheduling
            
        Returns:
            List of Keras callbacks
        """
        if use_enhanced_callbacks:
            # Use enhanced callbacks for better temporal learning
            return get_enhanced_temporal_callbacks(
                warmup_epochs=max(8, total_epochs // 10),
                total_epochs=total_epochs,
                max_lr=0.0005,  # Lower max learning rate for stability
                patience_early=patience_early,
                patience_lr=patience_lr
            )
        else:
            # Conservative standard callbacks
            return [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=patience_early,  # Patient for temporal models
                    restore_best_weights=True,
                    verbose=1,
                    min_delta=1e-6
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.8,  # Gentle LR reduction
                    patience=patience_lr,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
    def train(
        self, 
        scaled_train_data: np.ndarray, 
        scaled_validation_data: Optional[np.ndarray] = None,
        epochs: int = 100, 
        batch_size: int = 32,
        use_data_augmentation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the temporal autoencoder with enhanced techniques for long-term patterns.
        
        Args:
            scaled_train_data: Scaled training data
            scaled_validation_data: Scaled validation data (optional)
            epochs: Number of training epochs
            batch_size: Training batch size
            use_data_augmentation: Whether to use temporal data augmentation
            **kwargs: Additional training parameters
            
        Returns:
            Training history
            
        Raises:
            ValueError: If model is not built or training data is invalid
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
            
        # Validate training data
        self.validate_training_data(scaled_train_data, scaled_validation_data)
        self._log_training_start(scaled_train_data, scaled_validation_data)
        
        # Prepare sequences
        train_sequences, val_sequences = self._prepare_training_data(
            scaled_train_data, scaled_validation_data
        )
        
        # Apply conservative data augmentation only if specifically requested
        if use_data_augmentation and len(train_sequences) > 500:  # Only for larger datasets
            print("Applying conservative temporal data augmentation...")
            augmented_sequences = create_temporal_data_augmentation(
                train_sequences[:len(train_sequences)//2],  # Only augment half the data
                noise_level=0.002,  # Very conservative noise level
                time_shift_range=2   # Very small time shifts
            )
            # Combine original and augmented data
            train_sequences = np.concatenate([train_sequences, augmented_sequences], axis=0)
            print(f"Training sequences after conservative augmentation: {len(train_sequences)}")
        else:
            print("Skipping data augmentation for conservative training")
        
        # Prepare validation data for Keras
        validation_data = None
        if val_sequences is not None:
            validation_data = (val_sequences, val_sequences)
        
        # Get conservative callbacks
        callbacks = self._get_training_callbacks(
            patience_early=max(25, epochs // 3),  # Very patient
            patience_lr=max(10, epochs // 8),
            use_enhanced_callbacks=False,  # Use simple callbacks for stability
            total_epochs=epochs
        )
        
        print(f"Training {self.model_name} with enhanced temporal learning...")
        print(f"Training sequences: {len(train_sequences)}")
        print(f"Validation sequences: {len(val_sequences) if val_sequences is not None else 0}")
        print(f"Batch size: {batch_size}")
        print(f"Target epochs: {epochs} (with early stopping)")
        
        # Train the model
        self.history = self.model.fit(
            train_sequences, train_sequences,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )
        
        self.is_trained = True
        final_epoch = len(self.history.history['loss'])
        print(f"Training completed after {final_epoch} epochs")
        
        # Print final training metrics
        final_train_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history.get('val_loss', [None])[-1]
        print(f"Final training loss: {final_train_loss:.6f}")
        if final_val_loss is not None:
            print(f"Final validation loss: {final_val_loss:.6f}")
        
        return self.history
        
    def predict(self, scaled_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get reconstructions and calculate MSE for sequences.
        
        Args:
            scaled_data: Scaled input data
            
        Returns:
            Tuple of (reconstructions, mse_values)
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create sequences
        sequences = self.sequence_processor.create_sequences(scaled_data)
        
        # Get reconstructions
        reconstructions = self.model.predict(sequences, verbose=0)
        
        # Calculate MSE for each sequence
        mse = self.sequence_processor.calculate_sequence_mse(sequences, reconstructions)
        
        return reconstructions, mse
        
    def calculate_threshold(
        self, 
        train_mse: np.ndarray, 
        val_mse: np.ndarray,
        strategy: str = 'exponential_threshold'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate anomaly detection threshold.
        
        Args:
            train_mse: MSE values from training data
            val_mse: MSE values from validation data
            strategy: Threshold calculation strategy
            
        Returns:
            Tuple of (selected_threshold, all_thresholds)
        """
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
        
    def get_metrics(
        self, 
        scaled_data: np.ndarray, 
        reconstructions: np.ndarray,
        mse: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate reconstruction metrics.
        
        Args:
            scaled_data: Scaled input data
            reconstructions: Model reconstructions
            mse: MSE values
            
        Returns:
            Dictionary of metrics
        """
        sequences = self.sequence_processor.create_sequences(scaled_data)
        mae = np.mean(np.abs(sequences - reconstructions))
        mse_value = np.mean(mse)
        rmse = np.sqrt(mse_value)
        
        return {
            'mae': mae,
            'mse': mse_value,
            'rmse': rmse
        }
    
    def analyze_temporal_importance(
        self, 
        scaled_data: np.ndarray, 
        feature_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Analyze temporal and feature importance.
        
        Args:
            scaled_data: Scaled input data
            feature_names: Names of the features
            
        Returns:
            Dictionary containing importance analysis results
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        sequences = self.sequence_processor.create_sequences(scaled_data)
        reconstructions = self.model.predict(sequences, verbose=0)
        
        # Calculate errors
        temporal_errors = np.mean(np.abs(sequences - reconstructions), axis=(0, 2))  # Error per timestep
        feature_errors = np.mean(np.abs(sequences - reconstructions), axis=(0, 1))  # Error per feature
        
        print(f"\nTemporal Importance Analysis ({self.model_name}):")
        print("=" * 60)
        for t in range(min(10, len(temporal_errors))):
            print(f"Timestep {t+1:2d}: {temporal_errors[t]:.6f}")
        
        print(f"\nFeature Importance Analysis (Top 10):")
        print("=" * 60)
        importance_indices = np.argsort(feature_errors)[::-1]
        for i, idx in enumerate(importance_indices[:10]):
            feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature_{idx}"
            print(f"{i+1:2d}. {feature_name:<25} | Error: {feature_errors[idx]:.6f}")
        
        return {
            'temporal_errors': temporal_errors,
            'feature_errors': feature_errors,
            'feature_importance_order': importance_indices
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the model.
        
        Returns:
            Dictionary containing model summary
        """
        if self.model is None:
            return {"error": "Model not built"}
        
        return {
            'model_name': self.model_name,
            'sequence_length': self.sequence_length,
            'latent_dim': self.latent_dim,
            'total_parameters': self.model.count_params(),
            'trainable_parameters': sum([keras.backend.count_params(w) for w in self.model.trainable_weights]),
            'is_trained': self.is_trained,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape
        }
