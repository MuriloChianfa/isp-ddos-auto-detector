"""
Model artifacts management for saving and loading trained models.

This module provides functionality to save and load trained anomaly detection models
along with their associated metadata, scalers, and thresholds.
"""

import os
import json
import pickle
import joblib
import numpy as np
from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging

import tensorflow as tf
from tensorflow import keras

from ...utils import get_artifacts_path

logger = logging.getLogger(__name__)


class ModelArtifactsManager:
    """
    Manager class for saving and loading model artifacts.
    
    This class handles the persistence of trained models, including:
    - Model weights and architecture
    - Scalers and preprocessors
    - Thresholds and training metadata
    - Model configuration and parameters
    """
    
    def __init__(self, dataset_name: str, model_name: str, time_span: int):
        """
        Initialize the artifacts manager.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_span: Time span in seconds
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.time_span = time_span
        self.artifacts_dir = get_artifacts_path(dataset_name, model_name, time_span)
        
    def _ensure_artifacts_dir(self) -> None:
        """Create artifacts directory if it doesn't exist."""
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
    def save_model_artifacts(self, model, training_history: Optional[Dict] = None) -> str:
        """
        Save complete model artifacts.
        
        Args:
            model: Trained model instance
            training_history: Training history dictionary (optional)
            
        Returns:
            str: Path to the saved artifacts directory
        """
        self._ensure_artifacts_dir()
        
        logger.info(f"Saving model artifacts to: {self.artifacts_dir}")
        
        # Save model-specific components
        if hasattr(model, 'model') and model.model is not None:
            # For Keras/TensorFlow models (LSTM, TCN) - use .keras format for better compatibility
            model_path = os.path.join(self.artifacts_dir, 'model.keras')
            model.model.save(model_path, save_format='keras')
            logger.info(f"Saved Keras model to: {model_path}")
        elif hasattr(model, 'autoencoder') and model.autoencoder is not None:
            # For Autoencoder models - use .keras format for better compatibility
            model_path = os.path.join(self.artifacts_dir, 'model.keras')
            model.autoencoder.save(model_path, save_format='keras')
            logger.info(f"Saved Autoencoder model to: {model_path}")
        elif hasattr(model, 'estimator') and model.estimator is not None:
            # For scikit-learn models
            model_path = os.path.join(self.artifacts_dir, 'model.pkl')
            joblib.dump(model.estimator, model_path)
            logger.info(f"Saved sklearn model to: {model_path}")
        else:
            logger.warning(f"No model found to save for {model.__class__.__name__}")
        
        # Save scaler
        if hasattr(model, 'scaler') and model.scaler is not None:
            scaler_path = os.path.join(self.artifacts_dir, 'scaler.pkl')
            joblib.dump(model.scaler, scaler_path)
            logger.info(f"Saved scaler to: {scaler_path}")
        
        # Save threshold
        threshold_data = {
            'threshold': float(model.threshold) if model.threshold is not None else None,
            'is_trained': model.is_trained
        }
        threshold_path = os.path.join(self.artifacts_dir, 'threshold.json')
        with open(threshold_path, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        logger.info(f"Saved threshold to: {threshold_path}")
        
        # Save model configuration and metadata
        metadata = {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'time_span': self.time_span,
            'model_type': model.__class__.__name__,
            'save_timestamp': datetime.now().isoformat(),
            'model_info': model.get_model_info() if hasattr(model, 'get_model_info') else {}
        }
        
        # Add model-specific configuration
        if hasattr(model, 'get_model_config'):
            metadata['model_config'] = model.get_model_config()
        elif hasattr(model, 'latent_dim'):
            # For autoencoder models
            metadata['model_config'] = {
                'latent_dim': getattr(model, 'latent_dim', None),
                'input_dim': getattr(model, '_input_dim', None)
            }
        
        # Add training history if provided
        if training_history is not None:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in training_history.items():
                if isinstance(value, np.ndarray):
                    serializable_history[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    # Convert any numpy values in the list
                    serializable_history[key] = [
                        float(x) if isinstance(x, (np.float32, np.float64)) else x 
                        for x in value
                    ]
                else:
                    serializable_history[key] = value
            metadata['training_history'] = serializable_history
        
        metadata_path = os.path.join(self.artifacts_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_path}")
        
        # Create a summary file
        summary = {
            'artifacts_saved': True,
            'saved_at': datetime.now().isoformat(),
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'time_span': self.time_span,
            'threshold': threshold_data['threshold'],
            'files': {
                'model': 'model.h5' if hasattr(model, 'model') else 'model.pkl',
                'scaler': 'scaler.pkl',
                'threshold': 'threshold.json',
                'metadata': 'metadata.json'
            }
        }
        
        summary_path = os.path.join(self.artifacts_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Model artifacts successfully saved to: {self.artifacts_dir}")
        return self.artifacts_dir
    
    def update_threshold(self, threshold: float, all_thresholds: dict = None):
        """
        Update the threshold value and all strategies in the saved artifacts.
        
        This method is called after the threshold has been calculated during anomaly detection
        to ensure the artifacts contain the correct threshold value and all calculated strategies.
        
        Args:
            threshold: The calculated threshold value
            all_thresholds: Dictionary of all threshold strategies and their values
        """
        # Update threshold.json with selected threshold and all strategies
        threshold_data = {
            'selected_threshold': float(threshold),
            'selected_strategy': None,  # Will be filled if provided
            'is_trained': True
        }
        
        # Add all threshold strategies if provided
        if all_thresholds:
            threshold_data['all_strategies'] = {k: float(v) for k, v in all_thresholds.items()}
            # Find which strategy was selected
            for strategy_name, strategy_value in all_thresholds.items():
                if abs(float(strategy_value) - threshold) < 1e-10:  # Account for floating point precision
                    threshold_data['selected_strategy'] = strategy_name
                    break
        
        # Keep backward compatibility
        threshold_data['threshold'] = float(threshold)
        
        threshold_path = os.path.join(self.artifacts_dir, 'threshold.json')
        
        try:
            with open(threshold_path, 'w') as f:
                json.dump(threshold_data, f, indent=2)
            logger.info(f"Updated threshold to {threshold:.6f} in: {threshold_path}")
            
            # Also update the summary.json file if it exists
            summary_path = os.path.join(self.artifacts_dir, 'summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                summary['threshold'] = float(threshold)
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
                logger.info(f"Updated threshold in summary.json")
        except Exception as e:
            logger.error(f"Failed to update threshold in artifacts: {e}")
    
    def load_model_artifacts(self, model_class, **model_kwargs) -> Any:
        """
        Load complete model artifacts.
        
        Args:
            model_class: Model class to instantiate
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded and initialized model instance
            
        Raises:
            FileNotFoundError: If artifacts directory or required files don't exist
            ValueError: If artifacts are incompatible or corrupted
        """
        if not os.path.exists(self.artifacts_dir):
            raise FileNotFoundError(f"Artifacts directory not found: {self.artifacts_dir}")
        
        logger.info(f"Loading model artifacts from: {self.artifacts_dir}")
        
        # Load metadata first
        metadata_path = os.path.join(self.artifacts_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate compatibility
        if metadata['model_name'] != self.model_name:
            raise ValueError(f"Model name mismatch: expected {self.model_name}, got {metadata['model_name']}")
        if metadata['dataset_name'] != self.dataset_name:
            raise ValueError(f"Dataset name mismatch: expected {self.dataset_name}, got {metadata['dataset_name']}")
        if metadata['time_span'] != self.time_span:
            raise ValueError(f"Time span mismatch: expected {self.time_span}, got {metadata['time_span']}")
        
        # Initialize model with saved configuration
        model_config = metadata.get('model_config', {})
        init_kwargs = {**model_kwargs, **model_config}
        
        # Remove non-init parameters
        init_kwargs.pop('input_dim', None)
        
        model = model_class(**init_kwargs)
        
        # Load scaler
        scaler_path = os.path.join(self.artifacts_dir, 'scaler.pkl')
        if os.path.exists(scaler_path):
            model.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from: {scaler_path}")
        
        # Load threshold
        threshold_path = os.path.join(self.artifacts_dir, 'threshold.json')
        if os.path.exists(threshold_path):
            with open(threshold_path, 'r') as f:
                threshold_data = json.load(f)
            model.threshold = threshold_data.get('threshold')
            model.is_trained = threshold_data.get('is_trained', False)
            logger.info(f"Loaded threshold: {model.threshold}")
        
        # Load model weights/estimator - check for both .keras and .h5 formats
        keras_model_path = os.path.join(self.artifacts_dir, 'model.keras')
        h5_model_path = os.path.join(self.artifacts_dir, 'model.h5')
        sklearn_model_path = os.path.join(self.artifacts_dir, 'model.pkl')
        
        # Prefer .keras format, fallback to .h5
        model_path_to_load = None
        if os.path.exists(keras_model_path):
            model_path_to_load = keras_model_path
        elif os.path.exists(h5_model_path):
            model_path_to_load = h5_model_path
        
        if model_path_to_load:
            # Load Keras model
            loaded_model = keras.models.load_model(model_path_to_load)
            
            # Assign to appropriate attribute based on model type
            if hasattr(model, 'autoencoder'):
                model.autoencoder = loaded_model
                logger.info(f"Loaded Autoencoder model from: {model_path_to_load}")
            elif hasattr(model, 'model'):
                model.model = loaded_model
                logger.info(f"Loaded Keras model from: {model_path_to_load}")
            else:
                model.model = loaded_model
                logger.info(f"Loaded Keras model from: {model_path_to_load}")
            
            model.is_trained = True
            
            # Set input dimension from loaded model
            if hasattr(model, '_input_dim'):
                model._input_dim = loaded_model.input_shape[1]
        elif os.path.exists(sklearn_model_path):
            # Load sklearn model
            model.estimator = joblib.load(sklearn_model_path)
            model.is_trained = True
            logger.info(f"Loaded sklearn model from: {sklearn_model_path}")
        else:
            raise FileNotFoundError("No model file found (neither model.h5 nor model.pkl)")
        
        # Restore additional model-specific attributes
        if hasattr(model, 'sequence_length') and 'sequence_length' in model_config:
            model.sequence_length = model_config['sequence_length']
        if hasattr(model, '_feature_dim') and 'feature_dim' in model_config:
            model._feature_dim = model_config['feature_dim']
        
        logger.info(f"Model artifacts successfully loaded from: {self.artifacts_dir}")
        logger.info(f"Model saved at: {metadata.get('save_timestamp', 'unknown')}")
        
        return model
    
    def artifacts_exist(self) -> bool:
        """
        Check if model artifacts exist.
        
        Returns:
            bool: True if artifacts directory and essential files exist
        """
        if not os.path.exists(self.artifacts_dir):
            return False
        
        # Check for essential files
        essential_files = ['metadata.json', 'threshold.json', 'scaler.pkl']
        for file in essential_files:
            if not os.path.exists(os.path.join(self.artifacts_dir, file)):
                return False
        
        # Check for at least one model file - support multiple formats
        keras_model = os.path.exists(os.path.join(self.artifacts_dir, 'model.keras'))
        h5_model = os.path.exists(os.path.join(self.artifacts_dir, 'model.h5'))
        sklearn_model = os.path.exists(os.path.join(self.artifacts_dir, 'model.pkl'))
        
        return keras_model or h5_model or sklearn_model
    
    def get_artifacts_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about existing artifacts.
        
        Returns:
            Dict with artifacts information or None if no artifacts exist
        """
        if not self.artifacts_exist():
            return None
        
        try:
            metadata_path = os.path.join(self.artifacts_dir, 'metadata.json')
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            summary_path = os.path.join(self.artifacts_dir, 'summary.json')
            summary = {}
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
            
            return {
                'artifacts_dir': self.artifacts_dir,
                'save_timestamp': metadata.get('save_timestamp'),
                'model_type': metadata.get('model_type'),
                'model_config': metadata.get('model_config', {}),
                'threshold': summary.get('threshold'),
                'has_training_history': 'training_history' in metadata,
                'files': summary.get('files', {})
            }
        except Exception as e:
            logger.error(f"Error reading artifacts info: {e}")
            return None
    
    def remove_artifacts(self) -> bool:
        """
        Remove all model artifacts.
        
        Returns:
            bool: True if artifacts were successfully removed
        """
        try:
            if os.path.exists(self.artifacts_dir):
                import shutil
                shutil.rmtree(self.artifacts_dir)
                logger.info(f"Removed artifacts directory: {self.artifacts_dir}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing artifacts: {e}")
            return False


def save_model_artifacts(model, dataset_name: str, model_name: str, time_span: int, 
                        training_history: Optional[Dict] = None) -> str:
    """
    Convenience function to save model artifacts.
    
    Args:
        model: Trained model instance
        dataset_name: Name of the dataset
        model_name: Name of the model
        time_span: Time span in seconds
        training_history: Training history dictionary (optional)
        
    Returns:
        str: Path to the saved artifacts directory
    """
    manager = ModelArtifactsManager(dataset_name, model_name, time_span)
    return manager.save_model_artifacts(model, training_history)


def load_model_artifacts(model_class, dataset_name: str, model_name: str, time_span: int, 
                        **model_kwargs) -> Any:
    """
    Convenience function to load model artifacts.
    
    Args:
        model_class: Model class to instantiate
        dataset_name: Name of the dataset
        model_name: Name of the model
        time_span: Time span in seconds
        **model_kwargs: Additional arguments for model initialization
        
    Returns:
        Loaded and initialized model instance
    """
    manager = ModelArtifactsManager(dataset_name, model_name, time_span)
    return manager.load_model_artifacts(model_class, **model_kwargs)


def artifacts_exist(dataset_name: str, model_name: str, time_span: int) -> bool:
    """
    Convenience function to check if model artifacts exist.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        time_span: Time span in seconds
        
    Returns:
        bool: True if artifacts exist
    """
    manager = ModelArtifactsManager(dataset_name, model_name, time_span)
    return manager.artifacts_exist()


def get_artifacts_info(dataset_name: str, model_name: str, time_span: int) -> Optional[Dict[str, Any]]:
    """
    Convenience function to get artifacts information.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        time_span: Time span in seconds
        
    Returns:
        Dict with artifacts information or None if no artifacts exist
    """
    manager = ModelArtifactsManager(dataset_name, model_name, time_span)
    return manager.get_artifacts_info()