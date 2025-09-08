"""
Model factory for creating and managing different anomaly detection models.

This module provides a factory pattern implementation for creating instances
of different anomaly detection models with proper configuration and validation.
"""

from typing import Dict, Type, List, Any, Optional
import logging
from .template import BaseAnomalyDetector

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating and managing anomaly detection models.
    
    This class implements the factory pattern to provide a centralized way
    to create model instances with proper configuration and validation.
    """
    
    def __init__(self):
        """Initialize the model factory."""
        self._models: Dict[str, Type[BaseAnomalyDetector]] = {}
        self._model_configs: Dict[str, Dict[str, Any]] = {}
        self._register_default_models()
    
    def register_model(self, name: str, model_class: Type[BaseAnomalyDetector], 
                      default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a new model with the factory.
        
        Args:
            name: Unique name for the model
            model_class: Model class that inherits from BaseAnomalyDetector
            default_config: Default configuration parameters for the model
            
        Raises:
            ValueError: If model name already exists or invalid model class
        """
        if name in self._models:
            raise ValueError(f"Model '{name}' is already registered")
            
        if not issubclass(model_class, BaseAnomalyDetector):
            raise ValueError(f"Model class must inherit from BaseAnomalyDetector")
        
        self._models[name] = model_class
        self._model_configs[name] = default_config or {}
        
        logger.info(f"Registered model: {name}")
    
    def create_model(self, name: str, **kwargs) -> BaseAnomalyDetector:
        """
        Create an instance of the specified model.
        
        Args:
            name: Name of the model to create
            **kwargs: Model-specific parameters (override defaults)
            
        Returns:
            Instance of the requested model
            
        Raises:
            ValueError: If model name is not found
        """
        if name not in self._models:
            available_models = list(self._models.keys())
            raise ValueError(f"Unknown model: '{name}'. Available models: {available_models}")
        
        # Merge default config with provided kwargs
        config = self._model_configs[name].copy()
        config.update(kwargs)
        
        model_class = self._models[name]
        
        try:
            model_instance = model_class(**config)
            logger.info(f"Created model instance: {name}")
            return model_instance
        except Exception as e:
            logger.error(f"Failed to create model '{name}': {str(e)}")
            raise
    
    def list_models(self) -> List[str]:
        """
        Get list of available model names.
        
        Returns:
            List of registered model names
        """
        return list(self._models.keys())
    
    def get_model_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about models.
        
        Args:
            name: Specific model name (optional, returns all if None)
            
        Returns:
            Dictionary containing model information
            
        Raises:
            ValueError: If specific model name is not found
        """
        if name is not None:
            if name not in self._models:
                raise ValueError(f"Unknown model: '{name}'")
            
            model_class = self._models[name]
            return {
                'name': name,
                'class': model_class.__name__,
                'module': model_class.__module__,
                'description': getattr(model_class, '__doc__', 'No description available'),
                'default_config': self._model_configs[name]
            }
        
        # Return info for all models
        return {
            model_name: self.get_model_info(model_name) 
            for model_name in self._models.keys()
        }
    
    def validate_model_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a specific model.
        
        Args:
            name: Model name
            config: Configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If model name not found or config invalid
        """
        if name not in self._models:
            raise ValueError(f"Unknown model: '{name}'")
        
        # Try to create model with config to validate
        try:
            self.create_model(name, **config)
            return True
        except Exception as e:
            logger.error(f"Invalid configuration for model '{name}': {str(e)}")
            raise ValueError(f"Invalid configuration: {str(e)}")
    
    def _register_default_models(self) -> None:
        """Register default models that come with the framework."""
        try:
            # Import and register autoencoder
            from ..autoencoder import AutoencoderAnomalyDetector
            self.register_model(
                'autoencoder', 
                AutoencoderAnomalyDetector,
                {'latent_dim': 42}
            )
        except ImportError as e:
            logger.warning(f"Could not register autoencoder model: {e}")
        
        try:
            # Import and register LSTM autoencoder
            from ..lstm_autoencoder import LSTMAutoencoder
            self.register_model(
                'lstm_autoencoder',
                LSTMAutoencoder,
                {
                    'sequence_length': 60,
                    'latent_dim': 32,
                    'encoder_units': [128, 64],
                    'decoder_units': [64, 128],
                    'dropout_rate': 0.1
                }
            )
        except ImportError as e:
            logger.warning(f"Could not register LSTM autoencoder model: {e}")
        
        try:
            # Import and register TCN autoencoder
            from ..tcn_autoencoder import TCNAutoencoder
            self.register_model(
                'tcn_autoencoder',
                TCNAutoencoder,
                {
                    'sequence_length': 60,
                    'latent_dim': 8,   # Ultra-small for maximum compression
                    'num_blocks': 3,   
                    'filters': 24,     # Optimized filters
                    'kernel_size': 3,
                    'dropout_rate': 0.25,  # Higher regularization
                    'l2_reg': 2e-4     # Stronger L2 regularization
                }
            )
        except ImportError as e:
            logger.warning(f"Could not register TCN autoencoder model: {e}")
        
        try:
            # Import and register isolation forest
            from ..isolation_forest import IsolationForestAnomalyDetector
            self.register_model(
                'isolation_forest',
                IsolationForestAnomalyDetector,
                {'contamination': 0.1, 'n_estimators': 100, 'random_state': 42}
            )
        except ImportError as e:
            logger.warning(f"Could not register isolation forest model: {e}")
        
        try:
            # Import and register one-class SVM
            from ..one_class_svm import OneClassSVMAnomalyDetector
            self.register_model(
                'one_class_svm',
                OneClassSVMAnomalyDetector,
                {'nu': 0.1, 'kernel': 'rbf', 'gamma': 'scale'}
            )
        except ImportError as e:
            logger.warning(f"Could not register one-class SVM model: {e}")


# Global factory instance
_model_factory = ModelFactory()


def create_model(name: str, **kwargs) -> BaseAnomalyDetector:
    """
    Convenience function to create a model using the global factory.
    
    Args:
        name: Name of the model to create
        **kwargs: Model-specific parameters
        
    Returns:
        Instance of the requested model
    """
    return _model_factory.create_model(name, **kwargs)


def register_model(name: str, model_class: Type[BaseAnomalyDetector], 
                  default_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Convenience function to register a model with the global factory.
    
    Args:
        name: Unique name for the model
        model_class: Model class that inherits from BaseAnomalyDetector
        default_config: Default configuration parameters for the model
    """
    _model_factory.register_model(name, model_class, default_config)


def list_available_models() -> List[str]:
    """
    Get list of available model names.
    
    Returns:
        List of registered model names
    """
    return _model_factory.list_models()


def get_model_info(name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about available models.
    
    Args:
        name: Specific model name (optional)
        
    Returns:
        Dictionary containing model information
    """
    return _model_factory.get_model_info(name)


def validate_model_config(name: str, config: Dict[str, Any]) -> bool:
    """
    Validate configuration for a specific model.
    
    Args:
        name: Model name
        config: Configuration to validate
        
    Returns:
        True if configuration is valid
    """
    return _model_factory.validate_model_config(name, config)


# Model descriptions for user-friendly display
MODEL_DESCRIPTIONS = {
    'autoencoder': 'Neural network autoencoder for unsupervised anomaly detection using reconstruction error',
    'lstm_autoencoder': 'LSTM-based temporal autoencoder for sequential anomaly detection in network traffic',
    'tcn_autoencoder': 'Optimized Temporal Convolutional Network autoencoder with maximum compression (8D latent) specifically tuned for SYN flood detection',
    'isolation_forest': 'Ensemble method using isolation trees to identify anomalies by isolation efficiency',
    'one_class_svm': 'Support Vector Machine trained on normal data to identify outliers in feature space',
}


def get_model_descriptions() -> Dict[str, str]:
    """
    Get user-friendly descriptions of available models.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    available_models = list_available_models()
    return {name: MODEL_DESCRIPTIONS.get(name, 'No description available') 
            for name in available_models}
