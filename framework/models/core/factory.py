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
    
    def create_model_with_config(self, model_name, time_span, train_features, use_fixed_threshold=False):
        """Create a model with appropriate parameters based on model type and time span.
        
        Args:
            model_name (str): Name of the model to create
            time_span (int): Time span in seconds for feature aggregation
            train_features (pd.DataFrame): Training features for sequence length calculation
            use_fixed_threshold (bool): Whether to use fixed threshold for TCN autoencoder
            
        Returns:
            Model instance configured with appropriate parameters
        """
        # Import here to avoid circular imports
        from .. import create_model
        from ...utils import get_time_span_description, get_sequence_multiplier
        
        if model_name == 'autoencoder':
            return create_model(model_name, latent_dim=42)
        
        elif model_name == 'lstm_autoencoder':
            # Configure LSTM autoencoder with time-span specific parameters
            base_sequence_length = min(80, max(50, len(train_features) // 8))
            sequence_multiplier = get_sequence_multiplier(time_span)
            sequence_length = int(base_sequence_length * sequence_multiplier)
            
            # Adjust latent dimension based on time span granularity
            if time_span == 1:
                latent_dim = 96  # Largest latent space for real-time patterns
            elif time_span == 10:
                latent_dim = 64  # Larger latent space for micro-patterns
            elif time_span == 60:
                latent_dim = 48  # Standard latent space
            else:  # 300 seconds
                latent_dim = 48  # Standard latent space
            
            model = create_model(
                model_name, 
                sequence_length=sequence_length,
                latent_dim=latent_dim,
                encoder_units=[128, 64],  # Consistent architecture
                decoder_units=[64, 128],  # Consistent architecture
                dropout_rate=0.2  # Standard dropout for generalization
            )
            time_desc = get_time_span_description(time_span)
            print(f"LSTM Autoencoder configured for {time_desc} windows:")
            print(f"  - Sequence length: {sequence_length} (base: {base_sequence_length}, multiplier: {sequence_multiplier})")
            print(f"  - Latent dimension: {latent_dim}")
            return model
        
        elif model_name == 'tcn_autoencoder':
            # Configure TCN autoencoder with time-span specific parameters
            if time_span == 1:
                sequence_length = 2  # Shortest for real-time patterns
                latent_dim = 6      # Smallest for real-time patterns
                filters = 16        # Fewest filters for real-time detection
            elif time_span == 10:
                sequence_length = 3  # Very short for micro-patterns
                latent_dim = 8      # Smaller for micro-patterns
                filters = 24        # Fewer filters for fine-grained detection
            elif time_span == 60:
                sequence_length = 5
                latent_dim = 12
                filters = 32
            else:  # 300 seconds
                sequence_length = 5
                latent_dim = 12
                filters = 32
            
            model = create_model(
                model_name,
                sequence_length=sequence_length,
                latent_dim=latent_dim,
                num_blocks=2,
                filters=filters,
                kernel_size=3,
                dropout_rate=0.3,
                l2_reg=1e-4,
                use_fixed_threshold=use_fixed_threshold
            )
            time_desc = get_time_span_description(time_span)
            print(f"TCN Autoencoder configured for {time_desc} windows:")
            print(f"  - Sequence length: {sequence_length}")
            print(f"  - Latent dimension: {latent_dim}")
            print(f"  - Filters: {filters}")
            print(f"  - Threshold mode: {'Fixed' if use_fixed_threshold else 'Adaptive'}")
            return model
        
        elif model_name == 'isolation_forest':
            return create_model(model_name, contamination=0.1, n_estimators=100)
        
        elif model_name == 'one_class_svm':
            return create_model(model_name, nu=0.1, kernel='rbf')
        
        else:
            return create_model(model_name)

    def load_model_from_artifacts(self, model, model_name, dataset_name, time_span, artifacts_info, use_fixed_threshold=False):
        """Load a model from artifacts with appropriate configuration.
        
        Args:
            model: Model instance (used for type information)
            model_name (str): Name of the model to load
            dataset_name (str): Name of the dataset
            time_span (int): Time span in seconds
            artifacts_info (dict): Artifacts information containing model configuration
            use_fixed_threshold (bool): Whether to use fixed threshold for TCN autoencoder
            
        Returns:
            Loaded model instance with saved configuration
        """
        # Import here to avoid circular imports
        from .artifacts import load_model_artifacts
        
        model_config = artifacts_info.get('model_config', {})
        
        if model_name == 'autoencoder':
            return load_model_artifacts(
                type(model), dataset_name, model_name, time_span,
                latent_dim=model_config.get('latent_dim', 42)
            )
        
        elif model_name == 'lstm_autoencoder':
            return load_model_artifacts(
                type(model), dataset_name, model_name, time_span,
                sequence_length=model_config.get('sequence_length', 50),
                latent_dim=model_config.get('latent_dim', 48),
                encoder_units=model_config.get('encoder_units', [128, 64]),
                decoder_units=model_config.get('decoder_units', [64, 128]),
                dropout_rate=model_config.get('dropout_rate', 0.2)
            )
        
        elif model_name == 'tcn_autoencoder':
            return load_model_artifacts(
                type(model), dataset_name, model_name, time_span,
                sequence_length=model_config.get('sequence_length', 5),
                latent_dim=model_config.get('latent_dim', 12),
                num_blocks=model_config.get('num_blocks', 2),
                filters=model_config.get('filters', 32),
                kernel_size=model_config.get('kernel_size', 3),
                dropout_rate=model_config.get('dropout_rate', 0.3),
                l2_reg=model_config.get('l2_reg', 1e-4),
                use_fixed_threshold=use_fixed_threshold
            )
        
        elif model_name == 'isolation_forest':
            return load_model_artifacts(
                type(model), dataset_name, model_name, time_span,
                contamination=0.1, n_estimators=100
            )
        
        elif model_name == 'one_class_svm':
            return load_model_artifacts(
                type(model), dataset_name, model_name, time_span,
                nu=0.1, kernel='rbf'
            )
        
        else:
            return load_model_artifacts(
                type(model), dataset_name, model_name, time_span
            )
    
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


def create_model_with_config(model_name, time_span, train_features, use_fixed_threshold=False):
    """
    Convenience function to create a model with configuration using the global factory.
    
    Args:
        model_name (str): Name of the model to create
        time_span (int): Time span in seconds for feature aggregation
        train_features (pd.DataFrame): Training features for sequence length calculation
        use_fixed_threshold (bool): Whether to use fixed threshold for TCN autoencoder
        
    Returns:
        Model instance configured with appropriate parameters
    """
    return _model_factory.create_model_with_config(model_name, time_span, train_features, use_fixed_threshold)


def load_model_from_artifacts(model, model_name, dataset_name, time_span, artifacts_info, use_fixed_threshold=False):
    """
    Convenience function to load a model from artifacts using the global factory.
    
    Args:
        model: Model instance (used for type information)
        model_name (str): Name of the model to load
        dataset_name (str): Name of the dataset
        time_span (int): Time span in seconds
        artifacts_info (dict): Artifacts information containing model configuration
        use_fixed_threshold (bool): Whether to use fixed threshold for TCN autoencoder
        
    Returns:
        Loaded model instance with saved configuration
    """
    return _model_factory.load_model_from_artifacts(model, model_name, dataset_name, time_span, artifacts_info, use_fixed_threshold)


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
