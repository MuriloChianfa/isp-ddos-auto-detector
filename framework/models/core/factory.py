"""
Model factory for creating and managing different anomaly detection models.

This module provides a factory pattern implementation for creating instances
of different anomaly detection models with proper configuration and validation.
"""

from typing import Dict, Type, List, Any, Optional
import logging
from .template import BaseAnomalyDetector

logger = logging.getLogger(__name__)


def get_model_params(model_name: str, dataset_name: Optional[str] = None, 
                     time_span: Optional[int] = None) -> Dict[str, Any]:
    """
    Get model parameters with automatic optimal parameter loading.
    
    Priority order:
    1. Dataset-specific window params from config (highest priority - explicit overrides)
    2. Optimal params from hyperparameter optimization (auto-loaded from results/)
    3. Default params from config.MODEL_DEFAULT_PARAMS (fallback)
    
    Args:
        model_name: Name of the model
        dataset_name: Optional dataset name for dataset-specific params
        time_span: Optional time span for window-specific params
        
    Returns:
        Dictionary of model parameters
    """
    # Import here to avoid circular imports
    from config import MODEL_DEFAULT_PARAMS, DATASETS
    from framework.optimal import load_optimal_params
    
    # Start with default params
    params = MODEL_DEFAULT_PARAMS.get(model_name, {}).copy()
    param_source = "defaults"
    
    # Try to load optimal params if dataset and time_span provided
    if dataset_name and time_span:
        optimal_params = load_optimal_params(dataset_name, str(time_span), model_name)
        if optimal_params:
            params.update(optimal_params)
            param_source = "optimal"
            logger.debug(f"Loaded optimal params for {model_name} on {dataset_name} ({time_span}s)")
        else:
            logger.warning(f"No optimal params found for {model_name} on {dataset_name} ({time_span}s), using defaults")
    
    # Override with explicit config params if available (highest priority)
    if dataset_name and time_span:
        dataset_config = DATASETS.get(dataset_name, {})
        window_config = dataset_config.get('windows', {}).get(str(time_span), {})
        window_params = window_config.get('params', {}).get(model_name, {})
        
        if window_params:
            params.update(window_params)
            logger.info(f"Using config override params for {model_name} on {dataset_name} ({time_span}s)")
            param_source = "config"
    
    logger.debug(f"Final params source for {model_name}: {param_source}")
    return params


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
    
    def create_model_with_config(self, model_name, time_span, train_features, use_fixed_threshold=False, dataset_name=None):
        """Create a model with appropriate parameters based on model type and time span.
        
        Args:
            model_name (str): Name of the model to create
            time_span (int): Time span in seconds for feature aggregation
            train_features (pd.DataFrame): Training features for sequence length calculation
            use_fixed_threshold (bool): Ignored - kept for backward compatibility
            dataset_name (str): Optional dataset name for dataset-specific params
            
        Returns:
            Model instance configured with appropriate parameters
        """
        # Import here to avoid circular imports
        from .. import create_model
        
        # Get parameters (with optional dataset/window-specific overrides)
        params = get_model_params(model_name, dataset_name, time_span)
        return create_model(model_name, **params)

    def load_model_from_artifacts(self, model, model_name, dataset_name, time_span, artifacts_info, use_fixed_threshold=False, input_dim=None):
        """Load a model from artifacts with appropriate configuration.
        
        Args:
            model: Model instance (used for type information)
            model_name (str): Name of the model to load
            dataset_name (str): Name of the dataset
            time_span (int): Time span in seconds
            artifacts_info (dict): Artifacts information containing model configuration
            use_fixed_threshold (bool): Ignored - kept for backward compatibility
            input_dim (int, optional): Expected input dimension for validation
            
        Returns:
            Loaded model instance with saved configuration
        """
        # Import here to avoid circular imports
        from .artifacts import load_model_artifacts
        
        # Get parameters (with optional dataset/window-specific overrides)
        # Prefer saved config from artifacts if available, otherwise use defaults
        model_config = artifacts_info.get('model_config', {}).copy()
        default_params = get_model_params(model_name, dataset_name, time_span)
        
        # Remove input_dim from model_config if present, we'll add it separately for validation
        model_config.pop('input_dim', None)
        
        # Merge: use saved config values if present, otherwise use defaults
        params = {**default_params, **model_config}
        
        # Add input_dim for validation if provided
        if input_dim is not None:
            params['input_dim'] = input_dim
        
        return load_model_artifacts(
            type(model), dataset_name, model_name, time_span, **params
        )
    
    def _register_default_models(self) -> None:
        """Register default models that come with the framework."""
        # Import config here to avoid circular imports
        from config import MODEL_DEFAULT_PARAMS
        
        try:
            # Import and register autoencoder
            from ..autoencoder import AutoencoderAnomalyDetector
            self.register_model(
                'autoencoder', 
                AutoencoderAnomalyDetector,
                MODEL_DEFAULT_PARAMS.get('autoencoder', {})
            )
        except ImportError as e:
            logger.warning(f"Could not register autoencoder model: {e}")
        
        try:
            # Import and register isolation forest
            from ..isolation_forest import IsolationForestAnomalyDetector
            self.register_model(
                'isolation_forest',
                IsolationForestAnomalyDetector,
                MODEL_DEFAULT_PARAMS.get('isolation_forest', {})
            )
        except ImportError as e:
            logger.warning(f"Could not register isolation forest model: {e}")
        
        try:
            # Import and register one-class SVM
            from ..one_class_svm import OneClassSVMAnomalyDetector
            self.register_model(
                'one_class_svm',
                OneClassSVMAnomalyDetector,
                MODEL_DEFAULT_PARAMS.get('one_class_svm', {})
            )
        except ImportError as e:
            logger.warning(f"Could not register one-class SVM model: {e}")
        
        try:
            # Import and register local outlier factor
            from ..local_outlier_factor import LocalOutlierFactorAnomalyDetector
            self.register_model(
                'local_outlier_factor',
                LocalOutlierFactorAnomalyDetector,
                MODEL_DEFAULT_PARAMS.get('local_outlier_factor', {})
            )
        except ImportError as e:
            logger.warning(f"Could not register local outlier factor model: {e}")


def create_model_with_config(model_name, time_span, train_features, use_fixed_threshold=False, dataset_name=None):
    """
    Convenience function to create a model with configuration using the global factory.
    
    Args:
        model_name (str): Name of the model to create
        time_span (int): Time span in seconds for feature aggregation
        train_features (pd.DataFrame): Training features for sequence length calculation
        use_fixed_threshold (bool): Whether to use fixed threshold for TCN autoencoder
        dataset_name (str): Optional dataset name for dataset-specific params
        
    Returns:
        Model instance configured with appropriate parameters
    """
    return _model_factory.create_model_with_config(model_name, time_span, train_features, use_fixed_threshold, dataset_name)


def load_model_from_artifacts(model, model_name, dataset_name, time_span, artifacts_info, use_fixed_threshold=False, input_dim=None):
    """
    Convenience function to load a model from artifacts using the global factory.
    
    Args:
        model: Model instance (used for type information)
        model_name (str): Name of the model to load
        dataset_name (str): Name of the dataset
        time_span (int): Time span in seconds
        artifacts_info (dict): Artifacts information containing model configuration
        use_fixed_threshold (bool): Whether to use fixed threshold for TCN autoencoder
        input_dim (int, optional): Expected input dimension for validation
        
    Returns:
        Loaded model instance with saved configuration
    """
    return _model_factory.load_model_from_artifacts(model, model_name, dataset_name, time_span, artifacts_info, use_fixed_threshold, input_dim)


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
    'isolation_forest': 'Ensemble method using isolation trees to identify anomalies by isolation efficiency',
    'one_class_svm': 'Support Vector Machine trained on normal data to identify outliers in feature space',
    'local_outlier_factor': 'Density-based method that identifies anomalies by measuring local density deviation from neighbors',
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
