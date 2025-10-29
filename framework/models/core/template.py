"""
Base classes and interfaces for anomaly detection models.

This module defines the abstract base class that all anomaly detection models
must inherit from, ensuring a consistent interface across different algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseAnomalyDetector(ABC):
    """
    Abstract base class for all anomaly detection models.
    
    This class defines the interface that all anomaly detection models must implement,
    ensuring consistency across different algorithms and making it easy to swap
    between different models.
    
    Attributes:
        scaler: Data scaler for feature normalization
        threshold: Anomaly detection threshold
        is_trained: Flag indicating if the model has been trained
        model_name: Name of the model for identification
    """
    
    def __init__(self, model_name: str = "base_model"):
        """
        Initialize the base anomaly detector.
        
        Args:
            model_name: Name identifier for the model
        """
        self.scaler = None
        self.threshold = None
        self.is_trained = False
        self.model_name = model_name
        self._training_history = None
        
    @abstractmethod
    def build_model(self, input_dim: int) -> None:
        """
        Build the model architecture.
        
        Args:
            input_dim: Number of input features
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def fit_scaler(self, training_features) -> None:
        """
        Fit the data scaler on training data.
        
        Args:
            training_features: Training feature matrix (DataFrame or numpy array)
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def transform_data(self, features) -> np.ndarray:
        """
        Transform features using the fitted scaler.
        
        Args:
            features: Feature matrix to transform (DataFrame or numpy array)
            
        Returns:
            Transformed feature matrix
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def train(self, train_data: np.ndarray, validation_data: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            **kwargs: Model-specific training parameters
            
        Returns:
            Training history or metrics
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions and return reconstructions/predictions and anomaly scores.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Tuple of (reconstructions/predictions, anomaly_scores)
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    @abstractmethod
    def calculate_threshold(self, train_scores: np.ndarray, val_scores: np.ndarray, 
                          strategy: str = 'exponential_threshold') -> Tuple[float, Dict[str, float]]:
        """
        Calculate anomaly detection threshold.
        
        Args:
            train_scores: Anomaly scores from training data
            val_scores: Anomaly scores from validation data
            strategy: Threshold calculation strategy
            
        Returns:
            Tuple of (selected_threshold, all_threshold_strategies)
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def detect_anomalies(self, scores: np.ndarray) -> np.ndarray:
        """
        Detect anomalies based on threshold.
        
        Args:
            scores: Anomaly scores
            
        Returns:
            Boolean array indicating anomalies
            
        Raises:
            ValueError: If threshold not set
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call calculate_threshold first.")
        return scores > self.threshold
    
    @abstractmethod
    def get_metrics(self, data: np.ndarray, reconstructions: np.ndarray, 
                   scores: np.ndarray) -> Dict[str, float]:
        """
        Calculate model performance metrics.
        
        Args:
            data: Original data
            reconstructions: Model reconstructions/predictions
            scores: Anomaly scores
            
        Returns:
            Dictionary of performance metrics
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass
    
    def analyze_feature_importance(self, data: np.ndarray, 
                                 feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze feature importance (default implementation).
        
        This provides a default implementation that can be overridden by specific models
        that have their own feature importance analysis methods.
        
        Args:
            data: Input data
            feature_names: List of feature names
            
        Returns:
            Tuple of (feature_errors, importance_indices)
        """
        logger.warning(f"{self.model_name} doesn't provide specific feature importance analysis. "
                      "Using default uniform importance.")
        
        # Default implementation - uniform importance
        feature_errors = np.ones(len(feature_names))
        importance_indices = np.arange(len(feature_names))
        
        return feature_errors, importance_indices
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and parameters.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'has_threshold': self.threshold is not None,
            'threshold_value': self.threshold
        }
    
    def save_artifacts(self, dataset_name: str, time_span: int, training_history: Optional[Dict] = None) -> str:
        """
        Save model artifacts using the artifacts manager.
        
        Args:
            dataset_name: Name of the dataset
            time_span: Time span in seconds
            training_history: Training history dictionary (optional)
            
        Returns:
            str: Path to the saved artifacts directory
        """
        from .artifacts import save_model_artifacts
        return save_model_artifacts(self, dataset_name, self.model_name, time_span, training_history)
    
    @classmethod
    def load_artifacts(cls, dataset_name: str, model_name: str, time_span: int, **model_kwargs):
        """
        Load model artifacts using the artifacts manager.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_span: Time span in seconds
            **model_kwargs: Additional arguments for model initialization
            
        Returns:
            Loaded and initialized model instance
        """
        from .artifacts import load_model_artifacts
        return load_model_artifacts(cls, dataset_name, model_name, time_span, **model_kwargs)
    
    @staticmethod
    def artifacts_exist(dataset_name: str, model_name: str, time_span: int) -> bool:
        """
        Check if model artifacts exist.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_span: Time span in seconds
            
        Returns:
            bool: True if artifacts exist
        """
        from .artifacts import artifacts_exist
        return artifacts_exist(dataset_name, model_name, time_span)
    
    @staticmethod
    def get_artifacts_info(dataset_name: str, model_name: str, time_span: int) -> Optional[Dict[str, Any]]:
        """
        Get information about existing artifacts.
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_span: Time span in seconds
            
        Returns:
            Dict with artifacts information or None if no artifacts exist
        """
        from .artifacts import get_artifacts_info
        return get_artifacts_info(dataset_name, model_name, time_span)


class DummyTrainingHistory:
    """
    Simple wrapper to make scikit-learn models compatible with Keras-style training history.
    
    This class provides a simple interface that mimics Keras History objects
    for models that don't have traditional training histories.
    """
    
    def __init__(self, history_dict: Dict[str, Any]):
        """
        Initialize dummy training history.
        
        Args:
            history_dict: Dictionary containing training metrics
        """
        self.history = history_dict
    
    def _log_training_start(self, train_data: np.ndarray, 
                           validation_data: Optional[np.ndarray] = None) -> None:
        """
        Log training information.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
        """
        logger.info(f"Starting training for {self.model_name}")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Features: {train_data.shape[1]}")
        
        if validation_data is not None:
            logger.info(f"Validation samples: {len(validation_data)}")
    
    def _log_training_complete(self) -> None:
        """Log training completion."""
        logger.info(f"Training completed for {self.model_name}")
        self.is_trained = True


class ModelValidationMixin:
    """
    Mixin class providing common validation methods for models.
    """
    
    @staticmethod
    def validate_input_dimensions(data: np.ndarray, expected_dim: Optional[int] = None) -> None:
        """
        Validate input data dimensions.
        
        Args:
            data: Input data to validate
            expected_dim: Expected number of features (optional)
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if data.ndim != 2:
            raise ValueError(f"Input data must be 2D, got {data.ndim}D")
            
        if expected_dim is not None and data.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {data.shape[1]}")
    
    @staticmethod
    def validate_threshold_strategy(strategy: str) -> None:
        """
        Validate threshold calculation strategy.
        
        Args:
            strategy: Threshold strategy name
            
        Raises:
            ValueError: If strategy is not supported
        """
        valid_strategies = [
            'normal_mse_mean', 'mean_plus_1std', 'mean_plus_2std', 'mean_plus_3std',
            'mse_plus_3std', 'mse_plus_5std', 'mse_plus_8std', 'exponential_threshold', 'sigmoid_threshold',
            'percentile_95', 'percentile_99', 'percentile_99_5', 'percentile_99_9'
        ]
        
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown threshold strategy: {strategy}. "
                           f"Valid strategies: {valid_strategies}")

    def validate_training_data(self, train_data: np.ndarray, 
                             validation_data: Optional[np.ndarray] = None) -> None:
        """
        Validate training data before training.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
            
        Raises:
            ValueError: If data validation fails
        """
        if train_data is None or len(train_data) == 0:
            raise ValueError("Training data cannot be empty")
            
        if np.any(np.isnan(train_data)):
            raise ValueError("Training data contains NaN values")
            
        if np.any(np.isinf(train_data)):
            raise ValueError("Training data contains infinite values")
            
        if validation_data is not None:
            if train_data.shape[1] != validation_data.shape[1]:
                raise ValueError("Training and validation data must have the same number of features")
                
            if np.any(np.isnan(validation_data)):
                raise ValueError("Validation data contains NaN values")
                
            if np.any(np.isinf(validation_data)):
                raise ValueError("Validation data contains infinite values")

    def _log_training_start(self, train_data: np.ndarray, 
                           validation_data: Optional[np.ndarray] = None) -> None:
        """
        Log training information.
        
        Args:
            train_data: Training data
            validation_data: Validation data (optional)
        """
        logger.info(f"Starting training for {self.model_name}")
        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Features: {train_data.shape[1]}")
        
        if validation_data is not None:
            logger.info(f"Validation samples: {len(validation_data)}")

    def _log_training_complete(self) -> None:
        """Log training completion."""
        logger.info(f"Training completed for {self.model_name}")
        self.is_trained = True


class ThresholdCalculatorMixin:
    """
    Mixin class providing common threshold calculation methods.
    """
    
    @staticmethod
    def calculate_threshold_strategies(scores: np.ndarray) -> Dict[str, float]:
        """
        Calculate all available threshold strategies.
        
        Args:
            scores: Anomaly scores from normal data
            
        Returns:
            Dictionary of threshold strategies and their values
        """
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        strategies = {
            'normal_mse_mean': mean_score,
            'mean_plus_1std': mean_score + std_score,
            'mean_plus_2std': mean_score + 2 * std_score,
            'mean_plus_3std': mean_score + 3 * std_score,
            'mse_plus_3std': mean_score + 3 * std_score,  # Linear: μ + 3σ
            'mse_plus_5std': mean_score + 5 * std_score,  # Linear: μ + 5σ
            'mse_plus_8std': mean_score + 8 * std_score,  # Linear: μ + 8σ
            'exponential_threshold': np.exp(mean_score + 10 * std_score),  # Exponential: e^(μ + 10σ)
            'sigmoid_threshold': 1 / (1 + np.exp(-(mean_score + 3 * std_score))),  # Sigmoid: 1/(1 + e^-(μ + 3σ))
            'percentile_95': np.percentile(scores, 95),
            'percentile_99': np.percentile(scores, 99),
            'percentile_99_5': np.percentile(scores, 99.5),
            'percentile_99_9': np.percentile(scores, 99.9)
        }
        
        return strategies
    
    @staticmethod
    def log_threshold_info(scores: np.ndarray, selected_threshold: float, 
                          strategy: str) -> None:
        """
        Log threshold calculation information.
        
        Args:
            scores: Anomaly scores
            selected_threshold: Selected threshold value
            strategy: Selected strategy name
        """
        logger.info(f"Threshold Statistics:")
        logger.info(f"  Mean Score: {np.mean(scores):.6f}")
        logger.info(f"  Std Score:  {np.std(scores):.6f}")
        logger.info(f"  Min Score:  {np.min(scores):.6f}")
        logger.info(f"  Max Score:  {np.max(scores):.6f}")
        logger.info(f"  Selected Threshold ({strategy}): {selected_threshold:.6f}")
