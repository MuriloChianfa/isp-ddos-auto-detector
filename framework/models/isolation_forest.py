"""
Isolation Forest implementation for anomaly detection.

This module implements an Isolation Forest-based anomaly detection model that
identifies anomalies by their isolation efficiency in randomly generated trees.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, Optional, List
import logging

from .core.template import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin, DummyTrainingHistory

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector(BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin):
    """
    Isolation Forest-based anomaly detection model.
    
    This model uses the Isolation Forest algorithm to identify anomalies by
    measuring how easily samples can be isolated in randomly generated trees.
    Anomalous samples require fewer splits to be isolated compared to normal samples.
    
    Attributes:
        contamination: Expected proportion of outliers in the dataset
        n_estimators: Number of base estimators in the ensemble
        random_state: Random state for reproducibility
        model: The scikit-learn IsolationForest model
        scaler: Standard scaler for feature normalization
    """
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, 
                 random_state: int = 42, max_samples: str = "auto", 
                 max_features: float = 1.0, bootstrap: bool = False):
        """
        Initialize the Isolation Forest anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers in the dataset
            n_estimators: Number of base estimators in the ensemble
            random_state: Random state for reproducibility
            max_samples: Number of samples to draw to train each base estimator
            max_features: Number of features to draw to train each base estimator
            bootstrap: Whether to use bootstrap sampling
        """
        super().__init__(model_name="isolation_forest")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, input_dim: int) -> None:
        """Build the Isolation Forest model"""
        print(f"Building Isolation Forest for {input_dim} features...")
        
        if input_dim <= 0:
            raise ValueError(f"Invalid input dimension: {input_dim}")
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=-1,  # Use all available cores
            verbose=0
        )
        
        print(f"Isolation Forest parameters:")
        print(f"  Features: {input_dim}")
        print(f"  Estimators: {self.n_estimators}")
        print(f"  Contamination: {self.contamination}")
        print(f"  Max samples: {self.max_samples}")
        print(f"  Max features: {self.max_features}")
        print(f"  Bootstrap: {self.bootstrap}")
        
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
        
    def train(self, train_data: np.ndarray, validation_data: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        """Train the Isolation Forest"""
        
        # Validate training data
        self.validate_training_data(train_data, validation_data)
        self._log_training_start(train_data, validation_data)
        
        print("Training Isolation Forest...")
        print(f"Training samples: {len(train_data)}")
        
        if self.model is None:
            raise ValueError("Model not built. Call build_model first.")
        
        # Train the model
        self.model.fit(train_data)
        
        self._log_training_complete()
        
        # Create dummy history for compatibility with other models
        history_dict = {
            'loss': [0.1],  # Dummy loss value
            'val_loss': [0.1] if validation_data is not None else None
        }
        self.history = DummyTrainingHistory(history_dict)
        
        print("Training completed")
        return self.history
        
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get anomaly scores (negative values = more anomalous)"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        self.validate_input_dimensions(data)
        
        # Get decision function scores (higher = more normal)
        decision_scores = self.model.decision_function(data)
        
        # Convert to anomaly scores (higher = more anomalous)
        # Isolation Forest returns negative scores for anomalies, so we negate them
        anomaly_scores = -decision_scores
        
        # Return original data as "reconstructions" for compatibility
        return data, anomaly_scores
        
    def calculate_threshold(self, train_scores: np.ndarray, val_scores: np.ndarray, 
                          strategy: str = 'exponential_threshold') -> Tuple[float, Dict[str, float]]:
        """Calculate anomaly detection threshold"""
        self.validate_threshold_strategy(strategy)
        
        # Combine scores from training and validation data
        all_scores = np.concatenate([train_scores, val_scores])
        all_strategies = self.calculate_threshold_strategies(all_scores)
        
        self.threshold = all_strategies[strategy]
        
        # Log threshold information
        self.log_threshold_info(all_scores, self.threshold, strategy)
        
        print(f"Anomaly Score Statistics:")
        print(f"  Mean Score: {np.mean(all_scores):.6f}")
        print(f"  Std Score:  {np.std(all_scores):.6f}")
        print(f"  Selected Threshold ({strategy}): {self.threshold:.6f}")
        
        return self.threshold, all_strategies
        
    def get_metrics(self, data: np.ndarray, reconstructions: np.ndarray, 
                   scores: np.ndarray) -> Dict[str, float]:
        """Calculate metrics (adapted for Isolation Forest)"""
        
        # For Isolation Forest, we don't have reconstructions in the traditional sense
        # So we calculate metrics based on the anomaly scores
        return {
            'mae': np.mean(np.abs(scores)),
            'mse': np.mean(scores**2),
            'rmse': np.sqrt(np.mean(scores**2)),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores)
        }
    
    def analyze_feature_importance(self, data: np.ndarray, 
                                 feature_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Feature importance for Isolation Forest.
        
        Note: Isolation Forest doesn't provide direct feature importance like tree-based
        models, so this returns uniform importance as a placeholder.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Return uniform importance as placeholder
        num_features = len(feature_names)
        feature_errors = np.ones(num_features)
        importance_indices = np.arange(num_features)
        
        return feature_errors, importance_indices
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """
        Get Isolation Forest specific information.
        
        Returns:
            Dictionary containing model-specific information
        """
        base_info = self.get_model_info()
        
        isolation_forest_info = {
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'random_state': self.random_state,
            'max_samples': self.max_samples,
        }
        
        if self.is_trained and self.model is not None:
            isolation_forest_info.update({
                'n_features_in_': getattr(self.model, 'n_features_in_', None),
                'max_samples_': getattr(self.model, 'max_samples_', None),
            })
        
        base_info.update(isolation_forest_info)
        return base_info
