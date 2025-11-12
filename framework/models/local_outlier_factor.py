"""
Local Outlier Factor implementation for anomaly detection.

This module implements a Local Outlier Factor (LOF) based anomaly detection model that
identifies anomalies by measuring the local density deviation of a sample with respect 
to its neighbors.
"""

import numpy as np
import logging
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, Optional, List

from .core.template import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin, DummyTrainingHistory

logger = logging.getLogger(__name__)


class LocalOutlierFactorAnomalyDetector(BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin):
    """
    Local Outlier Factor (LOF) based anomaly detection model.
    
    This model uses the Local Outlier Factor algorithm to identify anomalies by
    measuring the local density deviation of a sample with respect to its neighbors.
    Anomalous samples have significantly lower density than their neighbors.
    
    Attributes:
        n_neighbors: Number of neighbors to use for LOF calculation
        contamination: Expected proportion of outliers in the dataset
        novelty: Whether to enable novelty detection mode (required for predict)
        random_state: Random state for reproducibility
        model: The scikit-learn LocalOutlierFactor model
        scaler: Standard scaler for feature normalization
    """
    
    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1, 
                 novelty: bool = True, random_state: int = 42,
                 algorithm: str = 'auto', leaf_size: int = 30,
                 metric: str = 'minkowski', p: int = 2):
        """
        Initialize the Local Outlier Factor anomaly detector.
        
        Args:
            n_neighbors: Number of neighbors to use for LOF calculation
            contamination: Expected proportion of outliers in the dataset
            novelty: Whether to enable novelty detection mode (must be True for prediction)
            random_state: Random state for reproducibility
            algorithm: Algorithm used to compute nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size: Leaf size passed to BallTree or KDTree (affects build and query time)
            metric: Distance metric to use ('minkowski', 'euclidean', 'manhattan', 'chebyshev', etc.)
            p: Power parameter for the Minkowski metric (1=Manhattan, 2=Euclidean)
        """
        super().__init__(model_name="local_outlier_factor")
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty
        self.random_state = random_state
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, input_dim: int) -> None:
        """Build the Local Outlier Factor model"""
        print(f"Building Local Outlier Factor for {input_dim} features...")
        
        if input_dim <= 0:
            raise ValueError(f"Input dimension must be positive, got {input_dim}")
        
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=self.novelty,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            n_jobs=-1  # Use all available cores
        )
        
        print(f"Local Outlier Factor parameters:")
        print(f"  Features: {input_dim}")
        print(f"  Neighbors: {self.n_neighbors}")
        print(f"  Contamination: {self.contamination}")
        print(f"  Novelty mode: {self.novelty}")
        print(f"  Algorithm: {self.algorithm}")
        print(f"  Leaf size: {self.leaf_size}")
        print(f"  Metric: {self.metric}")
        print(f"  p (Minkowski): {self.p}")
        print(f"  Novelty mode: {self.novelty}")
        
    def fit_scaler(self, training_features) -> None:
        """Fit the scaler on training data"""
        if training_features is None or len(training_features) == 0:
            raise ValueError("Training features cannot be None or empty")
        
        # Convert to numpy array if it's a DataFrame to avoid feature name warnings
        if hasattr(training_features, 'values'):
            training_data = training_features.values
        else:
            training_data = np.array(training_features)
            
        self.scaler.fit(training_data)
        logger.info(f"Fitted scaler on {len(training_features)} training samples")
        
    def transform_data(self, features) -> np.ndarray:
        """Transform features using the fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet")
        
        # Convert to numpy array if it's a DataFrame to avoid feature name warnings
        if hasattr(features, 'values'):
            feature_data = features.values
        else:
            feature_data = np.array(features)
            
        return self.scaler.transform(feature_data)
        
    def train(self, train_data: np.ndarray, validation_data: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        """Train the Local Outlier Factor model"""
        
        # Validate training data
        self.validate_training_data(train_data, validation_data)
        self._log_training_start(train_data, validation_data)
        
        print("Training Local Outlier Factor...")
        print(f"Training samples: {len(train_data)}")
        
        if self.model is None:
            raise ValueError("Model has not been built yet. Call build_model() first.")
        
        # Train the model (fit on normal data)
        self.model.fit(train_data)
        
        self._log_training_complete()
        
        # Create dummy history for compatibility with other models
        history_dict = {
            'loss': [0.1],  # Dummy loss for visualization
            'val_loss': [0.1] if validation_data is not None else None
        }
        self.history = DummyTrainingHistory(history_dict)
        
        print("Training completed")
        return self.history
        
    def predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get anomaly scores (negative values = more anomalous)"""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
            
        self.validate_input_dimensions(data)
        
        # Get decision function scores (higher = more normal, negative = anomalies)
        # In novelty mode, decision_function returns negative scores for anomalies
        decision_scores = self.model.decision_function(data)
        
        # Convert to anomaly scores (higher = more anomalous)
        # LOF returns negative scores for anomalies, so we negate them
        # anomaly_scores = -decision_scores
        anomaly_scores = -decision_scores
        
        # Return original data as "reconstructions" for compatibility
        return data, anomaly_scores
        
    def calculate_threshold(self, train_scores: np.ndarray, val_scores: np.ndarray, 
                          strategy: str = 'percentile_99_5') -> Tuple[float, Dict[str, float]]:
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
        """Calculate metrics (adapted for Local Outlier Factor)"""
        
        # For LOF, we don't have reconstructions in the traditional sense
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
        Feature importance for Local Outlier Factor.
        
        Note: LOF doesn't provide direct feature importance like tree-based
        models, so this returns uniform importance as a placeholder.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        # Return uniform importance as placeholder
        num_features = len(feature_names)
        importance = np.ones(num_features) / num_features
        std = np.zeros(num_features)
        
        return importance, std
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """Return LOF-specific model information"""
        return {
            'model_type': 'local_outlier_factor',
            'n_neighbors': self.n_neighbors,
            'contamination': self.contamination,
            'novelty': self.novelty,
            'algorithm': self.algorithm,
            'leaf_size': self.leaf_size,
            'metric': self.metric,
            'p': self.p
        }
