"""
One-Class SVM implementation for anomaly detection.

This module implements a One-Class Support Vector Machine for anomaly detection
that learns a decision boundary around normal data points in feature space.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, Optional, List
import logging

from .base_model import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin, DummyTrainingHistory

logger = logging.getLogger(__name__)


class OneClassSVMAnomalyDetector(BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin):
    """
    One-Class SVM-based anomaly detection model.
    
    This model uses a One-Class Support Vector Machine to learn a decision boundary
    around normal data points. It identifies anomalies as points that fall outside
    this learned boundary in the feature space.
    
    Attributes:
        nu: Upper bound on the fraction of training errors and lower bound of support vectors
        kernel: Kernel type to be used in the algorithm
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
        model: The scikit-learn OneClassSVM model
        scaler: Standard scaler for feature normalization
    """
    
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf', gamma: str = 'scale', 
                 degree: int = 3, coef0: float = 0.0):
        """
        Initialize the One-Class SVM anomaly detector.
        
        Args:
            nu: Upper bound on fraction of training errors and lower bound of support vectors
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient ('scale', 'auto' or float)
            degree: Degree of the polynomial kernel (ignored by other kernels)
            coef0: Independent term in kernel function (for 'poly' and 'sigmoid')
        """
        super().__init__(model_name="one_class_svm")
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, input_dim: int) -> None:
        """Build the One-Class SVM model"""
        print(f"Building One-Class SVM for {input_dim} features...")
        
        if input_dim <= 0:
            raise ValueError(f"Invalid input dimension: {input_dim}")
        
        self.model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            cache_size=200,  # Size of kernel cache (MB)
            verbose=False
        )
        
        logger.info(f"One-Class SVM parameters:")
        logger.info(f"  Features: {input_dim}")
        logger.info(f"  Nu: {self.nu}")
        logger.info(f"  Kernel: {self.kernel}")
        logger.info(f"  Gamma: {self.gamma}")
        logger.info(f"  Degree: {self.degree}")
        logger.info(f"  Coef0: {self.coef0}")
        
        print(f"One-Class SVM parameters:")
        print(f"  Features: {input_dim}")
        print(f"  Nu: {self.nu}")
        print(f"  Kernel: {self.kernel}")
        print(f"  Gamma: {self.gamma}")
        if self.kernel == 'poly':
            print(f"  Degree: {self.degree}")
        if self.kernel in ['poly', 'sigmoid']:
            print(f"  Coef0: {self.coef0}")
        
    def fit_scaler(self, training_features: np.ndarray) -> None:
        """Fit the scaler on training data"""
        if training_features is None or len(training_features) == 0:
            raise ValueError("Training features cannot be empty")
        
        self.scaler.fit(training_features)
        logger.info(f"Fitted scaler on {len(training_features)} training samples")
        
    def transform_data(self, features: np.ndarray) -> np.ndarray:
        """Transform features using the fitted scaler"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        return self.scaler.transform(features)
        
    def train(self, train_data: np.ndarray, validation_data: Optional[np.ndarray] = None, 
              **kwargs) -> Dict[str, Any]:
        """Train the One-Class SVM"""
        
        # Validate training data
        self.validate_training_data(train_data, validation_data)
        self._log_training_start(train_data, validation_data)
        
        print("Training One-Class SVM...")
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
        """Get anomaly scores"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        self.validate_input_dimensions(data)
        
        # Get decision function scores
        # Positive scores = normal, negative scores = anomalous
        decision_scores = self.model.decision_function(data)
        
        # Convert to anomaly scores (higher = more anomalous)
        # SVM returns negative values for anomalies, so we negate them
        anomaly_scores = -decision_scores
        
        # Return original data as "reconstructions" for compatibility
        return data, anomaly_scores
        
    def calculate_threshold(self, train_scores: np.ndarray, val_scores: np.ndarray, 
                          strategy: str = 'percentile_99') -> Tuple[float, Dict[str, float]]:
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
        """Calculate metrics (adapted for One-Class SVM)"""
        
        # For One-Class SVM, we don't have reconstructions in the traditional sense
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
        Feature importance for One-Class SVM.
        
        Note: One-Class SVM with non-linear kernels doesn't provide direct feature
        importance, so this returns uniform importance as a placeholder.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        num_features = len(feature_names)
        
        if self.kernel == 'linear':
            # For linear kernel, we can get feature weights
            if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                # Get absolute weights as feature importance
                feature_errors = np.abs(self.model.coef_[0])
                importance_indices = np.argsort(feature_errors)[::-1]
                
                print("\nOne-Class SVM Feature Importance (Linear Kernel):")
                print("=" * 60)
                print("Top 15 most important features:")
                for i, idx in enumerate(importance_indices[:15]):
                    print(f"{i+1:2d}. {feature_names[idx]:<30} | Weight: {feature_errors[idx]:.6f}")
            else:
                logger.warning("Linear SVM coefficients not available")
                feature_errors = np.ones(num_features)
                importance_indices = np.arange(num_features)
        else:
            # For non-linear kernels, feature importance is not directly available
            logger.warning(f"One-Class SVM with {self.kernel} kernel doesn't provide direct feature importance")
            
            feature_errors = np.ones(num_features)
            importance_indices = np.arange(num_features)
            
            print(f"\nOne-Class SVM Feature Analysis ({self.kernel} kernel):")
            print("=" * 60)
            print("Note: Non-linear SVM kernels don't provide direct feature importance.")
            print("All features are weighted equally in the decision function.")
            print(f"Total features: {num_features}")
        
        return feature_errors, importance_indices
    
    def get_model_specific_info(self) -> Dict[str, Any]:
        """
        Get One-Class SVM specific information.
        
        Returns:
            Dictionary containing model-specific information
        """
        base_info = self.get_model_info()
        
        svm_info = {
            'nu': self.nu,
            'kernel': self.kernel,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
        }
        
        if self.is_trained and self.model is not None:
            svm_info.update({
                'n_features_in_': getattr(self.model, 'n_features_in_', None),
                'n_support_': getattr(self.model, 'n_support_', None),
                'support_vectors_shape': getattr(self.model, 'support_vectors_', np.array([])).shape,
            })
        
        base_info.update(svm_info)
        return base_info
