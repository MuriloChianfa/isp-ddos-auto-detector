"""
One-Class SVM implementation for anomaly detection.

This module implements a One-Class Support Vector Machine for anomaly detection
that learns a decision boundary around normal data points in feature space.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Any, Optional, List
import logging

from .core.template import BaseAnomalyDetector, ModelValidationMixin, ThresholdCalculatorMixin, DummyTrainingHistory

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
                 degree: int = 3, coef0: float = 0.0, cache_size: int = 54512, 
                 tol: float = 1e-2, max_iter: int = -1, shrinking: bool = True,
                 max_samples: int = None, n_components: int = 100, verbose: bool = False, **kwargs):
        """
        Initialize the One-Class SVM anomaly detector.
        
        Args:
            nu: Upper bound on fraction of training errors and lower bound of support vectors
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid', 'sgd_rbf')
            gamma: Kernel coefficient ('scale', 'auto' or float)
            degree: Degree of the polynomial kernel (ignored by other kernels)
            coef0: Independent term in kernel function (for 'poly' and 'sigmoid')
            cache_size: Size of kernel cache in MB (default 8192MB/8GB for speed)
            tol: Tolerance for stopping criterion (default 1e-2 for speed)
            max_iter: Max iterations for solver (-1 = no limit)
            shrinking: Whether to use shrinking heuristic (can speed up training)
            max_samples: Max training samples to use (None = use all, int = subsample for speed)
            n_components: Number of components for RBF kernel approximation (for sgd_rbf)
            verbose: Enable verbose output during training
            **kwargs: Additional parameters (ignored, for compatibility)
        """
        super().__init__(model_name="one_class_svm")
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.cache_size = cache_size
        self.tol = tol
        self.max_iter = max_iter
        self.shrinking = shrinking
        self.max_samples = max_samples
        self.n_components = n_components
        self.verbose = verbose
        self.model = None
        self.scaler = StandardScaler()
        self.rbf_sampler = None  # For sgd_rbf kernel approximation
        
    def build_model(self, input_dim: int) -> None:
        """Build the One-Class SVM model"""
        print(f"Building One-Class SVM for {input_dim} features...")
        
        if input_dim <= 0:
            raise ValueError(f"Invalid input dimension: {input_dim}")
        
        if self.kernel == 'sgd_rbf':
            # Use RBFSampler + SGDOneClassSVM for scalable RBF kernel approximation
            print("Using SGDOneClassSVM with RBF kernel approximation...")
            
            # Initialize RBF sampler for kernel approximation
            # gamma calculation similar to sklearn's 'scale' option
            gamma_value = 1.0 / input_dim if self.gamma == 'scale' else (
                1.0 if self.gamma == 'auto' else float(self.gamma)
            )
            
            self.rbf_sampler = RBFSampler(
                gamma=gamma_value,
                n_components=self.n_components,
                random_state=42
            )
            
            self.model = SGDOneClassSVM(
                nu=self.nu,
                fit_intercept=True,
                max_iter=self.max_iter if self.max_iter > 0 else 1000000,
                tol=self.tol,
                shuffle=True,
                verbose=0 if not self.verbose else 1,
                random_state=42,
                learning_rate='optimal',
                eta0=0.0,
                power_t=0.5,
                warm_start=False,
                average=False
            )
        else:
            self.model = OneClassSVM(
                nu=self.nu,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                cache_size=self.cache_size,
                tol=self.tol,
                max_iter=self.max_iter,
                shrinking=self.shrinking,
                verbose=self.verbose
            )
        
        print(f"One-Class SVM parameters:")
        print(f"  Features: {input_dim}")
        print(f"  Nu: {self.nu}")
        print(f"  Kernel: {self.kernel}")
        
        if self.kernel == 'sgd_rbf':
            print(f"  Algorithm: SGD (Stochastic Gradient Descent)")
            print(f"  RBF Components: {self.n_components}")
            print(f"  Gamma: {gamma_value}")
            print(f"  Max Iterations: {self.max_iter if self.max_iter > 0 else 1000000}")
            print(f"  Tolerance: {self.tol}")
            print(f"  Learning Rate: optimal")
        else:
            print(f"  Gamma: {self.gamma}")
            if self.kernel == 'poly':
                print(f"  Degree: {self.degree}")
            if self.kernel in ['poly', 'sigmoid']:
                print(f"  Coef0: {self.coef0}")
            print(f"  Cache Size: {self.cache_size} MB")
            print(f"  Tolerance: {self.tol}")
            print(f"  Shrinking: {self.shrinking}")
        
    def fit_scaler(self, training_features) -> None:
        """Fit the scaler (and RBF sampler if using sgd_rbf) on training data"""
        if training_features is None or len(training_features) == 0:
            raise ValueError("Training features cannot be empty")
        
        # Convert to numpy array if it's a DataFrame to avoid feature name warnings
        if hasattr(training_features, 'values'):
            training_data = training_features.values
        else:
            training_data = training_features
            
        self.scaler.fit(training_data)
        
        # Fit RBF sampler if using sgd_rbf kernel
        if self.kernel == 'sgd_rbf' and self.rbf_sampler is not None:
            scaled_data = self.scaler.transform(training_data)
            self.rbf_sampler.fit(scaled_data)
            logger.info(f"Fitted RBF sampler with {self.n_components} components")
        
        logger.info(f"Fitted scaler on {len(training_features)} training samples")
        
    def transform_data(self, features) -> np.ndarray:
        """Transform features using the fitted scaler (and RBF sampler if using sgd_rbf)"""
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        # Convert to numpy array if it's a DataFrame to avoid feature name warnings
        if hasattr(features, 'values'):
            feature_data = features.values
        else:
            feature_data = features
        
        # Apply standard scaling
        scaled_data = self.scaler.transform(feature_data)
        
        # Apply RBF transformation if using sgd_rbf kernel
        if self.kernel == 'sgd_rbf' and self.rbf_sampler is not None:
            scaled_data = self.rbf_sampler.transform(scaled_data)
            
        return scaled_data
        
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
        
        # Subsample training data if max_samples is set
        if self.max_samples is not None and len(train_data) > self.max_samples:
            print(f"Subsampling training data from {len(train_data)} to {self.max_samples} samples for speed...")
            indices = np.random.choice(len(train_data), self.max_samples, replace=False)
            train_data_sampled = train_data[indices]
            print(f"Using {len(train_data_sampled)} samples for training")
        else:
            train_data_sampled = train_data
        
        # Train the model
        self.model.fit(train_data_sampled)
        
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
        # anomaly_scores = decision_scores
        # anomaly_scores = (-decision_scores) ** 2
        # anomaly_scores = np.exp(-decision_scores) - 1
        # anomaly_scores = 1 / (1 + np.exp(decision_scores))
        # anomaly_scores = -decision_scores * np.abs(decision_scores)
        
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
            # For linear kernel, we can get feature weights directly
            if hasattr(self.model, 'coef_') and self.model.coef_ is not None:
                # Get absolute weights as feature importance
                coef = self.model.coef_[0] if self.model.coef_.ndim > 1 else self.model.coef_
                feature_errors = np.abs(coef)
                importance_indices = np.argsort(feature_errors)[::-1]
                
                print(f"\nOne-Class SVM Feature Importance (Linear Kernel):")
                print("=" * 60)
                print("Top 15 most important features:")
                for i, idx in enumerate(importance_indices[:15]):
                    if idx < len(feature_names):
                        print(f"{i+1:2d}. {feature_names[idx]:<30} | Weight: {feature_errors[idx]:.6f}")
            else:
                logger.warning("Linear SVM coefficients not available")
                feature_errors = np.ones(num_features)
                importance_indices = np.arange(num_features)
        elif self.kernel == 'sgd_rbf':
            # For SGD RBF, coefficients are in transformed space, not original features
            # We cannot directly map RBF features back to original features
            logger.info("SGD RBF kernel uses transformed features - direct feature importance not available")
            print(f"\nOne-Class SVM Feature Analysis (SGD RBF Kernel):")
            print("=" * 60)
            print("Note: RBF kernel transformation maps features to a higher-dimensional space.")
            print(f"Original features: {num_features}, RBF features: {self.n_components}")
            print("Direct feature importance in original space is not available.")
            
            # Return uniform importance
            feature_errors = np.ones(num_features)
            importance_indices = np.arange(num_features)
        else:
            # For non-linear kernels, feature importance is not directly available
            # logger.warning(f"One-Class SVM with {self.kernel} kernel doesn't provide direct feature importance")
            
            feature_errors = np.ones(num_features)
            importance_indices = np.arange(num_features)
            
            # print(f"\nOne-Class SVM Feature Analysis ({self.kernel} kernel):")
            # print("=" * 60)
            # print("Note: Non-linear SVM kernels don't provide direct feature importance.")
            # print("All features are weighted equally in the decision function.")
            # print(f"Total features: {num_features}")
        
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
    
    def save_artifacts(self, dataset_name: str, time_span: int, training_history: Optional[Dict] = None, training_time_seconds: Optional[float] = None) -> str:
        """
        Save model artifacts including RBF sampler for sgd_rbf kernel.
        
        Overrides base class method to also save the RBFSampler when using sgd_rbf kernel.
        """
        import os
        import joblib
        
        # Call parent save method first
        artifacts_dir = super().save_artifacts(dataset_name, time_span, training_history, training_time_seconds)
        
        # Save RBF sampler if it exists (for sgd_rbf kernel)
        if self.kernel == 'sgd_rbf' and self.rbf_sampler is not None:
            rbf_sampler_path = os.path.join(artifacts_dir, 'rbf_sampler.pkl')
            joblib.dump(self.rbf_sampler, rbf_sampler_path)
            logger.info(f"Saved RBF sampler to: {rbf_sampler_path}")
        
        return artifacts_dir
    
    @classmethod
    def load_artifacts(cls, dataset_name: str, model_name: str, time_span: int, **model_kwargs):
        """
        Load model artifacts including RBF sampler for sgd_rbf kernel.
        
        Overrides base class method to also load the RBFSampler when using sgd_rbf kernel.
        """
        import os
        import joblib
        
        # Call parent load method first
        model = super().load_artifacts(dataset_name, model_name, time_span, **model_kwargs)
        
        # Determine artifacts directory
        from .core.artifacts import ArtifactsManager
        artifacts_manager = ArtifactsManager(dataset_name, model_name, time_span)
        artifacts_dir = artifacts_manager.artifacts_dir
        
        # Load RBF sampler if it exists (for sgd_rbf kernel)
        if model.kernel == 'sgd_rbf':
            rbf_sampler_path = os.path.join(artifacts_dir, 'rbf_sampler.pkl')
            if os.path.exists(rbf_sampler_path):
                model.rbf_sampler = joblib.load(rbf_sampler_path)
                logger.info(f"Loaded RBF sampler from: {rbf_sampler_path}")
            else:
                logger.warning(f"RBF sampler not found for sgd_rbf kernel at: {rbf_sampler_path}")
        
        return model
