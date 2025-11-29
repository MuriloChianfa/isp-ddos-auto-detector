"""
Hyperparameter optimization module for anomaly detection models
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import uniform, randint
import logging
from typing import Dict, Any, Tuple, List
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RandomSearchOptimizer:
    """Base class for random search hyperparameter optimization"""
    
    def __init__(self, n_iter=50, random_state=42, scoring='f1_score', y_val=None):
        """
        Args:
            n_iter: Number of parameter settings sampled
            random_state: Random state for reproducibility
            scoring: Scoring method (default: 'f1_score', options: 'f1_score', 'f2_score', 'anomaly_score')
            y_val: True labels for validation set (required for f1_score and f2_score)
        """
        self.n_iter = n_iter
        self.random_state = random_state
        self.scoring = scoring
        self.y_val = y_val
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []
        
    def get_param_distributions(self) -> Dict:
        """Define parameter distributions - to be implemented by subclasses"""
        raise NotImplementedError
        
    def fit(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[Dict, Any]:
        """
        Perform random search
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
        
        Returns:
            best_params: Best parameters found
            cv_results: All results from cross-validation
        """
        raise NotImplementedError
        
    def score_model(self, model, X_val: np.ndarray) -> float:
        """Calculate score for a trained model"""
        from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
        
        if self.scoring == 'anomaly_score':
            # Lower (more negative) mean score = better separation
            scores = model.score_samples(X_val)
            return -np.mean(scores)
        elif self.scoring in ['f1_score', 'f2_score']:
            # Need true labels for supervised metrics
            if self.y_val is None:
                raise ValueError(f"y_val is required for scoring method: {self.scoring}")
            
            # Get predictions (-1 for outliers/anomalies, 1 for inliers/normal)
            y_pred = model.predict(X_val)
            # Convert to binary: 1 for anomaly, 0 for normal
            y_pred_binary = (y_pred == -1).astype(int)
            
            if self.scoring == 'f1_score':
                # F1 score: balanced precision and recall
                score = f1_score(self.y_val, y_pred_binary, zero_division=0)
                logger.debug(f"  F1={score:.4f}, P={precision_score(self.y_val, y_pred_binary, zero_division=0):.4f}, R={recall_score(self.y_val, y_pred_binary, zero_division=0):.4f}")
            elif self.scoring == 'f2_score':
                # F2 score: weighs recall higher than precision (2x weight)
                score = fbeta_score(self.y_val, y_pred_binary, beta=2, zero_division=0)
                logger.debug(f"  F2={score:.4f}, P={precision_score(self.y_val, y_pred_binary, zero_division=0):.4f}, R={recall_score(self.y_val, y_pred_binary, zero_division=0):.4f}")
            
            return score
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring}")


class IsolationForestRandomSearch(RandomSearchOptimizer):
    """Random search for Isolation Forest hyperparameters"""
    
    def get_param_distributions(self) -> Dict:
        """Define parameter distributions for random search"""
        return {
            'n_estimators': [int(x) for x in np.linspace(50, 500, 20)],
            'max_samples': [int(x) for x in np.linspace(64, 512, 20)] + ['auto'],
            'contamination': [round(x, 3) for x in np.linspace(0.01, 0.20, 20)],
            'max_features': [round(x, 2) for x in np.linspace(0.5, 1.0, 10)],
            'bootstrap': [True, False]
        }
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[Dict, List]:
        """
        Perform random search for Isolation Forest
        
        Args:
            X_train: Training features
            X_val: Validation features (if None, uses X_train)
        
        Returns:
            best_params: Best parameters found
            cv_results: All results from search
        """
        if X_val is None:
            X_val = X_train
            
        # Get base parameter distributions
        param_distributions = self.get_param_distributions()
        
        # Adjust max_samples based on training data size
        n_samples = len(X_train)
        max_samples_limit = min(n_samples, 512)
        if max_samples_limit < 64:
            # For very small datasets, use smaller values
            param_distributions['max_samples'] = [int(x) for x in np.linspace(max(10, n_samples // 4), n_samples, 10)] + ['auto']
        else:
            # Adjust the upper limit to not exceed training samples
            param_distributions['max_samples'] = [int(x) for x in np.linspace(64, max_samples_limit, 20)] + ['auto']
        
        # Sample parameters
        param_list = list(ParameterSampler(
            param_distributions, 
            n_iter=self.n_iter, 
            random_state=self.random_state
        ))
        
        logger.info(f"Starting Isolation Forest random search with {self.n_iter} iterations...")
        
        best_score = float('-inf')
        best_params = None
        
        for i, params in enumerate(param_list, 1):
            try:
                print(f"\rOptimization progress: [{i}/{self.n_iter}] Testing parameters...", end='', flush=True)
                
                # Train model with current parameters
                model = IsolationForest(
                    random_state=self.random_state,
                    **params
                )
                model.fit(X_train)
                
                # Score on validation set
                score = self.score_model(model, X_val)
                
                # Store results
                result = {
                    'params': params,
                    'score': score,
                    'iteration': i
                }
                self.cv_results_.append(result)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params
                    print(f"\rIteration {i}/{self.n_iter}: New best score = {score:.6f}")
                    logger.info(f"  Params: {params}")
                else:
                    logger.debug(f"Iteration {i}/{self.n_iter}: Score = {score:.6f}")
                    
            except Exception as e:
                print(f"\rIteration {i}/{self.n_iter}: Failed")
                logger.warning(f"Iteration {i} failed with params {params}: {e}")
                continue
        
        # Clear progress line
        print()
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        logger.info(f"\nOptimization completed!")
        logger.info(f"Best score: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, self.cv_results_


class OneClassSVMRandomSearch(RandomSearchOptimizer):
    """Random search for One-Class SVM hyperparameters"""
    
    def __init__(self, n_iter=50, random_state=42, scoring='f1_score', y_val=None, n_jobs=-1, 
                 cache_size=4096, tol=1e-2, max_iter=-1, shrinking=True, kernel=None):
        """
        Initialize One-Class SVM optimizer with performance parameters
        
        Args:
            n_iter: Number of parameter settings sampled
            random_state: Random state for reproducibility
            scoring: Scoring method (default: 'f1_score')
            y_val: True labels for validation set (required for f1_score and f2_score)
            n_jobs: Number of parallel jobs (-1 uses all cores)
            cache_size: Size of kernel cache in MB (larger = faster but more memory)
            tol: Tolerance for stopping criterion (larger = faster but less accurate)
            max_iter: Max iterations for solver (-1 = no limit)
            shrinking: Whether to use shrinking heuristic (can speed up training)
            kernel: Kernel type to use (if specified, only this kernel will be optimized)
        """
        super().__init__(n_iter, random_state, scoring, y_val)
        self.n_jobs = n_jobs
        self.cache_size = cache_size
        self.tol = tol
        self.max_iter = max_iter
        self.shrinking = shrinking
        self.kernel = kernel
    
    def get_param_distributions(self) -> Dict:
        """Define parameter distributions for random search"""
        # If a specific kernel is configured, only optimize for that kernel
        if self.kernel is not None:
            # For sgd_rbf, optimize using rbf parameters since sgd_rbf is a custom
            # implementation that uses RBF kernel approximation
            if self.kernel == 'sgd_rbf':
                kernels = ['rbf']
            else:
                kernels = [self.kernel]
        else:
            # Otherwise, search across all standard kernel types (excluding sgd_rbf)
            kernels = ['rbf', 'sigmoid', 'poly']
        
        return {
            'nu': [round(x, 3) for x in np.linspace(0.01, 0.30, 20)],
            'kernel': kernels,
            'gamma': ['scale', 'auto'] + [round(x, 4) for x in np.logspace(-4, -1, 10)],
            'degree': [2, 3, 4, 5],  # Only for poly kernel
            'coef0': [round(x, 2) for x in np.linspace(-1.0, 1.0, 10)]  # For poly/sigmoid
        }
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[Dict, List]:
        """
        Perform random search for One-Class SVM
        
        Args:
            X_train: Training features
            X_val: Validation features (if None, uses X_train)
        
        Returns:
            best_params: Best parameters found
            cv_results: All results from search
        """
        if X_val is None:
            X_val = X_train
            
        param_distributions = self.get_param_distributions()
        
        # Sample parameters
        param_list = list(ParameterSampler(
            param_distributions, 
            n_iter=self.n_iter, 
            random_state=self.random_state
        ))
        
        logger.info(f"Starting One-Class SVM random search with {self.n_iter} iterations...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        logger.info(f"Performance settings: cache_size={self.cache_size}MB, tol={self.tol}, shrinking={self.shrinking}")
        
        # Initialize original_y_val
        original_y_val = None
        
        # For large datasets, sample training data for faster optimization
        # One-Class SVM is O(n^2) to O(n^3) in complexity, so large datasets are very slow
        max_train_samples = 50000  # Reasonable size for optimization
        if len(X_train) > max_train_samples:
            train_sample_indices = np.random.choice(len(X_train), max_train_samples, replace=False)
            X_train_sample = X_train[train_sample_indices]
            logger.info(f"Subsampling training data: {len(X_train)} -> {max_train_samples} samples for faster optimization")
            print(f"Note: Using {max_train_samples:,} training samples (of {len(X_train):,}) for faster optimization")
        else:
            X_train_sample = X_train
        
        # For large datasets, sample validation data for faster scoring
        max_val_samples = 25000
        if len(X_val) > max_val_samples:
            val_sample_indices = np.random.choice(len(X_val), max_val_samples, replace=False)
            X_val_sample = X_val[val_sample_indices]
            if self.y_val is not None:
                y_val_sample = self.y_val[val_sample_indices]
                # Temporarily store original y_val and use sample
                original_y_val = self.y_val
                self.y_val = y_val_sample
            logger.info(f"Subsampling validation data: {len(X_val)} -> {max_val_samples} samples for faster scoring")
            print(f"Note: Using {max_val_samples:,} validation samples (of {len(X_val):,}) for faster scoring")
        else:
            X_val_sample = X_val
        
        best_score = float('-inf')
        best_params = None
        
        for i, params in enumerate(param_list, 1):
            try:
                import time
                start_time = time.time()
                
                print(f"\rOptimization progress: [{i}/{self.n_iter}] Testing parameters...", end='', flush=True)
                
                # Remove irrelevant params based on kernel
                kernel = params['kernel']
                if kernel == 'rbf':
                    params.pop('degree', None)
                    params.pop('coef0', None)
                elif kernel == 'sigmoid':
                    params.pop('degree', None)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {i}/{self.n_iter}")
                logger.info(f"Parameters: {params}")
                
                # Add performance optimization parameters
                params['cache_size'] = self.cache_size
                params['tol'] = self.tol
                params['max_iter'] = self.max_iter
                params['shrinking'] = self.shrinking
                params['verbose'] = False
                    
                # Train model with current parameters
                print(f"\rIteration {i}/{self.n_iter}: Training with nu={params.get('nu'):.3f}, kernel={params.get('kernel')}...", end='', flush=True)
                model = OneClassSVM(**params)
                model.fit(X_train_sample)
                train_time = time.time() - start_time
                logger.info(f"Training completed in {train_time:.2f} seconds")
                
                # Score on validation set
                score_start = time.time()
                score = self.score_model(model, X_val_sample)
                score_time = time.time() - score_start
                total_time = time.time() - start_time
                logger.info(f"Scoring completed in {score_time:.2f} seconds (total: {total_time:.2f}s)")
                
                # Store results
                result = {
                    'params': params.copy(),
                    'score': score,
                    'iteration': i,
                    'train_time': train_time,
                    'score_time': score_time
                }
                self.cv_results_.append(result)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"\rIteration {i}/{self.n_iter}: New best score = {score:.6f} (time: {total_time:.1f}s)")
                    logger.info(f"  Params: {params}")
                else:
                    logger.debug(f"Iteration {i}/{self.n_iter}: Score = {score:.6f}")
                    
            except Exception as e:
                print(f"\rIteration {i}/{self.n_iter}: Failed")
                logger.warning(f"Iteration {i} failed with params {params}: {e}")
                continue
        
        # Clear progress line
        print()
        
        # Restore original y_val if it was sampled
        if original_y_val is not None:
            self.y_val = original_y_val
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        logger.info(f"\nOptimization completed!")
        logger.info(f"Best score: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, self.cv_results_


class LocalOutlierFactorRandomSearch(RandomSearchOptimizer):
    """Random search for Local Outlier Factor hyperparameters"""
    
    def get_param_distributions(self) -> Dict:
        """Define parameter distributions for random search"""
        return {
            'n_neighbors': [int(x) for x in np.linspace(20, 200, 12)],
            'contamination': [round(x, 3) for x in np.linspace(0.01, 0.30, 18)],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'algorithm': ['ball_tree', 'kd_tree'],
            'leaf_size': [int(x) for x in np.linspace(10, 200, 14)],
            'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
            'p': [1, 2, 3]  # For minkowski metric
        }
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[Dict, List]:
        """
        Perform random search for Local Outlier Factor
        
        Args:
            X_train: Training features
            X_val: Validation features (if None, uses X_train)
        
        Returns:
            best_params: Best parameters found
            cv_results: All results from search
        """
        if X_val is None:
            X_val = X_train
            
        param_distributions = self.get_param_distributions()
        
        # Sample parameters
        param_list = list(ParameterSampler(
            param_distributions, 
            n_iter=self.n_iter, 
            random_state=self.random_state
        ))
        
        logger.info(f"Starting Local Outlier Factor random search with {self.n_iter} iterations...")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Initialize original_y_val
        original_y_val = None
        
        # For large datasets, sample training data for faster optimization
        # LOF is O(n^2) in complexity, so large datasets are very slow
        max_train_samples = 100000  # Reasonable size for optimization
        if len(X_train) > max_train_samples:
            train_sample_indices = np.random.choice(len(X_train), max_train_samples, replace=False)
            X_train_sample = X_train[train_sample_indices]
            logger.info(f"Subsampling training data: {len(X_train)} -> {max_train_samples} samples for faster optimization")
            print(f"Note: Using {max_train_samples:,} training samples (of {len(X_train):,}) for faster optimization")
        else:
            X_train_sample = X_train
        
        # For large datasets, sample validation data for faster scoring
        max_val_samples = 50000
        if len(X_val) > max_val_samples:
            val_sample_indices = np.random.choice(len(X_val), max_val_samples, replace=False)
            X_val_sample = X_val[val_sample_indices]
            if self.y_val is not None:
                y_val_sample = self.y_val[val_sample_indices]
                # Temporarily store original y_val and use sample
                original_y_val = self.y_val
                self.y_val = y_val_sample
            logger.info(f"Subsampling validation data: {len(X_val)} -> {max_val_samples} samples for faster scoring")
            print(f"Note: Using {max_val_samples:,} validation samples (of {len(X_val):,}) for faster scoring")
        else:
            X_val_sample = X_val
        
        best_score = float('-inf')
        best_params = None
        
        for i, params in enumerate(param_list, 1):
            try:
                import time
                start_time = time.time()
                
                print(f"\rOptimization progress: [{i}/{self.n_iter}] Testing parameters...", end='', flush=True)
                
                # Add novelty=True for prediction
                params['novelty'] = True
                params['n_jobs'] = -1
                # Note: LocalOutlierFactor doesn't support random_state parameter, think later on how to fix it
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {i}/{self.n_iter}")
                logger.info(f"Parameters: {params}")
                
                # Train model with current parameters
                print(f"\rIteration {i}/{self.n_iter}: Training with n_neighbors={params.get('n_neighbors')}, contamination={params.get('contamination'):.3f}...", end='', flush=True)
                model = LocalOutlierFactor(**params)
                model.fit(X_train_sample)
                train_time = time.time() - start_time
                logger.info(f"Training completed in {train_time:.2f} seconds")
                
                # Score on validation set
                score_start = time.time()
                score = self.score_model(model, X_val_sample)
                score_time = time.time() - score_start
                total_time = time.time() - start_time
                logger.info(f"Scoring completed in {score_time:.2f} seconds (total: {total_time:.2f}s)")
                
                result = {
                    'params': params.copy(),
                    'score': score,
                    'iteration': i,
                    'train_time': train_time,
                    'score_time': score_time
                }
                self.cv_results_.append(result)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"\rIteration {i}/{self.n_iter}: New best score = {score:.6f} (time: {total_time:.1f}s)")
                    logger.info(f"  Params: {params}")
                else:
                    logger.debug(f"Iteration {i}/{self.n_iter}: Score = {score:.6f}")
                    
            except Exception as e:
                print(f"\rIteration {i}/{self.n_iter}: Failed")
                logger.warning(f"Iteration {i} failed with params {params}: {e}")
                continue
        
        # Clear progress line
        print()
        
        # Restore original y_val if it was sampled
        if original_y_val is not None:
            self.y_val = original_y_val
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        logger.info(f"\nOptimization completed!")
        logger.info(f"Best score: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, self.cv_results_


class AutoencoderRandomSearch(RandomSearchOptimizer):
    """Random search for Autoencoder hyperparameters"""
    
    def __init__(self, n_iter=50, random_state=42, scoring='f1_score', y_val=None, 
                 input_dim=None, epochs=50, use_early_stopping=True):
        """
        Initialize Autoencoder optimizer
        
        Args:
            n_iter: Number of parameter settings sampled
            random_state: Random state for reproducibility
            scoring: Scoring method (default: 'f1_score', options: 'f1_score', 'f2_score', 'mse_score')
            y_val: True labels for validation set (required for f1_score and f2_score)
            input_dim: Number of input features (required to scale architecture)
            epochs: Number of training epochs per iteration (default: 50 for faster optimization)
            use_early_stopping: Whether to use early stopping during training
        """
        super().__init__(n_iter, random_state, scoring, y_val)
        self.input_dim = input_dim
        self.epochs = epochs
        self.use_early_stopping = use_early_stopping
        
        if input_dim is None:
            raise ValueError("input_dim is required for autoencoder optimization")
    
    def get_param_distributions(self) -> Dict:
        """Define parameter distributions for random search"""
        # Generate hidden layer configurations based on input_dim
        hidden_layers_options = self._generate_hidden_layer_configs()
        
        return {
            'hidden_layers': hidden_layers_options,
            'latent_dim': [int(x) for x in np.linspace(4, min(32, self.input_dim // 2), 10)],
            'batch_size': [32, 64, 128],
            'learning_rate': [0.001, 0.005, 0.01, 0.05]
        }
    
    def _generate_hidden_layer_configs(self) -> List[List[int]]:
        """
        Generate hidden layer configurations scaled to input_dim
        
        Strategy:
        - Number of layers: 2-4
        - First layer: 60-95% of input_dim
        - Subsequent layers: 50-80% of previous layer
        - Ensure decreasing sizes
        """
        configs = []
        
        # 2-layer architectures
        for first_pct in [0.6, 0.7, 0.8, 0.9]:
            first_layer = min(self.input_dim - 1, max(4, int(self.input_dim * first_pct)))
            for second_pct in [0.5, 0.6, 0.7]:
                second_layer = max(4, int(first_layer * second_pct))
                if second_layer < first_layer:
                    configs.append([first_layer, second_layer])
        
        # 3-layer architectures
        for first_pct in [0.7, 0.8, 0.9, 0.95]:
            first_layer = min(self.input_dim - 1, max(4, int(self.input_dim * first_pct)))
            for second_pct in [0.6, 0.7, 0.8]:
                second_layer = max(4, int(first_layer * second_pct))
                for third_pct in [0.5, 0.6, 0.7]:
                    third_layer = max(4, int(second_layer * third_pct))
                    if third_layer < second_layer < first_layer:
                        configs.append([first_layer, second_layer, third_layer])
        
        # 4-layer architectures
        for first_pct in [0.8, 0.9]:
            first_layer = min(self.input_dim - 1, max(4, int(self.input_dim * first_pct)))
            second_layer = max(4, int(first_layer * 0.75))
            third_layer = max(4, int(second_layer * 0.7))
            fourth_layer = max(4, int(third_layer * 0.6))
            if fourth_layer < third_layer < second_layer < first_layer:
                configs.append([first_layer, second_layer, third_layer, fourth_layer])
        
        # Remove duplicates while preserving order
        unique_configs = []
        seen = set()
        for config in configs:
            config_tuple = tuple(config)
            if config_tuple not in seen:
                seen.add(config_tuple)
                unique_configs.append(config)
        
        return unique_configs
    
    def fit(self, X_train: np.ndarray, X_val: np.ndarray = None) -> Tuple[Dict, List]:
        """
        Perform random search for Autoencoder
        
        Args:
            X_train: Training features (already scaled)
            X_val: Validation features (already scaled, if None uses X_train)
        
        Returns:
            best_params: Best parameters found
            cv_results: All results from search
        """
        if X_val is None:
            X_val = X_train
        
        param_distributions = self.get_param_distributions()
        
        # Sample parameters
        param_list = list(ParameterSampler(
            param_distributions,
            n_iter=self.n_iter,
            random_state=self.random_state
        ))
        
        logger.info(f"Starting Autoencoder random search with {self.n_iter} iterations...")
        logger.info(f"Input dimension: {self.input_dim}")
        logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        logger.info(f"Epochs per iteration: {self.epochs}")
        
        best_score = float('-inf')
        best_params = None
        
        for i, params in enumerate(param_list, 1):
            try:
                import time
                start_time = time.time()
                
                print(f"\rOptimization progress: [{i}/{self.n_iter}] Testing parameters...", end='', flush=True)
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {i}/{self.n_iter}")
                logger.info(f"Parameters: {params}")
                
                # Extract parameters
                hidden_layers = params['hidden_layers']
                latent_dim = params['latent_dim']
                batch_size = params['batch_size']
                learning_rate = params['learning_rate']
                
                # Ensure latent_dim is smaller than last hidden layer
                if latent_dim >= hidden_layers[-1]:
                    latent_dim = max(2, hidden_layers[-1] // 2)
                    params['latent_dim'] = latent_dim
                
                # Build and train autoencoder
                from framework.models.autoencoder import AutoencoderAnomalyDetector
                import tensorflow as tf
                
                # Create model
                model = AutoencoderAnomalyDetector(
                    latent_dim=latent_dim,
                    hidden_layers=hidden_layers
                )
                
                # Build model
                model.build_model(self.input_dim)
                
                # Recompile with custom learning rate
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                )
                model.autoencoder.compile(
                    optimizer=optimizer,
                    loss='mse',
                    metrics=['mae']
                )
                
                # Train model
                print(f"\rIteration {i}/{self.n_iter}: Training architecture {hidden_layers} -> {latent_dim}...", end='', flush=True)
                history = model.train(
                    X_train, X_val,
                    epochs=self.epochs,
                    batch_size=batch_size
                )
                
                train_time = time.time() - start_time
                logger.info(f"Training completed in {train_time:.2f} seconds")
                
                # Score on validation set
                score_start = time.time()
                score = self.score_model(model, X_val)
                score_time = time.time() - score_start
                total_time = time.time() - start_time
                
                logger.info(f"Scoring completed in {score_time:.2f} seconds (total: {total_time:.2f}s)")
                
                # Store results
                result = {
                    'params': params.copy(),
                    'score': score,
                    'iteration': i,
                    'train_time': train_time,
                    'score_time': score_time,
                    'final_epoch': len(history.history['loss'])
                }
                self.cv_results_.append(result)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"\rIteration {i}/{self.n_iter}: New best score = {score:.6f} (time: {total_time:.1f}s)")
                    logger.info(f"  Architecture: {hidden_layers} -> {latent_dim}")
                    logger.info(f"  Batch size: {batch_size}, Learning rate: {learning_rate}")
                else:
                    logger.debug(f"Iteration {i}/{self.n_iter}: Score = {score:.6f}")
                
                # Clean up to prevent memory issues
                del model
                tf.keras.backend.clear_session()
                
            except Exception as e:
                print(f"\rIteration {i}/{self.n_iter}: Failed - {str(e)[:80]}")
                logger.warning(f"Iteration {i} failed with params {params}: {e}")
                logger.debug(f"Full traceback:", exc_info=True)
                # Clean up on error
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except:
                    pass
                continue
        
        # Clear progress line
        print()
        
        # Check if any iteration succeeded
        if best_params is None:
            logger.error("\nOptimization failed! All iterations failed.")
            logger.error("Check the logs above for specific error messages.")
            raise RuntimeError("All optimization iterations failed. No valid parameters found.")
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        logger.info(f"\nOptimization completed!")
        logger.info(f"Best score: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, self.cv_results_
    
    def score_model(self, model, X_val: np.ndarray) -> float:
        """Calculate score for a trained autoencoder"""
        from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score
        
        if self.scoring == 'mse_score':
            # Lower MSE = better, so negate for maximization
            _, mse = model.predict(X_val)
            mean_mse = np.mean(mse)
            # Return negative MSE so lower MSE = higher score
            return -mean_mse
            
        elif self.scoring in ['f1_score', 'f2_score']:
            # Need true labels for supervised metrics
            if self.y_val is None:
                raise ValueError(f"y_val is required for scoring method: {self.scoring}")
            
            # Get reconstruction errors
            _, mse = model.predict(X_val)
            
            # Calculate threshold using percentile method on MSE
            threshold = np.percentile(mse, 95)
            
            # Predict anomalies (1 = anomaly, 0 = normal)
            y_pred = (mse > threshold).astype(int)
            
            if self.scoring == 'f1_score':
                # F1 score: balanced precision and recall
                score = f1_score(self.y_val, y_pred, zero_division=0)
                logger.debug(f"  F1={score:.4f}, P={precision_score(self.y_val, y_pred, zero_division=0):.4f}, R={recall_score(self.y_val, y_pred, zero_division=0):.4f}")
            elif self.scoring == 'f2_score':
                # F2 score: weighs recall higher than precision (2x weight)
                score = fbeta_score(self.y_val, y_pred, beta=2, zero_division=0)
                logger.debug(f"  F2={score:.4f}, P={precision_score(self.y_val, y_pred, zero_division=0):.4f}, R={recall_score(self.y_val, y_pred, zero_division=0):.4f}")
            
            return score
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring}")


def save_optimization_results(results: List[Dict], output_path: str, best_params: Dict = None, best_score: float = None):
    """Save optimization results to JSON file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle empty results
    if not results:
        logger.warning("No optimization results to save (all iterations failed)")
        output_data = {
            'best_parameters': None,
            'all_iterations': [],
            'error': 'All optimization iterations failed'
        }
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        return
    
    # Convert results to serializable format
    serializable_results = []
    for result in results:
        serializable_result = {
            'iteration': result['iteration'],
            'score': float(result['score']),
            'params': {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) 
                      for k, v in result['params'].items()}
        }
        serializable_results.append(serializable_result)
    
    # Find best result if not provided
    if best_params is None or best_score is None:
        best_result = max(results, key=lambda x: x['score'])
        best_params = best_result['params']
        best_score = best_result['score']
    
    # Create output with best parameters at the top
    output_data = {
        'best_parameters': {
            'score': float(best_score),
            'params': {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) 
                      for k, v in best_params.items()}
        },
        'all_iterations': serializable_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Optimization results saved to {output_file}")


def save_optimal_parameters_py(best_params: Dict, best_score: float, dataset_name: str, 
                               time_span: int, model_name: str, output_dir: str, 
                               n_iter: int, scoring: str) -> str:
    """
    Save optimal parameters to a Python file for easy import in config.py
    
    Args:
        best_params: Best parameters found during optimization
        best_score: Best score achieved
        dataset_name: Name of the dataset
        time_span: Time span in seconds
        model_name: Name of the model
        output_dir: Base output directory
        n_iter: Number of iterations used
        scoring: Scoring method used
        
    Returns:
        Path to the saved Python file
    """
    params_dir = Path(output_dir) / "parameters"
    params_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = params_dir / "optimal_params.py"
    
    # Build the Python file content
    lines = []
    lines.append('"""')
    lines.append(f'Optimal parameters for {model_name} - {dataset_name} ({time_span}s window)')
    lines.append('')
    lines.append('Generated from hyperparameter optimization:')
    lines.append(f'- Optimization iterations: {n_iter}')
    lines.append(f'- Scoring method: {scoring}')
    lines.append(f'- Best score: {best_score:.6f}')
    lines.append('"""')
    lines.append('')
    
    # Add the parameter configuration
    lines.append('# Optimal parameters configuration')
    lines.append('OPTIMAL_PARAMS = {')
    
    # Format parameters with proper types and indentation
    for key, value in sorted(best_params.items()):
        if isinstance(value, str):
            lines.append(f"    '{key}': '{value}',")
        elif isinstance(value, bool):
            lines.append(f"    '{key}': {value},")
        elif isinstance(value, (int, float)):
            lines.append(f"    '{key}': {value},")
        else:
            lines.append(f"    '{key}': {repr(value)},")
    
    lines.append('}')
    lines.append('')
    
    # Add metadata
    lines.append('# Optimization metadata')
    lines.append('METADATA = {')
    lines.append(f"    'dataset': '{dataset_name}',")
    lines.append(f"    'time_span': '{time_span}s',")
    lines.append(f"    'model': '{model_name}',")
    lines.append(f"    'n_iterations': {n_iter},")
    lines.append(f"    'scoring_method': '{scoring}',")
    lines.append(f"    'best_score': {best_score:.6f},")
    lines.append('}')
    lines.append('')
    
    # Add usage instructions
    lines.append('# Usage example:')
    lines.append('#')
    lines.append('# In config.py, import and use this configuration:')
    lines.append('#')
    dataset_py_name = dataset_name.replace('-', '_')
    time_label = f"{time_span}seconds"
    lines.append(f"#   from results.{dataset_py_name}.{time_label}.models.{model_name}.parameters.optimal_params import OPTIMAL_PARAMS")
    lines.append('#')
    lines.append(f"#   DATASETS['{dataset_name}']['windows']['{time_span}']['params'] = {{")
    lines.append(f"#       '{model_name}': OPTIMAL_PARAMS")
    lines.append('#   }')
    lines.append('')
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"Optimal parameters saved to {output_path}")
    return str(output_path)


def load_optimization_results(input_path: str) -> List[Dict]:
    """Load optimization results from JSON file"""
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded {len(results)} optimization results from {input_path}")
    return results
