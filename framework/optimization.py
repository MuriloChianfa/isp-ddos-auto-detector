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
            
        param_distributions = self.get_param_distributions()
        
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
                    logger.info(f"Iteration {i}/{self.n_iter}: New best score = {score:.6f}")
                    logger.info(f"  Params: {params}")
                else:
                    logger.debug(f"Iteration {i}/{self.n_iter}: Score = {score:.6f}")
                    
            except Exception as e:
                logger.warning(f"Iteration {i} failed with params {params}: {e}")
                continue
        
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
        logger.info(f"Performance settings: cache_size={self.cache_size}MB, tol={self.tol}, shrinking={self.shrinking}")
        
        best_score = float('-inf')
        best_params = None
        
        for i, params in enumerate(param_list, 1):
            try:
                # Remove irrelevant params based on kernel
                kernel = params['kernel']
                if kernel == 'rbf':
                    params.pop('degree', None)
                    params.pop('coef0', None)
                elif kernel == 'sigmoid':
                    params.pop('degree', None)
                
                # Add performance optimization parameters
                params['cache_size'] = self.cache_size
                params['tol'] = self.tol
                params['max_iter'] = self.max_iter
                params['shrinking'] = self.shrinking
                params['verbose'] = False
                    
                # Train model with current parameters
                model = OneClassSVM(**params)
                model.fit(X_train)
                
                # Score on validation set
                score = self.score_model(model, X_val)
                
                # Store results
                result = {
                    'params': params.copy(),
                    'score': score,
                    'iteration': i
                }
                self.cv_results_.append(result)
                
                # Update best
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"Iteration {i}/{self.n_iter}: New best score = {score:.6f}")
                    logger.info(f"  Params: {params}")
                else:
                    logger.debug(f"Iteration {i}/{self.n_iter}: Score = {score:.6f}")
                    
            except Exception as e:
                logger.warning(f"Iteration {i} failed with params {params}: {e}")
                continue
        
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
            'n_neighbors': [int(x) for x in np.linspace(5, 100, 20)],
            'contamination': [round(x, 3) for x in np.linspace(0.01, 0.30, 20)],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [int(x) for x in np.linspace(10, 100, 10)],
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
        max_train_samples = 50000  # Reasonable size for optimization
        if len(X_train) > max_train_samples:
            train_sample_indices = np.random.choice(len(X_train), max_train_samples, replace=False)
            X_train_sample = X_train[train_sample_indices]
            logger.info(f"Subsampling training data: {len(X_train)} -> {max_train_samples} samples for faster optimization")
            print(f"Note: Using {max_train_samples:,} training samples (of {len(X_train):,}) for faster optimization")
        else:
            X_train_sample = X_train
        
        # For large datasets, sample validation data for faster scoring
        max_val_samples = 10000
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
                
                # Add novelty=True for prediction
                params['novelty'] = True
                params['n_jobs'] = -1
                # Note: LocalOutlierFactor doesn't support random_state parameter, think later on how to fix it
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Iteration {i}/{self.n_iter}")
                logger.info(f"Parameters: {params}")
                print(f"\nOptimization iteration {i}/{self.n_iter}...")
                
                # Train model with current parameters
                logger.info("Training model...")
                print(f"  Training with params: n_neighbors={params.get('n_neighbors')}, contamination={params.get('contamination'):.3f}...")
                model = LocalOutlierFactor(**params)
                model.fit(X_train_sample)
                train_time = time.time() - start_time
                logger.info(f"Training completed in {train_time:.2f} seconds")
                print(f"  Training completed in {train_time:.1f}s")
                
                # Score on validation set
                logger.info("Scoring on validation set...")
                print(f"  Scoring...")
                score_start = time.time()
                score = self.score_model(model, X_val_sample)
                score_time = time.time() - score_start
                total_time = time.time() - start_time
                logger.info(f"Scoring completed in {score_time:.2f} seconds (total: {total_time:.2f}s)")
                print(f"  Score: {score:.6f} (completed in {total_time:.1f}s total)")
                
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
                    logger.info(f"NEW BEST SCORE: {score:.6f}")
                else:
                    logger.info(f"Score: {score:.6f} (best so far: {best_score:.6f})")
                    
            except Exception as e:
                logger.warning(f"Iteration {i} failed with params {params}: {e}")
                continue
        
        # Restore original y_val if it was sampled
        if original_y_val is not None:
            self.y_val = original_y_val
        
        self.best_params_ = best_params
        self.best_score_ = best_score
        
        logger.info(f"\nOptimization completed!")
        logger.info(f"Best score: {best_score:.6f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, self.cv_results_


def save_optimization_results(results: List[Dict], output_path: str, best_params: Dict = None, best_score: float = None):
    """Save optimization results to JSON file"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
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


def load_optimization_results(input_path: str) -> List[Dict]:
    """Load optimization results from JSON file"""
    with open(input_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"Loaded {len(results)} optimization results from {input_path}")
    return results
