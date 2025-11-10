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
    
    def __init__(self, n_iter=50, random_state=42, scoring='anomaly_score'):
        """
        Args:
            n_iter: Number of parameter settings sampled
            random_state: Random state for reproducibility
            scoring: Scoring method (default: 'anomaly_score')
        """
        self.n_iter = n_iter
        self.random_state = random_state
        self.scoring = scoring
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
        if self.scoring == 'anomaly_score':
            # Lower (more negative) mean score = better separation
            scores = model.score_samples(X_val)
            return -np.mean(scores)
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
    
    def __init__(self, n_iter=50, random_state=42, scoring='anomaly_score', n_jobs=-1, 
                 cache_size=4096, tol=1e-2, max_iter=-1, shrinking=True):
        """
        Initialize One-Class SVM optimizer with performance parameters
        
        Args:
            n_iter: Number of parameter settings sampled
            random_state: Random state for reproducibility
            scoring: Scoring method (default: 'anomaly_score')
            n_jobs: Number of parallel jobs (-1 uses all cores)
            cache_size: Size of kernel cache in MB (larger = faster but more memory)
            tol: Tolerance for stopping criterion (larger = faster but less accurate)
            max_iter: Max iterations for solver (-1 = no limit)
            shrinking: Whether to use shrinking heuristic (can speed up training)
        """
        super().__init__(n_iter, random_state, scoring, n_jobs)
        self.cache_size = cache_size
        self.tol = tol
        self.max_iter = max_iter
        self.shrinking = shrinking
    
    def get_param_distributions(self) -> Dict:
        """Define parameter distributions for random search"""
        return {
            'nu': [round(x, 3) for x in np.linspace(0.01, 0.30, 20)],
            'kernel': ['rbf', 'sigmoid', 'poly'],
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
        
        best_score = float('-inf')
        best_params = None
        
        for i, params in enumerate(param_list, 1):
            try:
                # Add novelty=True for prediction
                params['novelty'] = True
                params['random_state'] = self.random_state
                
                # Train model with current parameters
                model = LocalOutlierFactor(**params)
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
