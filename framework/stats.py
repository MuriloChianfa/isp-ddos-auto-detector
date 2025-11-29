"""
Evaluation cache manager for tracking completed evaluations and avoiding redundant work.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class EvaluationCache:
    """Manages cache of completed evaluations to avoid redundant work."""
    
    def __init__(self, cache_dir: str = './cache'):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache metadata
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'evaluation_cache.json'
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def _generate_key(self, dataset: str, model: str, time_span: int, 
                     params: Optional[Dict] = None, 
                     feature_config: Optional[list] = None,
                     threshold_strategy: Optional[str] = None) -> str:
        """Generate a unique cache key for an evaluation.
        
        Args:
            dataset: Dataset name
            model: Model name
            time_span: Time span in seconds
            params: Model parameters
            feature_config: Feature configuration
            threshold_strategy: Threshold strategy name
            
        Returns:
            Unique cache key
        """
        # Create a dict with all parameters that affect the evaluation
        key_data = {
            'dataset': dataset,
            'model': model,
            'time_span': time_span,
            'params': params or {},
            'features': sorted(feature_config) if feature_config else [],
            'threshold': threshold_strategy or ''
        }
        
        # Create a hash of the parameters
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"{dataset}_{model}_{time_span}_{key_hash}"
    
    def is_cached(self, dataset: str, model: str, time_span: int,
                  params: Optional[Dict] = None,
                  feature_config: Optional[list] = None,
                  threshold_strategy: Optional[str] = None) -> bool:
        """Check if an evaluation is already cached.
        
        Args:
            dataset: Dataset name
            model: Model name
            time_span: Time span in seconds
            params: Model parameters
            feature_config: Feature configuration
            threshold_strategy: Threshold strategy name
            
        Returns:
            True if evaluation is cached and results exist
        """
        key = self._generate_key(dataset, model, time_span, params, 
                                feature_config, threshold_strategy)
        
        if key not in self.cache:
            return False
        
        # Check if the results file still exists
        cache_entry = self.cache[key]
        results_path = cache_entry.get('results_path')
        
        if not results_path or not os.path.exists(results_path):
            # Remove stale cache entry
            del self.cache[key]
            self._save_cache()
            return False
        
        return True
    
    def get_cached_info(self, dataset: str, model: str, time_span: int,
                       params: Optional[Dict] = None,
                       feature_config: Optional[list] = None,
                       threshold_strategy: Optional[str] = None) -> Optional[Dict]:
        """Get cached evaluation information.
        
        Returns:
            Cache entry dict or None if not cached
        """
        key = self._generate_key(dataset, model, time_span, params,
                                feature_config, threshold_strategy)
        return self.cache.get(key)
    
    def add_to_cache(self, dataset: str, model: str, time_span: int,
                    results_path: str,
                    params: Optional[Dict] = None,
                    feature_config: Optional[list] = None,
                    threshold_strategy: Optional[str] = None,
                    metrics: Optional[Dict] = None):
        """Add an evaluation to the cache.
        
        Args:
            dataset: Dataset name
            model: Model name
            time_span: Time span in seconds
            results_path: Path to results file
            params: Model parameters
            feature_config: Feature configuration
            threshold_strategy: Threshold strategy name
            metrics: Evaluation metrics
        """
        key = self._generate_key(dataset, model, time_span, params,
                                feature_config, threshold_strategy)
        
        self.cache[key] = {
            'dataset': dataset,
            'model': model,
            'time_span': time_span,
            'results_path': results_path,
            'cached_at': datetime.now().isoformat(),
            'params': params,
            'feature_config': feature_config,
            'threshold_strategy': threshold_strategy,
            'metrics': metrics
        }
        
        self._save_cache()
    
    def clear_cache(self):
        """Clear all cache entries."""
        self.cache = {}
        self._save_cache()
        print("Evaluation cache cleared")
    
    def remove_entry(self, dataset: str, model: str, time_span: int,
                    params: Optional[Dict] = None,
                    feature_config: Optional[list] = None,
                    threshold_strategy: Optional[str] = None):
        """Remove a specific cache entry."""
        key = self._generate_key(dataset, model, time_span, params,
                                feature_config, threshold_strategy)
        if key in self.cache:
            del self.cache[key]
            self._save_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        return {
            'total_entries': len(self.cache),
            'entries': list(self.cache.values())
        }
