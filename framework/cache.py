import os
import pickle
import hashlib
import pandas as pd
from typing import Dict, Any, Optional
import glob


class DataCache:
    """Cache system for network data loading and feature extraction"""
    
    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _get_cache_key(self, identifier: str, params: Dict[str, Any] = None) -> str:
        """Generate a cache key based on identifier and parameters"""
        if params:
            # Sort params for consistent hashing
            params_str = str(sorted(params.items()))
            hash_input = f"{identifier}_{params_str}"
        else:
            hash_input = identifier
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get the full path for a cache file"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _get_file_modification_times(self, file_pattern: str) -> Dict[str, float]:
        """Get modification times for files matching pattern"""
        files = glob.glob(file_pattern)
        return {f: os.path.getmtime(f) for f in files}
    
    def _is_cache_valid(self, cache_path: str, source_files_pattern: str) -> bool:
        """Check if cache is still valid (newer than source files)"""
        if not os.path.exists(cache_path):
            return False
            
        cache_mtime = os.path.getmtime(cache_path)
        source_files = glob.glob(source_files_pattern)
        
        # If no source files, cache is invalid
        if not source_files:
            return False
            
        # Check if any source file is newer than cache
        for source_file in source_files:
            if os.path.getmtime(source_file) > cache_mtime:
                return False
                
        return True
    
    def get_datasets_cache(self, data_path: str) -> Optional[Dict[str, pd.DataFrame]]:
        """Get cached datasets if valid"""
        cache_key = self._get_cache_key("datasets", {"data_path": data_path})
        cache_path = self._get_cache_path(cache_key)
        
        # Check if cache exists and is valid
        source_pattern = os.path.join(data_path, "*.csv")
        if self._is_cache_valid(cache_path, source_pattern):
            print(f"Loading cached datasets from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading cache: {e}")
                return None
        
        return None
    
    def save_datasets_cache(self, datasets: Dict[str, pd.DataFrame], data_path: str) -> None:
        """Save datasets to cache"""
        cache_key = self._get_cache_key("datasets", {"data_path": data_path})
        cache_path = self._get_cache_path(cache_key)
        
        print(f"Saving datasets to cache: {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(datasets, f)
        except Exception as e:
            print(f"Error saving datasets cache: {e}")
    
    def get_features_cache(self, datasets_hash: str, time_span: int) -> Optional[Dict[str, pd.DataFrame]]:
        """Get cached features if valid"""
        cache_key = self._get_cache_key("features", {
            "datasets_hash": datasets_hash, 
            "time_span": time_span
        })
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading features cache: {e}")
                return None
        
        return None
    
    def save_features_cache(self, features_dict: Dict[str, pd.DataFrame], 
                           datasets_hash: str, time_span: int) -> None:
        """Save features to cache"""
        cache_key = self._get_cache_key("features", {
            "datasets_hash": datasets_hash,
            "time_span": time_span
        })
        cache_path = self._get_cache_path(cache_key)
        
        print(f"Saving features to cache: {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features_dict, f)
        except Exception as e:
            print(f"Error saving features cache: {e}")
    
    def get_processed_features_cache(self, features_hash: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get cached processed features if valid"""
        cache_key = self._get_cache_key("processed_features", {"features_hash": features_hash})
        cache_path = self._get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            print(f"Loading cached processed features from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading processed features cache: {e}")
                return None
        
        return None
    
    def save_processed_features_cache(self, processed_features: Dict[str, Dict[str, Any]], 
                                    features_hash: str) -> None:
        """Save processed features to cache"""
        cache_key = self._get_cache_key("processed_features", {"features_hash": features_hash})
        cache_path = self._get_cache_path(cache_key)
        
        print(f"Saving processed features to cache: {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(processed_features, f)
        except Exception as e:
            print(f"Error saving processed features cache: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cache files"""
        cache_files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                print(f"Removed cache file: {cache_file}")
            except Exception as e:
                print(f"Error removing cache file {cache_file}: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached files"""
        cache_files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        info = {
            "cache_dir": self.cache_dir,
            "num_files": len(cache_files),
            "files": []
        }
        
        for cache_file in cache_files:
            file_info = {
                "name": os.path.basename(cache_file),
                "size_mb": os.path.getsize(cache_file) / (1024 * 1024),
                "modified": os.path.getmtime(cache_file)
            }
            info["files"].append(file_info)
        
        return info
    
    @staticmethod
    def hash_dataframes(datasets: Dict[str, pd.DataFrame]) -> str:
        """Create a hash of the datasets for cache validation"""
        hash_input = ""
        for split_name in sorted(datasets.keys()):
            df = datasets[split_name]
            # Use shape and column names for hash
            hash_input += f"{split_name}_{df.shape}_{list(df.columns)}"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
