"""
Run versioning and management module.
Handles versioned runs for result preservation and comparison.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime


class RunVersionManager:
    """Manages versioned runs for result preservation and comparison"""
    
    RUNS_INDEX_PATH = "./results/runs_index.json"
    VERSIONS_BASE_PATH = "./results/versions"
    
    def __init__(self):
        """Initialize the RunVersionManager"""
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Ensure the runs index file exists"""
        os.makedirs(os.path.dirname(self.RUNS_INDEX_PATH), exist_ok=True)
        if not os.path.exists(self.RUNS_INDEX_PATH):
            self._save_index({"runs": []})
    
    def _load_index(self) -> Dict:
        """Load the runs index"""
        with open(self.RUNS_INDEX_PATH, 'r') as f:
            return json.load(f)
    
    def _save_index(self, index_data: Dict):
        """Save the runs index"""
        with open(self.RUNS_INDEX_PATH, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def generate_run_id(self) -> str:
        """Generate a unique run ID based on timestamp"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def sanitize_run_name(self, run_name: str) -> str:
        """
        Sanitize run name for use as directory name
        
        Args:
            run_name: Human-readable run name
            
        Returns:
            Sanitized name safe for filesystem
        """
        import re
        # Replace spaces with underscores and remove unsafe characters
        sanitized = re.sub(r'[^\w\-_\.]', '_', run_name)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized or 'unnamed_run'
    
    def get_unique_run_name(self, base_name: str) -> str:
        """
        Get a unique run name by appending counter if needed
        
        Args:
            base_name: Base name for the run
            
        Returns:
            Unique run name (with counter if needed)
        """
        sanitized = self.sanitize_run_name(base_name)
        candidate = sanitized
        counter = 1
        
        while os.path.exists(os.path.join(self.VERSIONS_BASE_PATH, candidate)):
            candidate = f"{sanitized}_{counter}"
            counter += 1
        
        return candidate
    
    def create_version_directory(self, run_name: str) -> str:
        """
        Create a versioned directory structure
        
        Args:
            run_name: Run name (will be used as directory name)
            
        Returns:
            Path to the version directory
        """
        version_path = os.path.join(self.VERSIONS_BASE_PATH, run_name)
        os.makedirs(version_path, exist_ok=True)
        return version_path
    
    def register_run(self, run_name: str, metadata: Dict):
        """
        Register a new run in the index
        
        Args:
            run_name: Directory name for the run (sanitized)
            metadata: Dictionary containing run metadata
        """
        index = self._load_index()
        
        run_entry = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        index["runs"].append(run_entry)
        self._save_index(index)
    
    def list_runs(self) -> List[Dict]:
        """
        List all registered runs
        
        Returns:
            List of run entries
        """
        index = self._load_index()
        return index.get("runs", [])
    
    def get_run(self, run_name: str) -> Optional[Dict]:
        """
        Get a specific run by name
        
        Args:
            run_name: Run name (directory name)
            
        Returns:
            Run entry or None if not found
        """
        runs = self.list_runs()
        for run in runs:
            if run["run_name"] == run_name:
                return run
        return None
    
    def get_run_path(self, run_name: str) -> str:
        """
        Get the path to a versioned run
        
        Args:
            run_name: Run name (directory name)
            
        Returns:
            Path to the run directory
        """
        return os.path.join(self.VERSIONS_BASE_PATH, run_name)
    
    def copy_results_to_version(self, run_name: str, source_results_dir: str):
        """
        Copy results from standard location to versioned directory
        
        Args:
            run_name: Run name (directory name)
            source_results_dir: Source directory containing results
        """
        import shutil
        
        version_path = self.get_run_path(run_name)
        
        # Copy the entire results structure
        if os.path.exists(source_results_dir):
            # Extract the relative path from results/
            rel_path = os.path.relpath(source_results_dir, "./results")
            dest_path = os.path.join(version_path, rel_path)
            
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            
            shutil.copytree(source_results_dir, dest_path)
            
            return dest_path
        return None
