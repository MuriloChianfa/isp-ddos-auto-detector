"""
Settings validation and configuration management for the DDoS detection pipeline.
Handles validation of model parameters, dataset configuration, and feature settings.
"""

import os
from typing import Dict, List, Optional
from config import DATASETS, DEFAULT_DATASET
from framework.detector import MODEL_THRESHOLD_STRATEGIES

from typing import Dict, Optional, List
from config import DATASETS, DEFAULT_DATASET, MODEL_THRESHOLD_STRATEGIES
from framework.models import list_available_models, get_model_descriptions
from framework.utils import get_time_span_description, get_time_span_detailed_description
from framework.constants import SUPPORTED_TIME_SPANS


class SettingsManager:
    """Handles configuration validation and setup"""
    
    @staticmethod
    def validate_dataset(dataset_name: Optional[str] = None) -> Optional[Dict]:
        """
        Validate dataset configuration
        
        Args:
            dataset_name: Name of the dataset to validate
            
        Returns:
            Dataset configuration dict if valid, None if invalid
        """
        if dataset_name is None:
            dataset_name = DEFAULT_DATASET
            
        if dataset_name not in DATASETS:
            print(f"Error: Dataset '{dataset_name}' not found in configuration.")
            print(f"Available datasets: {', '.join(DATASETS.keys())}")
            return None
            
        return DATASETS[dataset_name]
    
    @staticmethod
    def validate_model(model_name: str) -> bool:
        """
        Validate model selection
        
        Args:
            model_name: Name of the model to validate
            
        Returns:
            True if model is valid, False otherwise
        """
        available_models = list_available_models()
        if model_name not in available_models:
            print(f"Error: Model '{model_name}' not found.")
            print(f"Available models: {', '.join(available_models)}")
            return False
        return True
    
    @staticmethod
    def validate_time_span(time_span: int) -> bool:
        """
        Validate time span parameter
        
        Args:
            time_span: Time span in seconds
            
        Returns:
            True if time span is valid, False otherwise
        """
        if time_span not in SUPPORTED_TIME_SPANS:
            print(f"Error: Time span '{time_span}' not supported.")
            print(f"Valid time spans: {SUPPORTED_TIME_SPANS}")
            return False
        return True
    
    @staticmethod
    def print_configuration_summary(dataset_name: str, model_name: str, time_span: int, 
                                   use_cache: bool = True, max_processes: Optional[int] = None):
        """
        Print comprehensive configuration summary
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_span: Time span in seconds
            use_cache: Whether caching is enabled
            max_processes: Maximum number of processes for parallel processing
        """
        dataset_config = DATASETS[dataset_name]
        feature_config = dataset_config.get('feature_config', {})
        default_threshold_strategy = MODEL_THRESHOLD_STRATEGIES.get(model_name, 'sigmoid_threshold')
        
        print(f"Using dataset: {dataset_name}")
        print(f"Description: {dataset_config['description']}")
        print(f"Path: {dataset_config['path']}")
        print(f"Model: {model_name}")
        print(f"Default threshold strategy: {default_threshold_strategy}")
        print(f"Time span: {time_span} seconds ({get_time_span_description(time_span)} - {get_time_span_detailed_description(time_span)} windows)")
        print("Using memory-efficient processing by default")
        
        # Display feature configuration info
        if feature_config:
            print(f"\nFeature Configuration:")
            include_groups = feature_config.get('include_groups', [])
            if include_groups:
                print(f"  Feature groups: {', '.join(include_groups)}")
            exclude_features = feature_config.get('exclude_features', [])
            if exclude_features:
                print(f"  Excluded features: {', '.join(exclude_features)}")
            custom_features = feature_config.get('custom_features', {})
            if custom_features:
                enabled_custom = [k for k, v in custom_features.items() if v]
                if enabled_custom:
                    print(f"  Custom features: {', '.join(enabled_custom)}")
    
    @staticmethod
    def list_available_datasets():
        """Print available datasets and their descriptions"""
        print("Available datasets:")
        print("=" * 50)
        for name, config in DATASETS.items():
            print(f"  {name}:")
            print(f"    Description: {config['description']}")
            print(f"    Path: {config['path']}")
            print(f"    Attack periods: {len(config.get('attack_periods', []))} defined")
            print()
    
    @staticmethod
    def list_available_models():
        """Print available models and their descriptions"""
        print("Available models:")
        print("=" * 50)
        model_descriptions = get_model_descriptions()
        for name, description in model_descriptions.items():
            print(f"  {name}:")
            print(f"    Description: {description}")
            print()
    
    @staticmethod
    def get_default_dataset() -> str:
        """Get the default dataset name"""
        return DEFAULT_DATASET
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available model names"""
        return list_available_models()
    
    @staticmethod
    def get_dataset_config(dataset_name: str) -> Dict:
        """Get dataset configuration by name"""
        return DATASETS.get(dataset_name, {})
    
    @staticmethod
    def get_model_threshold_strategy(model_name: str) -> str:
        """Get default threshold strategy for a model"""
        return MODEL_THRESHOLD_STRATEGIES.get(model_name, 'sigmoid_threshold')
    
    @staticmethod
    def validate_all_parameters(dataset_name: Optional[str], model_name: str, 
                               time_span: int) -> tuple[bool, Optional[Dict]]:
        """
        Validate all key parameters at once
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model 
            time_span: Time span in seconds
            
        Returns:
            Tuple of (is_valid, dataset_config)
        """
        # Validate dataset
        dataset_config = SettingsManager.validate_dataset(dataset_name)
        if dataset_config is None:
            return False, None
            
        # Validate model
        if not SettingsManager.validate_model(model_name):
            return False, None
            
        # Validate time span
        if not SettingsManager.validate_time_span(time_span):
            return False, None
            
        return True, dataset_config