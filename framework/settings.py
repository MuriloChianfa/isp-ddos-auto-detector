"""
Settings validation and configuration management for the DDoS detection pipeline.
Handles validation of model parameters, dataset configuration, and feature settings.
"""

import os
from typing import Dict, List, Optional
from config import DATASETS, DEFAULT_DATASET
from typing import Dict, Optional, List
from config import DATASETS, DEFAULT_DATASET
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
    def validate_window_config(dataset_name: str, time_span: int) -> bool:
        """
        Validate that the dataset has configuration for the specified timespan
        
        Args:
            dataset_name: Name of the dataset
            time_span: Time span in seconds
            
        Returns:
            True if window configuration exists, False otherwise
        """
        dataset_config = DATASETS.get(dataset_name, {})
        windows = dataset_config.get('windows', {})
        
        if str(time_span) not in windows:
            print(f"Warning: No window configuration found for dataset '{dataset_name}' with {time_span}s timespan.")
            available_windows = list(windows.keys())
            if available_windows:
                print(f"Available timespan configurations: {available_windows}")
            else:
                print("No window configurations defined for this dataset.")
            return False
        
        # Validate that the window configuration has required fields
        window_config = windows[str(time_span)]
        if 'threshold_strategies' not in window_config:
            print(f"Info: No threshold strategies defined for dataset '{dataset_name}' with {time_span}s timespan. Using default strategies.")
            
        if 'attack_periods' not in window_config:
            print(f"Warning: No attack periods defined for dataset '{dataset_name}' with {time_span}s timespan.")
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
        from framework.utils import get_model_threshold_strategy, get_attack_periods, get_threshold_strategies
        
        dataset_config = DATASETS[dataset_name]
        feature_config = dataset_config.get('feature_config', {})
        default_threshold_strategy = get_model_threshold_strategy(dataset_name, time_span, model_name)
        attack_periods = get_attack_periods(dataset_name, time_span)
        threshold_strategies = get_threshold_strategies(dataset_name, time_span)
        
        print(f"Using dataset: {dataset_name}")
        print(f"Description: {dataset_config['description']}")
        print(f"Path: {dataset_config['path']}")
        print(f"Model: {model_name}")
        print(f"Threshold strategy for {model_name}: {default_threshold_strategy}")
        print(f"Time span: {time_span} seconds ({get_time_span_description(time_span)} - {get_time_span_detailed_description(time_span)} windows)")
        print(f"Attack periods defined: {len(attack_periods)} periods")
        if threshold_strategies:
            print(f"Available threshold strategies: {list(threshold_strategies.keys())}")
        else:
            print("Available threshold strategies: Using default strategies from MODEL_THRESHOLD_STRATEGIES")
        
        # Display feature configuration info
        if feature_config:
            print(f"\nFeature Configuration:")
            include_groups = feature_config.get('include_groups', [])
            if include_groups:
                from framework.constants import FEATURE_GROUPS
                print(f"  Feature groups: {', '.join(include_groups)}")
                print(f"  Expanded features by group:")
                for group in include_groups:
                    if group in FEATURE_GROUPS:
                        features = FEATURE_GROUPS[group]
                        print(f"    {group}: {', '.join(features)}")
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
    def get_model_threshold_strategy(model_name: str, dataset_name: str = None, time_span: int = None) -> str:
        """Get threshold strategy for a model, optionally using dataset and timespan-specific configuration"""
        if dataset_name and time_span:
            from framework.utils import get_model_threshold_strategy as get_dataset_strategy
            return get_dataset_strategy(dataset_name, time_span, model_name)
        else:
            # Default strategy when no specific configuration is available
            return 'exponential_threshold'
    
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
            
        # Validate window configuration
        if not SettingsManager.validate_window_config(dataset_name or DEFAULT_DATASET, time_span):
            print("Note: Using fallback configurations where window-specific settings are missing.")
            
        return True, dataset_config