"""
Centralized loader for optimal feature configurations and model parameters.

This module dynamically loads FEATURE_CONFIG and OPTIMAL_PARAMS from generated files
without requiring __init__.py files in the results directory structure.
"""

import os
import importlib.util
from typing import List, Optional, Dict


def load_optimal_features(dataset: str, timespan: str) -> Optional[List[str]]:
    """
    Load optimal feature configuration for a specific dataset and timespan.
    
    Args:
        dataset: Dataset name (e.g., 'itp-downstream-http-flood')
        timespan: Timespan in seconds (e.g., '1', '10', '60', '300')
        
    Returns:
        List of feature names, or None if file doesn't exist
    """
    # Convert timespan to directory format (e.g., '1' -> '1seconds')
    timespan_dir = f"{timespan}seconds"
    
    # Build path to optimal_features.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_file = os.path.join(
        base_dir, 'results', dataset, timespan_dir, 
        'features', 'correlation', 'optimal_features.py'
    )
    
    if not os.path.exists(features_file):
        return None
    
    # Load module dynamically
    spec = importlib.util.spec_from_file_location(
        f"optimal_features_{dataset}_{timespan}", 
        features_file
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, 'FEATURE_CONFIG', None)
    
    return None


def load_optimal_params(dataset: str, timespan: str, model: str) -> Optional[Dict]:
    """
    Load optimal model parameters for a specific dataset, timespan, and model.
    
    Args:
        dataset: Dataset name (e.g., 'itp-downstream-http-flood')
        timespan: Timespan in seconds (e.g., '1', '10', '60', '300')
        model: Model name (e.g., 'one_class_svm', 'isolation_forest')
        
    Returns:
        Dictionary of optimal parameters, or None if file doesn't exist
    """
    # Convert timespan to directory format (e.g., '1' -> '1seconds')
    timespan_dir = f"{timespan}seconds"
    
    # Build path to optimal_params.py
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params_file = os.path.join(
        base_dir, 'results', dataset, timespan_dir, 
        'models', model, 'optimization', 'parameters', 'optimal_params.py'
    )
    
    if not os.path.exists(params_file):
        return None
    
    # Load module dynamically
    spec = importlib.util.spec_from_file_location(
        f"optimal_params_{dataset}_{timespan}_{model}", 
        params_file
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, 'OPTIMAL_PARAMS', None)
    
    return None


# Pre-load all available optimal feature configurations
# This allows them to be imported like: from framework.optimal_features_loader import HTTP_FLOOD_1S

# itp-downstream-http-flood
HTTP_FLOOD_1S = load_optimal_features('itp-downstream-http-flood', '1')
HTTP_FLOOD_10S = load_optimal_features('itp-downstream-http-flood', '10')
HTTP_FLOOD_60S = load_optimal_features('itp-downstream-http-flood', '60')
HTTP_FLOOD_300S = load_optimal_features('itp-downstream-http-flood', '300')

# itp-multivector-udp-100gbps-peak
MULTIVECTOR_1S = load_optimal_features('itp-multivector-udp-100gbps-peak', '1')
MULTIVECTOR_10S = load_optimal_features('itp-multivector-udp-100gbps-peak', '10')
MULTIVECTOR_60S = load_optimal_features('itp-multivector-udp-100gbps-peak', '60')
MULTIVECTOR_300S = load_optimal_features('itp-multivector-udp-100gbps-peak', '300')

# itp-synack-customer-outage
SYNACK_1S = load_optimal_features('itp-synack-customer-outage', '1')
SYNACK_10S = load_optimal_features('itp-synack-customer-outage', '10')
SYNACK_60S = load_optimal_features('itp-synack-customer-outage', '60')
SYNACK_300S = load_optimal_features('itp-synack-customer-outage', '300')
