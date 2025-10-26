"""
Utility functions for the framework
"""
from .constants import TIME_SPANS

def get_time_span_description(time_span):
    """Get human-readable description for a time span"""
    return TIME_SPANS.get(time_span, {}).get('description', f'{time_span}-second')

def get_time_span_detailed_description(time_span):
    """Get detailed description for a time span"""
    return TIME_SPANS.get(time_span, {}).get('detailed_description', 'custom resolution')

def get_time_span_frequency(time_span):
    """Get pandas frequency string for a time span"""
    return TIME_SPANS.get(time_span, {}).get('frequency', f'{time_span}s')

def get_time_span_floor(time_span):
    """Get pandas floor string for a time span"""
    return TIME_SPANS.get(time_span, {}).get('pandas_floor', f'{time_span}s')

def get_time_window_label(time_span):
    """Get standardized label for file naming"""
    return f"{time_span}seconds"

def get_results_path(dataset_name=None, model_name="autoencoder", time_span=300, path_type="models"):
    """
    Generate consistent results path with timespan included
    
    Args:
        dataset_name (str): Name of the dataset
        model_name (str): Name of the model
        time_span (int): Time span in seconds (10, 60 or 300)
        path_type (str): Type of path ("models", "features", "evaluation", "artifacts", etc.)
    
    Returns:
        str: Formatted results path
    """
    time_label = get_time_window_label(time_span)
    
    if dataset_name:
        if path_type == "models":
            return f"./results/{dataset_name}/{time_label}/models/{model_name}"
        elif path_type == "features":
            return f"./results/{dataset_name}/{time_label}/features"
        elif path_type == "evaluation":
            return f"./results/{dataset_name}/{time_label}/models/{model_name}/evaluation"
        elif path_type == "artifacts":
            return f"./results/{dataset_name}/{time_label}/models/{model_name}/artifacts"
        else:
            return f"./results/{dataset_name}/{time_label}/{path_type}"
    else:
        if path_type == "models":
            return f"./results/{time_label}/models/{model_name}"
        elif path_type == "features":
            return f"./results/{time_label}/features"
        elif path_type == "evaluation":
            return f"./results/{time_label}/models/{model_name}/evaluation"
        elif path_type == "artifacts":
            return f"./results/{time_label}/models/{model_name}/artifacts"
        else:
            return f"./results/{time_label}/{path_type}"

def get_artifacts_path(dataset_name, model_name, time_span):
    """
    Generate artifacts directory path for saving/loading trained models
    
    Args:
        dataset_name (str): Name of the dataset
        model_name (str): Name of the model
        time_span (int): Time span in seconds (10, 60 or 300)
    
    Returns:
        str: Artifacts directory path
    """
    return get_results_path(dataset_name, model_name, time_span, "artifacts")


# Configuration helper functions
def get_attack_periods(dataset_name: str, time_span: int):
    """
    Get attack periods for a specific dataset and time span.
    
    Args:
        dataset_name: Name of the dataset
        time_span: Time span in seconds
    
    Returns:
        List of attack periods tuples or empty list if not found
    """
    from config import DATASETS
    dataset = DATASETS.get(dataset_name, {})
    windows = dataset.get('windows', {})
    window_config = windows.get(str(time_span), {})
    return window_config.get('attack_periods', [])


def get_threshold_strategies(dataset_name: str, time_span: int):
    """
    Get threshold strategies for a specific dataset and time span.
    
    Args:
        dataset_name: Name of the dataset
        time_span: Time span in seconds
    
    Returns:
        Dictionary of model -> threshold strategy mappings
    """
    from config import DATASETS
    dataset = DATASETS.get(dataset_name, {})
    windows = dataset.get('windows', {})
    window_config = windows.get(str(time_span), {})
    return window_config.get('threshold_strategies', {})


def get_model_threshold_strategy(dataset_name: str, time_span: int, model_name: str):
    """
    Get specific threshold strategy for a model, dataset, and time span.
    
    Args:
        dataset_name: Name of the dataset
        time_span: Time span in seconds
        model_name: Name of the model
    
    Returns:
        Threshold strategy string or default strategy if not found
    """
    from config import MODEL_THRESHOLD_STRATEGIES
    strategies = get_threshold_strategies(dataset_name, time_span)
    return strategies.get(model_name, MODEL_THRESHOLD_STRATEGIES.get(model_name, 'exponential_threshold'))
