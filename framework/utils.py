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

def get_sequence_multiplier(time_span):
    """Get sequence length multiplier for temporal models based on time span"""
    return TIME_SPANS.get(time_span, {}).get('sequence_multiplier', 1.0)

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
