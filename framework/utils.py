"""
Utility functions for the framework
"""

def get_results_path(dataset_name=None, model_name="autoencoder", time_span=300, path_type="models"):
    """
    Generate consistent results path with timespan included
    
    Args:
        dataset_name (str): Name of the dataset
        model_name (str): Name of the model
        time_span (int): Time span in seconds (60 or 300)
        path_type (str): Type of path ("models", "features", "evaluation", etc.)
    
    Returns:
        str: Formatted results path
    """
    if dataset_name:
        if path_type == "models":
            return f"./results/{dataset_name}/{time_span}seconds/models/{model_name}"
        elif path_type == "features":
            return f"./results/{dataset_name}/{time_span}seconds/features"
        elif path_type == "evaluation":
            return f"./results/{dataset_name}/{time_span}seconds/models/{model_name}/evaluation"
        else:
            return f"./results/{dataset_name}/{time_span}seconds/{path_type}"
    else:
        if path_type == "models":
            return f"./results/{time_span}seconds/models/{model_name}"
        elif path_type == "features":
            return f"./results/{time_span}seconds/features"
        elif path_type == "evaluation":
            return f"./results/{time_span}seconds/models/{model_name}/evaluation"
        else:
            return f"./results/{time_span}seconds/{path_type}"
