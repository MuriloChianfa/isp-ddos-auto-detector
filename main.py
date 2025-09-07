import numpy as np
import pandas as pd
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from framework.loader import NetworkDataLoader
from framework.features import NetworkFeatureExtractor
from framework.models import create_model, list_available_models, get_model_descriptions
from framework.visualization.training_plots import TrainingVisualizer
from framework.visualization.anomaly_plots import AnomalyVisualizer
from framework.evaluation import GroundTruthEvaluator
from config import DATASETS, DEFAULT_DATASET
# from framework.visualization.dataset_feature_plots import DatasetFeatureVisualizer


def main(dataset_name=None, use_cache=True, time_span=300, force_regenerate=False, 
         max_processes=None, model_name='autoencoder'):
    # Select dataset configuration
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET
        
    if dataset_name not in DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found in configuration.")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return
    
    # Validate model selection
    available_models = list_available_models()
    if model_name not in available_models:
        print(f"Error: Model '{model_name}' not found.")
        print(f"Available models: {', '.join(available_models)}")
        return
    
    dataset_config = DATASETS[dataset_name]
    feature_config = dataset_config.get('feature_config', {})
    
    print(f"Using dataset: {dataset_name}")
    print(f"Description: {dataset_config['description']}")
    print(f"Path: {dataset_config['path']}")
    print(f"Model: {model_name}")
    print(f"Time span: {time_span} seconds ({'1-minute' if time_span == 60 else '5-minute'} windows)")
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
    
    print("Initializing data loader...")
    loader = NetworkDataLoader(dataset_config=dataset_config, use_cache=use_cache, max_processes=max_processes)
    
    print("Initializing feature extractor...")
    feature_extractor = NetworkFeatureExtractor(
        time_span=time_span, 
        use_cache=use_cache, 
        dataset_name=dataset_name, 
        max_processes=max_processes,
        feature_config=feature_config
    )
    
    print("\nExtracting features using memory-efficient approach...")
    features_dict = feature_extractor.extract_features_to_csv(loader, force_regenerate=force_regenerate)
    
    if not features_dict:
        print("Error: No features were extracted. Please check the data.")
        return
    
    processed_features = feature_extractor.prepare_training_data(features_dict)
    
    print(f"\nInitializing {model_name} model...")
    
    # Create model with appropriate parameters
    if model_name == 'autoencoder':
        model = create_model(model_name, latent_dim=42)
    elif model_name == 'isolation_forest':
        model = create_model(model_name, contamination=0.1, n_estimators=100)
    elif model_name == 'one_class_svm':
        model = create_model(model_name, nu=0.1, kernel='rbf')
    else:
        model = create_model(model_name)
    
    train_features = processed_features['train']['features']
    val_features = processed_features['validation']['features']
    test_features = processed_features['test']['features']
    
    model.fit_scaler(train_features)
    
    scaled_train_data = model.transform_data(train_features)
    scaled_validation_data = model.transform_data(val_features)
    scaled_test_data = model.transform_data(test_features)
    
    input_dim = scaled_train_data.shape[1]
    print(f"\nNumber of input features: {input_dim}")
    model.build_model(input_dim)
    
    print("\nTraining model...")
    history = model.train(scaled_train_data, scaled_validation_data)
    
    print("\nVisualizing training results...")
    training_viz = TrainingVisualizer(dataset_name=dataset_name)
    training_viz.plot_training_history(history)
    training_viz.print_training_summary(history)
    
    print("\nDetecting anomalies...")
    train_reconstructions, train_scores = model.predict(scaled_train_data)
    val_reconstructions, val_scores = model.predict(scaled_validation_data)
    test_reconstructions, test_scores = model.predict(scaled_test_data)
    
    # Feature importance analysis
    feature_names = train_features.columns.tolist()
    print("\nAnalyzing feature importance...")
    feature_errors, importance_indices = model.analyze_feature_importance(scaled_test_data, feature_names)
    
    # Create feature importance visualizations (only for models that support it)
    if model_name == 'autoencoder':
        print("Creating feature importance visualizations...")
        from framework.visualization.evaluation_plots import EvaluationVisualizer
        eval_viz = EvaluationVisualizer(f"./results/{dataset_name}/{model_name}")
        eval_viz.plot_feature_importance(feature_errors, feature_names, importance_indices, top_n=20)
        eval_viz.plot_feature_importance_detailed(feature_errors, feature_names, importance_indices, top_n=15)
    
    threshold, all_thresholds = model.calculate_threshold(train_scores, val_scores, 'mean_plus_3std')
    
    # Try different threshold methods for comparison
    print("\nThreshold method comparison:")
    threshold_methods = ['percentile_95', 'percentile_99', 'percentile_99_5', 'mean_plus_1std', 'mean_plus_2std', 'mean_plus_3std']
    for method in threshold_methods:
        temp_threshold, _ = model.calculate_threshold(train_scores, val_scores, method)
        test_anomalies = test_scores > temp_threshold
        anomaly_rate = np.sum(test_anomalies) / len(test_anomalies) * 100
        print(f"  {method}: {temp_threshold:.6f} -> {anomaly_rate:.2f}% anomalies detected")
    print(f"Selected threshold method: mean_plus_3std = {threshold:.6f}")
    print()
    
    results = {}
    all_features = []
    all_datasets = []
    
    for split_name, scaled_data, scores, features_data in [
        ('train', scaled_train_data, train_scores, processed_features['train']),
        ('validation', scaled_validation_data, val_scores, processed_features['validation']),
        ('test', scaled_test_data, test_scores, processed_features['test'])
    ]:
        anomalies = model.detect_anomalies(scores)
        reconstructions = model.predict(scaled_data)[0]
        metrics = model.get_metrics(scaled_data, reconstructions, scores)
        
        features_df = features_dict[split_name].copy()
        features_df['reconstruction_error'] = scores
        features_df['is_anomaly'] = anomalies
        features_df['dataset'] = split_name
        
        all_features.append(features_df)
        all_datasets.extend([split_name] * len(features_df))
        
        num_anomalies = np.sum(anomalies)
        total_samples = len(anomalies)
        anomaly_percentage = (num_anomalies / total_samples) * 100
        
        print(f"\nDataset: {split_name}")
        print(f"  Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        print(f"  Mean Squared Error (MSE):  {metrics['mse']:.6f}")
        print(f"  Root Mean Squared Error:   {metrics['rmse']:.6f}")
        print(f"  Anomalies detected:        {num_anomalies}/{total_samples} ({anomaly_percentage:.2f}%)")
        print(f"  Threshold used:            {threshold:.6f}")
    
    combined_features = pd.concat(all_features, ignore_index=True)
    
    print("\nVisualizing anomaly detection results...")
    anomaly_viz = AnomalyVisualizer(dataset_name=dataset_name, model_name=model_name)
    anomaly_viz.print_threshold_comparison(test_scores, all_thresholds)
    anomaly_viz.plot_anomaly_detection(combined_features, threshold)
    anomaly_viz.print_anomaly_statistics(combined_features, threshold)
    
    # Ground Truth Evaluation
    print("\nPerforming ground truth evaluation on test dataset...")
    evaluator = GroundTruthEvaluator(dataset_name=dataset_name, model_name=model_name)

    # Extract test dataset
    test_data = combined_features[combined_features['dataset'] == 'test'].copy()

    # Use dataset-specific attack periods
    attack_periods = dataset_config.get('attack_periods', [])
    if attack_periods:
        # Evaluate against ground truth using dataset-specific attack periods
        evaluation_metrics = evaluator.evaluate_test_dataset(test_data, threshold, attack_periods)
    else:
        print("Warning: No attack periods defined for this dataset. Skipping ground truth evaluation.")
    
    print(f"\nAnalysis completed. Results saved to ./results/{dataset_name}/")
    print(f"Features saved to ./datasets/{dataset_name}/features/")
    print(f"Dataset used: {dataset_name} ({dataset_config['description']})")


def list_datasets():
    """Print available datasets and their descriptions"""
    print("Available datasets:")
    print("=" * 50)
    for name, config in DATASETS.items():
        print(f"  {name}:")
        print(f"    Description: {config['description']}")
        print(f"    Path: {config['path']}")
        print(f"    Attack periods: {len(config.get('attack_periods', []))} defined")
        print()


def list_models():
    """Print available models and their descriptions"""
    print("Available models:")
    print("=" * 50)
    model_descriptions = get_model_descriptions()
    for name, description in model_descriptions.items():
        print(f"  {name}:")
        print(f"    Description: {description}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISP DDoS Auto Detector")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help=f'Dataset to use for analysis. Available: {", ".join(DATASETS.keys())}. Default: {DEFAULT_DATASET}'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='autoencoder',
        choices=list_available_models(),
        help=f'Model to use for anomaly detection. Available: {", ".join(list_available_models())}. Default: autoencoder'
    )
    parser.add_argument(
        '--time-span', '-t',
        type=int,
        choices=[60, 300],
        default=300,
        help='Time span for feature aggregation in seconds. Options: 60 or 300. Default: 300'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets and exit'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching system (force reload all data)'
    )
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of feature CSV files even if they exist'
    )
    parser.add_argument(
        '--max-processes',
        type=int,
        default=None,
        help='Maximum number of processes to use for parallel processing (default: 16)'
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        list_datasets()
        exit(0)
        
    if args.list_models:
        list_models()
        exit(0)
    
    use_cache = not args.no_cache
    
    if not use_cache:
        print("Caching disabled - will reload all data from scratch")
    
    main(
        dataset_name=args.dataset, 
        use_cache=use_cache, 
        time_span=args.time_span,
        force_regenerate=args.force_regenerate,
        max_processes=args.max_processes,
        model_name=args.model
    )
