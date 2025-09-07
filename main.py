import numpy as np
import pandas as pd
import os
import argparse

from framework.loader import NetworkDataLoader
from framework.features import NetworkFeatureExtractor
from framework.models.autoencoder import AutoencoderAnomalyDetector
from framework.visualization.training_plots import TrainingVisualizer
from framework.visualization.anomaly_plots import AnomalyVisualizer
from framework.evaluation import GroundTruthEvaluator
from config import DATASETS, DEFAULT_DATASET
# from framework.visualization.dataset_feature_plots import DatasetFeatureVisualizer


def main(dataset_name=None, use_cache=True, time_span=300, force_regenerate=False, max_processes=None):
    # Select dataset configuration
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET
        
    if dataset_name not in DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found in configuration.")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return
    
    dataset_config = DATASETS[dataset_name]
    print(f"Using dataset: {dataset_name}")
    print(f"Description: {dataset_config['description']}")
    print(f"Path: {dataset_config['path']}")
    print(f"Time span: {time_span} seconds ({'1-minute' if time_span == 60 else '5-minute'} windows)")
    print("Using memory-efficient processing by default")
    
    print("Initializing data loader...")
    loader = NetworkDataLoader(dataset_config=dataset_config, use_cache=use_cache, max_processes=max_processes)
    
    print("Initializing feature extractor...")
    feature_extractor = NetworkFeatureExtractor(time_span=time_span, use_cache=use_cache, dataset_name=dataset_name, max_processes=max_processes)
    
    print("\nExtracting features using memory-efficient approach...")
    features_dict = feature_extractor.extract_features_to_csv(loader, force_regenerate=force_regenerate)
    
    if not features_dict:
        print("Error: No features were extracted. Please check the data.")
        return
    
    processed_features = feature_extractor.prepare_training_data(features_dict)
    
    # print("\nGenerating dataset feature visualizations...")
    # feature_viz = DatasetFeatureVisualizer()
    # feature_viz.generate_all_feature_plots(features_dict, create_comparisons=True)
    
    print("\nInitializing autoencoder model...")
    model = AutoencoderAnomalyDetector(latent_dim=8)
    
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
    train_reconstructions, train_mse = model.predict(scaled_train_data)
    val_reconstructions, val_mse = model.predict(scaled_validation_data)
    test_reconstructions, test_mse = model.predict(scaled_test_data)
    
    threshold, all_thresholds = model.calculate_threshold(train_mse, val_mse, 'mean_plus_3std')
    
    results = {}
    all_features = []
    all_datasets = []
    
    for split_name, scaled_data, mse, features_data in [
        ('train', scaled_train_data, train_mse, processed_features['train']),
        ('validation', scaled_validation_data, val_mse, processed_features['validation']),
        ('test', scaled_test_data, test_mse, processed_features['test'])
    ]:
        anomalies = model.detect_anomalies(mse)
        metrics = model.get_metrics(scaled_data, 
                                  model.predict(scaled_data)[0], mse)
        
        features_df = features_dict[split_name].copy()
        features_df['reconstruction_error'] = mse
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
    anomaly_viz = AnomalyVisualizer(dataset_name=dataset_name)
    anomaly_viz.print_threshold_comparison(test_mse, all_thresholds)
    anomaly_viz.plot_anomaly_detection(combined_features, threshold)
    anomaly_viz.print_anomaly_statistics(combined_features, threshold)
    
    # Ground Truth Evaluation
    print("\nPerforming ground truth evaluation on test dataset...")
    evaluator = GroundTruthEvaluator(dataset_name=dataset_name)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISP DDoS Auto Detector")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help=f'Dataset to use for analysis. Available: {", ".join(DATASETS.keys())}. Default: {DEFAULT_DATASET}'
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
        help='Maximum number of processes to use for parallel processing (default: min(cpu_count(), files, 8))'
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        list_datasets()
        exit(0)
    
    use_cache = not args.no_cache
    
    if not use_cache:
        print("Caching disabled - will reload all data from scratch")
    
    main(
        dataset_name=args.dataset, 
        use_cache=use_cache, 
        time_span=args.time_span,
        force_regenerate=args.force_regenerate,
        max_processes=args.max_processes
    )
