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
from config import DATASETS, DEFAULT_DATASET, MODEL_THRESHOLD_STRATEGIES
# from framework.visualization.dataset_feature_plots import DatasetFeatureVisualizer


def save_anomalies_to_csv(combined_features, dataset_name, model_name, threshold):
    """Save detected anomalies to a CSV file for each model"""
    results_dir = f"./results/{dataset_name}/models/{model_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter only the detected anomalies
    anomalies_df = combined_features[combined_features['is_anomaly'] == True].copy()
    
    # Remove rows with NaN reconstruction_error (for temporal models)
    anomalies_df = anomalies_df.dropna(subset=['reconstruction_error'])
    
    # Select relevant columns for the anomalies CSV
    output_columns = [
        'timestamp', 'reconstruction_error', 'dataset', 'is_anomaly'
    ]
    
    # Add additional network flow information if available
    additional_columns = []
    available_columns = anomalies_df.columns.tolist()
    
    # Include basic network flow metrics if available
    flow_columns = [
        'total_flows', 'total_packets', 'total_bytes',
        'src_ip_count', 'dst_ip_count', 'src_port_count', 'dst_port_count',
        'tcp_flows', 'udp_flows', 'icmp_flows',
        'avg_packet_size', 'avg_flow_duration', 'max_packet_size'
    ]
    
    for col in flow_columns:
        if col in available_columns:
            additional_columns.append(col)
    
    # Include top source/destination IPs and ports if available
    ip_port_columns = [col for col in available_columns if 
                      col.startswith(('top_src_ip_', 'top_dst_ip_', 'top_src_port_', 'top_dst_port_'))]
    additional_columns.extend(ip_port_columns[:10])  # Limit to top 10 to avoid too many columns
    
    # Include protocol statistics if available
    protocol_columns = [col for col in available_columns if 
                       col.endswith(('_rate', '_ratio', '_percentage')) and 
                       not col.startswith('top_')]
    additional_columns.extend(protocol_columns[:10])  # Limit to avoid too many columns
    
    final_columns = output_columns + additional_columns
    
    # Select only existing columns
    final_columns = [col for col in final_columns if col in available_columns]
    
    anomalies_output = anomalies_df[final_columns].copy()
    
    # Sort by timestamp and reconstruction error (highest errors first)
    anomalies_output = anomalies_output.sort_values(['timestamp', 'reconstruction_error'], ascending=[True, False])
    
    # Add metadata columns
    anomalies_output.insert(0, 'model_name', model_name)
    anomalies_output.insert(1, 'threshold_used', threshold)
    anomalies_output.insert(2, 'anomaly_severity', 
                           pd.cut(anomalies_output['reconstruction_error'], 
                                 bins=[threshold, threshold*2, threshold*5, float('inf')],
                                 labels=['Low', 'Medium', 'High'],
                                 include_lowest=True))
    
    # Save to CSV
    csv_filename = os.path.join(results_dir, "anomalies_detected.csv")
    anomalies_output.to_csv(csv_filename, index=False)
    
    # Print summary statistics
    print(f"Anomalies saved to: {csv_filename}")
    print(f"Total anomalies detected: {len(anomalies_output)}")
    print(f"Anomalies by dataset:")
    for dataset in ['train', 'validation', 'test', 'horizon']:
        count = len(anomalies_output[anomalies_output['dataset'] == dataset])
        if count > 0:  # Only show datasets that have data
            print(f"  {dataset}: {count} anomalies")
    
    if len(anomalies_output) > 0:
        print(f"Severity distribution:")
        severity_counts = anomalies_output['anomaly_severity'].value_counts()
        for severity in ['Low', 'Medium', 'High']:
            count = severity_counts.get(severity, 0)
            percentage = (count / len(anomalies_output)) * 100 if len(anomalies_output) > 0 else 0
            print(f"  {severity}: {count} ({percentage:.1f}%)")
        
        print(f"Top 5 highest anomaly scores:")
        top_anomalies = anomalies_output.nlargest(5, 'reconstruction_error')
        for _, row in top_anomalies.iterrows():
            print(f"  {row['timestamp']}: {row['reconstruction_error']:.6f} ({row['dataset']} set)")
    
    return csv_filename


def main(dataset_name=None, use_cache=True, time_span=300, force_regenerate=False, 
         max_processes=None, model_name='autoencoder', use_fixed_threshold=False):
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
    
    # Get the default threshold strategy for this model
    default_threshold_strategy = MODEL_THRESHOLD_STRATEGIES.get(model_name, 'sigmoid_threshold')
    
    print(f"Using dataset: {dataset_name}")
    print(f"Description: {dataset_config['description']}")
    print(f"Path: {dataset_config['path']}")
    print(f"Model: {model_name}")
    print(f"Default threshold strategy: {default_threshold_strategy}")
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
    
    train_features = processed_features['train']['features']
    val_features = processed_features['validation']['features']
    test_features = processed_features['test']['features']
    
    # Check if horizon split exists (optional)
    horizon_features = None
    if 'horizon' in processed_features:
        horizon_features = processed_features['horizon']['features']
        print(f"Horizon features detected: {len(horizon_features)} samples")

    print(f"\nInitializing {model_name} model...")
    
    # Create model with appropriate parameters
    if model_name == 'autoencoder':
        model = create_model(model_name, latent_dim=42)
    elif model_name == 'lstm_autoencoder':
        # Configure LSTM autoencoder with more conservative parameters for stable learning
        sequence_length = min(80, max(50, len(train_features) // 8))  # More conservative sequence length
        model = create_model(
            model_name, 
            sequence_length=sequence_length,
            latent_dim=48,  # Reduced from 64 for simpler patterns
            encoder_units=[128, 64],  # Simpler architecture  
            decoder_units=[64, 128],  # Simpler architecture
            dropout_rate=0.2  # Higher dropout for better generalization
        )
        print(f"LSTM Autoencoder configured with sequence_length={sequence_length} (conservative mode for stability)")
    elif model_name == 'tcn_autoencoder':
        # Configure feature-focused autoencoder (not true TCN anymore)
        # Use minimal sequence length since we're focusing on features, not temporal patterns
        sequence_length = 5  # Very short - just for compatibility
        model = create_model(
            model_name,
            sequence_length=sequence_length,
            latent_dim=12,   # Smaller latent space for better compression
            num_blocks=2,    # Not used in new architecture
            filters=32,      # Not used in new architecture
            kernel_size=3,   # Not used in new architecture
            dropout_rate=0.3,   # Higher dropout for regularization
            l2_reg=1e-4,        # Moderate regularization
            use_fixed_threshold=use_fixed_threshold  # Pass the threshold parameter
        )
        print(f"Feature-focused autoencoder configured with sequence_length={sequence_length} (feature-based detection)")
        print(f"Threshold mode: {'Fixed' if use_fixed_threshold else 'Adaptive'}")
    elif model_name == 'isolation_forest':
        model = create_model(model_name, contamination=0.1, n_estimators=100)
    elif model_name == 'one_class_svm':
        model = create_model(model_name, nu=0.1, kernel='rbf')
    else:
        model = create_model(model_name)

    model.fit_scaler(train_features)
    
    scaled_train_data = model.transform_data(train_features)
    scaled_validation_data = model.transform_data(val_features)
    scaled_test_data = model.transform_data(test_features)
    
    input_dim = scaled_train_data.shape[1]
    print(f"\nNumber of input features: {input_dim}")
    
    # Build model with feature dimension
    if model_name in ['lstm_autoencoder', 'tcn_autoencoder']:
        model.build_model(feature_dim=input_dim)
        print(f"Model parameters: {model.get_model_summary()['total_parameters']:,}")
    else:
        model.build_model(input_dim)
    
    print("\nTraining model...")
    # Use moderate training duration with very patient early stopping for temporal models
    if model_name in ['lstm_autoencoder', 'tcn_autoencoder']:
        print(f"Training {model_name} with patient training strategy for stable pattern learning")
        history = model.train(scaled_train_data, scaled_validation_data, epochs=80)  # Moderate epochs with patient training
    else:
        history = model.train(scaled_train_data, scaled_validation_data)
    
    print("\nVisualizing training results...")
    training_viz = TrainingVisualizer(dataset_name=dataset_name, model_name=model_name)
    training_viz.plot_training_history(history, model_name=model_name)
    training_viz.print_training_summary(history)
    
    print("\nDetecting anomalies...")
    train_reconstructions, train_scores = model.predict(scaled_train_data)
    val_reconstructions, val_scores = model.predict(scaled_validation_data)
    test_reconstructions, test_scores = model.predict(scaled_test_data)
    
    # Process horizon data if available
    horizon_reconstructions, horizon_scores = None, None
    if horizon_features is not None:
        scaled_horizon_data = model.transform_data(horizon_features)
        horizon_reconstructions, horizon_scores = model.predict(scaled_horizon_data)
        print(f"Horizon anomaly scores computed: {len(horizon_scores)} samples")
    
    # Feature importance analysis
    feature_names = train_features.columns.tolist()
    print("\nAnalyzing feature importance...")
    
    # Use appropriate feature importance method based on model type
    if model_name in ['lstm_autoencoder', 'tcn_autoencoder']:
        # Temporal models have their own importance analysis method
        importance_analysis = model.analyze_temporal_importance(scaled_test_data, feature_names)
        feature_errors = importance_analysis['feature_errors']
        importance_indices = importance_analysis['feature_importance_order']
        
        # Print temporal importance summary
        print("\nTemporal Analysis Summary:")
        temporal_errors = importance_analysis['temporal_errors']
        print(f"  Most critical timesteps: {np.argsort(temporal_errors)[-5:][::-1] + 1}")
        print(f"  Temporal error range: {temporal_errors.min():.6f} - {temporal_errors.max():.6f}")
        
    else:
        # Standard feature importance for non-temporal models
        feature_errors, importance_indices = model.analyze_feature_importance(scaled_test_data, feature_names)
    
    # Create feature importance visualizations (only for models that support it)
    if model_name in ['autoencoder', 'lstm_autoencoder', 'tcn_autoencoder']:
        print("Creating feature importance visualizations...")
        from framework.visualization.evaluation_plots import EvaluationVisualizer
        eval_viz = EvaluationVisualizer(f"./results/{dataset_name}/models/{model_name}")
        eval_viz.plot_feature_importance(feature_errors, feature_names, importance_indices, top_n=20)
        eval_viz.plot_feature_importance_detailed(feature_errors, feature_names, importance_indices, top_n=15)
        
        # Additional visualizations for temporal models
        if model_name in ['lstm_autoencoder', 'tcn_autoencoder']:
            print("Creating temporal-specific visualizations...")
            # You can add temporal-specific plots here in the future
    
    # Get the default threshold strategy for this model
    default_strategy = MODEL_THRESHOLD_STRATEGIES.get(model_name, 'exponential_threshold')
    print(f"\nUsing default threshold strategy for {model_name}: {default_strategy}")
    
    # Calculate threshold using the default strategy
    threshold, all_thresholds = model.calculate_threshold(train_scores, val_scores, default_strategy)
    
    # For TCN autoencoder, decide between fixed and adaptive threshold
    if model_name == 'tcn_autoencoder':
        if use_fixed_threshold:
            print("Using TCN autoencoder's built-in FIXED threshold (set during training)...")
            if hasattr(model, 'threshold') and model.threshold is not None:
                print(f"Model's FIXED threshold: {model.threshold:.6f}")
                threshold = model.threshold
                print(f"Using model's FIXED threshold: {threshold:.6f}")
            else:
                print("No model threshold found, falling back to default strategy...")
                threshold, all_thresholds = model.calculate_threshold(train_scores, val_scores, default_strategy)
        else:
            print("Using ADAPTIVE threshold calculation for TCN autoencoder...")
            adaptive_threshold = model.calculate_adaptive_threshold(train_scores, val_scores, 'adaptive_percentile')
            robust_threshold = model.calculate_adaptive_threshold(train_scores, val_scores, 'robust_iqr')
            
            print(f"Adaptive thresholds:")
            print(f"  Adaptive percentile: {adaptive_threshold:.6f}")
            print(f"  Robust IQR: {robust_threshold:.6f}")
            print(f"  Default strategy ({default_strategy}): {threshold:.6f}")
            
            # Use the most conservative (highest) threshold
            final_threshold = max(adaptive_threshold, robust_threshold, threshold)
            print(f"Selected final ADAPTIVE threshold: {final_threshold:.6f}")
            threshold = final_threshold
    
    # Try different threshold methods for comparison
    print("\nThreshold method comparison:")
    threshold_methods = ['percentile_95', 'percentile_99', 'percentile_99_5', 'mean_plus_1std', 'mean_plus_2std', 'mean_plus_3std', 'exponential_threshold', 'sigmoid_threshold']
    for method in threshold_methods:
        temp_threshold, _ = model.calculate_threshold(train_scores, val_scores, method)
        test_anomalies = test_scores > temp_threshold
        anomaly_rate = np.sum(test_anomalies) / len(test_anomalies) * 100
        indicator = " (DEFAULT)" if method == default_strategy else ""
        print(f"  {method}: {temp_threshold:.6f} -> {anomaly_rate:.2f}% anomalies detected{indicator}")
    print(f"Selected threshold method: {default_strategy} = {threshold:.6f}")
    print()
    
    results = {}
    all_features = []
    all_datasets = []
    
    # Build the processing list dynamically
    processing_list = [
        ('train', scaled_train_data, train_scores, processed_features['train']),
        ('validation', scaled_validation_data, val_scores, processed_features['validation']),
        ('test', scaled_test_data, test_scores, processed_features['test'])
    ]
    
    # Add horizon if available
    if horizon_features is not None and horizon_scores is not None:
        processing_list.append(('horizon', scaled_horizon_data, horizon_scores, processed_features['horizon']))
    
    for split_name, scaled_data, scores, features_data in processing_list:
        # Ensure the model threshold is set before detection
        model.threshold = threshold
        # Use real-time mode for TCN autoencoder to get point-wise detection like Isolation Forest
        if model_name == 'tcn_autoencoder':
            anomalies = model.detect_anomalies(scores, real_time_mode=True)
        else:
            anomalies = model.detect_anomalies(scores)
        reconstructions = model.predict(scaled_data)[0]
        metrics = model.get_metrics(scaled_data, reconstructions, scores)
        
        # Handle temporal models - align features with sequences
        features_df = features_dict[split_name].copy()
        
        if model_name in ['lstm_autoencoder', 'tcn_autoencoder']:
            # For temporal models, we need to align the results with the original data
            # Since sequences are created with sliding windows, we need to properly align timestamps
            sequence_length = model.sequence_length
            
            # Initialize arrays with NaN/False
            padded_scores = np.full(len(features_df), np.nan)
            padded_anomalies = np.full(len(features_df), False, dtype=bool)
            
            # The key insight: each sequence prediction represents the anomaly status 
            # for a specific point in the sequence. For TCN models, we should consider
            # the receptive field to determine the optimal alignment point.
            if len(scores) > 0:
                # For TCN models, use receptive field information for better alignment
                if hasattr(model, 'get_receptive_field'):
                    receptive_field = model.get_receptive_field()
                    # Align to the center of the receptive field within the sequence
                    alignment_offset = min(sequence_length // 2, receptive_field // 2)
                else:
                    # Fallback to center alignment
                    alignment_offset = sequence_length // 2
                
                print(f"Using temporal alignment offset: {alignment_offset} timesteps")
                
                for i, (score, anomaly) in enumerate(zip(scores, anomalies)):
                    target_idx = i + alignment_offset
                    if target_idx < len(features_df):
                        padded_scores[target_idx] = score
                        padded_anomalies[target_idx] = anomaly
            
            features_df['reconstruction_error'] = padded_scores
            features_df['is_anomaly'] = padded_anomalies
        else:
            # Standard models - direct mapping
            features_df['reconstruction_error'] = scores
            features_df['is_anomaly'] = anomalies
            
        features_df['dataset'] = split_name
        
        all_features.append(features_df)
        all_datasets.extend([split_name] * len(features_df))
        
        # Calculate statistics, handling NaN values for temporal models
        if model_name in ['lstm_autoencoder', 'tcn_autoencoder']:
            # For temporal models, only count non-NaN anomalies
            valid_mask = ~np.isnan(features_df['reconstruction_error'])
            num_anomalies = np.sum(features_df['is_anomaly'] & valid_mask)
            total_samples = np.sum(valid_mask)
            sequence_length = model.sequence_length
        else:
            num_anomalies = np.sum(anomalies)
            total_samples = len(anomalies)
            sequence_length = None
        
        anomaly_percentage = (num_anomalies / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"\nDataset: {split_name}")
        print(f"  Mean Absolute Error (MAE): {metrics['mae']:.6f}")
        print(f"  Mean Squared Error (MSE):  {metrics['mse']:.6f}")
        print(f"  Root Mean Squared Error:   {metrics['rmse']:.6f}")
        if sequence_length:
            print(f"  Sequence length:           {sequence_length}")
            print(f"  Valid samples:             {total_samples} (excluding {len(features_df) - total_samples} initial timesteps)")
        print(f"  Anomalies detected:        {num_anomalies}/{total_samples} ({anomaly_percentage:.2f}%)")
        print(f"  Threshold used:            {threshold:.6f}")
    
    combined_features = pd.concat(all_features, ignore_index=True)
    
    # Save anomalies detected to CSV file
    print(f"\nSaving detected anomalies to CSV...")
    save_anomalies_to_csv(combined_features, dataset_name, model_name, threshold)
    
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
        '--use-fixed-threshold',
        action='store_true',
        help='Use the model\'s built-in fixed threshold instead of adaptive calculation (for TCN autoencoder)'
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
        model_name=args.model,
        use_fixed_threshold=args.use_fixed_threshold
    )
