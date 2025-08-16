import numpy as np
import pandas as pd

from framework.loader import NetworkDataLoader
from framework.features import NetworkFeatureExtractor
from framework.models.autoencoder import AutoencoderAnomalyDetector
from framework.visualization.training_plots import TrainingVisualizer
from framework.visualization.anomaly_plots import AnomalyVisualizer


def main():
    print("Loading network traffic data...")
    loader = NetworkDataLoader()
    datasets = loader.load_network_data_by_day()
    
    print("\nExtracting features...")
    feature_extractor = NetworkFeatureExtractor()
    features_dict = feature_extractor.process_datasets(datasets)
    processed_features = feature_extractor.prepare_training_data(features_dict)
    
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
    model.build_model(input_dim)
    
    print("\nTraining model...")
    history = model.train(scaled_train_data, scaled_validation_data)
    
    print("\nVisualizing training results...")
    training_viz = TrainingVisualizer()
    training_viz.plot_training_history(history)
    training_viz.print_training_summary(history)
    
    print("\nDetecting anomalies...")
    train_reconstructions, train_mse = model.predict(scaled_train_data)
    val_reconstructions, val_mse = model.predict(scaled_validation_data)
    test_reconstructions, test_mse = model.predict(scaled_test_data)
    
    threshold, all_thresholds = model.calculate_threshold(train_mse, val_mse, 'mean_plus_2std')
    
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
    anomaly_viz = AnomalyVisualizer()
    anomaly_viz.print_threshold_comparison(test_mse, all_thresholds)
    anomaly_viz.plot_anomaly_detection(combined_features, threshold)
    anomaly_viz.print_anomaly_statistics(combined_features, threshold)
    
    print("\nAnalysis completed. Results saved to ./results/autoencoder/")


if __name__ == "__main__":
    main()
