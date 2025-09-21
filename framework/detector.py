"""
Anomaly detection and evaluation functionality for DDoS detection pipeline.
Handles anomaly detection across all data splits, threshold calculation, and ground truth evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from framework.evaluation import GroundTruthEvaluator
from config import MODEL_THRESHOLD_STRATEGIES


class AnomalyDetector:
    """Handles anomaly detection and evaluation logic"""
    
    def __init__(self, model: Any, dataset_name: str, model_name: str, time_span: int):
        """
        Initialize AnomalyDetector
        
        Args:
            model: Trained model object
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_span: Time span in seconds
        """
        self.model = model
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.time_span = time_span
    
    def detect_anomalies_all_splits(self, processed_features: Dict, features_dict: Dict,
                                   use_fixed_threshold: bool = False) -> Tuple[pd.DataFrame, float, Dict]:
        """
        Detect anomalies across all data splits
        
        Args:
            processed_features: Dictionary containing processed features for each split
            features_dict: Dictionary containing original feature data for each split
            use_fixed_threshold: Whether to use fixed threshold for TCN autoencoder
            
        Returns:
            Tuple of (combined_features_df, threshold, all_thresholds)
        """
        print("\nDetecting anomalies...")
        
        # Get data for each split
        train_features = processed_features['train']['features']
        val_features = processed_features['validation']['features']
        test_features = processed_features['test']['features']
        horizon_features = processed_features.get('horizon', {}).get('features', None)
        
        # Transform data
        scaled_train_data = self.model.transform_data(train_features)
        scaled_validation_data = self.model.transform_data(val_features)
        scaled_test_data = self.model.transform_data(test_features)
        
        # Get predictions
        train_reconstructions, train_scores = self.model.predict(scaled_train_data)
        val_reconstructions, val_scores = self.model.predict(scaled_validation_data)
        test_reconstructions, test_scores = self.model.predict(scaled_test_data)
        
        # Process horizon data if available
        horizon_reconstructions, horizon_scores = None, None
        scaled_horizon_data = None
        if horizon_features is not None:
            scaled_horizon_data = self.model.transform_data(horizon_features)
            horizon_reconstructions, horizon_scores = self.model.predict(scaled_horizon_data)
            print(f"Horizon anomaly scores computed: {len(horizon_scores)} samples")
        
        # Calculate optimal threshold
        threshold, all_thresholds = self.calculate_optimal_threshold(
            train_scores, val_scores, use_fixed_threshold
        )
        
        # Build processing list
        processing_list = [
            ('train', scaled_train_data, train_scores, processed_features['train']),
            ('validation', scaled_validation_data, val_scores, processed_features['validation']),
            ('test', scaled_test_data, test_scores, processed_features['test'])
        ]
        
        if horizon_features is not None and horizon_scores is not None:
            processing_list.append(('horizon', scaled_horizon_data, horizon_scores, processed_features['horizon']))
        
        # Process each split
        all_features = []
        for split_name, scaled_data, scores, features_data in processing_list:
            features_df = self._process_split_anomalies(
                split_name, scaled_data, scores, features_dict[split_name], threshold
            )
            all_features.append(features_df)
        
        # Combine all results
        combined_features = pd.concat(all_features, ignore_index=True)
        
        return combined_features, threshold, all_thresholds
    
    def calculate_optimal_threshold(self, train_scores: np.ndarray, val_scores: np.ndarray,
                                   use_fixed_threshold: bool = False) -> Tuple[float, Dict]:
        """
        Calculate and compare different threshold strategies
        
        Args:
            train_scores: Training reconstruction scores
            val_scores: Validation reconstruction scores
            use_fixed_threshold: Whether to use fixed threshold for TCN autoencoder
            
        Returns:
            Tuple of (selected_threshold, all_thresholds_dict)
        """
        # Get the default threshold strategy for this model
        default_strategy = MODEL_THRESHOLD_STRATEGIES.get(self.model_name, 'exponential_threshold')
        print(f"\nUsing default threshold strategy for {self.model_name}: {default_strategy}")
        
        # Calculate threshold using the default strategy
        threshold, all_thresholds = self.model.calculate_threshold(train_scores, val_scores, default_strategy)
        
        # Update the threshold in model artifacts after calculation
        try:
            from framework.models.core.artifacts import ModelArtifactsManager
            artifacts_manager = ModelArtifactsManager(self.dataset_name, self.model_name, self.time_span)
            artifacts_manager.update_threshold(threshold, all_thresholds)
        except Exception as e:
            print(f"Warning: Could not update threshold in artifacts: {e}")
        
        # Special handling for TCN autoencoder
        if self.model_name == 'tcn_autoencoder':
            threshold = self._handle_tcn_threshold(
                train_scores, val_scores, threshold, all_thresholds, 
                default_strategy, use_fixed_threshold
            )
        
        # Compare different threshold methods
        self._print_threshold_comparison(train_scores, val_scores, default_strategy, threshold)
        
        return threshold, all_thresholds
    
    def _handle_tcn_threshold(self, train_scores: np.ndarray, val_scores: np.ndarray,
                             threshold: float, all_thresholds: Dict, default_strategy: str,
                             use_fixed_threshold: bool) -> float:
        """
        Handle special threshold calculation for TCN autoencoder
        
        Args:
            train_scores: Training reconstruction scores
            val_scores: Validation reconstruction scores  
            threshold: Default calculated threshold
            all_thresholds: Dictionary of all calculated thresholds
            default_strategy: Default threshold strategy name
            use_fixed_threshold: Whether to use fixed threshold
            
        Returns:
            Final threshold value
        """
        if use_fixed_threshold:
            print("Using TCN autoencoder's built-in FIXED threshold (set during training)...")
            if hasattr(self.model, 'threshold') and self.model.threshold is not None:
                print(f"Model's FIXED threshold: {self.model.threshold:.6f}")
                threshold = self.model.threshold
                print(f"Using model's FIXED threshold: {threshold:.6f}")
            else:
                print("No model threshold found, falling back to default strategy...")
        else:
            print("Using ADAPTIVE threshold calculation for TCN autoencoder...")
            adaptive_threshold = self.model.calculate_adaptive_threshold(train_scores, val_scores, 'adaptive_percentile')
            robust_threshold = self.model.calculate_adaptive_threshold(train_scores, val_scores, 'robust_iqr')
            
            print(f"Adaptive thresholds:")
            print(f"  Adaptive percentile: {adaptive_threshold:.6f}")
            print(f"  Robust IQR: {robust_threshold:.6f}")
            print(f"  Default strategy ({default_strategy}): {threshold:.6f}")
            
            # Use the most conservative (highest) threshold
            final_threshold = max(adaptive_threshold, robust_threshold, threshold)
            print(f"Selected final ADAPTIVE threshold: {final_threshold:.6f}")
            threshold = final_threshold
        
        return threshold
    
    def _print_threshold_comparison(self, train_scores: np.ndarray, val_scores: np.ndarray,
                                   default_strategy: str, selected_threshold: float):
        """
        Print comparison of different threshold methods
        
        Args:
            train_scores: Training reconstruction scores
            val_scores: Validation reconstruction scores
            default_strategy: Default threshold strategy name
            selected_threshold: The selected threshold value
        """
        print("\nThreshold method comparison:")
        threshold_methods = [
            'percentile_95', 'percentile_99', 'percentile_99_5', 
            'mean_plus_1std', 'mean_plus_2std', 'mean_plus_3std', 
            'exponential_threshold', 'sigmoid_threshold'
        ]
        
        # Get test scores for comparison (use validation scores as proxy)
        test_scores = val_scores
        
        for method in threshold_methods:
            temp_threshold, _ = self.model.calculate_threshold(train_scores, val_scores, method)
            test_anomalies = test_scores > temp_threshold
            anomaly_rate = np.sum(test_anomalies) / len(test_anomalies) * 100
            indicator = " (DEFAULT)" if method == default_strategy else ""
            print(f"  {method}: {temp_threshold:.6f} -> {anomaly_rate:.2f}% anomalies detected{indicator}")
        
        print(f"Selected threshold method: {default_strategy} = {selected_threshold:.6f}")
        print()
    
    def _process_split_anomalies(self, split_name: str, scaled_data: np.ndarray, 
                                scores: np.ndarray, features_df: pd.DataFrame,
                                threshold: float) -> pd.DataFrame:
        """
        Process anomaly detection for a specific data split
        
        Args:
            split_name: Name of the data split ('train', 'validation', 'test', 'horizon')
            scaled_data: Scaled input data
            scores: Reconstruction scores
            features_df: Original features dataframe
            threshold: Threshold for anomaly detection
            
        Returns:
            DataFrame with anomaly detection results
        """
        # Ensure the model threshold is set before detection
        self.model.threshold = threshold
        
        # Use real-time mode for TCN autoencoder to get point-wise detection
        if self.model_name == 'tcn_autoencoder':
            anomalies = self.model.detect_anomalies(scores, real_time_mode=True)
        else:
            anomalies = self.model.detect_anomalies(scores)
        
        reconstructions = self.model.predict(scaled_data)[0]
        metrics = self.model.get_metrics(scaled_data, reconstructions, scores)
        
        # Handle temporal models - align features with sequences
        features_result = features_df.copy()
        
        if self.model_name in ['lstm_autoencoder', 'tcn_autoencoder']:
            features_result = self._align_temporal_results(features_result, scores, anomalies)
        else:
            # Standard models - direct mapping
            features_result['reconstruction_error'] = scores
            features_result['is_anomaly'] = anomalies
        
        features_result['dataset'] = split_name
        
        # Print statistics
        self._print_split_statistics(split_name, features_result, metrics, anomalies)
        
        return features_result
    
    def _align_temporal_results(self, features_df: pd.DataFrame, scores: np.ndarray,
                               anomalies: np.ndarray) -> pd.DataFrame:
        """
        Align temporal model results with original feature timestamps
        
        Args:
            features_df: Original features dataframe
            scores: Reconstruction scores from temporal model
            anomalies: Anomaly predictions from temporal model
            
        Returns:
            DataFrame with properly aligned temporal results
        """
        sequence_length = self.model.sequence_length
        
        # Initialize arrays with NaN/False
        padded_scores = np.full(len(features_df), np.nan)
        padded_anomalies = np.full(len(features_df), False, dtype=bool)
        
        if len(scores) > 0:
            # For TCN models, use receptive field information for better alignment
            if hasattr(self.model, 'get_receptive_field'):
                receptive_field = self.model.get_receptive_field()
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
        
        return features_df
    
    def _print_split_statistics(self, split_name: str, features_df: pd.DataFrame,
                               metrics: Dict, anomalies: np.ndarray):
        """
        Print statistics for a data split
        
        Args:
            split_name: Name of the data split
            features_df: Features dataframe with results
            metrics: Model metrics dictionary
            anomalies: Anomaly predictions array
        """
        # Calculate statistics, handling NaN values for temporal models
        if self.model_name in ['lstm_autoencoder', 'tcn_autoencoder']:
            # For temporal models, only count non-NaN anomalies
            valid_mask = ~np.isnan(features_df['reconstruction_error'])
            num_anomalies = np.sum(features_df['is_anomaly'] & valid_mask)
            total_samples = np.sum(valid_mask)
            sequence_length = self.model.sequence_length
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
    
    def evaluate_performance(self, combined_features: pd.DataFrame, threshold: float,
                            dataset_config: Dict) -> Optional[Dict]:
        """
        Evaluate model performance with ground truth
        
        Args:
            combined_features: DataFrame containing all features and results
            threshold: Threshold used for anomaly detection
            dataset_config: Dataset configuration dictionary
            
        Returns:
            Evaluation metrics dictionary if successful, None otherwise
        """
        print("\nPerforming ground truth evaluation on test dataset...")
        evaluator = GroundTruthEvaluator(
            dataset_name=self.dataset_name, 
            model_name=self.model_name, 
            time_span=self.time_span
        )
        
        # Extract test dataset
        test_data = combined_features[combined_features['dataset'] == 'test'].copy()
        
        # Use dataset-specific attack periods
        attack_periods = dataset_config.get('attack_periods', [])
        if attack_periods:
            # Evaluate against ground truth using dataset-specific attack periods
            evaluation_metrics = evaluator.evaluate_test_dataset(test_data, threshold, attack_periods)
            return evaluation_metrics
        else:
            print("Warning: No attack periods defined for this dataset. Skipping ground truth evaluation.")
            return None
