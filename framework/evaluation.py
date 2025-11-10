import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score, fbeta_score,
    accuracy_score, f1_score
)
import os
from .visualization.evaluation_plots import EvaluationVisualizer
from .utils import get_results_path


class GroundTruthEvaluator:
    """
    Ground truth evaluator focused on test dataset evaluation.
    """
    
    def __init__(self, dataset_name=None, results_dir=None, model_name="autoencoder", time_span=300):
        if results_dir is None:
            self.results_dir = get_results_path(dataset_name, model_name, time_span, "models")
        else:
            self.results_dir = results_dir
        self.evaluation_dir = os.path.join(self.results_dir, "evaluation")
        os.makedirs(self.evaluation_dir, exist_ok=True)
        self.visualizer = EvaluationVisualizer(self.results_dir)
    
    def generate_ground_truth(self, test_data: pd.DataFrame, detection_threshold: float, attack_periods=None):
        """
        Generate ground truth based on predefined attack time periods.
        
        Args:
            test_data: DataFrame with test data
            detection_threshold: Threshold for anomaly detection
            attack_periods: List of tuples with (start_time, end_time) for attacks.
                           If None, no ground truth will be generated.
        """
        errors = test_data['reconstruction_error'].values
        detection_labels = errors > detection_threshold
        print("Using threshold detection")
        print(f"  Threshold: {detection_threshold}")
        print(f"  Samples above threshold: {np.sum(detection_labels)} anomalies")
        
        # DEBUG: Compare with model's is_anomaly if available
        if 'is_anomaly' in test_data.columns:
            model_labels = test_data['is_anomaly'].values
            print(f"  Model's is_anomaly detected: {np.sum(model_labels)} anomalies")
            print(f"  Difference (threshold - model): {np.sum(detection_labels) - np.sum(model_labels)} samples")
        
        # If no attack periods provided, return empty ground truth
        if not attack_periods:
            print("Warning: No attack periods provided. Cannot generate ground truth.")
            ground_truth_labels = np.zeros(len(test_data), dtype=bool)
            gt_threshold = "No attack periods defined"
            return ground_truth_labels, detection_labels, gt_threshold
        
        # Convert timestamps to datetime if they're strings
        attack_periods_dt = []
        for start, end in attack_periods:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            
            # Check if test_data timestamps are timezone-aware
            if hasattr(test_data['timestamp'].iloc[0], 'tz') and test_data['timestamp'].iloc[0].tz is not None:
                # If test data is timezone-aware, make attack periods timezone-aware too
                # Attack periods in config are already in local time, so just localize them to the same timezone
                timezone = test_data['timestamp'].iloc[0].tz
                start_dt = start_dt.tz_localize(timezone)
                end_dt = end_dt.tz_localize(timezone)
            
            attack_periods_dt.append((start_dt, end_dt))
        
        # Just count unique timestamps
        print(f"  Original test data size: {len(test_data)}")
        unique_timestamps = test_data['timestamp'].unique()
        print(f"  Unique timestamps: {len(unique_timestamps)}")
        
        # Count how many unique timestamps fall within attack periods
        unique_ground_truth_count = 0
        for start_time, end_time in attack_periods_dt:
            matching_timestamps = [ts for ts in unique_timestamps if start_time <= ts <= end_time]
            unique_ground_truth_count += len(matching_timestamps)
        print(f"  Unique timestamps in attack periods: {unique_ground_truth_count}")
        
        # Generate the original ground truth labels
        ground_truth_labels = np.zeros(len(test_data), dtype=bool)
        for start_time, end_time in attack_periods_dt:
            mask = (test_data['timestamp'] >= start_time) & (test_data['timestamp'] <= end_time)
            ground_truth_labels = ground_truth_labels | mask.values

        gt_threshold = "Hard-coded time periods"  # Not a numeric threshold
        
        print(f"Ground Truth Generation:")
        print(f"  Detection threshold: {detection_threshold:.6f}")
        print(f"  Ground truth: Hard-coded attack time periods")
        print(f"  Attack periods defined:")
        for i, (start, end) in enumerate(attack_periods_dt, 1):
            print(f"    {i}. {start} to {end}")
        
        return ground_truth_labels, detection_labels, gt_threshold
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics."""
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Additional comprehensive metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        missrate = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        fallout = fp / (fp + tn) if (fp + tn) > 0 else 0   # False Positive Rate (same as fpr)
        
        # ROC AUC score
        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            # Handle case where only one class is present
            auc = 0.0
        
        # F-beta scores
        f2_score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        
        return {
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'false_positive_rate': fpr,
            'matthews_corrcoef': mcc,
            'miss_rate': missrate,
            'fallout': fallout,
            'roc_auc_score': auc,
            'f2_score': f2_score,
            'total_samples': len(y_true),
            'attack_periods': np.sum(y_true),
            'detected_anomalies': np.sum(y_pred)
        }
    
    def print_evaluation_results(self, metrics):
        """Print evaluation results in a clean format."""
        print("\n" + "="*60)
        print("GROUND TRUTH EVALUATION RESULTS")
        print("="*60)
        
        print("\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"               Normal  Attack")
        print(f"Actual Normal   {metrics['true_negatives']:6d}  {metrics['false_positives']:6d}")
        print(f"       Attack   {metrics['false_negatives']:6d}  {metrics['true_positives']:6d}")
        
        print(f"\nBasic Performance Metrics:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1-Score:          {metrics['f1_score']:.4f}")
        print(f"  F2-Score:          {metrics['f2_score']:.4f}")
        
        print(f"\nAdvanced Performance Metrics:")
        print(f"  Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
        print(f"  ROC AUC Score:     {metrics['roc_auc_score']:.4f}")
        print(f"  Miss Rate (FNR):   {metrics['miss_rate']:.4f}")
        print(f"  Fallout (FPR):     {metrics['fallout']:.4f}")
        
        print(f"\nSample Distribution:")
        print(f"  Total Samples:     {metrics['total_samples']:,}")
        print(f"  Attack Periods:    {metrics['attack_periods']:,}")
        print(f"  Detected Anomalies: {metrics['detected_anomalies']:,}\n")
        
    def create_evaluation_plots(self, test_data, y_true, y_pred, detection_threshold, gt_threshold, metrics):
        """Create evaluation visualizations using the EvaluationVisualizer."""
        # Create ground truth comparison plot (threshold-based detection)
        self.visualizer.plot_ground_truth_evaluation(test_data, y_true, y_pred, detection_threshold, gt_threshold)
        
        # If model labels (is_anomaly) are available, create a separate comparison plot
        if 'is_anomaly' in test_data.columns:
            model_labels = test_data['is_anomaly'].values
            print("\nCreating additional evaluation plot for model's is_anomaly labels...")
            self.visualizer.plot_model_labels_evaluation(test_data, y_true, model_labels)
            
            # Also calculate and print metrics for model labels
            model_metrics = self.calculate_metrics(y_true, model_labels)
            print("\n" + "="*60)
            print("MODEL LABELS (is_anomaly) EVALUATION")
            print("="*60)
            self.print_evaluation_results(model_metrics)
        
        # Create confusion matrix heatmap
        self.visualizer.plot_confusion_matrix_heatmap(metrics)
        
        # Create ROC curve (using reconstruction errors as scores)
        # For ROC curve, we need continuous scores, not just binary predictions
        if 'reconstruction_error' in test_data.columns:
            y_scores = test_data['reconstruction_error'].values
            self.visualizer.plot_roc_curve(y_true, y_scores)
        
        # Create Precision-Recall curve
        if 'reconstruction_error' in test_data.columns:
            y_scores = test_data['reconstruction_error'].values
            self.visualizer.plot_precision_recall_curve(y_true, y_scores)
    
    def evaluate_test_dataset(self, test_data: pd.DataFrame, detection_threshold: float, attack_periods=None):
        """
        Complete evaluation of test dataset against ground truth.
        
        Args:
            test_data: DataFrame with test data
            detection_threshold: Threshold for anomaly detection
            attack_periods: List of tuples with (start_time, end_time) for attacks.
                           If None, evaluation will be skipped.
        """
        if not attack_periods:
            print("Warning: No attack periods provided. Skipping ground truth evaluation.")
            return None
            
        print("Evaluating test dataset against ground truth...")
        
        # Generate ground truth
        y_true, y_pred, gt_threshold = self.generate_ground_truth(test_data, detection_threshold, attack_periods)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Print results
        self.print_evaluation_results(metrics)
        
        # Create visualizations
        self.create_evaluation_plots(test_data, y_true, y_pred, detection_threshold, gt_threshold, metrics)
        
        return metrics
    
    def eval_learning(self, y_test, preds):
        """
        Comprehensive evaluation function similar to the suggested implementation.
        Returns all metrics as individual values for easy use.
        """
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, zero_division=0)
        prec = precision_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        mcc = matthews_corrcoef(y_test, preds)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        missrate = fn / (fn + tp) if (fn + tp) > 0 else 0
        fallout = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        try:
            auc = roc_auc_score(y_test, preds)
        except ValueError:
            auc = 0.0
            
        f2_value = fbeta_score(y_test, preds, beta=2, zero_division=0)

        return acc, rec, prec, f1, mcc, missrate, fallout, auc, f2_value
