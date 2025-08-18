import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import os
from .visualization.evaluation_plots import EvaluationVisualizer
from config import ATTACK_PERIODS


class GroundTruthEvaluator:
    """
    Ground truth evaluator focused on test dataset evaluation.
    """
    
    def __init__(self, results_dir="./results/autoencoder"):
        self.results_dir = results_dir
        self.evaluation_dir = os.path.join(results_dir, "evaluation")
        os.makedirs(self.evaluation_dir, exist_ok=True)
        self.visualizer = EvaluationVisualizer(results_dir)
    
    def generate_ground_truth(self, test_data: pd.DataFrame, detection_threshold: float):
        """
        Generate ground truth based on predefined attack time periods.
        Hard-codes specific timestamps when attacks occurred.
        """
        errors = test_data['reconstruction_error'].values
        
        # Detection labels: True for detected anomalies (above detection threshold)  
        detection_labels = errors > detection_threshold
        
        # Get attack periods from configuration
        attack_periods = ATTACK_PERIODS
        
        # Convert timestamps to datetime if they're strings
        attack_periods_dt = []
        for start, end in attack_periods:
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            attack_periods_dt.append((start_dt, end_dt))
        
        # Generate ground truth labels based on timestamps
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
        print(f"  Total ground truth anomalies: {np.sum(ground_truth_labels)}")
        print(f"  Total detected anomalies: {np.sum(detection_labels)}")
        
        return ground_truth_labels, detection_labels, gt_threshold
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics."""
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
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
        
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  Precision:         {metrics['precision']:.4f}")
        print(f"  Recall:            {metrics['recall']:.4f}")
        print(f"  F1-Score:          {metrics['f1_score']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}\n")
        
    def create_evaluation_plots(self, test_data, y_true, y_pred, detection_threshold, gt_threshold, metrics):
        """Create evaluation visualizations using the EvaluationVisualizer."""
        # Create ground truth comparison plot
        self.visualizer.plot_ground_truth_evaluation(test_data, y_true, y_pred, detection_threshold, gt_threshold)
        
        # Create confusion matrix heatmap
        self.visualizer.plot_confusion_matrix_heatmap(metrics)
    
    def evaluate_test_dataset(self, test_data: pd.DataFrame, detection_threshold: float):
        """
        Complete evaluation of test dataset against ground truth.
        """
        print("Evaluating test dataset against ground truth...")
        
        # Generate ground truth
        y_true, y_pred, gt_threshold = self.generate_ground_truth(test_data, detection_threshold)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Print results
        self.print_evaluation_results(metrics)
        
        # Create visualizations
        self.create_evaluation_plots(test_data, y_true, y_pred, detection_threshold, gt_threshold, metrics)
        
        return metrics
