import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os


class EvaluationVisualizer:
    """
    Visualization class for evaluation plots including ground truth comparisons.
    """
    
    def __init__(self, results_dir="./results/autoencoder"):
        self.results_dir = results_dir
        self.evaluation_dir = os.path.join(results_dir, "evaluation")
        os.makedirs(self.evaluation_dir, exist_ok=True)
    
    def plot_ground_truth_evaluation(self, test_data, y_true, y_pred, detection_threshold, gt_threshold):
        """
        Create evaluation visualization - binary comparison between ground truth and detection.
        
        Args:
            test_data: DataFrame containing test data with timestamps
            y_true: Ground truth binary labels
            y_pred: Predicted binary labels
            detection_threshold: Detection threshold used
            gt_threshold: Ground truth threshold used
            
        Returns:
            str: Path to saved plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        
        timestamps = test_data['timestamp']
        
        # Binary comparison plot
        ax.fill_between(timestamps, 0, y_true.astype(int), alpha=0.3, color='green', label='Ground Truth')
        ax.fill_between(timestamps, 1, 1 + y_pred.astype(int), alpha=0.3, color='red', label='Detected')
        
        ax.set_ylabel('Binary Labels')
        ax.set_xlabel('Time')
        ax.set_title('Ground Truth vs Detection Comparison', fontweight='bold')
        ax.set_ylim(-0.1, 2.1)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Ground Truth', 'Detected'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        filename = os.path.join(self.evaluation_dir, "ground_truth_evaluation.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plot saved to: {filename}")
        return filename
    
    def plot_confusion_matrix_heatmap(self, metrics):
        """
        Create a visual heatmap of the confusion matrix.
        
        Args:
            metrics: Dictionary containing confusion matrix values
            
        Returns:
            str: Path to saved plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Create confusion matrix array
        cm = np.array([[metrics['true_negatives'], metrics['false_positives']],
                       [metrics['false_negatives'], metrics['true_positives']]])
        
        # Create heatmap
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        
        # Add labels
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Normal', 'Attack'],
               yticklabels=['Normal', 'Attack'],
               title='Confusion Matrix Heatmap',
               ylabel='True Label',
               xlabel='Predicted Label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        
        filename = os.path.join(self.evaluation_dir, "confusion_matrix_heatmap.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix heatmap saved to: {filename}")
        return filename
