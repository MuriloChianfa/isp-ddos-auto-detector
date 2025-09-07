import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os


class EvaluationVisualizer:
    """
    Visualization class for evaluation plots including ground truth comparisons.
    """
    
    def __init__(self, results_dir):
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
    
    def plot_feature_importance(self, feature_errors, feature_names, importance_indices, top_n=20):
        """
        Create a feature importance visualization showing reconstruction errors.
        
        Args:
            feature_errors: Array of feature-wise reconstruction errors
            feature_names: List of feature names
            importance_indices: Indices sorted by importance (descending)
            top_n: Number of top features to display
            
        Returns:
            str: Path to saved plot
        """
        # Get top N features
        top_indices = importance_indices[:top_n]
        top_errors = feature_errors[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Horizontal bar chart for top features
        y_pos = np.arange(len(top_names))
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_names)))
        
        bars = ax1.barh(y_pos, top_errors, color=colors, alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_names, fontsize=10)
        ax1.invert_yaxis()  # Highest importance at top
        ax1.set_xlabel('Reconstruction Error', fontsize=12)
        ax1.set_title(f'Top {top_n} Most Important Features\n(by Reconstruction Error)', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, error) in enumerate(zip(bars, top_errors)):
            width = bar.get_width()
            ax1.text(width + 0.01 * max(top_errors), bar.get_y() + bar.get_height()/2, 
                    f'{error:.3f}', ha='left', va='center', fontsize=9)
        
        # Feature importance distribution histogram
        ax2.hist(feature_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(feature_errors), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(feature_errors):.3f}')
        ax2.axvline(np.median(feature_errors), color='orange', linestyle='--', 
                   label=f'Median: {np.median(feature_errors):.3f}')
        ax2.axvline(np.percentile(feature_errors, 95), color='green', linestyle='--', 
                   label=f'95th percentile: {np.percentile(feature_errors, 95):.3f}')
        
        ax2.set_xlabel('Reconstruction Error', fontsize=12)
        ax2.set_ylabel('Number of Features', fontsize=12)
        ax2.set_title('Feature Importance Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = f"""Feature Statistics:
Total Features: {len(feature_errors)}
Mean Error: {np.mean(feature_errors):.4f}
Std Error: {np.std(feature_errors):.4f}
Max Error: {np.max(feature_errors):.4f}
Min Error: {np.min(feature_errors):.4f}"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.evaluation_dir, "feature_importance.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: {filename}")
        return filename
    
    def plot_feature_importance_detailed(self, feature_errors, feature_names, importance_indices, 
                                       feature_categories=None, top_n=15):
        """
        Create a detailed feature importance visualization with categories.
        
        Args:
            feature_errors: Array of feature-wise reconstruction errors
            feature_names: List of feature names
            importance_indices: Indices sorted by importance (descending)
            feature_categories: Dict mapping feature names to categories (optional)
            top_n: Number of top features to display
            
        Returns:
            str: Path to saved plot
        """
        # Get top N features
        top_indices = importance_indices[:top_n]
        top_errors = feature_errors[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create categories if not provided
        if feature_categories is None:
            feature_categories = {}
            for name in top_names:
                if any(keyword in name.lower() for keyword in ['syn', 'flag', 'tcp']):
                    feature_categories[name] = 'TCP/Protocol'
                elif any(keyword in name.lower() for keyword in ['port', 'entropy']):
                    feature_categories[name] = 'Port/Entropy'
                elif any(keyword in name.lower() for keyword in ['ip', 'src', 'dst']):
                    feature_categories[name] = 'IP/Network'
                elif any(keyword in name.lower() for keyword in ['packet', 'flow', 'bytes']):
                    feature_categories[name] = 'Traffic Volume'
                elif any(keyword in name.lower() for keyword in ['time', 'inter', 'interval']):
                    feature_categories[name] = 'Temporal'
                else:
                    feature_categories[name] = 'Other'
        
        # Get categories for top features
        top_categories = [feature_categories.get(name, 'Other') for name in top_names]
        
        # Create color map for categories
        unique_categories = list(set(top_categories))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_categories)))
        category_colors = dict(zip(unique_categories, colors))
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        
        # Top features with categories
        y_pos = np.arange(len(top_names))
        bar_colors = [category_colors[cat] for cat in top_categories]
        
        bars = ax1.barh(y_pos, top_errors, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        ax1.set_yticks(y_pos)
        
        # Create better labels with ranking
        ranked_labels = [f"{i+1:2d}. {name}" for i, name in enumerate(top_names)]
        ax1.set_yticklabels(ranked_labels, fontsize=10)
        ax1.invert_yaxis()
        ax1.set_xlabel('Reconstruction Error', fontsize=12)
        ax1.set_title(f'Top {top_n} Most Important Features by Category', 
                      fontsize=14, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, error) in enumerate(zip(bars, top_errors)):
            width = bar.get_width()
            ax1.text(width + 0.01 * max(top_errors), bar.get_y() + bar.get_height()/2, 
                    f'{error:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Create legend for categories
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=category_colors[cat], 
                                       alpha=0.8, label=cat) for cat in unique_categories]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Feature importance by category (aggregated)
        category_errors = {}
        category_counts = {}
        
        for i, name in enumerate(feature_names):
            cat = feature_categories.get(name, 'Other')
            if cat not in category_errors:
                category_errors[cat] = []
                category_counts[cat] = 0
            category_errors[cat].append(feature_errors[i])
            category_counts[cat] += 1
        
        # Update color map to include all categories found
        all_categories = list(category_errors.keys())
        colors_all = plt.cm.Set3(np.linspace(0, 1, len(all_categories)))
        category_colors_all = dict(zip(all_categories, colors_all))
        
        # Calculate mean errors per category
        cat_names = list(category_errors.keys())
        cat_mean_errors = [np.mean(category_errors[cat]) for cat in cat_names]
        cat_max_errors = [np.max(category_errors[cat]) for cat in cat_names]
        cat_std_errors = [np.std(category_errors[cat]) for cat in cat_names]
        
        # Sort by mean error
        sorted_indices = np.argsort(cat_mean_errors)[::-1]
        cat_names = [cat_names[i] for i in sorted_indices]
        cat_mean_errors = [cat_mean_errors[i] for i in sorted_indices]
        cat_max_errors = [cat_max_errors[i] for i in sorted_indices]
        cat_std_errors = [cat_std_errors[i] for i in sorted_indices]
        
        # Plot category comparison
        x_pos = np.arange(len(cat_names))
        colors_sorted = [category_colors_all[cat] for cat in cat_names]
        
        bars2 = ax2.bar(x_pos, cat_mean_errors, color=colors_sorted, alpha=0.8, 
                       yerr=cat_std_errors, capsize=5, edgecolor='black', linewidth=0.5)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(cat_names, rotation=45, ha='right')
        ax2.set_ylabel('Mean Reconstruction Error', fontsize=12)
        ax2.set_title('Feature Importance by Category (Mean ± Std)', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean_err, count) in enumerate(zip(bars2, cat_mean_errors, 
                                                      [category_counts[cat] for cat in cat_names])):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + cat_std_errors[i] + 0.01,
                    f'{mean_err:.3f}\n({count} features)', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.evaluation_dir, "feature_importance_detailed.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed feature importance plot saved to: {filename}")
        return filename
