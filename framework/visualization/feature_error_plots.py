import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
from multiprocessing import Pool, cpu_count, Manager
import sys
import time
from ..utils import get_results_path


class FeatureErrorVisualizer:
    """
    Visualizer for creating individual feature reconstruction error charts.
    Creates detailed error analysis plots for each feature and saves them organized by type.
    """
    
    def __init__(self, dataset_name=None, results_dir=None, model_name="autoencoder", time_span=300, max_processes=None):
        if results_dir is None:
            self.results_dir = get_results_path(dataset_name, model_name, time_span, "models")
        else:
            self.results_dir = results_dir
        
        self.max_processes = max_processes if max_processes is not None else 12
        
        # Create error-specific directory structure
        self.error_dir = os.path.join(self.results_dir, "evaluation", "error")
        self.setup_directories()
        
        # Set plotting style consistent with other visualizations
        plt.style.use('default')
        sns.set_palette("husl")
        
    def setup_directories(self):
        """Create directory structure for organizing error plots"""
        subdirs = ['individual_features', 'feature_distributions', 'temporal_errors', 'comparative']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.error_dir, subdir), exist_ok=True)
    
    def plot_individual_feature_error(self, original_data, reconstructed_data, feature_name, 
                                    feature_idx, dataset_type='test', timestamps=None):
        """
        Create comprehensive error analysis plot for a single feature.
        
        Args:
            original_data: Original feature values (numpy array)
            reconstructed_data: Reconstructed feature values (numpy array)
            feature_name: Name of the feature
            feature_idx: Index of the feature
            dataset_type: Type of dataset (train/validation/test)
            timestamps: Optional timestamps for time series plot
            
        Returns:
            str: Path to saved plot
        """
        # Calculate different types of errors
        absolute_errors = np.abs(original_data - reconstructed_data)
        squared_errors = np.square(original_data - reconstructed_data)
        relative_errors = np.abs((original_data - reconstructed_data) / (original_data + 1e-8))
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Reconstruction Error Analysis: {feature_name} ({dataset_type.title()})', 
                     fontsize=16, fontweight='bold')
        
        # 1. Original vs Reconstructed scatter plot
        axes[0, 0].scatter(original_data, reconstructed_data, alpha=0.6, s=20, color='steelblue')
        axes[0, 0].plot([original_data.min(), original_data.max()], 
                       [original_data.min(), original_data.max()], 'r--', linewidth=2, label='Perfect Reconstruction')
        axes[0, 0].set_xlabel('Original Values')
        axes[0, 0].set_ylabel('Reconstructed Values')
        axes[0, 0].set_title('Original vs Reconstructed')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Calculate and display R² score
        ss_res = np.sum((original_data - reconstructed_data) ** 2)
        ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        mae = np.mean(absolute_errors)
        mse = np.mean(squared_errors)
        
        stats_text = f'R² = {r2_score:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}'
        axes[0, 0].text(0.05, 0.95, stats_text, transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), 
                       verticalalignment='top')
        
        # 2. Time series comparison (if timestamps available)
        if timestamps is not None:
            timestamps = pd.to_datetime(timestamps) if not isinstance(timestamps[0], pd.Timestamp) else timestamps
            axes[0, 1].plot(timestamps, original_data, label='Original', alpha=0.8, linewidth=1.5, color='blue')
            axes[0, 1].plot(timestamps, reconstructed_data, label='Reconstructed', alpha=0.8, linewidth=1.5, color='orange')
            axes[0, 1].fill_between(timestamps, original_data, reconstructed_data, 
                                   alpha=0.3, color='red', label='Error Region')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Feature Value')
            axes[0, 1].set_title('Time Series Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            # If no timestamps, show sample-wise comparison
            sample_indices = np.arange(len(original_data))
            axes[0, 1].plot(sample_indices, original_data, label='Original', alpha=0.8, linewidth=1.5, color='blue')
            axes[0, 1].plot(sample_indices, reconstructed_data, label='Reconstructed', alpha=0.8, linewidth=1.5, color='orange')
            axes[0, 1].set_xlabel('Sample Index')
            axes[0, 1].set_ylabel('Feature Value')
            axes[0, 1].set_title('Sample-wise Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Absolute error distribution
        axes[0, 2].hist(absolute_errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 2].axvline(np.mean(absolute_errors), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(absolute_errors):.4f}')
        axes[0, 2].axvline(np.median(absolute_errors), color='orange', linestyle='--', 
                          label=f'Median: {np.median(absolute_errors):.4f}')
        axes[0, 2].axvline(np.percentile(absolute_errors, 95), color='green', linestyle='--', 
                          label=f'95th %ile: {np.percentile(absolute_errors, 95):.4f}')
        axes[0, 2].set_xlabel('Absolute Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Absolute Error Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Error time series (if timestamps available)
        if timestamps is not None:
            axes[1, 0].plot(timestamps, absolute_errors, color='red', alpha=0.8, linewidth=1.5)
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Absolute Error')
            axes[1, 0].set_title('Error Over Time')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add rolling mean for better trend visualization
            if len(absolute_errors) > 10:
                window_size = max(1, len(absolute_errors) // 20)
                rolling_mean = pd.Series(absolute_errors).rolling(window=window_size, center=True).mean()
                axes[1, 0].plot(timestamps, rolling_mean, color='darkred', linewidth=2, 
                               label=f'Rolling Mean (window={window_size})')
                axes[1, 0].legend()
        else:
            sample_indices = np.arange(len(absolute_errors))
            axes[1, 0].plot(sample_indices, absolute_errors, color='red', alpha=0.8, linewidth=1.5)
            axes[1, 0].set_xlabel('Sample Index')
            axes[1, 0].set_ylabel('Absolute Error')
            axes[1, 0].set_title('Error by Sample')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Box plot and violin plot for different error types
        box_data = [absolute_errors, squared_errors, relative_errors]
        box_labels = ['Absolute', 'Squared', 'Relative']
        
        # Use violin plot for better distribution visualization
        parts = axes[1, 1].violinplot(box_data, positions=[1, 2, 3], showmeans=True, showmedians=True)
        
        # Customize violin plot colors
        colors = ['lightcoral', 'lightblue', 'lightgreen']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        axes[1, 1].set_xticks([1, 2, 3])
        axes[1, 1].set_xticklabels(box_labels)
        axes[1, 1].set_ylabel('Error Magnitude')
        axes[1, 1].set_title('Error Type Comparison')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Q-Q plot for error normality assessment
        stats.probplot(absolute_errors, dist="norm", plot=axes[1, 2])
        axes[1, 2].set_title('Q-Q Plot (Error Normality)')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Add Shapiro-Wilk test result if sample size is appropriate
        if 3 <= len(absolute_errors) <= 5000:
            try:
                stat, p_value = stats.shapiro(absolute_errors)
                normality_text = f'Shapiro-Wilk Test:\nStat: {stat:.4f}\np-value: {p_value:.4f}'
                if p_value > 0.05:
                    normality_text += '\n(Likely Normal)'
                else:
                    normality_text += '\n(Not Normal)'
                axes[1, 2].text(0.05, 0.95, normality_text, transform=axes[1, 2].transAxes,
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                               verticalalignment='top', fontsize=9)
            except:
                pass
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.error_dir, "individual_features", 
                               f"{feature_name}_{dataset_type}_error_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature error analysis plot saved to: {filename}")
        return filename
    
    def plot_feature_error_summary(self, feature_errors_dict, feature_names, dataset_type='test'):
        """
        Create summary visualization comparing reconstruction errors across all features.
        
        Args:
            feature_errors_dict: Dictionary with feature_name -> error_array mapping
            feature_names: List of feature names
            dataset_type: Type of dataset (train/validation/test)
            
        Returns:
            str: Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Feature Reconstruction Error Summary ({dataset_type.title()})', 
                     fontsize=16, fontweight='bold')
        
        # Calculate statistics for each feature
        feature_stats = {}
        for feature_name in feature_names:
            if feature_name in feature_errors_dict:
                errors = feature_errors_dict[feature_name]
                feature_stats[feature_name] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'median': np.median(errors),
                    'q95': np.percentile(errors, 95),
                    'max': np.max(errors)
                }
        
        if not feature_stats:
            print("No feature error data available for summary plot")
            plt.close()
            return None
        
        # 1. Mean error comparison
        means = [feature_stats[f]['mean'] for f in feature_names if f in feature_stats]
        feature_labels = [f for f in feature_names if f in feature_stats]
        
        bars1 = axes[0, 0].bar(range(len(means)), means, color='steelblue', alpha=0.7)
        axes[0, 0].set_xlabel('Features')
        axes[0, 0].set_ylabel('Mean Absolute Error')
        axes[0, 0].set_title('Mean Reconstruction Error by Feature')
        axes[0, 0].set_xticks(range(len(feature_labels)))
        axes[0, 0].set_xticklabels(feature_labels, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, means):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Error distribution comparison (violin plot)
        error_data = [feature_errors_dict[f] for f in feature_names if f in feature_errors_dict]
        if error_data:
            parts = axes[0, 1].violinplot(error_data, positions=range(len(error_data)), 
                                         showmeans=True, showmedians=True)
            
            # Customize colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(error_data)))
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.7)
            
            axes[0, 1].set_xlabel('Features')
            axes[0, 1].set_ylabel('Error Distribution')
            axes[0, 1].set_title('Error Distribution by Feature')
            axes[0, 1].set_xticks(range(len(feature_labels)))
            axes[0, 1].set_xticklabels(feature_labels, rotation=45, ha='right')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error variance comparison
        stds = [feature_stats[f]['std'] for f in feature_names if f in feature_stats]
        bars2 = axes[1, 0].bar(range(len(stds)), stds, color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Features')
        axes[1, 0].set_ylabel('Error Standard Deviation')
        axes[1, 0].set_title('Error Variance by Feature')
        axes[1, 0].set_xticks(range(len(feature_labels)))
        axes[1, 0].set_xticklabels(feature_labels, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars2, stds):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 4. 95th percentile comparison (outlier sensitivity)
        q95s = [feature_stats[f]['q95'] for f in feature_names if f in feature_stats]
        bars3 = axes[1, 1].bar(range(len(q95s)), q95s, color='red', alpha=0.7)
        axes[1, 1].set_xlabel('Features')
        axes[1, 1].set_ylabel('95th Percentile Error')
        axes[1, 1].set_title('Error Outliers by Feature (95th Percentile)')
        axes[1, 1].set_xticks(range(len(feature_labels)))
        axes[1, 1].set_xticklabels(feature_labels, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars3, q95s):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.error_dir, "comparative", 
                               f"feature_error_summary_{dataset_type}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature error summary plot saved to: {filename}")
        return filename
    
    def plot_temporal_error_heatmap(self, feature_errors_matrix, feature_names, timestamps=None, dataset_type='test'):
        """
        Create a temporal heatmap showing how reconstruction errors evolve over time for each feature.
        
        Args:
            feature_errors_matrix: 2D array where rows are time points and columns are features
            feature_names: List of feature names
            timestamps: Optional timestamps
            dataset_type: Type of dataset (train/validation/test)
            
        Returns:
            str: Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle(f'Temporal Feature Reconstruction Error Heatmap ({dataset_type.title()})', 
                     fontsize=16, fontweight='bold')
        
        # 1. Full temporal heatmap
        im1 = ax1.imshow(feature_errors_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_ylabel('Features')
        ax1.set_title('Reconstruction Error Over Time (All Features)')
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names)
        
        # Format x-axis for time
        if timestamps is not None:
            # Show fewer time labels for readability
            n_labels = min(10, len(timestamps))
            label_indices = np.linspace(0, len(timestamps)-1, n_labels, dtype=int)
            ax1.set_xticks(label_indices)
            if isinstance(timestamps[0], pd.Timestamp):
                ax1.set_xticklabels([timestamps[i].strftime('%H:%M') for i in label_indices], rotation=45)
            else:
                ax1.set_xticklabels([str(timestamps[i]) for i in label_indices], rotation=45)
            ax1.set_xlabel('Time')
        else:
            ax1.set_xlabel('Sample Index')
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Reconstruction Error')
        
        # 2. Aggregated temporal view (binned errors)
        # Bin the temporal data for better visualization if we have many time points
        if feature_errors_matrix.shape[0] > 100:
            # Bin into ~50 time bins
            bin_size = max(1, feature_errors_matrix.shape[0] // 50)
            binned_errors = []
            binned_times = []
            
            for i in range(0, feature_errors_matrix.shape[0], bin_size):
                end_idx = min(i + bin_size, feature_errors_matrix.shape[0])
                bin_data = feature_errors_matrix[i:end_idx]
                binned_errors.append(np.mean(bin_data, axis=0))
                
                if timestamps is not None:
                    binned_times.append(timestamps[i])
                else:
                    binned_times.append(i)
            
            binned_errors = np.array(binned_errors)
            
            im2 = ax2.imshow(binned_errors.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax2.set_ylabel('Features')
            ax2.set_title(f'Binned Reconstruction Error (bin_size={bin_size})')
            ax2.set_yticks(range(len(feature_names)))
            ax2.set_yticklabels(feature_names)
            
            # Format x-axis for binned time
            n_labels = min(10, len(binned_times))
            label_indices = np.linspace(0, len(binned_times)-1, n_labels, dtype=int)
            ax2.set_xticks(label_indices)
            if timestamps is not None and isinstance(binned_times[0], pd.Timestamp):
                ax2.set_xticklabels([binned_times[i].strftime('%H:%M') for i in label_indices], rotation=45)
            else:
                ax2.set_xticklabels([str(binned_times[i]) for i in label_indices], rotation=45)
            ax2.set_xlabel('Time (Binned)')
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Mean Reconstruction Error')
        else:
            # If not many time points, just duplicate the original plot
            im2 = ax2.imshow(feature_errors_matrix.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax2.set_ylabel('Features')
            ax2.set_title('Reconstruction Error Over Time (Same as above)')
            ax2.set_yticks(range(len(feature_names)))
            ax2.set_yticklabels(feature_names)
            ax2.set_xlabel('Sample Index')
            
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Reconstruction Error')
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.error_dir, "temporal_errors", 
                               f"temporal_error_heatmap_{dataset_type}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal error heatmap saved to: {filename}")
        return filename
    
    def plot_feature_distributions(self, original_data, reconstructed_data, feature_names, dataset_type='test'):
        """
        Create feature distribution comparison plots showing original vs reconstructed values.
        
        Args:
            original_data: Original feature matrix (samples x features)
            reconstructed_data: Reconstructed feature matrix (samples x features)
            feature_names: List of feature names
            dataset_type: Type of dataset (train/validation/test)
            
        Returns:
            list: List of saved plot file paths
        """
        saved_plots = []
        
        # Create distribution plots for top features with highest reconstruction errors
        n_features = len(feature_names)
        feature_errors = []
        
        for i in range(min(n_features, original_data.shape[1], reconstructed_data.shape[1])):
            error = np.mean(np.abs(original_data[:, i] - reconstructed_data[:, i]))
            feature_errors.append((error, i, feature_names[i]))
        
        # Sort by error and take top 12 features for distribution analysis
        feature_errors.sort(reverse=True)
        top_features = feature_errors[:12]
        
        # Create a 4x3 grid for the top 12 features
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        fig.suptitle(f'Feature Distributions: Original vs Reconstructed ({dataset_type.title()} Set)', 
                     fontsize=16, fontweight='bold')
        
        for idx, (error, feature_idx, feature_name) in enumerate(top_features):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if feature_idx < original_data.shape[1] and feature_idx < reconstructed_data.shape[1]:
                original_values = original_data[:, feature_idx]
                reconstructed_values = reconstructed_data[:, feature_idx]
                
                # Create histograms
                bins = min(30, len(np.unique(original_values)) if len(np.unique(original_values)) < 30 else 30)
                
                ax.hist(original_values, bins=bins, alpha=0.7, label='Original', 
                       color='skyblue', density=True, edgecolor='black', linewidth=0.5)
                ax.hist(reconstructed_values, bins=bins, alpha=0.7, label='Reconstructed', 
                       color='lightcoral', density=True, edgecolor='black', linewidth=0.5)
                
                # Add statistics
                orig_mean = np.mean(original_values)
                recon_mean = np.mean(reconstructed_values)
                mean_diff = abs(orig_mean - recon_mean)
                
                ax.set_title(f'{feature_name}\nMAE: {error:.4f}, Mean Δ: {mean_diff:.4f}', 
                           fontsize=10, fontweight='bold')
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Add vertical lines for means
                ax.axvline(orig_mean, color='blue', linestyle='--', alpha=0.8, linewidth=1)
                ax.axvline(recon_mean, color='red', linestyle='--', alpha=0.8, linewidth=1)
        
        # Hide empty subplots if we have fewer than 12 features
        for idx in range(len(top_features), 12):
            row = idx // 3
            col = idx % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save the plot
        filename = os.path.join(self.error_dir, "feature_distributions", 
                               f"feature_distributions_{dataset_type}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        saved_plots.append(filename)
        print(f"Feature distributions plot saved to: {filename}")
        
        # Create a second plot showing error distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Reconstruction Error Analysis ({dataset_type.title()} Set)', 
                     fontsize=14, fontweight='bold')
        
        # Calculate overall reconstruction errors
        all_errors = np.abs(original_data - reconstructed_data)
        mean_errors_per_sample = np.mean(all_errors, axis=1)
        mean_errors_per_feature = np.mean(all_errors, axis=0)
        
        # 1. Distribution of mean errors per sample
        ax1 = axes[0, 0]
        ax1.hist(mean_errors_per_sample, bins=30, alpha=0.7, color='lightblue', 
                edgecolor='black', linewidth=0.5)
        ax1.set_title('Distribution of Mean Errors per Sample')
        ax1.set_xlabel('Mean Reconstruction Error')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(mean_errors_per_sample), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(mean_errors_per_sample):.4f}')
        ax1.legend()
        
        # 2. Distribution of mean errors per feature
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(mean_errors_per_feature)), mean_errors_per_feature, 
                      alpha=0.7, color='lightgreen', edgecolor='black', linewidth=0.5)
        ax2.set_title('Mean Reconstruction Error per Feature')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Mean Reconstruction Error')
        ax2.grid(True, alpha=0.3)
        
        # Highlight top error features
        top_error_indices = np.argsort(mean_errors_per_feature)[-5:]
        for idx in top_error_indices:
            bars[idx].set_color('red')
            bars[idx].set_alpha(0.8)
        
        # 3. Cumulative error distribution
        ax3 = axes[1, 0]
        sorted_errors = np.sort(mean_errors_per_sample)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax3.plot(sorted_errors, cumulative, linewidth=2, color='navy')
        ax3.set_title('Cumulative Distribution of Sample Errors')
        ax3.set_xlabel('Reconstruction Error')
        ax3.set_ylabel('Cumulative Probability')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(np.percentile(sorted_errors, 95), color='red', linestyle='--', 
                   label='95th percentile')
        ax3.legend()
        
        # 4. Error correlation heatmap (top 10 features)
        ax4 = axes[1, 1]
        if len(feature_names) > 1:
            top_10_indices = np.argsort(mean_errors_per_feature)[-10:]
            error_matrix = all_errors[:, top_10_indices]
            correlation_matrix = np.corrcoef(error_matrix.T)
            
            im = ax4.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax4.set_title('Error Correlation (Top 10 Features)')
            
            # Set ticks and labels
            tick_labels = [feature_names[i][:10] + '...' if len(feature_names[i]) > 10 
                          else feature_names[i] for i in top_10_indices]
            ax4.set_xticks(range(len(tick_labels)))
            ax4.set_yticks(range(len(tick_labels)))
            ax4.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax4.set_yticklabels(tick_labels)
            
            # Add colorbar
            plt.colorbar(im, ax=ax4, label='Correlation')
        else:
            ax4.text(0.5, 0.5, 'Not enough features\nfor correlation analysis', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Error Correlation Analysis')
        
        plt.tight_layout()
        
        # Save the error analysis plot
        filename2 = os.path.join(self.error_dir, "feature_distributions", 
                                f"error_analysis_{dataset_type}.png")
        plt.savefig(filename2, dpi=300, bbox_inches='tight')
        plt.close()
        
        saved_plots.append(filename2)
        print(f"Error analysis plot saved to: {filename2}")
        
        return saved_plots
    
    def _plot_feature_error_task(self, task_info):
        """
        Worker function for parallel error plot generation.
        
        Args:
            task_info: Tuple containing (feature_idx, feature_name, original_col, reconstructed_col, 
                       dataset_type, timestamps, progress_info)
        
        Returns:
            Tuple: (success: bool, filepath: str, feature_name: str, dataset_type: str)
        """
        try:
            feature_idx = task_info[0]
            feature_name = task_info[1]
            original_col = task_info[2]
            reconstructed_col = task_info[3]
            dataset_type = task_info[4]
            timestamps = task_info[5]
            progress_info = task_info[6]
            
            # Get process ID for better tracking
            pid = os.getpid()
            
            # Print progress message with process info
            if progress_info and 'counter' in progress_info:
                # Thread-safe increment (Manager.Value handles synchronization automatically)
                progress_info['counter'].value += 1
                current_count = progress_info['counter'].value
                total_count = progress_info['total']
                progress_pct = (current_count / total_count) * 100
                print(f"[PID {pid:5}] [{current_count:3d}/{total_count}] ({progress_pct:5.1f}%) Processing error plot: {feature_name} ({dataset_type})")
                sys.stdout.flush()  # Force immediate output
            
            # Create individual feature error plot
            filepath = self.plot_individual_feature_error(
                original_col, reconstructed_col, 
                feature_name, feature_idx, dataset_type, timestamps
            )
            
            if filepath:
                sys.stdout.flush()
            
            return (True, filepath, feature_name, dataset_type)
            
        except Exception as e:
            print(f"Error creating plot for {task_info[1]}: {e}")
            return (False, None, task_info[1], task_info[4])
    
    def create_all_feature_error_plots(self, original_data, reconstructed_data, feature_names, 
                                     timestamps=None, dataset_type='test', use_parallel=True):
        """
        Create all error plots for the given dataset with optional parallel processing.
        
        Args:
            original_data: Original feature matrix (samples x features)
            reconstructed_data: Reconstructed feature matrix (samples x features)
            feature_names: List of feature names
            timestamps: Optional timestamps
            dataset_type: Type of dataset (train/validation/test)
            use_parallel: Whether to use parallel processing for individual feature plots
            
        Returns:
            dict: Dictionary with plot types and their saved file paths
        """
        saved_plots = {}
        
        # Calculate feature-wise errors for later use
        feature_errors_dict = {}
        
        # Prepare tasks for parallel processing of individual feature plots
        tasks = []
        for i, feature_name in enumerate(feature_names):
            if i < original_data.shape[1] and i < reconstructed_data.shape[1]:
                errors = np.abs(original_data[:, i] - reconstructed_data[:, i])
                feature_errors_dict[feature_name] = errors
                
                # Add task for individual plot generation
                tasks.append((
                    i,
                    feature_name,
                    original_data[:, i],
                    reconstructed_data[:, i],
                    dataset_type,
                    timestamps,
                    None  # progress_info (will be added later)
                ))
        
        # Process individual feature plots
        if tasks:
            if use_parallel and self.max_processes > 1:
                # Setup shared progress tracking for multiprocessing
                manager = Manager()
                progress_counter = manager.Value('i', 0)  # Shared integer counter
                progress_info = {
                    'counter': progress_counter,
                    'total': len(tasks)
                }
                
                # Update tasks to include progress_info
                tasks = [task[:-1] + (progress_info,) for task in tasks]
                
                # Determine actual number of processes to use
                actual_processes = min(self.max_processes, len(tasks), cpu_count())
                
                print(f"\nGenerating individual feature error plots using {actual_processes} processes...")
                print(f"  Total tasks: {len(tasks)}")
                print(f"  Estimated time: {len(tasks) // actual_processes + 1} batches")
                print("=" * 80)
                print("Starting parallel processing...")
                print("=" * 80)
                
                start_time = time.time()
                
                # Process tasks in parallel
                with Pool(processes=actual_processes) as pool:
                    results = pool.map(self._plot_feature_error_task, tasks)
                
                processing_time = time.time() - start_time
                print("=" * 80)
                print(f"Parallel processing completed in {processing_time:.2f} seconds")
                print("=" * 80)
                
                # Process results
                successful_plots = 0
                failed_plots = 0
                for success, filepath, feature_name, ds_type in results:
                    if success and filepath:
                        saved_plots[f"individual_{feature_name}"] = filepath
                        successful_plots += 1
                    else:
                        failed_plots += 1
                        print(f"  FAILED: {feature_name}")
                
                print(f"\nIndividual feature error plots: {successful_plots} successful, {failed_plots} failed")
                if successful_plots > 0:
                    print(f"  Average time per plot: {processing_time/successful_plots:.3f} seconds")
                    
            else:
                # Sequential processing
                print(f"\nGenerating individual feature error plots sequentially...")
                for i, feature_name in enumerate(feature_names):
                    if i < original_data.shape[1] and i < reconstructed_data.shape[1]:
                        try:
                            print(f"  [{i+1:3d}/{len(feature_names)}] Processing: {feature_name}")
                            plot_path = self.plot_individual_feature_error(
                                original_data[:, i], reconstructed_data[:, i], 
                                feature_name, i, dataset_type, timestamps
                            )
                            saved_plots[f"individual_{feature_name}"] = plot_path
                        except Exception as e:
                            print(f"  Warning: Could not create individual plot for {feature_name}: {e}")
        
        # Create summary plot (not parallelized - single plot)
        print(f"\nGenerating summary plot...")
        try:
            summary_path = self.plot_feature_error_summary(
                feature_errors_dict, feature_names, dataset_type
            )
            saved_plots["summary"] = summary_path
        except Exception as e:
            print(f"  Warning: Could not create summary plot: {e}")
        
        # Create temporal heatmap (not parallelized - single plot)
        print(f"\nGenerating temporal heatmap...")
        try:
            # Calculate error matrix for heatmap
            feature_errors_matrix = np.abs(original_data - reconstructed_data)
            heatmap_path = self.plot_temporal_error_heatmap(
                feature_errors_matrix, feature_names, timestamps, dataset_type
            )
            saved_plots["temporal_heatmap"] = heatmap_path
        except Exception as e:
            print(f"  Warning: Could not create temporal heatmap: {e}")
        
        # Create feature distribution plots (not parallelized - creates 2 summary plots)
        print(f"\nGenerating feature distribution plots...")
        try:
            distribution_paths = self.plot_feature_distributions(
                original_data, reconstructed_data, feature_names, dataset_type
            )
            for i, path in enumerate(distribution_paths):
                saved_plots[f"distribution_{i+1}"] = path
        except Exception as e:
            print(f"  Warning: Could not create feature distribution plots: {e}")
        
        print(f"\n{'='*60}")
        print(f"ERROR PLOT GENERATION COMPLETED!")
        print(f"{'='*60}")
        print(f"Generated {len(saved_plots)} error plot(s) for {dataset_type} dataset:")
        for plot_type, path in saved_plots.items():
            print(f"  - {plot_type}: {path}")
        
        return saved_plots


def generate_feature_reconstruction_error_plots(model, processing_list, features_dict, feature_names, 
                                               dataset_name, model_name, time_span):
    """
    Generate comprehensive feature reconstruction error visualizations for all dataset splits.
    
    Args:
        model: Trained model instance
        processing_list: List of tuples containing (split_name, scaled_data, scores, _)
        features_dict: Dictionary containing feature DataFrames for each split
        feature_names: List of feature names
        dataset_name: Name of the dataset
        model_name: Name of the model
        time_span: Time span in seconds
        
    Returns:
        dict: Summary of generated plots by split
    """
    print("\nCreating feature reconstruction error visualizations...")
    error_viz = FeatureErrorVisualizer(dataset_name=dataset_name, model_name=model_name, time_span=time_span)
    
    plot_summary = {}
    
    # Generate error visualizations for each dataset split
    for split_name, scaled_data, scores, _ in processing_list:
        try:
            print(f"  Creating error plots for {split_name} dataset...")
            
            # Get reconstructions for this split
            reconstructions = model.predict(scaled_data)[0]
            
            # Get the actual features dataframe from features_dict
            features_data = features_dict[split_name]
            
            # Extract timestamps if available
            timestamps = None
            if 'timestamp' in features_data.columns:
                timestamps = features_data['timestamp'].values
            
            # Standard models - direct processing
            error_plots = error_viz.create_all_feature_error_plots(
                scaled_data, reconstructions, feature_names,
                timestamps, split_name
            )
            
            plot_summary[split_name] = {
                "status": "success",
                "plots_generated": len(error_plots),
                "plot_paths": error_plots
            }
            print(f"    Generated {len(error_plots)} error visualizations for {split_name}")
            
        except Exception as e:
            print(f"    Warning: Could not create error plots for {split_name}: {e}")
            plot_summary[split_name] = {"status": "error", "error": str(e)}
            continue
    
    # Print summary
    total_plots = sum(summary.get("plots_generated", 0) for summary in plot_summary.values())
    successful_splits = sum(1 for summary in plot_summary.values() if summary["status"] == "success")
    
    print(f"\nFeature reconstruction error visualization summary:")
    print(f"  Total plots generated: {total_plots}")
    print(f"  Successful splits: {successful_splits}/{len(processing_list)}")
    
    for split_name, summary in plot_summary.items():
        status = summary["status"]
        if status == "success":
            print(f"  {split_name}: ✓ {summary['plots_generated']} plots")
        elif status == "skipped":
            print(f"  {split_name}: ⚠ skipped ({summary['reason']})")
        else:
            print(f"  {split_name}: ✗ error ({summary.get('error', 'unknown')})")
    
    return plot_summary