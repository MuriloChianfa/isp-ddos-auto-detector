import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os
import gc
from ..utils import get_results_path

plt.ioff()  # Disable interactive mode


class AnomalyVisualizer:
    def __init__(self, dataset_name=None, results_dir=None, model_name="autoencoder", time_span=300):
        if results_dir is None:
            self.results_dir = get_results_path(dataset_name, model_name, time_span, "models")
        else:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def plot_anomaly_detection(self, combined_features, threshold, model_name="autoencoder", attack_periods=None):
        """Create anomaly detection visualization focused on test dataset"""
        test_mask = combined_features['dataset'] == 'test'
        test_data = combined_features[test_mask]
        
        # Create main test dataset plot using common function
        self._create_anomaly_plot(
            test_data, combined_features, threshold,
            'DDoS Detection Event Timeline',
            'anomaly_detection.png',
            'green', 'Anomaly Scores',
            attack_periods=attack_periods
        )
        
        # Create a second plot focused on the threshold region for better visibility
        self._plot_threshold_focused_view(combined_features, threshold, model_name, attack_periods=attack_periods)
        
        # Generate separate plots for train and validation datasets
        self._plot_train_validation_splits(combined_features, threshold, model_name)
        
        filename = os.path.join(self.results_dir, "anomaly_detection.png")
        print(f"Anomaly detection plot saved to: {filename}")
        return filename
        
    def _plot_threshold_focused_view(self, combined_features, threshold, model_name="autoencoder", attack_periods=None):
        """Create a focused view of the threshold region for better visibility"""
        test_mask = combined_features['dataset'] == 'test'
        test_data = combined_features[test_mask]
        
        # Create figure with two subplots - main plot and ground truth bar
        fig, (ax_main, ax_gt) = plt.subplots(2, 1, figsize=(20, 9), 
                                              gridspec_kw={'height_ratios': [15, 1], 'hspace': 0.05},
                                              sharex=True)
        
        # Use common helper functions
        timestamps = pd.to_datetime(test_data['timestamp'])
        anomaly_scores = test_data['reconstruction_error']
        anomalies_mask = test_data['is_anomaly']
        
        # Add anomaly highlights (red)
        self._add_anomaly_highlights(ax_main, timestamps, anomalies_mask)
        
        ax_main.step(timestamps, anomaly_scores, where='post',
                color='green', alpha=0.8, linewidth=1.5, 
                label='Anomaly Scores', zorder=5)
        
        # Add threshold lines
        self._add_threshold_lines(ax_main, combined_features, threshold)
        
        # Calculate focused y-range around threshold region
        normal_mask = (combined_features['dataset'] == 'train') | (combined_features['dataset'] == 'validation')
        normal_anomaly_scores = combined_features[normal_mask]['reconstruction_error']
        mse_plus_std_anomaly_score = np.mean(normal_anomaly_scores) + np.std(normal_anomaly_scores)
        max_threshold = max(threshold, mse_plus_std_anomaly_score)
        focused_y_limit = max_threshold * 4
        ax_main.set_ylim(bottom=-0.005, top=focused_y_limit)
        
        # Configure appearance with focused title
        self._configure_plot_appearance(ax_main, 'Network Traffic Anomaly Detection - Threshold Focused View')
        
        # Add ground truth bar at the bottom
        self._add_ground_truth_bar(ax_gt, timestamps, attack_periods)
        
        focused_filename = os.path.join(self.results_dir, f"anomaly_detection_focused.png")
        plt.savefig(focused_filename, dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        gc.collect()
        
        print(f"Threshold-focused anomaly plot saved to: {focused_filename}")
        return focused_filename
        
    def print_anomaly_statistics(self, combined_features, threshold):
        """Print anomaly detection statistics"""
        normal_mask = (combined_features['dataset'] == 'train') | (combined_features['dataset'] == 'validation')
        normal_anomaly_scores = combined_features[normal_mask]['reconstruction_error']
        
        test_mask = combined_features['dataset'] == 'test'
        test_data = combined_features[test_mask]
        
        print(f"Normal Traffic (Train + Validation) Statistics:")
        print(f"  Mean Reconstruction Error: {np.mean(normal_anomaly_scores):.6f}")
        print(f"  MAE (Normal Traffic):      {np.mean(np.abs(normal_anomaly_scores)):.6f}")
        print(f"  Standard Deviation:        {np.std(normal_anomaly_scores):.6f}")
        print(f"  Anomaly Threshold:         {threshold:.6f}")
        print(f"\nTest Set Results:")
        print(f"  Total test windows:        {len(test_data)}")
        print(f"  Detected anomalies:        {np.sum(test_data['is_anomaly'])}")
        print(f"  Anomaly rate:              {(np.sum(test_data['is_anomaly'])/len(test_data)*100):.2f}%")
        
    def print_threshold_comparison(self, test_mse, thresholds):
        """Print threshold comparison statistics"""
        print("THRESHOLD COMPARISON ON TEST DATA")
        print("="*60)
        for name, thresh in thresholds.items():
            test_anomalies = np.sum(test_mse > thresh)
            test_rate = (test_anomalies / len(test_mse)) * 100
            print(f"{name:20s}: {test_anomalies:3d}/{len(test_mse)} ({test_rate:5.2f}%) - Threshold: {thresh:.6f}")
        print("="*60)

    def _plot_train_validation_splits(self, combined_features, threshold, model_name="autoencoder"):
        """Create separate plots for train, validation, and horizon datasets"""
        
        # Plot training data
        self._plot_dataset_split(combined_features, threshold, 'train', model_name)
        
        # Plot validation data
        self._plot_dataset_split(combined_features, threshold, 'validation', model_name)
        
        # Plot horizon data if available
        if 'horizon' in combined_features['dataset'].values:
            self._plot_dataset_split(combined_features, threshold, 'horizon', model_name)
    
    def _plot_dataset_split(self, combined_features, threshold, dataset_split, model_name="autoencoder"):
        """Create a plot for a specific dataset split (train/validation)"""
        # Filter data for the specific dataset split
        split_mask = combined_features['dataset'] == dataset_split
        split_data = combined_features[split_mask]
        
        if len(split_data) == 0:
            print(f"Warning: No data found for {dataset_split} dataset")
            return None
        
        # Use the common plotting function with dataset-specific parameters
        color_map = {
            'train': 'blue',
            'validation': 'orange', 
            'test': 'green',
            'horizon': 'purple'
        }
        color = color_map.get(dataset_split, 'gray')
        title = f'Network Traffic Anomaly Detection - {dataset_split.capitalize()} Dataset'
        filename = f"anomaly_detection_{dataset_split}.png"
        
        return self._create_anomaly_plot(
            split_data, combined_features, threshold, 
            title, filename, color, dataset_split.capitalize()
        )
    
    def _create_anomaly_plot(self, plot_data, combined_features, threshold, title, filename, 
                           line_color='green', data_label='Anomaly Scores', attack_periods=None):
        """Common function to create anomaly detection plots with broken y-axis and ground truth bar"""
        # Ensure timestamps are datetime objects
        timestamps = pd.to_datetime(plot_data['timestamp'])
        anomaly_scores = plot_data['reconstruction_error']
        anomalies_mask = plot_data['is_anomaly']
        
        # Determine best legend position - default to upper left, switch to upper right if anomalies on left
        legend_loc = 'upper left'
        if anomalies_mask.any():
            anomaly_timestamps = timestamps[anomalies_mask]
            time_range = timestamps.max() - timestamps.min()
            midpoint = timestamps.min() + time_range / 2
            # Count anomalies in left half vs right half
            left_anomalies = (anomaly_timestamps <= midpoint).sum()
            right_anomalies = (anomaly_timestamps > midpoint).sum()
            # If more anomalies on the left, put legend on the right
            if left_anomalies > right_anomalies:
                legend_loc = 'upper right'
        
        # Calculate the break points for the y-axis
        # Bottom section: focused on threshold region
        normal_mask = (combined_features['dataset'] == 'train') | (combined_features['dataset'] == 'validation')
        normal_anomaly_scores = combined_features[normal_mask]['reconstruction_error']
        
        if len(normal_anomaly_scores) > 0:
            mse_plus_std = np.mean(normal_anomaly_scores) + np.std(normal_anomaly_scores)
            max_threshold = max(threshold, mse_plus_std)
        else:
            max_threshold = threshold
        
        # Define y-axis ranges
        bottom_ylim = (-0.005, max_threshold * 4)  # Focused view near threshold
        
        # Find max anomaly score for top section
        max_score = anomaly_scores.max()
        
        # Only use broken axis if there are significant spikes above the bottom view
        use_broken_axis = max_score > bottom_ylim[1] * 1.5
        
        if use_broken_axis:
            # Create figure with three subplots - top (peaks), bottom (focused), ground truth bar
            fig, (ax_top, ax_bottom, ax_gt) = plt.subplots(3, 1, figsize=(20, 12), 
                                                  gridspec_kw={'height_ratios': [2, 8, 0.5], 'hspace': 0.02},
                                                  sharex=True)
            
            # Top section: show peaks
            top_ylim = (bottom_ylim[1] * 1.2, max_score * 1.1)
            
            # Plot on both axes
            for ax in [ax_top, ax_bottom]:
                # Highlight detected anomaly periods (red)
                self._add_anomaly_highlights(ax, timestamps, anomalies_mask)
                
                # Plot anomaly scores
                ax.step(timestamps, anomaly_scores, where='post',
                        color=line_color, alpha=0.8, linewidth=1.5, 
                        label=f'{data_label}', zorder=5)
                
                # Add threshold line
                self._add_threshold_lines(ax, combined_features, threshold)
            
            # Set y-limits for each section
            ax_top.set_ylim(top_ylim)
            ax_bottom.set_ylim(bottom_ylim)
            
            # Hide the spines between the two plots
            ax_top.spines['bottom'].set_visible(False)
            ax_bottom.spines['top'].set_visible(False)
            ax_top.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            ax_top.tick_params(axis='y', labelsize=12)
            ax_bottom.tick_params(axis='y', labelsize=12)
            
            # Add break indicators (diagonal lines)
            d = 0.015  # Size of diagonal lines
            kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)  # Bottom-left diagonal
            ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)  # Bottom-right diagonal
            
            kwargs.update(transform=ax_bottom.transAxes)
            ax_bottom.plot((-d, +d), (1-d, 1+d), **kwargs)  # Top-left diagonal
            ax_bottom.plot((1-d, 1+d), (1-d, 1+d), **kwargs)  # Top-right diagonal
            
            # Configure appearance
            ax_top.set_title(title, fontsize=18, fontweight='bold')
            ax_top.grid(True, alpha=0.3)
            ax_top.legend(fontsize=16, loc=legend_loc)
            
            ax_bottom.set_ylabel('Anomaly Score (Reconstruction Error)', fontsize=14, fontweight='bold')
            ax_bottom.grid(True, alpha=0.3)
            ax_bottom.tick_params(axis='x', labelbottom=False)
            
        else:
            # Regular plot without broken axis
            fig, (ax_bottom, ax_gt) = plt.subplots(2, 1, figsize=(20, 11), 
                                                  gridspec_kw={'height_ratios': [20, 1], 'hspace': 0.05},
                                                  sharex=True)
            
            # Highlight detected anomaly periods (red)
            self._add_anomaly_highlights(ax_bottom, timestamps, anomalies_mask)
            
            # Plot anomaly scores
            ax_bottom.step(timestamps, anomaly_scores, where='post',
                    color=line_color, alpha=0.8, linewidth=1.5, 
                    label=f'{data_label}', zorder=5)
            
            # Add threshold line
            self._add_threshold_lines(ax_bottom, combined_features, threshold)
            
            # Configure main plot appearance
            self._configure_plot_appearance(ax_bottom, title, legend_loc=legend_loc)
        
        # Add ground truth bar at the bottom
        self._add_ground_truth_bar(ax_gt, timestamps, attack_periods)
        
        # Save plot
        full_filename = os.path.join(self.results_dir, filename)
        plt.savefig(full_filename, dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        gc.collect()
        
        print(f"{data_label} plot saved to: {full_filename}")
        return full_filename
    
    def _add_anomaly_highlights(self, ax, timestamps, anomalies_mask):
        """Add detected anomaly period highlights to the plot (red)"""
        anomaly_timestamps = timestamps[anomalies_mask]
        for i, anomaly_time in enumerate(anomaly_timestamps):
            window_start = anomaly_time - pd.Timedelta(minutes=2.5)
            window_end = anomaly_time + pd.Timedelta(minutes=2.5)
            ax.axvspan(window_start, window_end, alpha=0.25, color='red', 
                       label='Detected Anomaly Period' if i == 0 else "", zorder=1)
    
    def _add_ground_truth_bar(self, ax, timestamps, attack_periods):
        """Add ground truth horizontal bar at the bottom of the plot"""
        # Set up the ground truth bar
        ax.set_ylim(0, 1)
        ax.set_ylabel('GT', fontsize=12, fontweight='bold')
        ax.set_yticks([])
        ax.set_facecolor('white')
        
        # Get time range from data
        time_min = timestamps.min()
        time_max = timestamps.max()
        
        # Draw background (no attack)
        ax.axhspan(0, 1, color='lightgray', alpha=0.3)
        
        if attack_periods:
            # Convert attack periods to datetime
            for start, end in attack_periods:
                start_dt = pd.to_datetime(start)
                end_dt = pd.to_datetime(end)
                
                # Check if timestamps are timezone-aware
                if len(timestamps) > 0 and hasattr(timestamps.iloc[0], 'tz') and timestamps.iloc[0].tz is not None:
                    timezone = timestamps.iloc[0].tz
                    start_dt = start_dt.tz_localize(timezone)
                    end_dt = end_dt.tz_localize(timezone)
                
                # Draw ground truth period as green bar
                ax.axvspan(start_dt, end_dt, color='green', alpha=0.7)
        
        # Configure x-axis for the ground truth bar with dynamic interval
        time_range_hours = (time_max - time_min).total_seconds() / 3600
        if time_range_hours > 120:  # More than 5 days
            major_interval = 24  # Show every day
            minor_interval = 6
        elif time_range_hours > 48:  # More than 2 days
            major_interval = 12
            minor_interval = 3
        else:
            major_interval = 4
            minor_interval = 1
        
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=major_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=minor_interval))
        ax.tick_params(axis='x', labelsize=12, rotation=0)
        ax.set_xlabel('Date Time (Aggregated Windows)', fontsize=14, fontweight='bold')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
    
    def _add_threshold_lines(self, ax, combined_features, threshold):
        # Add threshold line
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Anomaly Threshold: {threshold:.6f}')
    
    def _configure_plot_appearance(self, ax, title, legend_loc='upper right'):
        """Configure common plot appearance settings"""
        ax.set_ylabel('Anomaly Score (Reconstruction Error)', fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        
        # Keep original auto-scaling for the main plot to show all data
        ax.set_ylim(bottom=-0.01)  # Only set bottom limit, let top auto-scale
        
        # Hide x-axis labels on main plot (will be shown on GT bar)
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(axis='y', labelsize=12)
        
        ax.legend(fontsize=16, loc=legend_loc)
        ax.grid(True, alpha=0.3)
