import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os


class AnomalyVisualizer:
    def __init__(self, dataset_name=None, results_dir=None, model_name="autoencoder"):
        if results_dir is None:
            if dataset_name:
                self.results_dir = f"./results/{dataset_name}/{model_name}"
            else:
                self.results_dir = f"./results/{model_name}"
        else:
            self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def plot_anomaly_detection(self, combined_features, threshold, model_name="autoencoder"):
        """Create anomaly detection visualization focused on test dataset"""
        test_mask = combined_features['dataset'] == 'test'
        test_data = combined_features[test_mask]
        
        # Create main test dataset plot using common function
        self._create_anomaly_plot(
            test_data, combined_features, threshold,
            'Network Traffic Anomaly Detection',
            'anomaly_detection.png',
            'green', 'Anomaly Scores'
        )
        
        # Create a second plot focused on the threshold region for better visibility
        self._plot_threshold_focused_view(combined_features, threshold, model_name)
        
        # Generate separate plots for train and validation datasets
        self._plot_train_validation_splits(combined_features, threshold, model_name)
        
        filename = os.path.join(self.results_dir, "anomaly_detection.png")
        print(f"Anomaly detection plot saved to: {filename}")
        return filename
        
    def _plot_threshold_focused_view(self, combined_features, threshold, model_name="autoencoder"):
        """Create a focused view of the threshold region for better visibility"""
        test_mask = combined_features['dataset'] == 'test'
        test_data = combined_features[test_mask]
        
        plt.figure(figsize=(20, 8))
        
        # Use common helper functions
        timestamps = pd.to_datetime(test_data['timestamp'])
        anomaly_scores = test_data['reconstruction_error']
        anomalies_mask = test_data['is_anomaly']
        
        # Add anomaly highlights and plot data
        self._add_anomaly_highlights(timestamps, anomalies_mask)
        
        plt.plot(timestamps, anomaly_scores, 
                color='green', alpha=0.8, linewidth=1.5, 
                label='Anomaly Scores', zorder=5)
        
        # Add threshold lines
        self._add_threshold_lines(combined_features, threshold)
        
        # Calculate focused y-range around threshold region
        normal_mask = (combined_features['dataset'] == 'train') | (combined_features['dataset'] == 'validation')
        normal_anomaly_scores = combined_features[normal_mask]['reconstruction_error']
        mse_plus_std_anomaly_score = np.mean(normal_anomaly_scores) + np.std(normal_anomaly_scores)
        max_threshold = max(threshold, mse_plus_std_anomaly_score)
        focused_y_limit = max_threshold * 4
        plt.ylim(bottom=-0.005, top=focused_y_limit)
        
        # Configure appearance with focused title
        self._configure_plot_appearance('Network Traffic Anomaly Detection - Threshold Focused View')
        
        focused_filename = os.path.join(self.results_dir, f"anomaly_detection_focused.png")
        plt.savefig(focused_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
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
        """Create separate plots for train and validation datasets"""
        
        # Plot training data
        self._plot_dataset_split(combined_features, threshold, 'train', model_name)
        
        # Plot validation data
        self._plot_dataset_split(combined_features, threshold, 'validation', model_name)
    
    def _plot_dataset_split(self, combined_features, threshold, dataset_split, model_name="autoencoder"):
        """Create a plot for a specific dataset split (train/validation)"""
        # Filter data for the specific dataset split
        split_mask = combined_features['dataset'] == dataset_split
        split_data = combined_features[split_mask]
        
        if len(split_data) == 0:
            print(f"Warning: No data found for {dataset_split} dataset")
            return None
        
        # Use the common plotting function with dataset-specific parameters
        color = 'blue' if dataset_split == 'train' else 'orange'
        title = f'Network Traffic Anomaly Detection - {dataset_split.capitalize()} Dataset'
        filename = f"anomaly_detection_{dataset_split}.png"
        
        return self._create_anomaly_plot(
            split_data, combined_features, threshold, 
            title, filename, color, dataset_split.capitalize()
        )
    
    def _create_anomaly_plot(self, plot_data, combined_features, threshold, title, filename, 
                           line_color='green', data_label='Anomaly Scores'):
        """Common function to create anomaly detection plots"""
        plt.figure(figsize=(20, 10))
        
        # Ensure timestamps are datetime objects
        timestamps = pd.to_datetime(plot_data['timestamp'])
        anomaly_scores = plot_data['reconstruction_error']
        anomalies_mask = plot_data['is_anomaly']
        
        # Highlight anomaly periods
        self._add_anomaly_highlights(timestamps, anomalies_mask)
        
        # Plot anomaly scores
        plt.plot(timestamps, anomaly_scores, 
                color=line_color, alpha=0.8, linewidth=1.5, 
                label=f'{data_label}', zorder=5)
        
        # Add threshold and reference lines
        self._add_threshold_lines(combined_features, threshold)
        
        # Configure plot appearance
        self._configure_plot_appearance(title)
        
        # Save plot
        full_filename = os.path.join(self.results_dir, filename)
        plt.savefig(full_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{data_label} plot saved to: {full_filename}")
        return full_filename
    
    def _add_anomaly_highlights(self, timestamps, anomalies_mask):
        """Add anomaly period highlights to the plot"""
        anomaly_timestamps = timestamps[anomalies_mask]
        for i, anomaly_time in enumerate(anomaly_timestamps):
            window_start = anomaly_time - pd.Timedelta(minutes=2.5)
            window_end = anomaly_time + pd.Timedelta(minutes=2.5)
            plt.axvspan(window_start, window_end, alpha=0.25, color='red', 
                       label='Detected Anomaly Period' if i == 0 else "", zorder=1)
    
    def _add_threshold_lines(self, combined_features, threshold):
        """Add threshold and reference lines to the plot"""
        # Add threshold line
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Anomaly Threshold: {threshold:.6f}')
        
        # Calculate statistics for normal data (train + validation)
        normal_mask = (combined_features['dataset'] == 'train') | (combined_features['dataset'] == 'validation')
        normal_anomaly_scores = combined_features[normal_mask]['reconstruction_error']
        
        mae_anomaly_score = np.mean(np.abs(normal_anomaly_scores))
        mse_anomaly_score = np.mean(normal_anomaly_scores)
        mse_plus_std_anomaly_score = np.mean(normal_anomaly_scores) + np.std(normal_anomaly_scores)
        
        plt.axhline(y=mae_anomaly_score, color='purple', linestyle='-.', alpha=0.7, linewidth=1.5, 
                   label=f'MAE: {mae_anomaly_score:.6f}')
        plt.axhline(y=mse_anomaly_score, color='orange', linestyle='-.', alpha=0.7, linewidth=1.5, 
                   label=f'MSE: {mse_anomaly_score:.6f}')
        plt.axhline(y=mse_plus_std_anomaly_score, color='brown', linestyle='-.', alpha=0.7, linewidth=1.5, 
                   label=f'MSE + STD: {mse_plus_std_anomaly_score:.6f}')
    
    def _configure_plot_appearance(self, title):
        """Configure common plot appearance settings"""
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Anomaly Score (Reconstruction Error)', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Keep original auto-scaling for the main plot to show all data
        plt.ylim(bottom=-0.01)  # Only set bottom limit, let top auto-scale
        
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        
        plt.legend(fontsize=9, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
