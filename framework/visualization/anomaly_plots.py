import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import os


class AnomalyVisualizer:
    def __init__(self, results_dir="./results/autoencoder"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def plot_anomaly_detection(self, combined_features, threshold, model_name="autoencoder"):
        """Create anomaly detection visualization focused on test dataset"""
        plt.figure(figsize=(20, 10))
        
        test_mask = combined_features['dataset'] == 'test'
        test_data = combined_features[test_mask]
        
        test_timestamps = test_data['timestamp']
        test_anomaly_scores = test_data['reconstruction_error']
        test_anomalies_mask = test_data['is_anomaly']
        
        anomaly_timestamps = test_timestamps[test_anomalies_mask]
        for i, anomaly_time in enumerate(anomaly_timestamps):
            window_start = anomaly_time - pd.Timedelta(minutes=2.5)
            window_end = anomaly_time + pd.Timedelta(minutes=2.5)
            plt.axvspan(window_start, window_end, alpha=0.25, color='red', 
                       label='Detected Anomaly Period' if i == 0 else "", zorder=1)
        
        plt.plot(test_timestamps, test_anomaly_scores, 
                color='green', alpha=0.8, linewidth=1.5, 
                label='Anomaly Scores', zorder=5)
        
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Anomaly Threshold: {threshold:.6f}')
        
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
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Anomaly Score (Reconstruction Error)', fontsize=12)
        plt.title('Network Traffic Anomaly Detection', fontsize=14, fontweight='bold')
        
        # Keep original auto-scaling for the main plot to show all data
        plt.ylim(bottom=-0.01)  # Only set bottom limit, let top auto-scale
        
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        
        plt.legend(fontsize=9, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filename = os.path.join(self.results_dir, f"anomaly_detection.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a second plot focused on the threshold region for better visibility
        self._plot_threshold_focused_view(combined_features, threshold, model_name)
        
        print(f"Anomaly detection plot saved to: {filename}")
        return filename
        
    def _plot_threshold_focused_view(self, combined_features, threshold, model_name="autoencoder"):
        """Create a focused view of the threshold region for better visibility"""
        plt.figure(figsize=(20, 8))
        
        test_mask = combined_features['dataset'] == 'test'
        test_data = combined_features[test_mask]
        
        test_timestamps = test_data['timestamp']
        test_anomaly_scores = test_data['reconstruction_error']
        test_anomalies_mask = test_data['is_anomaly']
        
        # Calculate statistics for focused view
        normal_mask = (combined_features['dataset'] == 'train') | (combined_features['dataset'] == 'validation')
        normal_anomaly_scores = combined_features[normal_mask]['reconstruction_error']
        
        mae_anomaly_score = np.mean(np.abs(normal_anomaly_scores))
        mse_anomaly_score = np.mean(normal_anomaly_scores)
        mse_plus_std_anomaly_score = np.mean(normal_anomaly_scores) + np.std(normal_anomaly_scores)
        
        # Plot data with focused y-range
        plt.plot(test_timestamps, test_anomaly_scores, 
                color='green', alpha=0.8, linewidth=1.5, 
                label='Anomaly Scores', zorder=5)
        
        # Highlight anomaly periods
        anomaly_timestamps = test_timestamps[test_anomalies_mask]
        for i, anomaly_time in enumerate(anomaly_timestamps):
            window_start = anomaly_time - pd.Timedelta(minutes=2.5)
            window_end = anomaly_time + pd.Timedelta(minutes=2.5)
            plt.axvspan(window_start, window_end, alpha=0.25, color='red', 
                       label='Detected Anomaly Period' if i == 0 else "", zorder=1)
        
        # Add threshold lines
        plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, linewidth=2, 
                   label=f'Anomaly Threshold: {threshold:.6f}')
        plt.axhline(y=mae_anomaly_score, color='purple', linestyle='-.', alpha=0.7, linewidth=1.5, 
                   label=f'MAE: {mae_anomaly_score:.6f}')
        plt.axhline(y=mse_anomaly_score, color='orange', linestyle='-.', alpha=0.7, linewidth=1.5, 
                   label=f'MSE: {mse_anomaly_score:.6f}')
        plt.axhline(y=mse_plus_std_anomaly_score, color='brown', linestyle='-.', alpha=0.7, linewidth=1.5, 
                   label=f'MSE + STD: {mse_plus_std_anomaly_score:.6f}')
        
        # Set focused y-range around threshold region
        max_threshold = max(threshold, mse_plus_std_anomaly_score)
        # Focus on threshold region but allow some visibility above
        focused_y_limit = max_threshold * 4
        plt.ylim(bottom=-0.005, top=focused_y_limit)
        
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Anomaly Score (Reconstruction Error)', fontsize=12)
        plt.title('Network Traffic Anomaly Detection - Threshold Focused View', fontsize=14, fontweight='bold')
        
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        
        plt.legend(fontsize=9, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
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
