"""
Results management and CSV saving functionality for DDoS detection pipeline.
Handles saving detected anomalies and printing comprehensive analysis summaries.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from framework.utils import get_results_path, get_artifacts_path, get_time_span_description, get_time_window_label


class ResultsManager:
    """Handles saving and managing analysis results"""
    
    def __init__(self, dataset_name: str, model_name: str, time_span: int):
        """
        Initialize ResultsManager
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_span: Time span in seconds
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.time_span = time_span
        self.results_dir = get_results_path(dataset_name, model_name, time_span, "models")
    
    def save_anomalies_to_csv(self, combined_features: pd.DataFrame, threshold: float) -> str:
        """
        Save detected anomalies to a CSV file
        
        Args:
            combined_features: DataFrame containing all features and anomaly detection results
            threshold: Threshold value used for anomaly detection
            
        Returns:
            Path to the saved CSV file
        """
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Filter only the detected anomalies
        anomalies_df = combined_features[combined_features['is_anomaly'] == True].copy()
        
        # Remove rows with NaN reconstruction_error (for temporal models)
        anomalies_df = anomalies_df.dropna(subset=['reconstruction_error'])
        
        # Select relevant columns for the anomalies CSV
        output_columns = [
            'timestamp', 'reconstruction_error', 'dataset', 'is_anomaly'
        ]
        
        # Add additional network flow information if available
        additional_columns = []
        available_columns = anomalies_df.columns.tolist()
        
        # Include basic network flow metrics if available
        flow_columns = [
            'total_flows', 'total_packets', 'total_bytes',
            'src_ip_count', 'dst_ip_count', 'src_port_count', 'dst_port_count',
            'tcp_flows', 'udp_flows', 'icmp_flows',
            'avg_packet_size', 'avg_flow_duration', 'max_packet_size'
        ]
        
        for col in flow_columns:
            if col in available_columns:
                additional_columns.append(col)
        
        # Include top source/destination IPs and ports if available
        ip_port_columns = [col for col in available_columns if 
                          col.startswith(('top_src_ip_', 'top_dst_ip_', 'top_src_port_', 'top_dst_port_'))]
        additional_columns.extend(ip_port_columns[:10])  # Limit to top 10 to avoid too many columns
        
        # Include protocol statistics if available
        protocol_columns = [col for col in available_columns if 
                           col.endswith(('_rate', '_ratio', '_percentage')) and 
                           not col.startswith('top_')]
        additional_columns.extend(protocol_columns[:10])  # Limit to avoid too many columns
        
        final_columns = output_columns + additional_columns
        
        # Select only existing columns
        final_columns = [col for col in final_columns if col in available_columns]
        
        anomalies_output = anomalies_df[final_columns].copy()
        
        # Sort by timestamp and reconstruction error (highest errors first)
        anomalies_output = anomalies_output.sort_values(['timestamp', 'reconstruction_error'], ascending=[True, False])
        
        # Add metadata columns
        anomalies_output.insert(0, 'model_name', self.model_name)
        anomalies_output.insert(1, 'threshold_used', threshold)
        anomalies_output.insert(2, 'anomaly_severity', 
                               pd.cut(anomalies_output['reconstruction_error'], 
                                     bins=[threshold, threshold*2, threshold*5, float('inf')],
                                     labels=['Low', 'Medium', 'High'],
                                     include_lowest=True))
        
        # Save to CSV with time span information
        csv_filename = os.path.join(self.results_dir, f"anomalies_detected.csv")
        anomalies_output.to_csv(csv_filename, index=False)
        
        # Print summary statistics
        self._print_anomaly_summary(anomalies_output, csv_filename)
        
        return csv_filename
    
    def _print_anomaly_summary(self, anomalies_output: pd.DataFrame, csv_filename: str):
        """
        Print summary statistics for detected anomalies
        
        Args:
            anomalies_output: DataFrame containing detected anomalies
            csv_filename: Path to the saved CSV file
        """
        print(f"Anomalies saved to: {csv_filename}")
        print(f"Total anomalies detected: {len(anomalies_output)}")
        print(f"Anomalies by dataset:")
        for dataset in ['train', 'validation', 'test', 'horizon']:
            count = len(anomalies_output[anomalies_output['dataset'] == dataset])
            if count > 0:  # Only show datasets that have data
                print(f"  {dataset}: {count} anomalies")
        
        if len(anomalies_output) > 0:
            # print(f"Severity distribution:")
            severity_counts = anomalies_output['anomaly_severity'].value_counts()
            for severity in ['Low', 'Medium', 'High']:
                count = severity_counts.get(severity, 0)
                percentage = (count / len(anomalies_output)) * 100 if len(anomalies_output) > 0 else 0
                # print(f"  {severity}: {count} ({percentage:.1f}%)")
            
            print(f"Top 5 highest anomaly scores:")
            top_anomalies = anomalies_output.nlargest(5, 'reconstruction_error')
            for _, row in top_anomalies.iterrows():
                print(f"  {row['timestamp']}: {row['reconstruction_error']:.6f} ({row['dataset']} set)")
    
    def print_analysis_completion_summary(self, dataset_config: Dict):
        """
        Print final analysis completion summary
        
        Args:
            dataset_config: Configuration dictionary for the dataset
        """
        time_window_label = get_time_window_label(self.time_span)
        time_desc = get_time_span_description(self.time_span)
        
        print(f"\nAnalysis completed. Results saved to ./results/{self.dataset_name}/{time_window_label}/")
        print(f"Features saved to ./datasets/{self.dataset_name}/features/")
        print(f"Dataset used: {self.dataset_name} ({dataset_config['description']})")
        print(f"Time window configuration: {time_desc} ({self.time_span} seconds)")
    
    def save_analysis_metadata(self, analysis_results: Dict, additional_config: Optional[Dict] = None):
        """
        Save metadata about the analysis to a JSON file in the artifacts directory
        
        Args:
            analysis_results: Dictionary containing analysis results and metadata
            additional_config: Optional dictionary with additional configuration details
        """
        # Get artifacts directory path
        artifacts_dir = get_artifacts_path(self.dataset_name, self.model_name, self.time_span)
        os.makedirs(artifacts_dir, exist_ok=True)
        
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'time_span': self.time_span,
            'time_window_label': get_time_window_label(self.time_span),
            'results_directory': self.results_dir,
            'artifacts_directory': artifacts_dir,
            **analysis_results
        }
        
        # Add additional configuration if provided
        if additional_config:
            metadata['configuration'] = additional_config

        metadata_file = os.path.join(artifacts_dir, "analysis.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"Analysis metadata saved to: {metadata_file}")
    
    def get_results_directory(self) -> str:
        """Get the results directory path"""
        return self.results_dir
    
    def print_summary_statistics(self, combined_features: pd.DataFrame, threshold: float):
        """
        Print comprehensive summary statistics for the analysis
        
        Args:
            combined_features: DataFrame containing all features and results
            threshold: Threshold used for anomaly detection
        """
        print(f"\n{'='*60}")
        print("ANALYSIS SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        # Overall statistics
        total_samples = len(combined_features)
        total_anomalies = len(combined_features[combined_features['is_anomaly'] == True])
        anomaly_rate = (total_anomalies / total_samples) * 100 if total_samples > 0 else 0
        
        print(f"Model: {self.model_name}")
        print(f"Dataset: {self.dataset_name}")
        print(f"Time span: {self.time_span} seconds")
        print(f"Threshold used: {threshold:.6f}")
        print(f"Total samples: {total_samples:,}")
        print(f"Total anomalies: {total_anomalies:,} ({anomaly_rate:.2f}%)")
        
        # Per-dataset statistics
        print(f"\nPer-dataset breakdown:")
        for dataset in ['train', 'validation', 'test', 'horizon']:
            subset = combined_features[combined_features['dataset'] == dataset]
            if len(subset) > 0:
                subset_anomalies = len(subset[subset['is_anomaly'] == True])
                subset_rate = (subset_anomalies / len(subset)) * 100 if len(subset) > 0 else 0
                print(f"  {dataset}: {len(subset):,} samples, {subset_anomalies:,} anomalies ({subset_rate:.2f}%)")
        
        # Reconstruction error statistics
        if 'reconstruction_error' in combined_features.columns:
            valid_errors = combined_features['reconstruction_error'].dropna()
            if len(valid_errors) > 0:
                print(f"\nReconstruction error statistics:")
                print(f"  Mean: {valid_errors.mean():.6f}")
                print(f"  Std: {valid_errors.std():.6f}")
                print(f"  Min: {valid_errors.min():.6f}")
                print(f"  Max: {valid_errors.max():.6f}")
                print(f"  Median: {valid_errors.median():.6f}")
                print(f"  95th percentile: {valid_errors.quantile(0.95):.6f}")
                print(f"  99th percentile: {valid_errors.quantile(0.99):.6f}")