import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import sys
import time
from ..utils import get_results_path


class DatasetFeatureVisualizer:
    """
    Visualizer for creating individual feature charts across train, validation, and test datasets.
    Creates separate plots for each feature and saves them organized by dataset type.
    """
    
    def __init__(self, dataset_name=None, results_dir=None, save_format="png", time_span=300, max_processes=None):
        if results_dir is None:
            self.results_dir = get_results_path(dataset_name, None, time_span, "features")
        else:
            self.results_dir = results_dir
        self.save_format = save_format.lower()
        self.max_processes = max_processes if max_processes is not None else 12
        self.setup_directories()
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def setup_directories(self):
        """Create directory structure for organizing plots"""
        for dataset_type in ['train', 'validation', 'test', 'horizon']:
            dataset_dir = os.path.join(self.results_dir, dataset_type)
            os.makedirs(dataset_dir, exist_ok=True)
            
    def plot_feature_distribution(self, feature_data, feature_name, dataset_type, timestamps=None):
        """
        Create a comprehensive plot for a single feature showing distribution and time series.
        
        Args:
            feature_data: Series or array with feature values
            feature_name: Name of the feature
            dataset_type: 'train', 'validation', or 'test'
            timestamps: Optional timestamps for time series plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{feature_name} - {dataset_type.title()} Dataset', fontsize=16, fontweight='bold')
        
        # Remove any infinite or NaN values for plotting
        clean_data = pd.Series(feature_data).replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) == 0:
            plt.close(fig)
            print(f"Warning: No valid data for feature {feature_name} in {dataset_type} dataset")
            return None
            
        # Histogram with density curve
        axes[0, 0].hist(clean_data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution Histogram')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {clean_data.mean():.4f}\nStd: {clean_data.std():.4f}\nMin: {clean_data.min():.4f}\nMax: {clean_data.max():.4f}'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Box plot
        axes[0, 1].boxplot(clean_data, vert=True, patch_artist=True, 
                          boxprops=dict(facecolor='lightcoral', alpha=0.7))
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series plots
        if timestamps is not None and len(timestamps) == len(feature_data):
            # Align timestamps with clean data indices
            clean_timestamps = pd.Series(timestamps).iloc[clean_data.index]
            
            # Parse timestamps to datetime objects
            clean_timestamps_parsed = pd.to_datetime(clean_timestamps)
            
            # Plot the time series
            axes[1, 0].plot(clean_timestamps_parsed, clean_data.values, linewidth=1, alpha=0.8, color='green')
            axes[1, 0].set_title('Time Series')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Value')
            
            # Format x-axis dates based on time range
            time_span = clean_timestamps_parsed.max() - clean_timestamps_parsed.min()
            
            # Adjust formatting based on data density and time span
            if time_span.total_seconds() < 3600:  # Less than 1 hour
                # Show minute:second format
                axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                axes[1, 0].xaxis.set_major_locator(mdates.AutoDateLocator())
            elif time_span.total_seconds() < 86400:  # Less than 1 day
                # Show hour:minute format
                axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                axes[1, 0].xaxis.set_major_locator(mdates.HourLocator(interval=max(1, int(time_span.total_seconds() / 7200))))
            else:  # Multiple days
                # Show day and hour
                axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                axes[1, 0].xaxis.set_major_locator(mdates.HourLocator(interval=6))
            
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Improve layout to prevent label cutoff
            plt.setp(axes[1, 0].xaxis.get_majorticklabels(), ha='right')
        else:
            axes[1, 0].plot(range(len(clean_data)), clean_data.values, linewidth=1, alpha=0.8, color='green')
            axes[1, 0].set_title('Sequential Plot')
            axes[1, 0].set_xlabel('Sample Index')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot for normality assessment
        from scipy import stats
        stats.probplot(clean_data, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        filename = f"{feature_name.replace('/', '_').replace(' ', '_')}.{self.save_format}"
        filepath = os.path.join(self.results_dir, dataset_type, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return filepath
        
    def plot_feature_comparison(self, features_dict, feature_name):
        """
        Create a comparison plot for a single feature across all three datasets.
        
        Args:
            features_dict: Dictionary with 'train', 'validation', 'test' keys containing feature DataFrames
            feature_name: Name of the feature to compare
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{feature_name} - Cross-Dataset Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data for comparison
        comparison_data = []
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        for dataset_type in ['train', 'validation', 'test']:
            if dataset_type in features_dict and feature_name in features_dict[dataset_type].columns:
                data = features_dict[dataset_type][feature_name].replace([np.inf, -np.inf], np.nan).dropna()
                comparison_data.append((dataset_type, data))
        
        if len(comparison_data) == 0:
            plt.close(fig)
            print(f"Warning: No valid data for feature {feature_name} across datasets")
            return None
        
        # Overlaid histograms
        for i, (dataset_type, data) in enumerate(comparison_data):
            axes[0, 0].hist(data, bins=30, alpha=0.6, label=dataset_type.title(), 
                           density=True, color=colors[i])
        axes[0, 0].set_title('Distribution Comparison')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plots side by side
        box_data = [data for _, data in comparison_data]
        box_labels = [dataset_type.title() for dataset_type, _ in comparison_data]
        axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True,
                          boxprops=dict(alpha=0.7))
        axes[0, 1].set_title('Box Plot Comparison')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Statistics comparison table
        stats_data = []
        for dataset_type, data in comparison_data:
            stats_data.append({
                'Dataset': dataset_type.title(),
                'Mean': f"{data.mean():.4f}",
                'Std': f"{data.std():.4f}",
                'Min': f"{data.min():.4f}",
                'Max': f"{data.max():.4f}",
                'Count': len(data)
            })
        
        stats_df = pd.DataFrame(stats_data)
        axes[1, 0].axis('tight')
        axes[1, 0].axis('off')
        table = axes[1, 0].table(cellText=stats_df.values, colLabels=stats_df.columns,
                                cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 0].set_title('Statistical Summary')
        
        # Violin plots
        violin_data = [data for _, data in comparison_data]
        violin_labels = [dataset_type.title() for dataset_type, _ in comparison_data]
        axes[1, 1].violinplot(violin_data, positions=range(1, len(violin_data) + 1))
        axes[1, 1].set_xticks(range(1, len(violin_labels) + 1))
        axes[1, 1].set_xticklabels(violin_labels)
        axes[1, 1].set_title('Distribution Shape Comparison')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison plot in a separate comparisons directory
        comparison_dir = os.path.join(self.results_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        filename = f"{feature_name.replace('/', '_').replace(' ', '_')}_comparison.{self.save_format}"
        filepath = os.path.join(comparison_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return filepath

    def _plot_feature_task(self, task_info):
        """
        Worker function for parallel plot generation.
        
        Args:
            task_info: Tuple containing (task_type, feature_name, dataset_type, feature_data, timestamps, progress_info)
                      or (task_type, feature_name, features_dict, progress_info) for comparisons
        
        Returns:
            Tuple: (success: bool, filepath: str, feature_name: str, dataset_type: str)
        """
        try:
            task_type = task_info[0]
            feature_name = task_info[1]
            progress_info = task_info[-1]  # Always the last element
            
            import os
            pid = os.getpid()
            
            if task_type == "distribution":
                dataset_type, feature_data, timestamps = task_info[2], task_info[3], task_info[4]
                
                # Print start message with process info
                if progress_info and 'counter' in progress_info:
                    # Thread-safe increment (Manager.Value handles synchronization automatically)
                    progress_info['counter'].value += 1
                    current_count = progress_info['counter'].value
                    total_count = progress_info['total']
                    progress_pct = (current_count / total_count) * 100
                    print(f"[PID {pid:5}] [{current_count:3d}/{total_count}] ({progress_pct:5.1f}%) Processing: {feature_name} ({dataset_type})")
                    sys.stdout.flush()  # Force immediate output
                
                filepath = self.plot_feature_distribution(feature_data, feature_name, dataset_type, timestamps)
                if filepath:
                    # print(f"[PID {pid:5}] Completed: {os.path.basename(filepath)}")
                    sys.stdout.flush()
                
                return (True, filepath, feature_name, dataset_type)
                
            elif task_type == "comparison":
                features_dict = task_info[2]
                
                # Print start message with process info
                if progress_info and 'counter' in progress_info:
                    # Thread-safe increment (Manager.Value handles synchronization automatically)
                    progress_info['counter'].value += 1
                    current_count = progress_info['counter'].value
                    total_count = progress_info['total']
                    progress_pct = (current_count / total_count) * 100
                    print(f"[PID {pid:5}] [{current_count:3d}/{total_count}] ({progress_pct:5.1f}%) Processing: {feature_name}")
                    sys.stdout.flush()
                
                filepath = self.plot_feature_comparison(features_dict, feature_name)
                
                # Print completion message
                if filepath:
                    # print(f"[PID {pid:5}] Completed comparison: {os.path.basename(filepath)}")
                    sys.stdout.flush()
                
                return (True, filepath, feature_name, "comparison")
                
        except Exception as e:
            print(f"[PID {pid:5}] ERROR generating plot for {feature_name}: {e}")
            sys.stdout.flush()
            dataset_type_for_error = task_info[2] if len(task_info) > 4 and task_type == "distribution" else "comparison"
            return (False, None, feature_name, dataset_type_for_error)
        
    def generate_all_feature_plots(self, features_dict, create_comparisons=True):
        """
        Generate plots for all features across all datasets using parallel processing.
        
        Args:
            features_dict: Dictionary with 'train', 'validation', 'test' keys containing feature DataFrames
            create_comparisons: Whether to create cross-dataset comparison plots
        """
        print(f"Generating individual feature plots for each dataset (using {self.max_processes} processes)...")
        
        # Get all unique feature names
        all_features = set()
        timestamps_dict = {}
        
        for dataset_type, df in features_dict.items():
            feature_cols = [col for col in df.columns if col != 'timestamp']
            all_features.update(feature_cols)
            
            # Store timestamps if available
            if 'timestamp' in df.columns:
                timestamps_dict[dataset_type] = df['timestamp']
        
        total_plots = len(all_features) * len(features_dict)
        if create_comparisons:
            total_plots += len(all_features)
        
        # Prepare tasks for parallel processing
        print("Preparing plot generation tasks...")
        tasks = []
        
        # Individual feature distribution tasks
        distribution_tasks = 0
        for feature_name in sorted(all_features):
            for dataset_type, df in features_dict.items():
                if feature_name in df.columns:
                    timestamps = timestamps_dict.get(dataset_type, None)
                    tasks.append(("distribution", feature_name, dataset_type, df[feature_name], timestamps))
                    distribution_tasks += 1
        
        # Comparison plot tasks
        comparison_tasks = 0
        if create_comparisons:
            for feature_name in sorted(all_features):
                tasks.append(("comparison", feature_name, features_dict))
                comparison_tasks += 1
        
        # Now setup shared progress tracking for multiprocessing
        manager = Manager()
        progress_counter = manager.Value('i', 0)  # Shared integer counter
        progress_info = {
            'counter': progress_counter,
            'total': len(tasks)
        }
        
        # Update tasks to include progress_info
        updated_tasks = []
        for task in tasks:
            if task[0] == "distribution":
                updated_tasks.append(task + (progress_info,))
            else:  # comparison
                updated_tasks.append(task + (progress_info,))
        tasks = updated_tasks
        
        print(f"Prepared {len(tasks)} total tasks:")
        print(f"  - Individual plots: {distribution_tasks} tasks")
        print(f"  - Comparison plots: {comparison_tasks} tasks")
        
        # Determine actual number of processes to use
        actual_processes = min(self.max_processes, len(tasks), cpu_count())
        
        print(f"Processing {len(tasks)} plot generation tasks using {actual_processes} processes...")
        print(f"  - Individual feature plots: {len(tasks) - (len(all_features) if create_comparisons else 0)}")
        if create_comparisons:
            print(f"  - Comparison plots: {len(all_features)}")
        print(f"  - Estimated time: {len(tasks) // actual_processes + 1} batches")
        print("=" * 80)
        print("Starting parallel processing...")
        print("=" * 80)
        
        # Process tasks in parallel
        successful_plots = 0
        failed_plots = 0
        
        start_time = time.time()
        
        if actual_processes > 1:
            with Pool(processes=actual_processes) as pool:
                results = pool.map(self._plot_feature_task, tasks)
        else:
            # Fallback to sequential processing if only 1 process
            print("Using sequential processing (single process)...")
            results = []
            for i, task in enumerate(tasks):
                print(f"  [{i+1:3d}/{len(tasks)}] Processing: {task[1]} ({task[2] if task[0] == 'distribution' else 'comparison'})")
                results.append(self._plot_feature_task(task))
        
        processing_time = time.time() - start_time
        print("=" * 80)
        print(f"Parallel processing completed in {processing_time:.2f} seconds")
        
        # Process results and generate summary
        distribution_count = 0
        comparison_count = 0
        
        print("\n" + "=" * 60)
        print("PROCESSING RESULTS SUMMARY")
        print("=" * 60)
        
        for i, (success, filepath, feature_name, dataset_type) in enumerate(results):
            if success and filepath:
                successful_plots += 1
                if dataset_type == "comparison":
                    comparison_count += 1
                else:
                    distribution_count += 1
            else:
                failed_plots += 1
                print(f"  FAILED: {feature_name} ({dataset_type})")
        
        print(f"\n{'='*60}")
        print(f"FEATURE VISUALIZATION COMPLETED!")
        print(f"{'='*60}")
        print(f"  Successfully generated: {successful_plots} plots")
        print(f"    - Individual feature plots: {distribution_count}")
        print(f"    - Comparison plots: {comparison_count}")
        if failed_plots > 0:
            print(f"  Failed to generate: {failed_plots} plots")
        print(f"  Total processing time: {processing_time:.2f} seconds")
        print(f"  Average time per plot: {processing_time/successful_plots:.3f} seconds") if successful_plots > 0 else None
        print(f"  Results saved to: {os.path.abspath(self.results_dir)}")
        
        # Print detailed summary
        self.print_generation_summary(features_dict, all_features)
        
    def print_generation_summary(self, features_dict, all_features):
        """Print a summary of the generated plots"""
        print("\n" + "="*60)
        print("FEATURE VISUALIZATION SUMMARY")
        print("="*60)
        
        print(f"Total features analyzed: {len(all_features)}")
        print(f"Datasets processed: {', '.join(features_dict.keys())}")
        
        # Print directory structure
        print(f"\nGenerated directory structure:")
        for root, dirs, files in os.walk(self.results_dir):
            level = root.replace(self.results_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            plot_files = [f for f in files if f.endswith(f'.{self.save_format}')]
            if plot_files:
                print(f"{subindent}{len(plot_files)} plot files generated")
        
        print(f"\nPlots are organized as follows:")
        print(f"  • Individual dataset plots: {self.results_dir}/[train|validation|test]/*.{self.save_format}")
        print(f"  • Cross-dataset comparisons: {self.results_dir}/comparisons/*.{self.save_format}")
        
        # Feature categories
        feature_categories = self.categorize_features(all_features)
        print(f"\nFeature categories:")
        for category, features in feature_categories.items():
            print(f"  • {category}: {len(features)} features")
            
    def categorize_features(self, features):
        """Categorize features for better organization"""
        categories = {
            'Traffic Volume': [],
            'Rate Metrics': [],
            'Entropy Measures': [],
            'Protocol Distribution': [],
            'Network Diversity': [],
            'Geographic Features': [],
            'TCP Flags': [],
            'AS Features': [],
            'Connection Patterns': [],
            'Statistical Measures': []
        }
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(word in feature_lower for word in ['total_flows', 'total_packets', 'total_bytes']):
                categories['Traffic Volume'].append(feature)
            elif any(word in feature_lower for word in ['rate', '_per_']):
                categories['Rate Metrics'].append(feature)
            elif 'entropy' in feature_lower:
                categories['Entropy Measures'].append(feature)
            elif any(word in feature_lower for word in ['tcp_ratio', 'udp_ratio', 'icmp_ratio', 'proto']):
                categories['Protocol Distribution'].append(feature)
            elif any(word in feature_lower for word in ['unique_', '_ip']):
                categories['Network Diversity'].append(feature)
            elif any(word in feature_lower for word in ['geo', 'border']):
                categories['Geographic Features'].append(feature)
            elif any(word in feature_lower for word in ['flag', 'syn_', 'ack_', 'fin_', 'rst_', 'psh_', 'urg_']):
                categories['TCP Flags'].append(feature)
            elif 'as' in feature_lower:
                categories['AS Features'].append(feature)
            elif any(word in feature_lower for word in ['port', 'connection']):
                categories['Connection Patterns'].append(feature)
            elif any(word in feature_lower for word in ['avg_', 'max_', 'std_', 'duration']):
                categories['Statistical Measures'].append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
