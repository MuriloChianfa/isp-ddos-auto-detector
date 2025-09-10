import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from pathlib import Path
from ..utils import get_results_path


class DatasetFeatureVisualizer:
    """
    Visualizer for creating individual feature charts across train, validation, and test datasets.
    Creates separate plots for each feature and saves them organized by dataset type.
    """
    
    def __init__(self, dataset_name=None, results_dir=None, save_format="png", time_span=300):
        if results_dir is None:
            self.results_dir = get_results_path(dataset_name, None, time_span, "features")
        else:
            self.results_dir = results_dir
        self.save_format = save_format.lower()
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
            
        # 1. Histogram with density curve
        axes[0, 0].hist(clean_data, bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution Histogram')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Mean: {clean_data.mean():.4f}\nStd: {clean_data.std():.4f}\nMin: {clean_data.min():.4f}\nMax: {clean_data.max():.4f}'
        axes[0, 0].text(0.02, 0.98, stats_text, transform=axes[0, 0].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Box plot
        axes[0, 1].boxplot(clean_data, vert=True, patch_artist=True, 
                          boxprops=dict(facecolor='lightcoral', alpha=0.7))
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Time series (if timestamps available)
        if timestamps is not None and len(timestamps) == len(feature_data):
            # Align timestamps with clean data indices
            clean_timestamps = pd.Series(timestamps).iloc[clean_data.index]
            axes[1, 0].plot(clean_timestamps, clean_data.values, linewidth=1, alpha=0.8, color='green')
            axes[1, 0].set_title('Time Series')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].plot(range(len(clean_data)), clean_data.values, linewidth=1, alpha=0.8, color='green')
            axes[1, 0].set_title('Sequential Plot')
            axes[1, 0].set_xlabel('Sample Index')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Q-Q plot for normality assessment
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
        
        # 1. Overlaid histograms
        for i, (dataset_type, data) in enumerate(comparison_data):
            axes[0, 0].hist(data, bins=30, alpha=0.6, label=dataset_type.title(), 
                           density=True, color=colors[i])
        axes[0, 0].set_title('Distribution Comparison')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plots side by side
        box_data = [data for _, data in comparison_data]
        box_labels = [dataset_type.title() for dataset_type, _ in comparison_data]
        axes[0, 1].boxplot(box_data, labels=box_labels, patch_artist=True,
                          boxprops=dict(alpha=0.7))
        axes[0, 1].set_title('Box Plot Comparison')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Statistics comparison table
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
        
        # 4. Violin plots
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
        
    def generate_all_feature_plots(self, features_dict, create_comparisons=True):
        """
        Generate plots for all features across all datasets.
        
        Args:
            features_dict: Dictionary with 'train', 'validation', 'test' keys containing feature DataFrames
            create_comparisons: Whether to create cross-dataset comparison plots
        """
        print("Generating individual feature plots for each dataset...")
        
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
            
        plot_count = 0
        
        # Generate individual plots for each feature in each dataset
        for feature_name in sorted(all_features):
            for dataset_type, df in features_dict.items():
                if feature_name in df.columns:
                    timestamps = timestamps_dict.get(dataset_type, None)
                    filepath = self.plot_feature_distribution(
                        df[feature_name], feature_name, dataset_type, timestamps
                    )
                    if filepath:
                        plot_count += 1
                        print(f"  [{plot_count}/{total_plots}] Generated: {os.path.basename(filepath)} for {dataset_type}")
        
        # Generate comparison plots
        if create_comparisons:
            print("\nGenerating cross-dataset comparison plots...")
            for feature_name in sorted(all_features):
                filepath = self.plot_feature_comparison(features_dict, feature_name)
                if filepath:
                    plot_count += 1
                    print(f"  [{plot_count}/{total_plots}] Generated comparison: {os.path.basename(filepath)}")
        
        print(f"\nFeature visualization completed! Generated {plot_count} plots.")
        print(f"Results saved to: {os.path.abspath(self.results_dir)}")
        
        # Print summary
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
