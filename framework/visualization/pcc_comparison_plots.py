"""
PCC Comparison Visualization Module

Generates AUPRC comparison charts across different PCC (Pearson Correlation Coefficient)
threshold versions for the same model/dataset/time span configuration.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tabulate import tabulate


class PCCComparisonVisualizer:
    """
    Visualizes AUPRC performance comparisons across different PCC threshold versions.
    
    This class collects precision-recall curve data from multiple version directories
    and generates comparison charts showing how feature correlation filtering (PCC threshold)
    affects model performance.
    """
    
    # Color map for PCC thresholds
    PCC_COLORMAP = {
        0.50: '#e74c3c',  # Red
        0.70: '#f39c12',  # Orange
        0.80: '#3498db',  # Blue
        0.90: '#2ecc71'   # Green
    }
    
    def __init__(self, results_base_dir: str = './results', output_dir: str = './results/pcc_comparison'):
        """
        Initialize the PCC Comparison Visualizer.
        
        Args:
            results_base_dir: Base directory containing results and versions
            output_dir: Directory to save comparison plots and reports
        """
        self.results_base_dir = Path(results_base_dir)
        self.versions_dir = self.results_base_dir / 'versions'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PCC Comparison Visualizer initialized")
        print(f"Results base: {self.results_base_dir}")
        print(f"Versions dir: {self.versions_dir}")
        print(f"Output dir: {self.output_dir}")
    
    def discover_pcc_versions(self) -> List[Dict[str, any]]:
        """
        Automatically discover all available PCC versions in the versions directory.
        
        Returns:
            List of dicts containing version info: name, threshold, n_iter
        """
        if not self.versions_dir.exists():
            print(f"Warning: Versions directory does not exist: {self.versions_dir}")
            return []
        
        versions = []
        for version_dir in self.versions_dir.iterdir():
            if version_dir.is_dir() and '_pcc_' in version_dir.name:
                try:
                    # Parse version name: {int}_{decimal}_pcc_n_iter_{n}
                    # Example: 0_90_pcc_n_iter_5 -> threshold=0.90, n_iter=5
                    parts = version_dir.name.split('_')
                    
                    if len(parts) >= 5 and parts[2] == 'pcc':
                        # Format: 0_90_pcc_n_iter_5
                        # parts[0] = '0', parts[1] = '90', parts[2] = 'pcc'
                        threshold = float(f"{parts[0]}.{parts[1]}")
                        
                        # Find n_iter value
                        n_iter = int(parts[-1])
                        
                        versions.append({
                            'name': version_dir.name,
                            'threshold': threshold,
                            'n_iter': n_iter,
                            'path': version_dir
                        })
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse version directory name: {version_dir.name} - {e}")
                    continue
        
        # Sort by threshold
        versions.sort(key=lambda x: x['threshold'])
        
        print(f"\nDiscovered {len(versions)} PCC versions:")
        for v in versions:
            print(f"  - {v['name']} (threshold={v['threshold']}, n_iter={v['n_iter']})")
        
        return versions
    
    def collect_pcc_version_curves(self, 
                                    dataset: str,
                                    model: str, 
                                    window: str,
                                    pcc_versions: Optional[List[str]] = None) -> List[Dict]:
        """
        Collect PR curve data for same model/dataset/window across PCC versions.
        
        Args:
            dataset: Dataset name (e.g., 'itp-multivector-udp-100gbps-peak')
            model: Model name (e.g., 'autoencoder')
            window: Time window (e.g., '1seconds')
            pcc_versions: Optional list of specific version names to use.
                         If None, auto-discovers all available versions.
        
        Returns:
            List of curve data dicts with added 'pcc_version' and 'pcc_threshold' fields
        """
        # Discover or use specified versions
        if pcc_versions is None:
            available_versions = self.discover_pcc_versions()
            pcc_versions = [v['name'] for v in available_versions]
        else:
            available_versions = self.discover_pcc_versions()
            # Validate specified versions exist
            available_names = [v['name'] for v in available_versions]
            pcc_versions = [v for v in pcc_versions if v in available_names]
        
        curves = []
        
        for version_name in pcc_versions:
            # Construct path to PR curve data
            file_path = (self.versions_dir / version_name / dataset / 
                        window / 'models' / model / 'evaluation' / 
                        'precision_recall_curve_data.json')
            
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Extract PCC threshold from version name
                    # Format: {int}_{decimal}_pcc_n_iter_{n}
                    # Example: 0_90_pcc_n_iter_5 -> 0.90
                    parts = version_name.split('_')
                    if len(parts) >= 3 and parts[2] == 'pcc':
                        # parts[0] = '0', parts[1] = '90', parts[2] = 'pcc'
                        pcc_threshold = float(f"{parts[0]}.{parts[1]}")
                    else:
                        pcc_threshold = None
                    
                    data['pcc_version'] = version_name
                    data['pcc_threshold'] = pcc_threshold
                    data['dataset'] = dataset
                    data['model'] = model
                    data['window'] = window
                    
                    curves.append(data)
                    print(f"  Loaded: {version_name} (PCC={pcc_threshold}, AP={data.get('average_precision', 'N/A')})")
                    
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Error reading {file_path}: {e}")
            else:
                print(f"  Not found: {file_path}")
        
        if not curves:
            print(f"\nWarning: No PR curve data found for {dataset}/{window}/{model}")
        
        return curves
    
    def plot_pcc_auprc_comparison(self, 
                                   dataset: str,
                                   model: str, 
                                   window: str,
                                   pcc_versions: Optional[List[str]] = None,
                                   save: bool = True) -> Optional[str]:
        """
        Generate AUPRC comparison chart for a single configuration across PCC versions.
        
        Args:
            dataset: Dataset name
            model: Model name
            window: Time window (e.g., '1seconds')
            pcc_versions: Optional list of specific version names
            save: Whether to save the plot to disk
        
        Returns:
            Path to saved plot file, or None if no data available
        """
        print(f"\n{'='*80}")
        print(f"Generating PCC AUPRC Comparison")
        print(f"Dataset: {dataset}")
        print(f"Model: {model}")
        print(f"Window: {window}")
        print(f"{'='*80}\n")
        
        # Collect curve data
        curves = self.collect_pcc_version_curves(dataset, model, window, pcc_versions)
        
        if not curves:
            print("No data available for comparison.")
            return None
        
        # Sort by PCC threshold for consistent ordering
        curves.sort(key=lambda x: x['pcc_threshold'])
        
        # Create figure with scientific styling
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot each PCC version's curve
        for curve_data in curves:
            precision = np.array(curve_data['precision'])
            recall = np.array(curve_data['recall'])
            ap_score = curve_data['average_precision']
            pcc_threshold = curve_data['pcc_threshold']
            
            # Get color for this PCC threshold
            color = self.PCC_COLORMAP.get(pcc_threshold, '#95a5a6')
            
            # Create label with PCC threshold and AP score
            label = f"PCC={pcc_threshold:.2f} (AP={ap_score:.4f})"
            
            ax.plot(recall, precision, 
                   label=label, 
                   linewidth=2.5, 
                   alpha=0.85,
                   color=color)
        
        # Scientific styling (matching evaluation_plots.py patterns)
        ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
        
        time_value = window.replace('seconds', 's')
        title = f'AUPRC: PCC Threshold Comparison\n'
        # title += f'{model.replace("_", " ").title()} - {dataset} - {time_value}'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.legend(loc='lower left', fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        # Save plot
        if save:
            filename = f'pr_curve_pcc_comparison_{dataset}_{window}_{model}.png'
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"\nSaved comparison plot: {filepath}")
            plt.close()
            return str(filepath)
        else:
            plt.show()
            return None
    
    def plot_all_pcc_comparisons(self,
                                 datasets: Optional[List[str]] = None,
                                 models: Optional[List[str]] = None,
                                 windows: Optional[List[str]] = None,
                                 pcc_versions: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Generate all AUPRC comparison charts for specified configurations.
        
        Args:
            datasets: List of dataset names (None = auto-discover)
            models: List of model names (None = auto-discover)
            windows: List of time windows (None = use standard set)
            pcc_versions: List of PCC version names (None = auto-discover)
        
        Returns:
            Dict mapping configuration to list of generated plot paths
        """
        from framework.constants import SUPPORTED_TIME_SPANS
        from framework.models import list_available_models
        from config import DATASETS
        
        # Auto-discover if not specified
        if datasets is None:
            datasets = list(DATASETS.keys())
        
        if models is None:
            models = list_available_models()
        
        if windows is None:
            windows = [f"{ts}seconds" for ts in SUPPORTED_TIME_SPANS]
        
        print(f"\n{'='*80}")
        print(f"Generating All PCC AUPRC Comparisons")
        print(f"{'='*80}")
        print(f"Datasets: {datasets}")
        print(f"Models: {models}")
        print(f"Windows: {windows}")
        print(f"{'='*80}\n")
        
        results = {}
        total = len(datasets) * len(models) * len(windows)
        current = 0
        
        for dataset in datasets:
            for model in models:
                for window in windows:
                    current += 1
                    config_key = f"{dataset}/{window}/{model}"
                    
                    print(f"\n[{current}/{total}] Processing: {config_key}")
                    
                    plot_path = self.plot_pcc_auprc_comparison(
                        dataset=dataset,
                        model=model,
                        window=window,
                        pcc_versions=pcc_versions,
                        save=True
                    )
                    
                    if plot_path:
                        results[config_key] = [plot_path]
        
        print(f"\n{'='*80}")
        print(f"Completed: Generated {len(results)} comparison charts")
        print(f"{'='*80}\n")
        
        return results
    
    def generate_pcc_summary_table(self,
                                   datasets: Optional[List[str]] = None,
                                   models: Optional[List[str]] = None,
                                   windows: Optional[List[str]] = None,
                                   pcc_versions: Optional[List[str]] = None,
                                   export_csv: bool = True,
                                   format: str = 'github') -> pd.DataFrame:
        """
        Generate summary table with AUPRC scores across PCC versions.
        
        Args:
            datasets: List of dataset names (None = auto-discover)
            models: List of model names (None = auto-discover)
            windows: List of time windows (None = use standard set)
            pcc_versions: List of PCC version names (None = auto-discover)
            export_csv: Whether to export results to CSV
            format: Table format for display
        
        Returns:
            DataFrame containing summary statistics
        """
        from framework.constants import SUPPORTED_TIME_SPANS
        from framework.models import list_available_models
        from config import DATASETS
        
        # Auto-discover if not specified
        if datasets is None:
            datasets = list(DATASETS.keys())
        
        if models is None:
            models = list_available_models()
        
        if windows is None:
            windows = [f"{ts}seconds" for ts in SUPPORTED_TIME_SPANS]
        
        print(f"\n{'='*80}")
        print(f"Generating PCC Summary Table")
        print(f"{'='*80}\n")
        
        # Discover available PCC versions
        available_versions = self.discover_pcc_versions()
        if pcc_versions:
            available_versions = [v for v in available_versions if v['name'] in pcc_versions]
        
        rows = []
        
        for dataset in datasets:
            for model in models:
                for window in windows:
                    curves = self.collect_pcc_version_curves(dataset, model, window, 
                                                             [v['name'] for v in available_versions])
                    
                    if not curves:
                        continue
                    
                    row = {
                        'Dataset': dataset,
                        'Model': model,
                        'Window': window
                    }
                    
                    ap_scores = {}
                    for curve in curves:
                        pcc_threshold = curve['pcc_threshold']
                        ap_score = curve['average_precision']
                        row[f'PCC_{pcc_threshold:.2f}'] = ap_score
                        ap_scores[pcc_threshold] = ap_score
                    
                    # Calculate best PCC and improvement
                    if ap_scores:
                        best_pcc = max(ap_scores.items(), key=lambda x: x[1])
                        worst_pcc = min(ap_scores.items(), key=lambda x: x[1])
                        
                        row['Best_PCC'] = best_pcc[0]
                        row['Best_AP'] = best_pcc[1]
                        row['Worst_AP'] = worst_pcc[1]
                        row['Improvement'] = best_pcc[1] - worst_pcc[1]
                        row['Improvement_Pct'] = ((best_pcc[1] - worst_pcc[1]) / worst_pcc[1] * 100) if worst_pcc[1] > 0 else 0
                    
                    rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if df.empty:
            print("No data available for summary table.")
            return df
        
        print(f"\n{'='*80}")
        print("PCC Threshold Comparison Summary")
        print(f"{'='*80}\n")
        print(tabulate(df, headers='keys', tablefmt=format, showindex=False, floatfmt='.4f'))
        print(f"\n{'='*80}\n")
        
        if export_csv:
            csv_path = self.output_dir / 'pcc_comparison_summary.csv'
            df.to_csv(csv_path, index=False)
            print(f"Exported summary to: {csv_path}\n")
        
        print(f"\n{'='*80}")
        print("Aggregate Statistics")
        print(f"{'='*80}\n")
        
        stats = {
            'Total Configurations': len(df),
            'Avg Improvement': df['Improvement'].mean() if 'Improvement' in df.columns else 0,
            'Max Improvement': df['Improvement'].max() if 'Improvement' in df.columns else 0,
            'Avg Improvement %': df['Improvement_Pct'].mean() if 'Improvement_Pct' in df.columns else 0,
        }
        
        for key, value in stats.items():
            print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        
        print(f"\n{'='*80}\n")
        
        return df


def run_pcc_comparison(
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    windows: Optional[List[str]] = None,
    pcc_versions: Optional[List[str]] = None,
    generate_plots: bool = True,
    generate_summary: bool = True,
    export_csv: bool = True,
    summary_format: str = 'github'
):
    """
    Main entry point for PCC comparison analysis.
    
    Args:
        datasets: List of dataset names to analyze
        models: List of model names to analyze
        windows: List of time windows to analyze
        pcc_versions: List of specific PCC version names to compare
        generate_plots: Whether to generate comparison plots
        generate_summary: Whether to generate summary table
        export_csv: Whether to export summary to CSV
        summary_format: Table format for summary display
    """
    visualizer = PCCComparisonVisualizer()
    
    results = {}
    
    if generate_plots:
        print("\n" + "="*80)
        print("GENERATING PCC COMPARISON PLOTS")
        print("="*80 + "\n")
        
        plot_results = visualizer.plot_all_pcc_comparisons(
            datasets=datasets,
            models=models,
            windows=windows,
            pcc_versions=pcc_versions
        )
        results['plots'] = plot_results
    
    if generate_summary:
        print("\n" + "="*80)
        print("GENERATING PCC SUMMARY TABLE")
        print("="*80 + "\n")
        
        summary_df = visualizer.generate_pcc_summary_table(
            datasets=datasets,
            models=models,
            windows=windows,
            pcc_versions=pcc_versions,
            export_csv=export_csv,
            format=summary_format
        )
        results['summary'] = summary_df
    
    return results
