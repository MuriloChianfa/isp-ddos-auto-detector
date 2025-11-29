"""
Cross-Evaluation Tool for DDoS Detection Models
Generates comprehensive comparisons of ROC-AUC, Precision-Recall and other metrics
across multiple models, datasets, and time windows.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Dict
import json
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CrossEvaluator:
    """Cross-evaluation of multiple models across datasets and time windows"""
    
    def __init__(self, results_base_dir='./results', output_dir='./results/cross_evaluation'):
        """
        Initialize CrossEvaluator
        
        Args:
            results_base_dir: Base directory containing all model results
            output_dir: Directory to save cross-evaluation outputs
        """
        self.results_base_dir = Path(results_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Cross-Evaluator initialized")
        print(f"  Results directory: {self.results_base_dir}")
        print(f"  Output directory: {self.output_dir}")
    
    def collect_metrics_from_analysis(self,
                                     datasets: Optional[List[str]] = None,
                                     windows: Optional[List[str]] = None,
                                     models: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Collect evaluation metrics from analysis.json files
        
        Args:
            datasets: List of dataset names to include (None = all)
            windows: List of time windows to include (e.g., ['1seconds', '60seconds'])
            models: List of model names to include (None = all)
            
        Returns:
            DataFrame with metrics from all matching models
        """
        print("\nCollecting evaluation metrics from analysis files...")
        all_metrics = []
        
        # Search for analysis.json files
        analysis_files = list(self.results_base_dir.rglob("**/artifacts/analysis.json"))
        
        print(f"Found {len(analysis_files)} analysis files")
        
        for file_path in analysis_files:
            try:
                # Skip files in the versions folder
                if 'versions' in file_path.parts:
                    continue
                
                parts = file_path.parts
                
                if 'results' in parts and 'models' in parts:
                    results_idx = parts.index('results')
                    models_idx = parts.index('models')
                    
                    dataset = parts[results_idx + 1]
                    window = parts[results_idx + 2]
                    model = parts[models_idx + 1]
                else:
                    continue
                
                # Apply filters
                if datasets and dataset not in datasets:
                    continue
                if windows and window not in windows:
                    continue
                if models and model not in models:
                    continue
                
                # Read the analysis file
                with open(file_path, 'r') as f:
                    analysis_data = json.load(f)
                
                # Extract evaluation metrics if available
                if 'evaluation_metrics' in analysis_data:
                    metrics = analysis_data['evaluation_metrics']
                    
                    # Create a row for this model
                    row = {
                        'dataset': dataset,
                        'window': window,
                        'model': model,
                        'roc_auc': float(metrics.get('roc_auc_score', 0)),
                        'precision': float(metrics.get('precision', 0)),
                        'recall': float(metrics.get('recall', 0)),
                        'f1_score': float(metrics.get('f1_score', 0)),
                        'f2_score': float(metrics.get('f2_score', 0)),
                        'accuracy': float(metrics.get('accuracy', 0)),
                        'fpr': float(metrics.get('false_positive_rate', 0)),
                        'mcc': float(metrics.get('matthews_corrcoef', 0)),
                        'tp': int(metrics.get('true_positives', 0)),
                        'tn': int(metrics.get('true_negatives', 0)),
                        'fp': int(metrics.get('false_positives', 0)),
                        'fn': int(metrics.get('false_negatives', 0)),
                        'total_samples': int(metrics.get('total_samples', 0)),
                        'attack_periods': int(metrics.get('attack_periods', 0)),
                        'detected_anomalies': int(metrics.get('detected_anomalies', 0)),
                        'threshold': float(analysis_data.get('threshold_used', 0))
                    }
                    
                    all_metrics.append(row)
                    print(f"  Loaded metrics from {dataset}/{window}/{model}")
                
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
                continue
        
        if not all_metrics:
            print("\nNo metrics found!")
            return pd.DataFrame()
        
        metrics_df = pd.DataFrame(all_metrics)
        
        print(f"\nSuccessfully collected metrics from {len(metrics_df)} model configurations")
        print(f"  Datasets: {metrics_df['dataset'].nunique()} ({', '.join(metrics_df['dataset'].unique())})")
        print(f"  Windows: {metrics_df['window'].nunique()} ({', '.join(metrics_df['window'].unique())})")
        print(f"  Models: {metrics_df['model'].nunique()} ({', '.join(metrics_df['model'].unique())})")
        
        return metrics_df
    
    def regenerate_missing_curve_data(self,
                                     datasets: Optional[List[str]] = None,
                                     windows: Optional[List[str]] = None,
                                     models: Optional[List[str]] = None,
                                     auto_regenerate: bool = False):
        """
        Regenerate curve data for evaluations that are missing JSON files
        
        Args:
            datasets: List of dataset names to include (None = all)
            windows: List of time windows to include
            models: List of model names to include
            auto_regenerate: If True, skip confirmation prompt
        """
        print("\nChecking for missing curve data...")
        
        # Find all evaluation directories
        eval_dirs = list(self.results_base_dir.rglob("**/models/*/evaluation"))
        
        missing = []
        for eval_dir in eval_dirs:
            roc_file = eval_dir / "roc_curve_data.json"
            pr_file = eval_dir / "precision_recall_curve_data.json"
            
            # Check if both files exist
            if not (roc_file.exists() and pr_file.exists()):
                parts = eval_dir.parts
                if 'results' in parts and 'models' in parts:
                    results_idx = parts.index('results')
                    models_idx = parts.index('models')
                    
                    dataset = parts[results_idx + 1]
                    window = parts[results_idx + 2]
                    model = parts[models_idx + 1]
                    
                    # Apply filters
                    if datasets and dataset not in datasets:
                        continue
                    if windows and window not in windows:
                        continue
                    if models and model not in models:
                        continue
                    
                    missing.append((dataset, window, model))
        
        if not missing:
            print("  All evaluations have curve data!")
            return
        
        print(f"  Found {len(missing)} evaluations missing curve data")
        
        if not auto_regenerate:
            print(f"\nRegenerate curve data for these {len(missing)} evaluations? (y/n): ", end='')
            
            import sys
            response = input().strip().lower()
            if response != 'y':
                print("  Skipped regeneration")
                return
        else:
            print(f"  Auto-regenerating curve data...")
        
        print(f"\nRegenerating curve data...")
        import subprocess
        
        success = 0
        failed = 0
        
        for idx, (dataset, window, model) in enumerate(missing, 1):
            # Extract time span from window string (e.g., "300seconds" -> 300)
            timespan = int(window.replace('seconds', ''))
            
            print(f"\n[{idx}/{len(missing)}] {dataset}/{window}/{model}")
            
            cmd = ['python', 'main.py', '-d', dataset, '-m', model, '-t', str(timespan)]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(self.results_base_dir.parent)
                )
                
                if result.returncode == 0:
                    print(f"  Successfully regenerated")
                    success += 1
                else:
                    print(f"  Failed")
                    failed += 1
                    
            except subprocess.TimeoutExpired:
                print(f"  Timeout")
                failed += 1
            except Exception as e:
                print(f"  Error: {e}")
                failed += 1
        
        print(f"\n  Summary: {success} success, {failed} failed")
    
    def collect_curve_data(self,
                          datasets: Optional[List[str]] = None,
                          windows: Optional[List[str]] = None,
                          models: Optional[List[str]] = None) -> Dict:
        """
        Collect ROC and PR curve data from evaluation directories
        
        Args:
            datasets: List of dataset names to include (None = all)
            windows: List of time windows to include (e.g., ['1seconds', '60seconds'])
            models: List of model names to include (None = all)
            
        Returns:
            Dictionary with roc_curves and pr_curves data
        """
        print("\nCollecting ROC and PR curve data from evaluation directories...")
        roc_curves = []
        pr_curves = []
        
        # Search for curve data files
        roc_files = list(self.results_base_dir.rglob("**/evaluation/roc_curve_data.json"))
        pr_files = list(self.results_base_dir.rglob("**/evaluation/precision_recall_curve_data.json"))
        
        print(f"Found {len(roc_files)} ROC curve data files")
        print(f"Found {len(pr_files)} PR curve data files")
        
        # Process ROC curves
        for file_path in roc_files:
            try:
                # Skip files in the versions folder
                if 'versions' in file_path.parts:
                    continue
                
                parts = file_path.parts
                
                if 'results' in parts and 'models' in parts:
                    results_idx = parts.index('results')
                    models_idx = parts.index('models')
                    
                    dataset = parts[results_idx + 1]
                    window = parts[results_idx + 2]
                    model = parts[models_idx + 1]
                else:
                    continue
                
                # Apply filters
                if datasets and dataset not in datasets:
                    continue
                if windows and window not in windows:
                    continue
                if models and model not in models:
                    continue
                
                # Read the curve data
                with open(file_path, 'r') as f:
                    curve_data = json.load(f)
                
                curve_data['dataset'] = dataset
                curve_data['window'] = window
                curve_data['model'] = model
                roc_curves.append(curve_data)
                print(f"  Loaded ROC curve from {dataset}/{window}/{model}")
                
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
                continue
        
        # Process PR curves
        for file_path in pr_files:
            try:
                # Skip files in the versions folder
                if 'versions' in file_path.parts:
                    continue
                
                parts = file_path.parts
                
                if 'results' in parts and 'models' in parts:
                    results_idx = parts.index('results')
                    models_idx = parts.index('models')
                    
                    dataset = parts[results_idx + 1]
                    window = parts[results_idx + 2]
                    model = parts[models_idx + 1]
                else:
                    continue
                
                # Apply filters
                if datasets and dataset not in datasets:
                    continue
                if windows and window not in windows:
                    continue
                if models and model not in models:
                    continue
                
                # Read the curve data
                with open(file_path, 'r') as f:
                    curve_data = json.load(f)
                
                curve_data['dataset'] = dataset
                curve_data['window'] = window
                curve_data['model'] = model
                pr_curves.append(curve_data)
                print(f"  Loaded PR curve from {dataset}/{window}/{model}")
                
            except Exception as e:
                print(f"  Error reading {file_path}: {e}")
                continue
        
        print(f"\nCollected {len(roc_curves)} ROC curves and {len(pr_curves)} PR curves")
        
        return {
            'roc_curves': roc_curves,
            'pr_curves': pr_curves
        }
    
    def plot_all_roc_curves(self, curve_data: Dict):
        """
        Plot ROC curves grouped by dataset and time window
        
        Args:
            curve_data: Dictionary containing roc_curves list
        """
        if not curve_data['roc_curves']:
            print("No ROC curve data available to plot")
            return
        
        print("\nGenerating ROC curves plots by dataset and time window...")
        
        # Convert to DataFrame for easier grouping
        curves_df = pd.DataFrame(curve_data['roc_curves'])
        
        # Group by dataset and window
        grouped = curves_df.groupby(['dataset', 'window'])
        
        for (dataset, window), group in grouped:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Use a colormap for different models
            colors = plt.cm.tab10(np.linspace(0, 1, len(group)))
            
            for idx, (_, row) in enumerate(group.iterrows()):
                fpr = np.array(row['fpr'])
                tpr = np.array(row['tpr'])
                auc_score = row['auc']
                model = row['model']
                
                label = f"{model} (AUC={auc_score:.3f})"
                ax.plot(fpr, tpr, label=label, linewidth=2.5, alpha=0.8, color=colors[idx])
            
            # Plot diagonal reference line (random classifier)
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.5, label='Random Classifier (AUC=0.500)')
            
            # Styling
            ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
            
            # Clean up dataset name for title
            dataset_clean = dataset.replace('-', ' ').title()
            ax.set_title(f'ROC Curve - {dataset_clean}\nTime Window: {window}', 
                         fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            
            # Legend
            ax.legend(loc='lower right', fontsize=10, frameon=True, 
                     fancybox=True, shadow=True, framealpha=0.9)
            
            plt.tight_layout()
            
            # Save with descriptive filename
            safe_dataset = dataset.replace('/', '_').replace(' ', '_')
            output_path = self.output_dir / f'roc_curve_{safe_dataset}_{window}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  Saved ROC curve for {dataset}/{window}")
            plt.close()
        
        print(f"  Generated {len(grouped)} ROC curve plots")
    
    def plot_all_pr_curves(self, curve_data: Dict):
        """
        Plot Precision-Recall curves grouped by dataset and time window
        
        Args:
            curve_data: Dictionary containing pr_curves list
        """
        if not curve_data['pr_curves']:
            print("No PR curve data available to plot")
            return
        
        print("\nGenerating Precision-Recall curves plots by dataset and time window...")
        
        # Convert to DataFrame for easier grouping
        curves_df = pd.DataFrame(curve_data['pr_curves'])
        
        # Group by dataset and window
        grouped = curves_df.groupby(['dataset', 'window'])
        
        for (dataset, window), group in grouped:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Use a colormap for different models
            colors = plt.cm.tab10(np.linspace(0, 1, len(group)))
            
            for idx, (_, row) in enumerate(group.iterrows()):
                precision = np.array(row['precision'])
                recall = np.array(row['recall'])
                ap_score = row['average_precision']
                model = row['model']
                
                label = f"{model} (AP={ap_score:.3f})"
                ax.plot(recall, precision, label=label, linewidth=2.5, alpha=0.8, color=colors[idx])
            
            # Styling
            ax.set_xlabel('Recall (Sensitivity, True Positive Rate)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12, fontweight='bold')
            
            # Clean up dataset name for title
            dataset_clean = dataset.replace('-', ' ').title()
            ax.set_title(f'Precision-Recall Curve - {dataset_clean}\nTime Window: {window}', 
                         fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            
            # Legend
            ax.legend(loc='best', fontsize=10, frameon=True, 
                     fancybox=True, shadow=True, framealpha=0.9)
            
            plt.tight_layout()
            
            # Save with descriptive filename
            safe_dataset = dataset.replace('/', '_').replace(' ', '_')
            output_path = self.output_dir / f'pr_curve_{safe_dataset}_{window}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"  Saved PR curve for {dataset}/{window}")
            plt.close()
        
        print(f"  Generated {len(grouped)} Precision-Recall curve plots")
    
    def plot_roc_auc_comparison(self, metrics_df: pd.DataFrame):
        """
        Generate ROC-AUC comparison bar charts and visualizations
        
        Args:
            metrics_df: DataFrame with metrics from all models
        """
        print("\nGenerating ROC-AUC comparison visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ROC-AUC Score Comparison', fontsize=16, fontweight='bold')
        
        # By Model (averaged across datasets and windows)
        model_avg = metrics_df.groupby('model')['roc_auc'].mean().sort_values(ascending=False)
        axes[0, 0].bar(range(len(model_avg)), model_avg.values, color='steelblue', alpha=0.7)
        axes[0, 0].set_xticks(range(len(model_avg)))
        axes[0, 0].set_xticklabels(model_avg.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Average ROC-AUC Score')
        axes[0, 0].set_title('By Model')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 1])
        
        # Add value labels on bars
        for i, v in enumerate(model_avg.values):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # By Dataset
        dataset_avg = metrics_df.groupby('dataset')['roc_auc'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(dataset_avg)), dataset_avg.values, color='coral', alpha=0.7)
        axes[0, 1].set_xticks(range(len(dataset_avg)))
        axes[0, 1].set_xticklabels([d[:20] for d in dataset_avg.index], rotation=45, ha='right')
        axes[0, 1].set_ylabel('Average ROC-AUC Score')
        axes[0, 1].set_title('By Dataset')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1])
        
        for i, v in enumerate(dataset_avg.values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # By Window
        window_avg = metrics_df.groupby('window')['roc_auc'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(window_avg)), window_avg.values, color='mediumseagreen', alpha=0.7)
        axes[1, 0].set_xticks(range(len(window_avg)))
        axes[1, 0].set_xticklabels(window_avg.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Average ROC-AUC Score')
        axes[1, 0].set_title('By Time Window')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1])
        
        for i, v in enumerate(window_avg.values):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Distribution plot
        axes[1, 1].hist(metrics_df['roc_auc'], bins=20, color='mediumpurple', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(metrics_df['roc_auc'].mean(), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {metrics_df["roc_auc"].mean():.3f}')
        axes[1, 1].axvline(metrics_df['roc_auc'].median(), color='green', linestyle='--',
                           linewidth=2, label=f'Median: {metrics_df["roc_auc"].median():.3f}')
        axes[1, 1].set_xlabel('ROC-AUC Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of ROC-AUC Scores')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_path = self.output_dir / 'roc_auc_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved ROC-AUC comparison to {output_path}")
        plt.close()
    
    def plot_precision_recall_comparison(self, metrics_df: pd.DataFrame):
        """
        Generate Precision-Recall comparison visualizations
        
        Args:
            metrics_df: DataFrame with metrics from all models
        """
        print("\nGenerating Precision-Recall comparison visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Precision & Recall Comparison', fontsize=16, fontweight='bold')
        
        # Precision by Model
        model_precision = metrics_df.groupby('model')['precision'].mean().sort_values(ascending=False)
        axes[0, 0].bar(range(len(model_precision)), model_precision.values, color='steelblue', alpha=0.7)
        axes[0, 0].set_xticks(range(len(model_precision)))
        axes[0, 0].set_xticklabels(model_precision.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Average Precision')
        axes[0, 0].set_title('Precision by Model')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        axes[0, 0].set_ylim([0, 1])
        
        for i, v in enumerate(model_precision.values):
            axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Recall by Model
        model_recall = metrics_df.groupby('model')['recall'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(model_recall)), model_recall.values, color='coral', alpha=0.7)
        axes[0, 1].set_xticks(range(len(model_recall)))
        axes[0, 1].set_xticklabels(model_recall.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Average Recall')
        axes[0, 1].set_title('Recall by Model')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_ylim([0, 1])
        
        for i, v in enumerate(model_recall.values):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # F1-Score by Model
        model_f1 = metrics_df.groupby('model')['f1_score'].mean().sort_values(ascending=False)
        axes[1, 0].bar(range(len(model_f1)), model_f1.values, color='mediumseagreen', alpha=0.7)
        axes[1, 0].set_xticks(range(len(model_f1)))
        axes[1, 0].set_xticklabels(model_f1.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Average F1-Score')
        axes[1, 0].set_title('F1-Score by Model')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        axes[1, 0].set_ylim([0, 1])
        
        for i, v in enumerate(model_f1.values):
            axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Precision vs Recall scatter
        for model in metrics_df['model'].unique():
            model_data = metrics_df[metrics_df['model'] == model]
            axes[1, 1].scatter(model_data['recall'], model_data['precision'], 
                             label=model, alpha=0.6, s=100)
        
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Recall by Model')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim([-0.05, 1.05])
        axes[1, 1].set_ylim([-0.05, 1.05])
        
        plt.tight_layout()
        output_path = self.output_dir / 'precision_recall_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved Precision-Recall comparison to {output_path}")
        plt.close()
    
    def generate_summary_report(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Save comprehensive summary statistics
        
        Args:
            metrics_df: DataFrame with metrics
            
        Returns:
            Summary DataFrame
        """
        print("\nGenerating summary report...")
        
        # Sort by ROC-AUC descending
        summary_df = metrics_df.sort_values('roc_auc', ascending=False)
        
        # Save to CSV
        output_path = self.output_dir / 'cross_evaluation_summary.csv'
        summary_df.to_csv(output_path, index=False)
        print(f"  Saved summary to {output_path}")
        
        return summary_df
    
    def generate_ranking_reports(self, summary_df: pd.DataFrame):
        """
        Generate ranking reports for best performers
        
        Args:
            summary_df: Summary DataFrame with metrics
        """
        print("\nGenerating ranking reports...")
        
        # Top performers by various metrics
        best_roc = summary_df.nlargest(10, 'roc_auc')
        best_precision = summary_df.nlargest(10, 'precision')
        best_recall = summary_df.nlargest(10, 'recall')
        best_f1 = summary_df.nlargest(10, 'f1_score')
        
        # Save rankings as JSON
        rankings = {
            'generated_at': datetime.now().isoformat(),
            'total_configurations': len(summary_df),
            'best_roc_auc': best_roc.to_dict('records'),
            'best_precision': best_precision.to_dict('records'),
            'best_recall': best_recall.to_dict('records'),
            'best_f1_score': best_f1.to_dict('records')
        }
        
        output_path = self.output_dir / 'best_performers.json'
        with open(output_path, 'w') as f:
            json.dump(rankings, f, indent=2)
        print(f"  Saved rankings to {output_path}")
        
        # Print top performers
        print("\n" + "="*70)
        print("TOP 5 MODELS BY ROC-AUC")
        print("="*70)
        for idx, row in best_roc.head(5).iterrows():
            print(f"{row['dataset']:30} | {row['window']:12} | {row['model']:20} | AUC: {row['roc_auc']:.4f}")
        
        print("\n" + "="*70)
        print("TOP 5 MODELS BY F1-SCORE")
        print("="*70)
        for idx, row in best_f1.head(5).iterrows():
            print(f"{row['dataset']:30} | {row['window']:12} | {row['model']:20} | F1: {row['f1_score']:.4f}")
        
        print("\n" + "="*70)
        print("TOP 5 MODELS BY RECALL")
        print("="*70)
        for idx, row in best_recall.head(5).iterrows():
            print(f"{row['dataset']:30} | {row['window']:12} | {row['model']:20} | Recall: {row['recall']:.4f}")
        print("="*70 + "\n")
    
    def generate_heatmaps(self, summary_df: pd.DataFrame):
        """
        Generate heatmaps for ROC-AUC and other metrics across different dimensions
        
        Args:
            summary_df: Summary DataFrame with metrics
        """
        print("\nGenerating heatmaps...")
        
        # Heatmap: Models vs Datasets (averaged across windows)
        pivot_roc = summary_df.pivot_table(
            index='model', 
            columns='dataset', 
            values='roc_auc', 
            aggfunc='mean'
        )
        
        pivot_f1 = summary_df.pivot_table(
            index='model', 
            columns='dataset', 
            values='f1_score', 
            aggfunc='mean'
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        # ROC-AUC heatmap
        sns.heatmap(pivot_roc, annot=True, fmt='.3f', cmap='RdYlGn', 
                   ax=axes[0], vmin=0, vmax=1, cbar_kws={'label': 'ROC-AUC'})
        axes[0].set_title('ROC-AUC: Models vs Datasets', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Dataset', fontsize=11)
        axes[0].set_ylabel('Model', fontsize=11)
        
        # F1-Score heatmap
        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'F1-Score'})
        axes[1].set_title('F1-Score: Models vs Datasets', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Dataset', fontsize=11)
        axes[1].set_ylabel('Model', fontsize=11)
        
        plt.tight_layout()
        output_path = self.output_dir / 'heatmap_models_vs_datasets.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved heatmap to {output_path}")
        plt.close()
        
        # Heatmap: Models vs Windows (averaged across datasets)
        pivot_roc_windows = summary_df.pivot_table(
            index='model', 
            columns='window', 
            values='roc_auc', 
            aggfunc='mean'
        )
        
        pivot_f1_windows = summary_df.pivot_table(
            index='model', 
            columns='window', 
            values='f1_score', 
            aggfunc='mean'
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        sns.heatmap(pivot_roc_windows, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[0], vmin=0, vmax=1, cbar_kws={'label': 'ROC-AUC'})
        axes[0].set_title('ROC-AUC: Models vs Time Windows', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Time Window', fontsize=11)
        axes[0].set_ylabel('Model', fontsize=11)
        
        sns.heatmap(pivot_f1_windows, annot=True, fmt='.3f', cmap='RdYlGn',
                   ax=axes[1], vmin=0, vmax=1, cbar_kws={'label': 'F1-Score'})
        axes[1].set_title('F1-Score: Models vs Time Windows', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Time Window', fontsize=11)
        axes[1].set_ylabel('Model', fontsize=11)
        
        plt.tight_layout()
        output_path = self.output_dir / 'heatmap_models_vs_windows.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved heatmap to {output_path}")
        plt.close()
    
    def run_cross_evaluation(self, 
                            datasets: Optional[List[str]] = None,
                            windows: Optional[List[str]] = None,
                            models: Optional[List[str]] = None,
                            auto_regenerate: bool = False):
        """
        Main execution method for cross-evaluation
        
        Args:
            datasets: List of datasets to include (None = all)
            windows: List of time windows to include (None = all)
            models: List of models to include (None = all)
            auto_regenerate: If True, automatically regenerate missing curve data
        """
        print("\n" + "="*70)
        print("CROSS-EVALUATION: ROC-AUC & PRECISION-RECALL ANALYSIS")
        print("="*70)
        
        # Step 1: Collect metrics from analysis files
        metrics_df = self.collect_metrics_from_analysis(datasets, windows, models)
        
        if metrics_df.empty:
            print("\nNo metrics found. Please train and evaluate models first.")
            return
        
        # Step 2: Collect curve data for detailed ROC and PR plots
        curve_data = self.collect_curve_data(datasets, windows, models)
        
        # Check if we're missing curve data and offer to regenerate
        if not curve_data['roc_curves'] or not curve_data['pr_curves']:
            print("\nSome evaluations are missing curve data.")
            self.regenerate_missing_curve_data(datasets, windows, models, auto_regenerate)
            # Recollect after regeneration
            curve_data = self.collect_curve_data(datasets, windows, models)
        
        # Step 3: Plot combined ROC curves (all models on one plot)
        if curve_data['roc_curves']:
            self.plot_all_roc_curves(curve_data)
        else:
            print("\nNo ROC curve data found. Curves will be generated on next evaluation run.")
        
        # Step 4: Plot combined PR curves (all models on one plot)
        if curve_data['pr_curves']:
            self.plot_all_pr_curves(curve_data)
        else:
            print("\nNo PR curve data found. Curves will be generated on next evaluation run.")
        
        # Step 5: Generate ROC-AUC comparison
        self.plot_roc_auc_comparison(metrics_df)
        
        # Step 6: Generate Precision-Recall comparison
        self.plot_precision_recall_comparison(metrics_df)
        
        # Step 7: Generate summary report
        summary_df = self.generate_summary_report(metrics_df)
        
        # Step 8: Generate rankings
        self.generate_ranking_reports(summary_df)
        
        # Step 9: Generate heatmaps
        self.generate_heatmaps(summary_df)
        
        print("\n" + "="*70)
        print(f"Cross-evaluation complete!")
        print(f"  Results saved to: {self.output_dir}")
        print("="*70 + "\n")


def run_cross_evaluation(datasets=None, windows=None, models=None, 
                        results_dir='./results', output_dir='./results/cross_evaluation',
                        regenerate_curves=False):
    """
    Main function to run cross-evaluation analysis
    
    Args:
        datasets: List of dataset names to include (None = all)
        windows: List of time windows to include (None = all)
        models: List of model names to include (None = all)
        results_dir: Base directory containing model results
        output_dir: Output directory for cross-evaluation results
        regenerate_curves: If True, automatically regenerate missing curve data
    """
    evaluator = CrossEvaluator(
        results_base_dir=results_dir,
        output_dir=output_dir
    )
    
    evaluator.run_cross_evaluation(
        datasets=datasets,
        windows=windows,
        models=models,
        auto_regenerate=regenerate_curves
    )
