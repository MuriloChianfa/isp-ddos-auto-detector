"""
Comparison module for analyzing and comparing multiple experimental runs.
Provides functionality to load, compare, and visualize results from different parameter configurations.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class RunComparator:
    """Compares multiple experimental runs and generates comparison reports"""
    
    def __init__(self):
        """Initialize the RunComparator"""
        self.runs_data = []
    
    def load_run(self, run_id: str, run_path: Optional[str] = None) -> Dict:
        """
        Load analysis data for a specific run
        
        Args:
            run_id: Run identifier
            run_path: Optional path to the run directory (auto-detected if not provided)
            
        Returns:
            Dictionary containing run data
        """
        if run_path is None:
            run_path = f"./results/versions/{run_id}"
        
        run_data = {
            "run_id": run_id,
            "run_path": run_path,
            "analyses": []
        }
        
        # Find all analysis.json files in the run directory
        for root, dirs, files in os.walk(run_path):
            if "analysis.json" in files:
                analysis_file = os.path.join(root, "analysis.json")
                try:
                    with open(analysis_file, 'r') as f:
                        analysis = json.load(f)
                        analysis['_file_path'] = analysis_file
                        run_data["analyses"].append(analysis)
                except Exception as e:
                    print(f"Warning: Could not load {analysis_file}: {e}")
        
        return run_data
    
    def load_runs(self, run_ids: List[str]) -> List[Dict]:
        """
        Load multiple runs for comparison
        
        Args:
            run_ids: List of run identifiers
            
        Returns:
            List of run data dictionaries
        """
        self.runs_data = []
        for run_id in run_ids:
            run_data = self.load_run(run_id)
            if run_data["analyses"]:
                self.runs_data.append(run_data)
            else:
                print(f"Warning: No analyses found for run {run_id}")
        
        return self.runs_data
    
    def compare_metrics(self, metric_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compare specified metrics across all loaded runs
        
        Args:
            metric_names: List of metric names to compare (uses defaults if None)
            
        Returns:
            DataFrame with comparison results
        """
        if not self.runs_data:
            raise ValueError("No runs loaded. Call load_runs() first.")
        
        # Default metrics to compare
        if metric_names is None:
            metric_names = [
                'roc_auc_score', 'accuracy', 'precision', 'recall', 
                'f1_score', 'f2_score', 'matthews_corrcoef',
                'false_positive_rate', 'miss_rate'
            ]
        
        comparison_data = []
        
        for run_data in self.runs_data:
            for analysis in run_data["analyses"]:
                row = {
                    'run_id': run_data['run_id'],
                    'dataset': analysis.get('dataset_name', 'unknown'),
                    'model': analysis.get('model_name', 'unknown'),
                    'time_span': analysis.get('time_span', 0),
                    'timestamp': analysis.get('analysis_timestamp', ''),
                    'threshold': analysis.get('threshold_used', 0.0)
                }
                
                # Extract metrics from evaluation_metrics
                eval_metrics = analysis.get('evaluation_metrics', {})
                for metric in metric_names:
                    row[metric] = eval_metrics.get(metric, np.nan)
                
                # Extract sample counts
                row['total_samples'] = eval_metrics.get('total_samples', 0)
                row['detected_anomalies'] = eval_metrics.get('detected_anomalies', 0)
                row['attack_periods'] = eval_metrics.get('attack_periods', 0)
                
                # Extract confusion matrix values
                row['tp'] = eval_metrics.get('true_positives', 0)
                row['tn'] = eval_metrics.get('true_negatives', 0)
                row['fp'] = eval_metrics.get('false_positives', 0)
                row['fn'] = eval_metrics.get('false_negatives', 0)
                
                comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def compute_metric_deltas(self, baseline_run_id: str, metric_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Compute metric differences relative to a baseline run
        
        Args:
            baseline_run_id: Run ID to use as baseline
            metric_names: List of metrics to compute deltas for
            
        Returns:
            DataFrame with metric deltas
        """
        comparison_df = self.compare_metrics(metric_names)
        
        if baseline_run_id not in comparison_df['run_id'].values:
            raise ValueError(f"Baseline run {baseline_run_id} not found in loaded runs")
        
        # Get baseline metrics
        baseline_df = comparison_df[comparison_df['run_id'] == baseline_run_id].copy()
        
        # Compute deltas for each configuration
        delta_data = []
        
        for _, row in comparison_df.iterrows():
            if row['run_id'] == baseline_run_id:
                continue
            
            # Find matching baseline configuration
            baseline_match = baseline_df[
                (baseline_df['dataset'] == row['dataset']) &
                (baseline_df['model'] == row['model']) &
                (baseline_df['time_span'] == row['time_span'])
            ]
            
            if baseline_match.empty:
                continue
            
            baseline_row = baseline_match.iloc[0]
            
            delta_row = {
                'run_id': row['run_id'],
                'baseline_run_id': baseline_run_id,
                'dataset': row['dataset'],
                'model': row['model'],
                'time_span': row['time_span']
            }
            
            # Compute deltas for numeric metrics
            if metric_names is None:
                metric_names = ['roc_auc_score', 'accuracy', 'precision', 'recall', 
                               'f1_score', 'f2_score', 'matthews_corrcoef']
            
            for metric in metric_names:
                if metric in row and metric in baseline_row:
                    delta = row[metric] - baseline_row[metric]
                    pct_change = ((row[metric] - baseline_row[metric]) / baseline_row[metric] * 100) if baseline_row[metric] != 0 else 0
                    delta_row[f'{metric}_delta'] = delta
                    delta_row[f'{metric}_pct_change'] = pct_change
            
            delta_data.append(delta_row)
        
        return pd.DataFrame(delta_data)
    
    def generate_comparison_report(self, output_path: str, baseline_run_id: Optional[str] = None):
        """
        Generate a comprehensive comparison report and save to CSV
        
        Args:
            output_path: Path to save the comparison report
            baseline_run_id: Optional baseline run for delta calculations
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Get comparison metrics
        comparison_df = self.compare_metrics()
        
        # Save main comparison
        comparison_df.to_csv(output_path, index=False)
        print(f"Comparison report saved to: {output_path}")
        
        # If baseline provided, also save deltas
        if baseline_run_id:
            delta_df = self.compute_metric_deltas(baseline_run_id)
            delta_path = output_path.replace('.csv', '_deltas.csv')
            delta_df.to_csv(delta_path, index=False)
            print(f"Delta report saved to: {delta_path}")
        
        return comparison_df
    
    def print_comparison_summary(self):
        """Print a summary of the comparison"""
        if not self.runs_data:
            print("No runs loaded for comparison")
            return
        
        print(f"\n{'='*80}")
        print("RUN COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Total runs loaded: {len(self.runs_data)}")
        
        comparison_df = self.compare_metrics()
        
        print(f"\nRuns:")
        for run_data in self.runs_data:
            print(f"  - {run_data['run_id']}: {len(run_data['analyses'])} analyses")
        
        print(f"\nMetric Ranges:")
        metrics = ['roc_auc_score', 'f1_score', 'precision', 'recall', 'accuracy']
        for metric in metrics:
            if metric in comparison_df.columns:
                print(f"  {metric}:")
                print(f"    Min: {comparison_df[metric].min():.4f}")
                print(f"    Max: {comparison_df[metric].max():.4f}")
                print(f"    Mean: {comparison_df[metric].mean():.4f}")
                print(f"    Std: {comparison_df[metric].std():.4f}")
        
        print(f"\nBest performers by ROC-AUC:")
        if 'roc_auc_score' in comparison_df.columns:
            top_5 = comparison_df.nlargest(5, 'roc_auc_score')
            for idx, row in top_5.iterrows():
                print(f"  {row['run_id']} - {row['dataset']} - {row['model']} - "
                      f"{row['time_span']}s: ROC-AUC={row['roc_auc_score']:.4f}")
        
        print(f"{'='*80}\n")
    
    def get_best_configuration(self, metric: str = 'roc_auc_score') -> Dict:
        """
        Find the best configuration based on a specific metric
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            Dictionary with best configuration details
        """
        comparison_df = self.compare_metrics()
        
        if metric not in comparison_df.columns:
            raise ValueError(f"Metric {metric} not found in comparison data")
        
        best_idx = comparison_df[metric].idxmax()
        best_row = comparison_df.loc[best_idx]
        
        return best_row.to_dict()


def compare_runs(run_ids: List[str], output_dir: str = "./results/comparisons", 
                 baseline_run_id: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience function to compare multiple runs
    
    Args:
        run_ids: List of run identifiers to compare
        output_dir: Directory to save comparison reports
        baseline_run_id: Optional baseline run for delta calculations
        
    Returns:
        DataFrame with comparison results
    """
    comparator = RunComparator()
    comparator.load_runs(run_ids)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"comparison_{timestamp}.csv")
    
    comparison_df = comparator.generate_comparison_report(output_path, baseline_run_id)
    comparator.print_comparison_summary()
    
    return comparison_df
