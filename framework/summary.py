"""
Results Summary Module
Provides functionality to collect and display all model metrics from all databases 
and timestamps in a comprehensive table format.
"""

import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from tabulate import tabulate
from framework.evaluation_cache import EvaluationCache


class ResultsSummary:
    """Collects and displays comprehensive summary of all evaluation results"""
    
    def __init__(self, results_base_dir: str = './results', cache_dir: str = './cache'):
        """
        Initialize ResultsSummary
        
        Args:
            results_base_dir: Base directory containing all evaluation results
            cache_dir: Directory containing evaluation cache
        """
        self.results_base_dir = Path(results_base_dir)
        self.cache = EvaluationCache(cache_dir=cache_dir)
        
    def collect_all_metrics(self, 
                           datasets: Optional[List[str]] = None,
                           models: Optional[List[str]] = None,
                           time_spans: Optional[List[int]] = None,
                           use_cache: bool = True) -> pd.DataFrame:
        """
        Collect all evaluation metrics from analysis files or cache
        
        Args:
            datasets: List of dataset names to include (None = all)
            models: List of model names to include (None = all)
            time_spans: List of time spans to include (None = all)
            use_cache: If True, first try to collect from cache
            
        Returns:
            DataFrame with all metrics
        """
        all_metrics = []
        
        if use_cache:
            cache_stats = self.cache.get_stats()
            cache_entries = cache_stats.get('entries', [])
            
            if cache_entries:
                print(f"Loading metrics from cache...")
                for entry in cache_entries:
                    dataset = entry.get('dataset')
                    model = entry.get('model')
                    time_span = entry.get('time_span')
                    metrics = entry.get('metrics', {})
                    
                    # Apply filters
                    if datasets and dataset not in datasets:
                        continue
                    if models and model not in models:
                        continue
                    if time_spans and time_span not in time_spans:
                        continue
                    
                    if metrics:
                        row = self._parse_metrics_from_cache(entry)
                        if row:
                            all_metrics.append(row)
        
        # Also collect from analysis.json files
        print("Scanning analysis files...")
        analysis_files = list(self.results_base_dir.rglob("**/artifacts/analysis.json"))
        
        for file_path in analysis_files:
            try:
                # Parse path structure: results/dataset/timespan/models/model/artifacts/analysis.json
                parts = file_path.parts
                
                if 'results' not in parts or 'models' not in parts:
                    continue
                    
                results_idx = parts.index('results')
                models_idx = parts.index('models')
                
                dataset = parts[results_idx + 1]
                window_str = parts[results_idx + 2]
                model = parts[models_idx + 1]
                
                # Extract time span from window string (e.g., "300seconds" -> 300)
                time_span = int(window_str.replace('seconds', ''))
                
                # Apply filters
                if datasets and dataset not in datasets:
                    continue
                if models and model not in models:
                    continue
                if time_spans and time_span not in time_spans:
                    continue
                
                # Check if already in cache results
                if any(m.get('dataset') == dataset and 
                      m.get('model') == model and 
                      m.get('time_span') == time_span 
                      for m in all_metrics):
                    continue
                
                # Read the analysis file
                with open(file_path, 'r') as f:
                    analysis_data = json.load(f)
                
                row = self._parse_metrics_from_analysis(analysis_data, dataset, model, time_span)
                if row:
                    all_metrics.append(row)
                    
            except Exception as e:
                print(f"Warning: Error reading {file_path}: {e}")
                continue
        
        if not all_metrics:
            print("No metrics found!")
            return pd.DataFrame()
        
        # Create DataFrame and sort
        df = pd.DataFrame(all_metrics)
        df = df.sort_values(['dataset', 'time_span', 'model'])
        
        print(f"\nCollected metrics from {len(df)} evaluations:")
        print(f"  Datasets: {df['dataset'].nunique()} - {sorted(df['dataset'].unique())}")
        print(f"  Models: {df['model'].nunique()} - {sorted(df['model'].unique())}")
        print(f"  Time Spans: {df['time_span'].nunique()} - {sorted(df['time_span'].unique())}")
        
        return df
    
    def _parse_metrics_from_cache(self, cache_entry: Dict) -> Optional[Dict]:
        """Parse metrics from cache entry"""
        try:
            metrics = cache_entry.get('metrics', {})
            if not metrics:
                return None
            
            return {
                'dataset': cache_entry.get('dataset'),
                'model': cache_entry.get('model'),
                'time_span': cache_entry.get('time_span'),
                'accuracy': float(metrics.get('accuracy', 0)),
                'precision': float(metrics.get('precision', 0)),
                'recall': float(metrics.get('recall', 0)),
                'f1_score': float(metrics.get('f1_score', 0)),
                'f2_score': float(metrics.get('f2_score', 0)),
                'roc_auc': float(metrics.get('roc_auc_score', 0)),
                'mcc': float(metrics.get('matthews_corrcoef', 0)),
                'fpr': float(metrics.get('false_positive_rate', metrics.get('fpr', 0))),
                'fnr': float(metrics.get('miss_rate', 0)),
                'tp': int(metrics.get('true_positives', 0)),
                'tn': int(metrics.get('true_negatives', 0)),
                'fp': int(metrics.get('false_positives', 0)),
                'fn': int(metrics.get('false_negatives', 0)),
                'total_samples': int(metrics.get('total_samples', 0)),
                'attack_periods': int(metrics.get('attack_periods', 0)),
                'detected_anomalies': int(metrics.get('detected_anomalies', 0)),
                'threshold': float(metrics.get('threshold', cache_entry.get('threshold', 0))),
                'threshold_strategy': cache_entry.get('threshold_strategy', 'unknown'),
                'training_time_seconds': float(cache_entry.get('training_time_seconds', 0))
            }
        except Exception as e:
            print(f"Warning: Error parsing cache metrics: {e}")
            return None
    
    def _parse_metrics_from_analysis(self, analysis_data: Dict, 
                                     dataset: str, model: str, 
                                     time_span: int) -> Optional[Dict]:
        """Parse metrics from analysis.json file"""
        try:
            if 'evaluation_metrics' not in analysis_data:
                return None
            
            metrics = analysis_data['evaluation_metrics']
            model_info = analysis_data.get('model_info', {})
            
            # Helper function to safely convert to float
            def safe_float(value, default=0.0):
                if value is None:
                    return default
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            # Helper function to safely convert to int
            def safe_int(value, default=0):
                if value is None:
                    return default
                try:
                    return int(float(value))  # Convert through float to handle string floats
                except (ValueError, TypeError):
                    return default
            
            # Try to get training_time_seconds from model_info first, fallback to top-level
            training_time = model_info.get('training_time_seconds')
            if training_time is None:
                training_time = analysis_data.get('training_time_seconds')
            
            return {
                'dataset': dataset,
                'model': model,
                'time_span': time_span,
                'accuracy': safe_float(metrics.get('accuracy')),
                'precision': safe_float(metrics.get('precision')),
                'recall': safe_float(metrics.get('recall')),
                'f1_score': safe_float(metrics.get('f1_score')),
                'f2_score': safe_float(metrics.get('f2_score')),
                'roc_auc': safe_float(metrics.get('roc_auc_score')),
                'mcc': safe_float(metrics.get('matthews_corrcoef')),
                'fpr': safe_float(metrics.get('false_positive_rate')),
                'fnr': safe_float(metrics.get('miss_rate')),
                'tp': safe_int(metrics.get('true_positives')),
                'tn': safe_int(metrics.get('true_negatives')),
                'fp': safe_int(metrics.get('false_positives')),
                'fn': safe_int(metrics.get('false_negatives')),
                'total_samples': safe_int(metrics.get('total_samples')),
                'attack_periods': safe_int(metrics.get('attack_periods')),
                'detected_anomalies': safe_int(metrics.get('detected_anomalies')),
                'threshold': safe_float(analysis_data.get('threshold_used')),
                'threshold_strategy': analysis_data.get('threshold_strategy', 'unknown'),
                'training_time_seconds': safe_float(training_time)
            }
        except Exception as e:
            print(f"Warning: Error parsing analysis metrics: {e}")
            return None
    
    def display_summary_table(self, df: pd.DataFrame, 
                             format: str = 'grid',
                             metrics: Optional[List[str]] = None):
        """
        Display metrics summary in a formatted table
        
        Args:
            df: DataFrame with metrics
            format: Table format ('grid', 'simple', 'fancy_grid', 'pipe', 'html', 'latex')
            metrics: List of metric columns to display (None = key metrics)
        """
        if df.empty:
            print("No data to display")
            return
        
        # Default key metrics
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'fpr', 'mcc']
        
        # Select columns for display
        display_cols = ['dataset', 'model', 'time_span'] + metrics
        display_cols = [col for col in display_cols if col in df.columns]
        
        display_df = df[display_cols].copy()
        
        # Round numeric columns for better display
        for col in metrics:
            if col in display_df.columns and pd.api.types.is_numeric_dtype(display_df[col]):
                display_df[col] = display_df[col].round(4)
        
        print("\n" + "="*100)
        print("EVALUATION RESULTS SUMMARY | ALL MODELS, DATASETS, AND TIME SPANS")
        print("="*100)
        print(tabulate(display_df, headers='keys', tablefmt=format, showindex=False))
        print("="*100)
    
    def display_detailed_table(self, df: pd.DataFrame, format: str = 'grid'):
        """
        Display detailed metrics including confusion matrix values
        
        Args:
            df: DataFrame with metrics
            format: Table format
        """
        if df.empty:
            print("No data to display")
            return
        
        display_cols = [
            'dataset', 'model', 'time_span',
            'accuracy', 'precision', 'recall', 'f1_score', 'f2_score',
            'roc_auc', 'mcc', 'fpr', 'fnr',
            'tp', 'tn', 'fp', 'fn',
            'total_samples', 'attack_periods', 'detected_anomalies',
            'threshold', 'threshold_strategy'
        ]
        
        # Filter to existing columns
        display_cols = [col for col in display_cols if col in df.columns]
        display_df = df[display_cols].copy()
        
        # Round numeric columns
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'f2_score',
                       'roc_auc', 'mcc', 'fpr', 'fnr', 'threshold']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)
        
        print("\n" + "="*150)
        print("DETAILED EVALUATION RESULTS - ALL METRICS")
        print("="*150)
        print(tabulate(display_df, headers='keys', tablefmt=format, showindex=False))
        print("="*150)
    
    def display_by_dataset(self, df: pd.DataFrame, format: str = 'grid'):
        """
        Display results grouped by dataset
        
        Args:
            df: DataFrame with metrics
            format: Table format
        """
        if df.empty:
            print("No data to display")
            return
        
        datasets = sorted(df['dataset'].unique())
        
        for dataset in datasets:
            dataset_df = df[df['dataset'] == dataset].copy()
            
            display_cols = ['model', 'time_span', 'accuracy', 'precision', 
                          'recall', 'f1_score', 'roc_auc', 'fpr', 'mcc']
            display_cols = [col for col in display_cols if col in dataset_df.columns]
            
            display_df = dataset_df[display_cols].copy()
            
            # Round numeric columns
            for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'fpr', 'mcc']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(4)
            
            print("\n" + "="*100)
            print(f"DATASET: {dataset}")
            print("="*100)
            print(tabulate(display_df, headers='keys', tablefmt=format, showindex=False))
            print("="*100)
    
    def display_by_model(self, df: pd.DataFrame, format: str = 'grid'):
        """
        Display results grouped by model
        
        Args:
            df: DataFrame with metrics
            format: Table format
        """
        if df.empty:
            print("No data to display")
            return
        
        models = sorted(df['model'].unique())
        
        for model in models:
            model_df = df[df['model'] == model].copy()
            
            display_cols = ['dataset', 'time_span', 'accuracy', 'precision', 
                          'recall', 'f1_score', 'roc_auc', 'fpr', 'mcc']
            display_cols = [col for col in display_cols if col in model_df.columns]
            
            display_df = model_df[display_cols].copy()
            
            # Round numeric columns
            for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'fpr', 'mcc']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(4)
            
            print("\n" + "="*100)
            print(f"MODEL: {model}")
            print("="*100)
            print(tabulate(display_df, headers='keys', tablefmt=format, showindex=False))
            print("="*100)
    
    def display_by_timespan(self, df: pd.DataFrame, format: str = 'grid'):
        """
        Display results grouped by time span
        
        Args:
            df: DataFrame with metrics
            format: Table format
        """
        if df.empty:
            print("No data to display")
            return
        
        time_spans = sorted(df['time_span'].unique())
        
        for time_span in time_spans:
            ts_df = df[df['time_span'] == time_span].copy()
            
            display_cols = ['dataset', 'model', 'accuracy', 'precision', 
                          'recall', 'f1_score', 'roc_auc', 'fpr', 'mcc']
            display_cols = [col for col in display_cols if col in ts_df.columns]
            
            display_df = ts_df[display_cols].copy()
            
            # Round numeric columns
            for col in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'fpr', 'mcc']:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(4)
            
            print("\n" + "="*100)
            print(f"TIME SPAN: {time_span} seconds")
            print("="*100)
            print(tabulate(display_df, headers='keys', tablefmt=format, showindex=False))
            print("="*100)
    
    def display_training_times(self, df: pd.DataFrame, format: str = 'grid'):
        """
        Display training time statistics for all models
        
        Args:
            df: DataFrame with metrics
            format: Table format
        """
        if df.empty:
            print("No data to display")
            return
        
        if 'training_time_seconds' not in df.columns:
            print("Training time data not available")
            return
        
        # Filter out entries with 0 or None training time (loaded from artifacts)
        training_df = df[df['training_time_seconds'] > 0].copy()
        
        if training_df.empty:
            print("No training time data available (all models loaded from artifacts)")
            return
        
        # Add training time in minutes for readability
        training_df['training_time_minutes'] = training_df['training_time_seconds'] / 60
        
        display_cols = ['dataset', 'model', 'time_span', 'training_time_seconds', 'training_time_minutes']
        display_df = training_df[display_cols].copy()
        
        # Round time columns
        display_df['training_time_seconds'] = display_df['training_time_seconds'].round(2)
        display_df['training_time_minutes'] = display_df['training_time_minutes'].round(2)
        
        # Sort by training time
        display_df = display_df.sort_values('training_time_seconds', ascending=False)
        
        print("\n" + "="*100)
        print("TRAINING TIME STATISTICS")
        print("="*100)
        print(tabulate(display_df, headers='keys', tablefmt=format, showindex=False))
        print("="*100)
        
        # Print summary statistics
        print("\nTraining Time Summary:")
        print(f"  Total models trained: {len(training_df)}")
        print(f"  Average training time: {training_df['training_time_minutes'].mean():.2f} minutes ({training_df['training_time_seconds'].mean():.2f} seconds)")
        print(f"  Median training time: {training_df['training_time_minutes'].median():.2f} minutes ({training_df['training_time_seconds'].median():.2f} seconds)")
        print(f"  Fastest training: {training_df['training_time_minutes'].min():.2f} minutes ({training_df['training_time_seconds'].min():.2f} seconds)")
        print(f"  Slowest training: {training_df['training_time_minutes'].max():.2f} minutes ({training_df['training_time_seconds'].max():.2f} seconds)")
        print(f"  Total training time: {training_df['training_time_minutes'].sum():.2f} minutes ({training_df['training_time_seconds'].sum():.2f} seconds)")
        print("="*100)
    
    def get_best_performers(self, df: pd.DataFrame, 
                           metric: str = 'f1_score',
                           top_n: int = 10) -> pd.DataFrame:
        """
        Get top N best performing configurations
        
        Args:
            df: DataFrame with metrics
            metric: Metric to rank by
            top_n: Number of top performers to return
            
        Returns:
            DataFrame with top performers
        """
        if df.empty or metric not in df.columns:
            print(f"Cannot rank by metric '{metric}'")
            return pd.DataFrame()
        
        top_df = df.nlargest(top_n, metric)
        return top_df
    
    def get_worst_performers(self, df: pd.DataFrame,
                            metric: str = 'f1_score',
                            bottom_n: int = 10) -> pd.DataFrame:
        """
        Get bottom N worst performing configurations
        
        Args:
            df: DataFrame with metrics
            metric: Metric to rank by
            bottom_n: Number of bottom performers to return
            
        Returns:
            DataFrame with worst performers
        """
        if df.empty or metric not in df.columns:
            print(f"Cannot rank by metric '{metric}'")
            return pd.DataFrame()
        
        bottom_df = df.nsmallest(bottom_n, metric)
        return bottom_df
    
    def export_to_csv(self, df: pd.DataFrame, output_path: str = None):
        """
        Export metrics to CSV file
        
        Args:
            df: DataFrame with metrics
            output_path: Path to save CSV file (None = auto-generate)
        """
        if df.empty:
            print("No data to export")
            return
        
        if output_path is None:
            output_dir = self.results_base_dir / 'summary'
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / 'all_metrics_summary.csv'
        
        df.to_csv(output_path, index=False)
        print(f"\nMetrics exported to: {output_path}")
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics across all evaluations
        
        Args:
            df: DataFrame with metrics
            
        Returns:
            Dictionary with statistics
        """
        if df.empty:
            return {}
        
        numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 
                       'roc_auc', 'mcc', 'fpr', 'fnr']
        
        stats = {
            'total_evaluations': len(df),
            'datasets': sorted(df['dataset'].unique().tolist()),
            'models': sorted(df['model'].unique().tolist()),
            'time_spans': sorted(df['time_span'].unique().tolist()),
        }
        
        # Calculate statistics for each numeric metric
        for metric in numeric_cols:
            if metric in df.columns:
                stats[f'{metric}_mean'] = float(df[metric].mean())
                stats[f'{metric}_std'] = float(df[metric].std())
                stats[f'{metric}_min'] = float(df[metric].min())
                stats[f'{metric}_max'] = float(df[metric].max())
                stats[f'{metric}_median'] = float(df[metric].median())
        
        return stats
    
    def display_statistics(self, df: pd.DataFrame):
        """
        Display summary statistics
        
        Args:
            df: DataFrame with metrics
        """
        stats = self.generate_statistics(df)
        
        if not stats:
            print("No statistics available")
            return
        
        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)
        print(f"Total Evaluations: {stats['total_evaluations']}")
        print(f"Datasets ({len(stats['datasets'])}): {', '.join(stats['datasets'])}")
        print(f"Models ({len(stats['models'])}): {', '.join(stats['models'])}")
        print(f"Time Spans ({len(stats['time_spans'])}): {', '.join(map(str, stats['time_spans']))} seconds")
        
        # print("\nMetric Statistics (Mean ± Std):")
        # print("-" * 100)
        
        # metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'mcc', 'fpr', 'fnr']
        # for metric in metrics:
        #     mean_key = f'{metric}_mean'
        #     std_key = f'{metric}_std'
        #     min_key = f'{metric}_min'
        #     max_key = f'{metric}_max'
            
        #     if mean_key in stats:
        #         print(f"{metric:20s}: {stats[mean_key]:.4f} ± {stats[std_key]:.4f}  "
        #               f"[min: {stats[min_key]:.4f}, max: {stats[max_key]:.4f}]")
        
        print("="*100)


def display_all_results_summary(datasets: Optional[List[str]] = None,
                                models: Optional[List[str]] = None,
                                time_spans: Optional[List[int]] = None,
                                format: str = 'grid',
                                detailed: bool = False,
                                group_by: Optional[str] = None,
                                export_csv: bool = False,
                                show_stats: bool = True,
                                show_training_times: bool = True,
                                top_n: Optional[int] = None):
    """
    Convenience function to display comprehensive results summary
    
    Args:
        datasets: List of dataset names to include (None = all)
        models: List of model names to include (None = all)
        time_spans: List of time spans to include (None = all)
        format: Table format ('grid', 'simple', 'fancy_grid', 'pipe', 'html', 'latex')
        detailed: If True, show all metrics including confusion matrix
        group_by: Group results by 'dataset', 'model', 'timespan', or None for no grouping
        export_csv: If True, export results to CSV
        show_stats: If True, display summary statistics
        show_training_times: If True, display training time statistics
        top_n: If specified, show top N performers by F1 score
    """
    summary = ResultsSummary()
    
    # Collect all metrics
    df = summary.collect_all_metrics(datasets=datasets, models=models, time_spans=time_spans)
    
    if df.empty:
        print("No results found!")
        return
    
    # Display based on grouping
    if group_by == 'dataset':
        summary.display_by_dataset(df, format=format)
    elif group_by == 'model':
        summary.display_by_model(df, format=format)
    elif group_by == 'timespan' or group_by == 'time_span':
        summary.display_by_timespan(df, format=format)
    else:
        if detailed:
            summary.display_detailed_table(df, format=format)
        else:
            summary.display_summary_table(df, format=format)
    
    # Show top performers
    if top_n and top_n > 0:
        print("\n" + "="*100)
        print(f"TOP {top_n} PERFORMERS (by F1 Score)")
        print("="*100)
        top_df = summary.get_best_performers(df, metric='f1_score', top_n=top_n)
        summary.display_summary_table(top_df, format=format)
    
    # Show training times
    if show_training_times:
        summary.display_training_times(df, format=format)
    
    # Show statistics
    if show_stats:
        summary.display_statistics(df)
    
    # Export to CSV
    if export_csv:
        summary.export_to_csv(df)
    
    return df


if __name__ == "__main__":
    # Simple test when run directly
    display_all_results_summary()
