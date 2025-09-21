"""
ISP DDoS Auto Detector - Main Entry Point
A modular pipeline for detecting DDoS attacks using machine learning.
"""

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from framework.pipeline import DDoSDetectorPipeline
from framework.settings import SettingsManager
from framework.models import list_available_models
from config import DATASETS, DEFAULT_DATASET


def main(dataset_name=None, **kwargs):
    """Main entry point for DDoS detection analysis"""
    
    # Use default dataset if none specified
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET
    
    # Initialize and run pipeline
    pipeline = DDoSDetectorPipeline(
        dataset_name=dataset_name,
        **kwargs
    )
    
    try:
        # Run complete analysis
        results = pipeline.run_complete_analysis()
        
        # Print final summary
        print(f"\nAnalysis completed successfully!")
        summary = pipeline.get_results_summary()
        if summary.get('results_directory'):
            print(f"Results available in: {summary['results_directory']}")
        
        return results
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None


def list_datasets():
    """Print available datasets and their descriptions"""
    SettingsManager.list_available_datasets()


def list_models():
    """Print available models and their descriptions"""
    SettingsManager.list_available_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISP DDoS Auto Detector")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help=f'Dataset to use for analysis. Available: {", ".join(DATASETS.keys())}. Default: {DEFAULT_DATASET}'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='autoencoder',
        choices=list_available_models(),
        help=f'Model to use for anomaly detection. Available: {", ".join(list_available_models())}. Default: autoencoder'
    )
    parser.add_argument(
        '--time-span', '-t',
        type=int,
        choices=[10, 60, 300],
        default=300,
        help='Time span for feature aggregation in seconds. Options: 10, 60 or 300. Default: 300'
    )
    parser.add_argument(
        '--use-fixed-threshold',
        action='store_true',
        help='Use the model\'s built-in fixed threshold instead of adaptive calculation (for TCN autoencoder)'
    )
    parser.add_argument(
        '--generate-reconstruction-error',
        action='store_true',
        help='Generate detailed feature reconstruction error visualizations (creates many plots)'
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets and exit'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List all available models and exit'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching system (force reload all data)'
    )
    parser.add_argument(
        '--force-regenerate',
        action='store_true',
        help='Force regeneration of feature CSV files even if they exist'
    )
    parser.add_argument(
        '--max-processes',
        type=int,
        default=None,
        help='Maximum number of processes to use for parallel processing (default: 16)'
    )
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='Force retraining of the model even if saved artifacts exist'
    )
    parser.add_argument(
        '--evaluate-performance',
        action='store_true',
        help='Include real-time performance evaluation in the analysis'
    )
    parser.add_argument(
        '--performance-samples',
        type=int,
        default=1000,
        help='Number of samples to use for performance testing (default: 1000)'
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        list_datasets()
        exit(0)
        
    if args.list_models:
        list_models()
        exit(0)
    
    kwargs = {
        'model_name': args.model,
        'time_span': args.time_span,
        'use_cache': not args.no_cache,
        'force_regenerate': args.force_regenerate,
        'max_processes': args.max_processes,
        'use_fixed_threshold': args.use_fixed_threshold,
        'generate_reconstruction_error': args.generate_reconstruction_error,
        'force_retrain': args.force_retrain,
        'evaluate_performance': args.evaluate_performance,
        'performance_samples': args.performance_samples
    }

    if not kwargs['use_cache']:
        print("Caching disabled - will reload all data from scratch")

    main(dataset_name=args.dataset, **kwargs)
