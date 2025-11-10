import matplotlib
import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
matplotlib.rcParams['agg.path.chunksize'] = 10000

from framework.pipeline import DDoSDetectorPipeline
from framework.settings import SettingsManager
from framework.models import list_available_models
from framework.constants import SUPPORTED_TIME_SPANS
from framework.batch import run_batch_evaluation
from config import DATASETS, DEFAULT_DATASET, DEFAULT_TIME_SPAN


def main(dataset_name=None, **kwargs):
    """Main entry point for DDoS detection analysis"""
    
    # Use default dataset if none specified
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET

    pipeline = DDoSDetectorPipeline(
        dataset_name=dataset_name,
        **kwargs
    )
    
    try:
        results = pipeline.run_complete_analysis()
        
        print(f"\nAnalysis completed successfully!")
        summary = pipeline.get_results_summary()
        if summary.get('results_directory'):
            print(f"Results available in: {summary['results_directory']}")
        
        return results
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return None


def list_datasets():
    SettingsManager.list_available_datasets()


def list_models():
    SettingsManager.list_available_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ISP DDoS Auto Detector",
        epilog="""
Examples:
  # Single evaluation
  python main.py -d itp-multivector-udp-100gbps-peak -m isolation_forest -t 300
  
  # Batch evaluation - all combinations (64 total)
  python main.py --batch
  
  # Batch evaluation - specific dataset with all models and time spans
  python main.py --batch --batch-datasets itp-multivector-udp-100gbps-peak
  
  # Batch evaluation - specific models and time spans
  python main.py --batch \\
    --batch-models isolation_forest --batch-models one_class_svm \\
    --batch-time-spans 60 --batch-time-spans 300
  
  # Batch evaluation - full custom combination
  python main.py --batch \\
    --batch-datasets itp-multivector-udp-100gbps-peak \\
    --batch-models isolation_forest --batch-models one_class_svm \\
    --batch-time-spans 60 --batch-time-spans 300
  
  # Dry run to preview
  python main.py --batch-dry-run
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
        choices=SUPPORTED_TIME_SPANS,
        default=DEFAULT_TIME_SPAN,
        help=f'Time span for feature aggregation in seconds. Options: {", ".join(map(str, SUPPORTED_TIME_SPANS))}. Default: {DEFAULT_TIME_SPAN}'
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
        help='Maximum number of processes to use for parallel processing (default: 48)'
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
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run hyperparameter optimization before training'
    )
    parser.add_argument(
        '--optimize-n-iter',
        type=int,
        default=10,
        help='Number of iterations for random search optimization (default: 10)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run batch evaluation across all combinations of datasets, models, and time spans (or filtered subsets)'
    )
    parser.add_argument(
        '--batch-dry-run',
        action='store_true',
        help='Show what batch evaluations would run without executing them'
    )
    parser.add_argument(
        '--batch-datasets',
        type=str,
        action='append',
        dest='datasets',
        help='Datasets for batch mode (can be specified multiple times). If not specified, all datasets are used.'
    )
    parser.add_argument(
        '--batch-models',
        type=str,
        action='append',
        dest='models',
        help='Models for batch mode (can be specified multiple times). If not specified, all models are used.'
    )
    parser.add_argument(
        '--batch-time-spans',
        type=int,
        action='append',
        dest='time_spans',
        help='Time spans for batch mode (can be specified multiple times). If not specified, all time spans are used.'
    )
    
    args = parser.parse_args()
    
    if args.list_datasets:
        list_datasets()
        exit(0)
        
    if args.list_models:
        list_models()
        exit(0)
    
    # Batch evaluation mode
    if args.batch or args.batch_dry_run:
        if hasattr(args, 'datasets') and args.datasets:
            datasets = args.datasets
        elif args.dataset:
            datasets = [args.dataset]
        else:
            datasets = None  # Will use all datasets
            
        if hasattr(args, 'models') and args.models:
            models = args.models
        elif args.model != 'autoencoder':
            models = [args.model]
        else:
            models = None  # Will use all models
            
        if hasattr(args, 'time_spans') and args.time_spans:
            time_spans = args.time_spans
        elif args.time_span != DEFAULT_TIME_SPAN:
            time_spans = [args.time_span]
        else:
            time_spans = None  # Will use all time_spans
        
        kwargs = {
            'use_cache': not args.no_cache,
            'force_regenerate': args.force_regenerate,
            'max_processes': args.max_processes,
            'generate_reconstruction_error': args.generate_reconstruction_error,
            'force_retrain': args.force_retrain,
            'evaluate_performance': args.evaluate_performance,
            'performance_samples': args.performance_samples,
            'optimize': args.optimize,
            'optimize_n_iter': args.optimize_n_iter
        }
        
        run_batch_evaluation(
            main_func=main,
            datasets=datasets,
            models=models,
            time_spans=time_spans,
            dry_run=args.batch_dry_run,
            **kwargs
        )
        exit(0)
    
    kwargs = {
        'model_name': args.model,
        'time_span': args.time_span,
        'use_cache': not args.no_cache,
        'force_regenerate': args.force_regenerate,
        'max_processes': args.max_processes,
        'generate_reconstruction_error': args.generate_reconstruction_error,
        'force_retrain': args.force_retrain,
        'evaluate_performance': args.evaluate_performance,
        'performance_samples': args.performance_samples,
        'optimize': args.optimize,
        'optimize_n_iter': args.optimize_n_iter
    }

    if not kwargs['use_cache']:
        print("Caching disabled, we will reload all the data from scratch")

    main(dataset_name=args.dataset, **kwargs)
