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
  
  # Batch evaluation on all combinations (cached results will be skipped)
  python main.py --batch
  
  # Batch evaluation - force re-run everything
  python main.py --batch --force
  
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
  
  # Display summary of all results
  python main.py --summary
  
  # Display detailed summary with all metrics
  python main.py --summary --summary-detailed
  
  # Display summary grouped by dataset
  python main.py --summary --summary-group-by dataset
  
  # Display summary for specific datasets and models
  python main.py --summary \\
    --summary-datasets itp-multivector-udp-100gbps-peak \\
    --summary-models isolation_forest --summary-models autoencoder
  
  # Display top 10 performers and export to CSV
  python main.py --summary --summary-top-n 10 --summary-export-csv
  
  # Display summary in GitHub markdown format
  python main.py --summary --summary-format github
  
  # Clear evaluation cache
  python main.py --clear-cache
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
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-evaluation even if cached results exist (batch mode only)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear evaluation cache and exit'
    )
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Display summary of all evaluation results and exit'
    )
    parser.add_argument(
        '--summary-datasets',
        type=str,
        action='append',
        help='Filter summary by specific datasets (can be specified multiple times)'
    )
    parser.add_argument(
        '--summary-models',
        type=str,
        action='append',
        help='Filter summary by specific models (can be specified multiple times)'
    )
    parser.add_argument(
        '--summary-time-spans',
        type=int,
        action='append',
        help='Filter summary by specific time spans (can be specified multiple times)'
    )
    parser.add_argument(
        '--summary-format',
        type=str,
        choices=['grid', 'simple', 'fancy_grid', 'pipe', 'html', 'latex', 'github'],
        default='github',
        help='Table format for summary display (default: github)'
    )
    parser.add_argument(
        '--summary-detailed',
        action='store_true',
        help='Show detailed summary with all metrics'
    )
    parser.add_argument(
        '--summary-group-by',
        type=str,
        choices=['dataset', 'model', 'timespan'],
        help='Group summary results by dataset, model, or timespan'
    )
    parser.add_argument(
        '--summary-export-csv',
        action='store_true',
        help='Export summary to CSV file'
    )
    parser.add_argument(
        '--summary-top-n',
        type=int,
        help='Show top N best performers by F1 score'
    )
    parser.add_argument(
        '--summary-no-stats',
        action='store_true',
        help='Disable summary statistics display'
    )
    
    args = parser.parse_args()
    
    if args.clear_cache:
        from framework.evaluation_cache import EvaluationCache
        cache = EvaluationCache()
        cache.clear_cache()
        exit(0)
    
    if args.summary:
        from framework.summary import display_all_results_summary
        
        display_all_results_summary(
            datasets=args.summary_datasets,
            models=args.summary_models,
            time_spans=args.summary_time_spans,
            format=args.summary_format,
            detailed=args.summary_detailed,
            group_by=args.summary_group_by,
            export_csv=args.summary_export_csv,
            show_stats=not args.summary_no_stats,
            top_n=args.summary_top_n
        )
        exit(0)
    
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
            force=args.force,
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
