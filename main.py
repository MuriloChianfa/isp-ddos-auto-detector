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


def main(dataset_name=None, save_run_name=None, **kwargs):
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
        
        # Check if this was optimization-only mode
        if results and results.get('status') == 'optimization_complete':
            print(f"\n{results.get('message', 'Optimization completed')}")
            return results
        
        print(f"\nAnalysis completed successfully!")
        summary = pipeline.get_results_summary()
        if summary.get('results_directory'):
            print(f"Results available in: {summary['results_directory']}")
        
        # Save run if requested
        if save_run_name:
            from framework.versioning import RunVersionManager
            import glob
            
            manager = RunVersionManager()
            run_name = manager.get_unique_run_name(save_run_name)
            
            print(f"\nSaving run as: {run_name}")
            
            # Collect metadata about this run
            metadata = {
                'datasets': [dataset_name],
                'models': [kwargs.get('model_name', 'autoencoder')],
                'time_spans': [kwargs.get('time_span', DEFAULT_TIME_SPAN)],
                'total_analyses': 1,
                'optimized': kwargs.get('optimize', False),
                'optimization_iterations': kwargs.get('optimize_n_iter', 0) if kwargs.get('optimize', False) else 0
            }
            
            # Find and copy all result directories for this dataset/model/timespan
            results_base = f"./results/{dataset_name}"
            if os.path.exists(results_base):
                manager.copy_results_to_version(run_name, results_base)
            
            # Also copy cross-evaluation and summary results if they exist
            if os.path.exists("./results/cross_evaluation"):
                manager.copy_results_to_version(run_name, "./results/cross_evaluation")
            if os.path.exists("./results/summary"):
                manager.copy_results_to_version(run_name, "./results/summary")
            
            # Register the run
            manager.register_run(run_name, metadata)
            
            print(f"Run saved successfully!")
            print(f"Version path: {manager.get_run_path(run_name)}")
        
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

  # Batch optimization for all datasets/models/time spans
  python main.py --batch --optimize --optimize-n-iter 10
  
  # Batch optimization - specific combinations
  python main.py --batch --optimize --optimize-n-iter 10 \\
    --batch-datasets itp-downstream-http-flood \\
    --batch-models isolation_forest --batch-models one_class_svm \\
    --batch-time-spans 60 --batch-time-spans 300

  # Hyperparameter optimization only (saves params to importable .py file)
  python main.py -d itp-downstream-http-flood -t 300 -m one_class_svm --optimize --optimize-n-iter 10
  
  # Optimize and continue training with best parameters
  python main.py -d itp-downstream-http-flood -t 300 -m one_class_svm --optimize-and-train --optimize-n-iter 10

  # Dry run to preview
  python main.py --batch-dry-run
  
  # Display summary of all results
  python main.py --summary
  
  # Display only optimal parameters status
  python main.py --optimal-params-status
  
  # Display detailed summary with all metrics
  python main.py --summary --summary-detailed
  
  # Display summary without optimal params status
  python main.py --summary --summary-no-optimal-params
  
  # Display summary grouped by dataset
  python main.py --summary --summary-group-by dataset
  
  # Display summary for specific datasets and models
  python main.py --summary \\
    --summary-datasets itp-multivector-udp-100gbps-peak \\
    --summary-models isolation_forest --summary-models autoencoder
  
  # Display top 10 performers and export to CSV
  python main.py --summary --summary-top-n 10 --summary-export-csv
  
  # Display summary in LaTeX format
  python main.py --summary --summary-format latex
  
  # Clear evaluation cache
  python main.py --clear-cache
  
  # Display optimal parameters optimization status
  python main.py --optimal-params-status
  
  # Display optimal params status for specific combinations
  python main.py --optimal-params-status \\
    --summary-datasets itp-multivector-udp-100gbps-peak \\
    --summary-time-spans 1 --summary-time-spans 10
  
  # Generate cross-evaluation plots
  python main.py --cross-evaluation
  
  # Generate cross-evaluation plots with filters
  python main.py --cross-evaluation --cross-datasets itp-multivector-udp-100gbps-peak
  
  # Generate feature plots for dataset
  python main.py --generate-plots
  
  # Generate feature plots for specific dataset and time span
  python main.py --generate-plots -d itp-downstream-http-flood -t 60
  
  # Save current run as a versioned experiment
  python main.py -d itp-downstream-http-flood -t 300 -m autoencoder --save-run "baseline_experiment"
  
  # List all saved runs
  python main.py --list-runs
  
  # Compare multiple runs
  python main.py --compare-runs 20251116_120000 20251116_140000 --compare-baseline 20251116_120000
  
  # Analyze feature correlations
  python main.py --analyze-correlation
  
  # Analyze correlations for specific datasets
  python main.py --analyze-correlation \\
    --correlation-datasets itp-multivector-udp-100gbps-peak
  
  # Analyze correlations with custom threshold
  python main.py --analyze-correlation --correlation-threshold 0.90
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
        help='Run hyperparameter optimization and save results, then stop (do not train model). Use --optimize-and-train to continue training after optimization.'
    )
    parser.add_argument(
        '--optimize-n-iter',
        type=int,
        default=10,
        help='Number of iterations for random search optimization (default: 10)'
    )
    parser.add_argument(
        '--optimize-and-train',
        action='store_true',
        help='Run hyperparameter optimization, then continue training with the optimized parameters'
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
    parser.add_argument(
        '--summary-no-optimal-params',
        action='store_true',
        help='Disable optimal parameters status display'
    )
    parser.add_argument(
        '--optimal-params-status',
        action='store_true',
        help='Display optimal parameters status and exit'
    )
    parser.add_argument(
        '--analyze-correlation',
        action='store_true',
        help='Analyze feature correlations on training data and identify features to drop'
    )
    parser.add_argument(
        '--correlation-threshold',
        type=float,
        default=0.90,
        help='Correlation threshold for dropping features (default: 0.90)'
    )
    parser.add_argument(
        '--correlation-datasets',
        type=str,
        action='append',
        help='Datasets to analyze correlations for (can be specified multiple times). If not specified, all datasets are used.'
    )
    parser.add_argument(
        '--correlation-time-spans',
        type=int,
        action='append',
        help='Time spans to analyze correlations for (can be specified multiple times). If not specified, all time spans are used.'
    )
    parser.add_argument(
        '--cross-evaluation',
        action='store_true',
        help='Run cross-evaluation analysis to compare models across datasets and time windows'
    )
    parser.add_argument(
        '--cross-datasets',
        type=str,
        action='append',
        help='Datasets for cross-evaluation (can be specified multiple times). If not specified, all datasets are used.'
    )
    parser.add_argument(
        '--cross-windows',
        type=str,
        action='append',
        help='Time windows for cross-evaluation (e.g., 1seconds, 60seconds). If not specified, all windows are used.'
    )
    parser.add_argument(
        '--cross-models',
        type=str,
        action='append',
        help='Models for cross-evaluation (can be specified multiple times). If not specified, all models are used.'
    )
    parser.add_argument(
        '--cross-regenerate-curves',
        action='store_true',
        help='Automatically regenerate missing curve data during cross-evaluation'
    )
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        help='Generate feature visualization plots for dataset'
    )
    parser.add_argument(
        '--plots-no-comparisons',
        action='store_true',
        help='Skip generating cross-dataset comparison plots when using --generate-plots'
    )
    parser.add_argument(
        '--plots-max-processes',
        type=int,
        default=12,
        help='Maximum number of processes for parallel plot generation (default: 12)'
    )
    parser.add_argument(
        '--save-run',
        type=str,
        metavar='NAME',
        help='Save this run as a versioned experiment with the given name for later comparison'
    )
    parser.add_argument(
        '--list-runs',
        action='store_true',
        help='List all saved versioned runs and exit'
    )
    parser.add_argument(
        '--compare-runs',
        type=str,
        nargs='+',
        metavar='RUN_ID',
        help='Compare multiple saved runs by their IDs (space-separated)'
    )
    parser.add_argument(
        '--compare-baseline',
        type=str,
        metavar='RUN_ID',
        help='Specify baseline run ID for delta calculations in comparison'
    )
    parser.add_argument(
        '--compare-output',
        type=str,
        default='./results/comparisons',
        help='Output directory for comparison reports (default: ./results/comparisons)'
    )
    
    args = parser.parse_args()
    
    # Handle run versioning commands
    if args.list_runs:
        from framework.versioning import RunVersionManager
        manager = RunVersionManager()
        runs = manager.list_runs()
        
        if not runs:
            print("No saved runs found.")
        else:
            print(f"\n{'='*80}")
            print(f"SAVED RUNS ({len(runs)} total)")
            print(f"{'='*80}\n")
            
            for run in runs:
                print(f"Run Name: {run['run_name']}")
                print(f"Timestamp: {run['timestamp']}")
                
                metadata = run.get('metadata', {})
                if metadata:
                    print(f"Configuration:")
                    if 'datasets' in metadata:
                        print(f"  Datasets: {', '.join(metadata['datasets'])}")
                    if 'models' in metadata:
                        print(f"  Models: {', '.join(metadata['models'])}")
                    if 'time_spans' in metadata:
                        print(f"  Time spans: {', '.join(map(str, metadata['time_spans']))}")
                    if 'total_analyses' in metadata:
                        print(f"  Total analyses: {metadata['total_analyses']}")
                
                print()
        exit(0)
    
    if args.compare_runs:
        from framework.comparison import compare_runs
        
        print(f"\nComparing {len(args.compare_runs)} runs...")
        comparison_df = compare_runs(
            run_ids=args.compare_runs,
            output_dir=args.compare_output,
            baseline_run_id=args.compare_baseline
        )
        exit(0)
    
    if args.cross_evaluation:
        from framework.cross import run_cross_evaluation
        run_cross_evaluation(
            datasets=args.cross_datasets,
            windows=args.cross_windows,
            models=args.cross_models,
            results_dir='./results',
            output_dir='./results/cross_evaluation',
            regenerate_curves=args.cross_regenerate_curves
        )
        
        # Save run if requested
        if args.save_run:
            from framework.versioning import RunVersionManager
            manager = RunVersionManager()
            run_name = manager.get_unique_run_name(args.save_run)
            
            print(f"\nSaving cross-evaluation run as: {run_name}")
            
            metadata = {
                'operation': 'cross_evaluation',
                'datasets': args.cross_datasets,
                'windows': args.cross_windows,
                'models': args.cross_models
            }
            
            # Copy cross-evaluation results
            if os.path.exists("./results/cross_evaluation"):
                manager.copy_results_to_version(run_name, "./results/cross_evaluation")
            
            manager.register_run(run_name, metadata)
            print(f"Run saved successfully! Version path: {manager.get_run_path(run_name)}")
        
        exit(0)
    
    if args.generate_plots:
        from framework.plots import run_feature_plots
        result = run_feature_plots(
            dataset_name=args.dataset,
            time_span=args.time_span,
            no_comparisons=args.plots_no_comparisons,
            max_processes=args.plots_max_processes
        )
        
        # Save run if requested
        if args.save_run:
            from framework.versioning import RunVersionManager
            manager = RunVersionManager()
            run_name = manager.get_unique_run_name(args.save_run)
            
            print(f"\nSaving feature plots run as: {run_name}")
            
            metadata = {
                'operation': 'generate_plots',
                'dataset': args.dataset,
                'time_span': args.time_span
            }
            
            # Copy results
            if args.dataset:
                dataset_path = f"./results/{args.dataset}"
                if os.path.exists(dataset_path):
                    manager.copy_results_to_version(run_name, dataset_path)
            
            manager.register_run(run_name, metadata)
            print(f"Run saved successfully! Version path: {manager.get_run_path(run_name)}")
        
        exit(result)
    
    if args.analyze_correlation:
        from framework.correlation import run_correlation_analysis
        run_correlation_analysis(
            datasets=args.correlation_datasets,
            time_spans=args.correlation_time_spans,
            threshold=args.correlation_threshold
        )
        
        # Save run if requested
        if args.save_run:
            from framework.versioning import RunVersionManager
            manager = RunVersionManager()
            run_name = manager.get_unique_run_name(args.save_run)
            
            print(f"\nSaving correlation analysis run as: {run_name}")
            
            metadata = {
                'operation': 'correlation_analysis',
                'threshold': args.correlation_threshold,
                'datasets': args.correlation_datasets,
                'time_spans': args.correlation_time_spans
            }
            
            # Copy correlation results from all analyzed datasets
            datasets_to_save = args.correlation_datasets if args.correlation_datasets else list(DATASETS.keys())
            for dataset in datasets_to_save:
                dataset_path = f"./results/{dataset}"
                if os.path.exists(dataset_path):
                    manager.copy_results_to_version(run_name, dataset_path)
            
            manager.register_run(run_name, metadata)
            print(f"Run saved successfully! Version path: {manager.get_run_path(run_name)}")
        
        exit(0)
    
    if args.clear_cache:
        from framework.stats import EvaluationCache
        cache = EvaluationCache()
        cache.clear_cache()
        exit(0)
    
    if args.optimal_params_status:
        from framework.summary import ResultsSummary
        summary = ResultsSummary()
        summary.display_optimal_params_status(
            datasets=args.summary_datasets if hasattr(args, 'summary_datasets') else None,
            models=args.summary_models if hasattr(args, 'summary_models') else None,
            time_spans=args.summary_time_spans if hasattr(args, 'summary_time_spans') else None,
            format=args.summary_format if hasattr(args, 'summary_format') else 'github'
        )
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
            show_optimal_params_status=not args.summary_no_optimal_params,
            top_n=args.summary_top_n
        )
        exit(0)
    
    if args.list_datasets:
        list_datasets()
        exit(0)
        
    if args.list_models:
        list_models()
        exit(0)
    
    # Handle standalone --save-run (save current results without running anything)
    if args.save_run and not args.batch and not args.batch_dry_run and not args.dataset:
        from framework.versioning import RunVersionManager
        
        if not os.path.exists("./results") or not any(os.path.isdir(os.path.join("./results", d)) 
                                                       for d in os.listdir("./results") 
                                                       if d not in ["versions", "comparisons"]):
            print("Error: No results found to save. Run an analysis first, then use --save-run.")
            exit(1)
        
        manager = RunVersionManager()
        run_name = manager.get_unique_run_name(args.save_run)
        
        print(f"\nSaving current results as: {run_name}")
        
        metadata = {
            'operation': 'save_existing_results',
            'note': 'Saved existing results without running new analysis'
        }
        
        # Copy all results except versions and comparisons
        saved_items = []
        for item in os.listdir("./results"):
            item_path = os.path.join("./results", item)
            if os.path.isdir(item_path) and item not in ["versions", "comparisons"]:
                manager.copy_results_to_version(run_name, item_path)
                saved_items.append(item)
        
        metadata['saved_items'] = saved_items
        
        # Register the run
        manager.register_run(run_name, metadata)
        
        print(f"Results saved successfully!")
        print(f"Saved items: {', '.join(saved_items)}")
        print(f"Version path: {manager.get_run_path(run_name)}")
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
            'optimize': args.optimize or args.optimize_and_train,
            'optimize_n_iter': args.optimize_n_iter,
            'optimize_only': args.optimize and not args.optimize_and_train
        }
        
        batch_results = run_batch_evaluation(
            main_func=main,
            datasets=datasets,
            models=models,
            time_spans=time_spans,
            dry_run=args.batch_dry_run,
            force=args.force,
            **kwargs
        )
        
        # Save batch run if requested
        if args.save_run and not args.batch_dry_run and batch_results:
            from framework.versioning import RunVersionManager
            
            manager = RunVersionManager()
            run_name = manager.get_unique_run_name(args.save_run)
            
            print(f"\nSaving batch run as: {run_name}")
            
            # Collect metadata about this batch run
            metadata = {
                'datasets': datasets if datasets else list(DATASETS.keys()),
                'models': models if models else list_available_models(),
                'time_spans': time_spans if time_spans else SUPPORTED_TIME_SPANS,
                'total_analyses': batch_results.get('completed', 0),
                'batch_mode': True,
                'optimized': args.optimize or args.optimize_and_train,
                'optimization_iterations': args.optimize_n_iter if (args.optimize or args.optimize_and_train) else 0
            }
            
            # Copy all results
            if os.path.exists("./results"):
                for item in os.listdir("./results"):
                    item_path = os.path.join("./results", item)
                    if os.path.isdir(item_path) and item != "versions":
                        manager.copy_results_to_version(run_name, item_path)
            
            # Register the run
            manager.register_run(run_name, metadata)
            
            print(f"Batch run saved successfully!")
            print(f"Version path: {manager.get_run_path(run_name)}")
        
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
        'optimize': args.optimize or args.optimize_and_train,
        'optimize_n_iter': args.optimize_n_iter,
        'optimize_only': args.optimize and not args.optimize_and_train
    }

    if not kwargs['use_cache']:
        print("Caching disabled, we will reload all the data from scratch")

    main(dataset_name=args.dataset, save_run_name=args.save_run, **kwargs)
