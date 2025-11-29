"""
Batch Evaluation Module
Provides functionality to run evaluations across multiple combinations of datasets, models, and time spans.
"""

from itertools import product
from datetime import datetime
from framework.models import list_available_models
from framework.constants import SUPPORTED_TIME_SPANS
from framework.stats import EvaluationCache
from config import DATASETS, MODEL_DEFAULT_PARAMS, MODEL_THRESHOLD_STRATEGIES


def run_batch_evaluation(main_func, datasets=None, models=None, time_spans=None, dry_run=False, force=False, **kwargs):
    """
    Run batch evaluations across multiple combinations
    
    Args:
        main_func: The main evaluation function to call for each combination
        datasets: List of dataset names (None = all available)
        models: List of model names (None = all available)
        time_spans: List of time spans (None = all supported)
        dry_run: If True, only print what would be run
        force: If True, re-run even if cached
        **kwargs: Additional arguments passed to main_func()
    
    Returns:
        Dictionary with batch evaluation results and statistics
    """
    # Initialize cache
    cache = EvaluationCache()
    # Get all options if not specified
    if datasets is None:
        datasets = list(DATASETS.keys())
    if models is None:
        models = list_available_models()
    if time_spans is None:
        time_spans = SUPPORTED_TIME_SPANS
    
    # Generate all combinations
    # Put time_spans first in product, then reverse it so all 1-second runs execute last
    combinations = list(product(reversed(time_spans), datasets, models))
    # Reorder tuple to (dataset, model, time_span)
    combinations = [(dataset, model, time_span) for time_span, dataset, model in combinations]
    total = len(combinations)
    
    # Check cache status for all combinations
    cached_count = 0
    if not force and not dry_run:
        for dataset, model, time_span in combinations:
            dataset_config = DATASETS.get(dataset, {})
            window_config = dataset_config.get('windows', {}).get(str(time_span), {})
            
            params = window_config.get('params', {}).get(model)
            if params is None:
                params = MODEL_DEFAULT_PARAMS.get(model, {})
            
            feature_config_dict = window_config.get('feature_config', {})
            if isinstance(feature_config_dict, dict):
                feature_config = feature_config_dict.get(model)
            else:
                feature_config = None
            
            if feature_config is None:
                feature_config = dataset_config.get('feature_config')
            
            threshold_strategies = window_config.get('threshold_strategies', {})
            threshold_strategy = threshold_strategies.get(model)
            if threshold_strategy is None:
                threshold_strategy = MODEL_THRESHOLD_STRATEGIES.get(model)
            
            if cache.is_cached(dataset, model, time_span, params, feature_config, threshold_strategy):
                cached_count += 1
    
    print("="*80)
    print("BATCH EVALUATION MODE")
    print("="*80)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: {', '.join(models)}")
    print(f"Time Spans: {', '.join(map(str, time_spans))} seconds")
    print(f"Total combinations: {total}")
    
    # Check if this is optimization mode
    is_optimization = kwargs.get('optimize', False) and kwargs.get('optimize_only', False)
    
    if is_optimization:
        print(f"Mode: HYPERPARAMETER OPTIMIZATION")
        print(f"Iterations per combination: {kwargs.get('optimize_n_iter', 10)}")
    else:
        if cached_count > 0 and not force:
            print(f"Cached: {cached_count} ({cached_count/total*100:.1f}%)")
            print(f"To run: {total - cached_count} ({(total-cached_count)/total*100:.1f}%)")
        print(f"Mode: {'DRY RUN' if dry_run else 'EVALUATION'}")
    
    print("="*80)
    
    if not dry_run:
        import sys
        # Skip prompt if force flag is set, stdin is not interactive (e.g., in CI/CD or piped commands)
        if force:
            print(f"\nAuto-proceeding with {total} {'optimizations' if is_optimization else 'evaluations'} (force mode)")
        elif not sys.stdin.isatty():
            print(f"\nAuto-proceeding with {total} {'optimizations' if is_optimization else 'evaluations'} (non-interactive mode)")
        else:
            action = "optimizations" if is_optimization else "evaluations"
            response = input(f"\nProceed with {total} {action}? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Cancelled by user")
                return None
    
    # Track results
    successful = 0
    failed = 0
    cached = 0
    failures = []
    start_time = datetime.now()
    
    # Run evaluations
    for idx, (dataset, model, time_span) in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"{'Optimization' if is_optimization else 'Evaluation'} {idx}/{total}")
        print(f"Dataset: {dataset}, Model: {model}, Time Span: {time_span}s")
        print(f"{'='*80}")
        
        if dry_run:
            print(f"[DRY RUN] Would run: dataset={dataset}, model={model}, time_span={time_span}")
            successful += 1
            continue
        
        # Get configuration for cache check
        dataset_config = DATASETS.get(dataset, {})
        window_config = dataset_config.get('windows', {}).get(str(time_span), {})
        
        # Get model-specific parameters
        params = window_config.get('params', {}).get(model)
        if params is None:
            params = MODEL_DEFAULT_PARAMS.get(model, {})
        
        # Get model-specific features
        feature_config_dict = window_config.get('feature_config', {})
        if isinstance(feature_config_dict, dict):
            feature_config = feature_config_dict.get(model)
        else:
            feature_config = None
        
        if feature_config is None:
            feature_config = dataset_config.get('feature_config')
        
        # Get threshold strategy
        threshold_strategies = window_config.get('threshold_strategies', {})
        threshold_strategy = threshold_strategies.get(model)
        if threshold_strategy is None:
            threshold_strategy = MODEL_THRESHOLD_STRATEGIES.get(model)
        
        # Build results path - check if model directory exists
        results_dir = f"./results/{dataset}/{time_span}seconds/models/{model}"
        results_path = results_dir
        
        # In optimization mode, check if parameters already exist
        if is_optimization:
            import os
            params_file = f"{results_dir}/parameters/optimal_params.py"
            if os.path.exists(params_file) and not force:
                print(f"\nSKIPPED: Optimal parameters already exist")
                print(f"   File: {params_file}")
                print(f"   Use --force to re-optimize")
                cached += 1
                continue
        
        # Check cache (only for evaluation mode)
        if not is_optimization and not force and cache.is_cached(dataset, model, time_span, params, 
                                        feature_config, threshold_strategy):
            cached_info = cache.get_cached_info(dataset, model, time_span, params,
                                               feature_config, threshold_strategy)
            print(f"\nCACHED Skipping evaluation")
            print(f"   Cached at: {cached_info.get('cached_at', 'unknown')}")
            print(f"   Results: {cached_info.get('results_path', 'unknown')}")
            
            if cached_info.get('metrics'):
                metrics = cached_info['metrics']
                print(f"   Metrics: TPR={metrics.get('tpr', 0):.2%}, "
                      f"FPR={metrics.get('fpr', 0):.2%}, "
                      f"F1={metrics.get('f1_score', 0):.3f}")
            
            cached += 1
            continue
        
        try:
            # Update kwargs for this combination
            eval_kwargs = kwargs.copy()
            eval_kwargs['model_name'] = model
            eval_kwargs['time_span'] = time_span
            
            # Run the evaluation
            result = main_func(dataset_name=dataset, **eval_kwargs)
            
            # Handle optimization mode results
            if is_optimization:
                if result and result.get('status') == 'optimization_complete':
                    print(f"Optimization completed successfully")
                    print(f"   Parameters saved to: {result.get('params_file', 'unknown')}")
                    successful += 1
                else:
                    print(f"Optimization failed or returned unexpected result")
                    failed += 1
                    failures.append((dataset, model, time_span))
            else:
                # Handle normal evaluation results
                if result is not None:
                    print(f"Completed successfully")
                    successful += 1
                    
                    try:
                        metrics = None
                        if hasattr(result, 'metrics'):
                            metrics = result.metrics
                        elif isinstance(result, dict) and 'metrics' in result:
                            metrics = result['metrics']
                        
                        cache.add_to_cache(
                            dataset, model, time_span,
                            results_path,
                            params, feature_config, threshold_strategy,
                            metrics
                        )
                    except Exception as e:
                        print(f"Warning: Failed to cache result: {e}")
                    
                else:
                    print(f"Failed (returned None)")
                    failed += 1
                    failures.append((dataset, model, time_span))
        except KeyboardInterrupt:
            print("\n\nBatch evaluation interrupted by user")
            break
        except Exception as e:
            print(f"Failed with error: {e}")
            failed += 1
            failures.append((dataset, model, time_span))
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print(f"BATCH {'OPTIMIZATION' if is_optimization else 'EVALUATION'} SUMMARY")
    print("="*80)
    print(f"Total combinations: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"{'Skipped' if is_optimization else 'Cached'}: {cached} ({cached/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Duration: {duration}")
    
    if failures:
        print(f"\nFailed {'optimizations' if is_optimization else 'evaluations'}:")
        for dataset, model, time_span in failures:
            print(f"  - Dataset: {dataset}, Model: {model}, Time Span: {time_span}s")
    
    print("="*80)
    
    # Return results dictionary
    return {
        'total': total,
        'successful': successful,
        'cached': cached,
        'failed': failed,
        'failures': failures,
        'duration': duration,
        'start_time': start_time,
        'end_time': end_time
    }
