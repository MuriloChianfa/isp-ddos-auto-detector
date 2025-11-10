"""
Batch Evaluation Module
Provides functionality to run evaluations across multiple combinations of datasets, models, and time spans.
"""

from itertools import product
from datetime import datetime
from framework.models import list_available_models
from framework.constants import SUPPORTED_TIME_SPANS
from config import DATASETS


def run_batch_evaluation(main_func, datasets=None, models=None, time_spans=None, dry_run=False, **kwargs):
    """
    Run batch evaluations across multiple combinations
    
    Args:
        main_func: The main evaluation function to call for each combination
        datasets: List of dataset names (None = all available)
        models: List of model names (None = all available)
        time_spans: List of time spans (None = all supported)
        dry_run: If True, only print what would be run
        **kwargs: Additional arguments passed to main_func()
    
    Returns:
        Dictionary with batch evaluation results and statistics
    """
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
    
    print("="*80)
    print("BATCH EVALUATION MODE")
    print("="*80)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Models: {', '.join(models)}")
    print(f"Time Spans: {', '.join(map(str, time_spans))} seconds")
    print(f"Total combinations: {total}")
    print(f"Mode: {'DRY RUN' if dry_run else 'EXECUTION'}")
    print("="*80)
    
    if not dry_run:
        response = input(f"\nProceed with {total} evaluations? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled by user")
            return None
    
    # Track results
    successful = 0
    failed = 0
    failures = []
    start_time = datetime.now()
    
    # Run evaluations
    for idx, (dataset, model, time_span) in enumerate(combinations, 1):
        print(f"\n{'='*80}")
        print(f"Evaluation {idx}/{total}")
        print(f"Dataset: {dataset}, Model: {model}, Time Span: {time_span}s")
        print(f"{'='*80}")
        
        if dry_run:
            print(f"[DRY RUN] Would run: dataset={dataset}, model={model}, time_span={time_span}")
            successful += 1
            continue
        
        try:
            # Update kwargs for this combination
            eval_kwargs = kwargs.copy()
            eval_kwargs['model_name'] = model
            eval_kwargs['time_span'] = time_span
            
            # Run the evaluation
            result = main_func(dataset_name=dataset, **eval_kwargs)
            
            if result is not None:
                print(f"✓ Completed successfully")
                successful += 1
            else:
                print(f"✗ Failed (returned None)")
                failed += 1
                failures.append((dataset, model, time_span))
        except KeyboardInterrupt:
            print("\n\nBatch evaluation interrupted by user")
            break
        except Exception as e:
            print(f"✗ Failed with error: {e}")
            failed += 1
            failures.append((dataset, model, time_span))
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    print(f"Total evaluations: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Duration: {duration}")
    
    if failures:
        print(f"\nFailed evaluations:")
        for dataset, model, time_span in failures:
            print(f"  - Dataset: {dataset}, Model: {model}, Time Span: {time_span}s")
    
    print("="*80)
    
    # Return results dictionary
    return {
        'total': total,
        'successful': successful,
        'failed': failed,
        'failures': failures,
        'duration': duration,
        'start_time': start_time,
        'end_time': end_time
    }
