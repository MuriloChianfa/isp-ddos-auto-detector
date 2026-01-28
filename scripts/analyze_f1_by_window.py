#!/usr/bin/env python3
"""
Script to analyze average F1-score for each algorithm across time windows.
This script reads the cross-evaluation results and generates specific F1-score analysis.

Usage:
    python analyze_f1_by_window.py
    python analyze_f1_by_window.py --datasets itp-downstream-http-flood
    python analyze_f1_by_window.py --models autoencoder isolation_forest
    python analyze_f1_by_window.py --windows 1seconds 60seconds
    python analyze_f1_by_window.py --version 0_90_pcc_n_iter_5
"""

import argparse
from pathlib import Path
from framework.cross import CrossEvaluator


def main():
    parser = argparse.ArgumentParser(
        description='Analyze average F1-score for each algorithm across time windows',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Filter by specific datasets (e.g., itp-downstream-http-flood)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='Filter by specific models (e.g., autoencoder isolation_forest)'
    )
    
    parser.add_argument(
        '--windows',
        nargs='+',
        help='Filter by specific time windows (e.g., 1seconds 60seconds 300seconds)'
    )
    
    parser.add_argument(
        '--results-dir',
        default='./results',
        help='Base directory containing model results (default: ./results)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results/cross_evaluation',
        help='Output directory for analysis (default: ./results/cross_evaluation)'
    )
    
    parser.add_argument(
        '--version',
        help='Analyze a specific version (e.g., 0_90_pcc_n_iter_5). Will use results/versions/{version}/ as results directory'
    )
    
    args = parser.parse_args()
    
    # Adjust paths if version is specified
    if args.version:
        version_base = Path(args.results_dir) / 'versions' / args.version
        if not version_base.exists():
            print(f"\nError: Version directory not found: {version_base}")
            print(f"Available versions:")
            versions_dir = Path(args.results_dir) / 'versions'
            if versions_dir.exists():
                for v in sorted(versions_dir.iterdir()):
                    if v.is_dir():
                        print(f"  - {v.name}")
            else:
                print(f"  No versions directory found at {versions_dir}")
            return
        
        args.results_dir = str(version_base)
        if args.output_dir == './results/cross_evaluation':
            args.output_dir = str(version_base / 'cross_evaluation')
    
    print("\n" + "="*80)
    print("F1-SCORE ANALYSIS BY ALGORITHM AND TIME WINDOW")
    print("="*80)
    
    if args.version:
        print(f"Analyzing version: {args.version}")
        print(f"Results directory: {args.results_dir}")
    
    if args.datasets:
        print(f"Filtering datasets: {', '.join(args.datasets)}")
    if args.models:
        print(f"Filtering models: {', '.join(args.models)}")
    if args.windows:
        print(f"Filtering windows: {', '.join(args.windows)}")
    
    print()
    
    # Create evaluator
    evaluator = CrossEvaluator(
        results_base_dir=args.results_dir,
        output_dir=args.output_dir
    )
    
    # Collect metrics
    print("Collecting metrics from analysis files...")
    metrics_df = evaluator.collect_metrics_from_analysis(
        datasets=args.datasets,
        windows=args.windows,
        models=args.models
    )
    
    if metrics_df.empty:
        print("\nNo metrics found. Please run some evaluations first.")
        print("Example: python main.py --batch")
        return
    
    # Generate F1-score analysis
    f1_results = evaluator.generate_f1_by_algorithm_and_window(metrics_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated files in: {args.output_dir}")
    print("  - f1_score_by_algorithm_and_window.csv")
    print("  - f1_score_heatmap_algorithm_vs_window.png")
    print("  - f1_score_barplot_algorithm_vs_window.png")
    print("\n")


if __name__ == '__main__':
    main()
