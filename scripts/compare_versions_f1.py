#!/usr/bin/env python3
"""
Script to compare F1-scores across multiple versions.
This script allows you to compare how different algorithms perform across different experiment versions.

Usage:
    python compare_versions_f1.py --versions 0_50_pcc_n_iter_5 0_90_pcc_n_iter_5 --windows 1seconds 300seconds
    python compare_versions_f1.py --versions 0_50_pcc_n_iter_5 0_70_pcc_n_iter_5 0_90_pcc_n_iter_5 --windows 1seconds 300seconds
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from framework.cross import CrossEvaluator

# Set style for better-looking plots
sns.set_style("whitegrid")


def main():
    parser = argparse.ArgumentParser(
        description='Compare F1-scores across multiple experiment versions',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--versions',
        nargs='+',
        required=True,
        help='List of versions to compare (e.g., 0_50_pcc_n_iter_5 0_90_pcc_n_iter_5)'
    )
    
    parser.add_argument(
        '--datasets',
        nargs='+',
        help='Filter by specific datasets'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='Filter by specific models'
    )
    
    parser.add_argument(
        '--windows',
        nargs='+',
        help='Filter by specific time windows (e.g., 1seconds 300seconds)'
    )
    
    parser.add_argument(
        '--results-dir',
        default='./results',
        help='Base directory containing model results (default: ./results)'
    )
    
    parser.add_argument(
        '--output-dir',
        default='./results/version_comparison',
        help='Output directory for comparison results'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("F1-SCORE COMPARISON ACROSS VERSIONS")
    print("="*80)
    print(f"Comparing versions: {', '.join(args.versions)}")
    if args.windows:
        print(f"Filtering windows: {', '.join(args.windows)}")
    if args.models:
        print(f"Filtering models: {', '.join(args.models)}")
    print()
    
    # Collect data from all versions
    all_data = []
    
    for version in args.versions:
        version_path = Path(args.results_dir) / 'versions' / version
        if not version_path.exists():
            print(f"Warning: Version {version} not found at {version_path}, skipping...")
            continue
        
        print(f"\nCollecting metrics from version: {version}")
        evaluator = CrossEvaluator(
            results_base_dir=str(version_path),
            output_dir=args.output_dir
        )
        
        metrics_df = evaluator.collect_metrics_from_analysis(
            datasets=args.datasets,
            windows=args.windows,
            models=args.models
        )
        
        if not metrics_df.empty:
            # Calculate average F1 for each model-window combination across datasets
            f1_by_model_window = metrics_df.groupby(['model', 'window'])['f1_score'].mean().reset_index()
            f1_by_model_window['version'] = version
            all_data.append(f1_by_model_window)
            
            print(f"  Collected {len(metrics_df)} configurations from {version}")
    
    if not all_data:
        print("\nNo data found for any version!")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / 'version_comparison_f1_scores.csv'
    combined_df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison data to {csv_path}")
    
    # Map model names to abbreviations
    model_name_map = {
        'autoencoder': 'AE',
        'isolation_forest': 'IF',
        'local_outlier_factor': 'LOF',
        'one_class_svm': 'OCSVM'
    }
    combined_df['model_display'] = combined_df['model'].map(lambda x: model_name_map.get(x, x))
    
    # Simplify version names for display (show as 0.50, 0.90, etc.)
    version_display_map = {v: v.replace('_pcc_n_iter_5', '').replace('_', '.') for v in args.versions}
    combined_df['version_display'] = combined_df['version'].map(version_display_map)
    
    # Print summary table
    print("\n" + "="*80)
    print("AVERAGE F1-SCORE BY ALGORITHM, WINDOW, AND VERSION")
    print("="*80)
    for window in combined_df['window'].unique():
        print(f"\n{window}:")
        window_data = combined_df[combined_df['window'] == window]
        pivot_table = window_data.pivot(index='model', columns='version', values='f1_score')
        print(pivot_table.to_string())
    print("="*80)
    
    # Generate comparison bar charts - separate figure for each window
    windows = sorted(combined_df['window'].unique())
    
    models = combined_df['model'].unique()
    models_display = [model_name_map.get(m, m) for m in models]
    versions = combined_df['version_display'].unique()
    
    # Use color palette matching the dataset comparison charts
    if len(versions) == 2:
        colors = ['#8FD5A6', '#F1A7A7']  # Mint green and soft pink
    elif len(versions) == 3:
        colors = ['#8FD5A6', '#8FB4D6', '#F1A7A7']  # Mint green, light blue, soft pink
    else:
        colors = ['#8FD5A6', '#8FB4D6', '#F1A7A7', '#ECC9A1']  # Add beige for 4th version
    
    # Create a separate figure for each window
    for window in windows:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        window_data = combined_df[combined_df['window'] == window]
        
        x = np.arange(len(models))
        width = 0.35 if len(versions) == 2 else 0.25
        
        for i, version in enumerate(versions):
            version_data = window_data[window_data['version_display'] == version]
            f1_values = [version_data[version_data['model'] == model]['f1_score'].values[0] 
                        if len(version_data[version_data['model'] == model]) > 0 else 0 
                        for model in models]
            
            offset = width * (i - len(versions)/2 + 0.5)
            # bars = ax.bar(x + offset, f1_values, width, label=f'θ = {version}', color=colors[i], alpha=0.8)
            bars = ax.bar(x + offset, f1_values, width, label=f'θ = {version}', color=colors[i], alpha=1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=14)
        
        # Convert window to readable format (e.g., "1 second", "300 seconds")
        window_seconds = int(window.replace('seconds', ''))
        window_text = f"{window_seconds} second" if window_seconds == 1 else f"{window_seconds} seconds"
        
        ax.set_ylabel('Average F1-Score', fontsize=18, fontweight='bold')
        ax.set_title(f'Average F1-Score Comparison for ∆t = {window_text}', 
                    fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Model', fontsize=18, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models_display, rotation=0, fontsize=16)
        ax.legend(loc='upper right', fontsize=16)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        # Save with window-specific filename
        output_path = output_dir / f'version_comparison_{window}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comparison chart for {window} to {output_path}")
        plt.close()
    
    # Highlight best performer in each version and window
    print("\n" + "="*80)
    print("BEST PERFORMER IN EACH VERSION AND WINDOW")
    print("="*80)
    for version in args.versions:
        print(f"\n{version}:")
        version_data = combined_df[combined_df['version'] == version]
        for window in sorted(version_data['window'].unique()):
            window_data = version_data[version_data['window'] == window]
            if not window_data.empty:
                best = window_data.loc[window_data['f1_score'].idxmax()]
                best_model_display = model_name_map.get(best['model'], best['model'])
                print(f"  {window:15} → {best_model_display:6} (F1: {best['f1_score']:.4f})")
    print("="*80 + "\n")
    
    print("\nComparison complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
