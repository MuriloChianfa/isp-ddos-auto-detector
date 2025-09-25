#!/usr/bin/env python3
"""
Standalone script for generating dataset feature visualizations.
This script can be used independently to create feature plots without running the full pipeline.
"""

import argparse
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from framework.loader import NetworkDataLoader
from framework.features import NetworkFeatureExtractor
from framework.visualization.dataset_plots import DatasetFeatureVisualizer
from framework.utils import get_time_span_description, get_time_span_detailed_description
from framework.constants import SUPPORTED_TIME_SPANS
from config import DATASETS, DEFAULT_DATASET


def main():
    parser = argparse.ArgumentParser(
        description='Generate PNG visualizations for all network traffic dataset features',
        epilog="""
Examples:
  %(prog)s                                    # Generate all plots for default dataset
  %(prog)s -d isp-synflood-multiple-days     # Generate plots for specific dataset
  %(prog)s --no-comparisons                  # Skip cross-dataset comparison plots
  %(prog)s -t 60                             # Use 60-second time windows instead of default 300
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--dataset', '-d', default=None,
                       help=f'Dataset to analyze (default: {DEFAULT_DATASET}). Available: {", ".join(DATASETS.keys())}')
    parser.add_argument('--no-comparisons', action='store_true',
                       help='Skip generating cross-dataset comparison plots')
    parser.add_argument(
        '--time-span', '-t',
        type=int,
        choices=SUPPORTED_TIME_SPANS,
        default=300,
        help=f'Time span for feature aggregation in seconds. Options: {", ".join(map(str, SUPPORTED_TIME_SPANS))}. Default: 300'
    )
    
    args = parser.parse_args()
    
    # Select dataset configuration
    dataset_name = args.dataset if args.dataset else DEFAULT_DATASET
    
    if dataset_name not in DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found in configuration.")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return 1
    
    dataset_config = DATASETS[dataset_name]
    
    # Get time_span from arguments
    time_span = args.time_span
    
    # Create dataset-specific output directory for features
    output_dir = os.path.join("./results", dataset_name, f"{time_span}seconds", "features")
    
    print("=" * 60)
    print("NETWORK TRAFFIC FEATURE VISUALIZATION")
    print("=" * 60)
    print(f"Using dataset: {dataset_name}")
    print(f"Description: {dataset_config['description']}")
    print(f"Path: {dataset_config['path']}")
    print(f"Time span: {time_span} seconds ({get_time_span_description(time_span)} windows)")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 60)
    
    try:
        print("\nInitializing data loader...")
        loader = NetworkDataLoader(dataset_config=dataset_config)
        
        print("\nExtracting features...")
        feature_extractor = NetworkFeatureExtractor(
            time_span=time_span, 
            dataset_name=dataset_name
        )
        
        # Use the new CSV-based feature extraction method to get all features
        features_dict = feature_extractor.extract_features_to_csv(loader, force_regenerate=False, parallel=True)
        
        # Show summary of available features
        print(f"\nAnalyzing all available features:")
        total_features = 0
        for dataset_type in features_dict:
            available_features = [col for col in features_dict[dataset_type].columns if col != 'timestamp']
            total_features = len(available_features)
            print(f"  {dataset_type}: {len(available_features)} features")
        
        print(f"\nGenerating PNG visualizations for all {total_features} features...")
        
        feature_viz = DatasetFeatureVisualizer(
            dataset_name=dataset_name, 
            save_format='png',
            time_span=time_span
        )
        feature_viz.generate_all_feature_plots(
            features_dict, 
            create_comparisons=not args.no_comparisons
        )
        
        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Generated PNG visualizations for:")
        
        # Show summary of what was generated
        for dataset_type in features_dict:
            available_features = [col for col in features_dict[dataset_type].columns if col != 'timestamp']
            print(f"  - {dataset_type.capitalize()}: {len(available_features)} features")
        
        if not args.no_comparisons:
            all_features = set()
            for df in features_dict.values():
                all_features.update([col for col in df.columns if col != 'timestamp'])
            print(f"  - Cross-dataset comparisons: {len(all_features)} features")
        
        print(f"\nOutput directory: {os.path.abspath(output_dir)}")
        print("Plot types generated:")
        print("  - Distribution histograms with statistics")
        print("  - Box plots for outlier analysis") 
        print("  - Time series plots")
        print("  - Q-Q plots for normality assessment")
        if not args.no_comparisons:
            print("  - Cross-dataset comparison plots")
        print(f"\nAll plots saved with {time_span}-second time windows ({get_time_span_description(time_span)} aggregation)")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nVisualization interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
