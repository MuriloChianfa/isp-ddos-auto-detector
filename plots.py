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
from config import DATASETS, DEFAULT_DATASET


def main():
    parser = argparse.ArgumentParser(description='Generate feature visualizations for network traffic datasets')
    parser.add_argument('--dataset', '-d', default=None,
                       help=f'Dataset to analyze (default: {DEFAULT_DATASET}). Available: {", ".join(DATASETS.keys())}')
    parser.add_argument('--output-dir', '-o', default='./results',
                       help='Base output directory for visualizations (default: ./results)')
    parser.add_argument('--no-comparisons', action='store_true',
                       help='Skip generating cross-dataset comparison plots')
    parser.add_argument('--features', nargs='+',
                       help='Specific features to visualize (default: all features)')
    
    args = parser.parse_args()
    
    # Select dataset configuration
    dataset_name = args.dataset if args.dataset else DEFAULT_DATASET
    
    if dataset_name not in DATASETS:
        print(f"Error: Dataset '{dataset_name}' not found in configuration.")
        print(f"Available datasets: {', '.join(DATASETS.keys())}")
        return 1
    
    dataset_config = DATASETS[dataset_name]
    
    # Create dataset-specific output directory
    output_dir = os.path.join(args.output_dir, dataset_name)
    
    print("=" * 60)
    print("NETWORK TRAFFIC FEATURE VISUALIZATION")
    print("=" * 60)
    print(f"Using dataset: {dataset_name}")
    print(f"Description: {dataset_config['description']}")
    print(f"Path: {dataset_config['path']}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    print("=" * 60)
    
    try:
        print("\nLoading network traffic data...")
        loader = NetworkDataLoader(dataset_config=dataset_config)
        datasets = loader.load_network_data_by_day()
        
        print("\nExtracting features...")
        feature_extractor = NetworkFeatureExtractor()
        features_dict = feature_extractor.process_datasets(datasets)
        
        # Filter features if specified
        if args.features:
            print(f"\nFiltering to specified features: {', '.join(args.features)}")
            for dataset_type in features_dict:
                available_features = [col for col in features_dict[dataset_type].columns if col != 'timestamp']
                requested_features = [f for f in args.features if f in available_features]
                missing_features = [f for f in args.features if f not in available_features]
                
                if missing_features:
                    print(f"Warning: Features not found in {dataset_type}: {missing_features}")
                
                if requested_features:
                    # Keep timestamp and requested features
                    cols_to_keep = ['timestamp'] + requested_features
                    features_dict[dataset_type] = features_dict[dataset_type][cols_to_keep]
                else:
                    print(f"Error: No valid features found for {dataset_type} dataset")
                    return 1
        
        print(f"\nGenerating visualizations...")
        
        feature_viz = DatasetFeatureVisualizer(results_dir=output_dir)
        feature_viz.generate_all_feature_plots(
            features_dict, 
            create_comparisons=not args.no_comparisons
        )
        
        print("\n" + "=" * 60)
        print("VISUALIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
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
