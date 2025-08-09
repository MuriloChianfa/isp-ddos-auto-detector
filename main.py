#!/usr/bin/env python3
"""
Enhanced DDoS Detection System - Main Script
Supports multiple attack types and improved detection sensitivity
"""

# Configure TensorFlow logging at the very beginning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_ENABLE_XLA'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import argparse

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').setLevel(logging.ERROR)

from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

try:
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        for gpu in gpu_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("TensorFlow GPU memory growth configured")
except Exception as e:
    print(f"TensorFlow early configuration failed: {e}")

from detection_system import DDoSDetectionSystem
from visualization import DDoSVisualizer


def main():
    """Main function with multi-attack detection"""
    parser = argparse.ArgumentParser(description='DDoS Detection System')
    parser.add_argument('--no-cache', action='store_true', 
                       help='Force retraining even if models exist')
    args = parser.parse_args()
    
    print("DDoS Detection System Starting...")
    print("=" * 60)
    
    detector = DDoSDetectionSystem()
    
    print("Loading and processing netflow data...")
    start_time = time.time()
    features_df = detector.load_and_process_data()
    load_time = time.time() - start_time
    print(f"Total time windows: {len(features_df)}")
    print(f"Time range: {features_df.ts_bin.min()} to {features_df.ts_bin.max()}")
    
    print("\nSplitting data with enhanced attack period handling...")
    train_data, attack_data, test_data, attack_info, additional_normal_data = detector.split_data_enhanced(features_df)
    
    print(f"Training data: {len(train_data)} windows (normal traffic)")
    print(f"Attack data: {len(attack_data)} windows total")
    for attack_name, data in attack_info.items():
        attack_type = data['attack_type'].iloc[0] if len(data) > 0 else "unknown"
        print(f"  - {attack_name} ({attack_type}): {len(data)} windows")
    print(f"Test data: {len(test_data)} windows (normal traffic)")
    print(f"Additional normal data: {len(additional_normal_data)} windows")
    
    # Check if models exist and determine whether to retrain
    models_exist = detector.check_saved_models()
    retrain = True
    
    if models_exist and not args.no_cache:
        print("\nUsing existing models (use --no-cache to force retraining)...")
        retrain = False
    elif models_exist and args.no_cache:
        print("\nForcing retraining (--no-cache specified)...")
        retrain = True
    else:
        print("\nNo saved models found. Training new models...")
        retrain = True
    
    if retrain:
        print("\nTraining enhanced models...")
        training_start = time.time()
        detector.train_models_enhanced(train_data, attack_data)
        training_time = time.time() - training_start
        print(f"Training completed in {training_time:.2f} seconds")
        
        print("Saving enhanced models...")
        detector.save_models()
        print("Models saved successfully")
    else:
        print("\nLoading existing models...")
        detector.load_models()
    
    print("\nEvaluating enhanced models...")
    eval_start = time.time()
    results = detector.evaluate_models_enhanced(attack_info, test_data, additional_normal_data)
    eval_time = time.time() - eval_start
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    print("\n" + "=" * 60)
    print("ENHANCED DETECTION RESULTS")
    print("=" * 60)
    
    for model_name, model_results in results.items():
        print(f"\n{model_name.upper().replace('_', ' ')}")
        print("-" * 40)
        
        for attack_name, attack_results in model_results['attack_detection_by_type'].items():
            attack_type = attack_info[attack_name]['attack_type'].iloc[0]
            windows_detected = attack_results['windows_detected']
            total_windows = attack_results['total_windows']
            standard_rate = attack_results['standard_detection_rate'] * 100
            flow_rate = attack_results['flow_sensitive_rate'] * 100
            
            print(f"  {attack_name} ({attack_type}):")
            print(f"     Standard detection: {standard_rate:.1f}% ({windows_detected}/{total_windows} windows)")
            print(f"     Flow-sensitive:     {flow_rate:.1f}% ({windows_detected}/{total_windows} windows)")
        
        fp_rate = model_results['false_positive_rate'] * 100
        additional_fp_rate = model_results['additional_fp_rate'] * 100
        print(f"  False positives:")
        print(f"     Test data:    {fp_rate:.1f}%")
        print(f"     Additional normal: {additional_fp_rate:.1f}%")
    
    print(f"\nGenerating charts and error metrics...")

    try:
        visualizer = DDoSVisualizer(detector)
        combined_attack_data = pd.concat([attack_info[name] for name in attack_info.keys()], ignore_index=True)
        visualizer.plot_detailed_timeline(combined_attack_data, test_data, features_df)
        print("Timeline charts saved to results/ directory.")

        for model_name in detector.models.keys():
            # Get model threshold
            threshold = detector.thresholds.get(model_name, 0.5)
            
            # Get test data errors
            X_test = detector.prepare_data_for_prediction(test_data)
            errors = detector.models[model_name].predict(X_test)
            
            # Calculate metrics
            mse = np.mean(errors)
            mae = np.mean(np.abs(errors))
            std_dev = np.std(errors)
            
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print(f"  MSE:              {mse:.5f}")
            print(f"  MAE:              {mae:.5f}")
            print(f"  Standard Deviation: {std_dev:.5f}")
            print(f"  Threshold:        {threshold:.5f}")
            print(f"  Error Range:      [{np.min(errors):.5f}, {np.max(errors):.5f}]")
        
        print("\nAll charts saved to results/ directory.")
        
    except Exception as e:
        print(f"Visualization error: {e}")


if __name__ == "__main__":
    main()
