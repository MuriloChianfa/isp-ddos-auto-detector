#!/usr/bin/env python3
"""
Performance Test for Real-time DDoS Detection
Measures inference time and detection latency
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_ENABLE_XLA'] = '0'

import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_system import DDoSDetectionSystem
from config import AGG_PERIOD

def measure_inference_performance():
    """Measure inference performance for real-time detection"""
    print("Performance Test for Real-time DDoS Detection")
    print("=" * 60)
    
    # Initialize detection system
    detector = DDoSDetectionSystem()
    
    # Load models
    print("Loading pre-trained models...")
    detector.load_models()
    
    # Load a small sample of data for testing
    print("Loading test data...")
    features_df = detector.load_and_process_data()
    
    # Take a small sample for performance testing
    test_sample = features_df.head(100)  # 100 time windows
    
    print(f"Testing with {len(test_sample)} time windows")
    print(f"Time window size: {AGG_PERIOD}")
    
    # Warm up the models
    print("Warming up models...")
    for _ in range(5):
        detector.detect_anomalies_enhanced(test_sample.head(10), 'standard_ae')
        detector.detect_anomalies_enhanced(test_sample.head(10), 'lstm_ae')
    
    # Performance tests
    results = {}
    
    for model_name in ['standard_ae', 'lstm_ae']:
        print(f"\nTesting {model_name}...")
        
        # Single prediction timing
        single_times = []
        for i in range(10):
            sample = test_sample.iloc[i:i+1]
            start_time = time.time()
            anomalies, errors = detector.detect_anomalies_enhanced(sample, model_name)
            end_time = time.time()
            single_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Batch prediction timing
        batch_sizes = [1, 10, 50, 100]
        batch_times = {}
        
        for batch_size in batch_sizes:
            if batch_size <= len(test_sample):
                batch_data = test_sample.head(batch_size)
                start_time = time.time()
                anomalies, errors = detector.detect_anomalies_enhanced(batch_data, model_name)
                end_time = time.time()
                batch_time = (end_time - start_time) * 1000  # Convert to milliseconds
                batch_times[batch_size] = batch_time
        
        results[model_name] = {
            'single_prediction': {
                'mean_ms': np.mean(single_times),
                'std_ms': np.std(single_times),
                'min_ms': np.min(single_times),
                'max_ms': np.max(single_times)
            },
            'batch_predictions': batch_times
        }
    
    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Single Prediction:")
        print(f"    Mean: {result['single_prediction']['mean_ms']:.2f} ms")
        print(f"    Std:  {result['single_prediction']['std_ms']:.2f} ms")
        print(f"    Min:  {result['single_prediction']['min_ms']:.2f} ms")
        print(f"    Max:  {result['single_prediction']['max_ms']:.2f} ms")
        
        print(f"  Batch Predictions:")
        for batch_size, time_ms in result['batch_predictions'].items():
            throughput = batch_size / (time_ms / 1000)  # predictions per second
            print(f"    {batch_size} samples: {time_ms:.2f} ms ({throughput:.1f} pred/s)")
    
    # Real-time analysis
    print("\n" + "=" * 60)
    print("REAL-TIME DETECTION ANALYSIS")
    print("=" * 60)
    
    # Calculate time window duration
    if AGG_PERIOD == "1Min":
        window_duration_ms = 60 * 1000  # 60 seconds in milliseconds
    elif AGG_PERIOD == "30S":
        window_duration_ms = 30 * 1000
    elif AGG_PERIOD == "10S":
        window_duration_ms = 10 * 1000
    else:
        window_duration_ms = 60 * 1000  # Default to 1 minute
    
    print(f"Time window duration: {window_duration_ms/1000:.1f} seconds")
    
    for model_name, result in results.items():
        mean_inference_time = result['single_prediction']['mean_ms']
        
        # Detection latency analysis
        if mean_inference_time < window_duration_ms:
            latency_percentage = (mean_inference_time / window_duration_ms) * 100
            print(f"\n{model_name.upper()}:")
            print(f"  ✓ Real-time capable")
            print(f"  ✓ Inference time: {mean_inference_time:.2f} ms")
            print(f"  ✓ Uses {latency_percentage:.1f}% of time window")
            print(f"  ✓ Available processing time: {window_duration_ms - mean_inference_time:.2f} ms")
            
            # Calculate maximum throughput
            max_throughput = 1000 / mean_inference_time  # predictions per second
            print(f"  ✓ Maximum throughput: {max_throughput:.1f} predictions/second")
            
        else:
            print(f"\n{model_name.upper()}:")
            print(f"  ✗ Not real-time capable")
            print(f"  ✗ Inference time: {mean_inference_time:.2f} ms")
            print(f"  ✗ Exceeds time window by: {mean_inference_time - window_duration_ms:.2f} ms")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR REAL-TIME DEPLOYMENT")
    print("=" * 60)
    
    best_model = min(results.keys(), key=lambda x: results[x]['single_prediction']['mean_ms'])
    best_time = results[best_model]['single_prediction']['mean_ms']
    
    print(f"Recommended model: {best_model}")
    print(f"Best inference time: {best_time:.2f} ms")
    
    if best_time < window_duration_ms:
        print(f"✓ System can detect attacks within {window_duration_ms/1000:.1f} second time windows")
        print(f"✓ Detection latency: {best_time:.2f} ms")
        
        # Calculate buffer time
        buffer_time = window_duration_ms - best_time
        print(f"✓ Buffer time available: {buffer_time:.2f} ms")
        
        if buffer_time > 1000:  # More than 1 second buffer
            print("✓ Sufficient buffer for additional processing")
        else:
            print("⚠ Limited buffer time - consider optimization")
            
    else:
        print("✗ System cannot meet real-time requirements")
        print("Recommendations:")
        print("  - Reduce time window size")
        print("  - Optimize model architecture")
        print("  - Use hardware acceleration")
        print("  - Implement model quantization")

if __name__ == "__main__":
    measure_inference_performance()
