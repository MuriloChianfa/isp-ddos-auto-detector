#!/usr/bin/env python3
"""
Real-time DDoS Detection Analysis
Comprehensive analysis of detection latency for different scenarios
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

def analyze_realtime_capabilities():
    """Analyze real-time detection capabilities for different scenarios"""
    print("Real-time DDoS Detection Analysis")
    print("=" * 60)
    
    # Initialize detection system
    detector = DDoSDetectionSystem()
    detector.load_models()
    
    # Load test data
    features_df = detector.load_and_process_data()
    test_sample = features_df.head(100)
    
    # Performance measurements
    print("Measuring inference performance...")
    
    # Single prediction times
    standard_times = []
    lstm_times = []
    
    for i in range(20):  # More samples for better statistics
        sample = test_sample.iloc[i:i+1]
        
        # Standard AE
        start_time = time.time()
        detector.detect_anomalies_enhanced(sample, 'standard_ae')
        end_time = time.time()
        standard_times.append((end_time - start_time) * 1000)
        
        # LSTM AE
        start_time = time.time()
        detector.detect_anomalies_enhanced(sample, 'lstm_ae')
        end_time = time.time()
        lstm_times.append((end_time - start_time) * 1000)
    
    # Calculate statistics
    standard_mean = np.mean(standard_times)
    standard_std = np.std(standard_times)
    lstm_mean = np.mean(lstm_times)
    lstm_std = np.std(lstm_times)
    
    print(f"\nInference Performance:")
    print(f"Standard Autoencoder: {standard_mean:.2f} ± {standard_std:.2f} ms")
    print(f"LSTM Autoencoder: {lstm_mean:.2f} ± {lstm_std:.2f} ms")
    
    # Analyze different time window scenarios
    time_windows = {
        "1 second": 1000,
        "5 seconds": 5000,
        "10 seconds": 10000,
        "30 seconds": 30000,
        "1 minute": 60000,
        "5 minutes": 300000
    }
    
    print("\n" + "=" * 60)
    print("REAL-TIME DETECTION SCENARIOS")
    print("=" * 60)
    
    scenarios = []
    
    for window_name, window_ms in time_windows.items():
        print(f"\n{window_name.upper()} TIME WINDOW:")
        print(f"Window duration: {window_ms} ms")
        
        # Standard AE analysis
        if standard_mean < window_ms:
            latency_pct = (standard_mean / window_ms) * 100
            buffer_time = window_ms - standard_mean
            scenarios.append({
                'window': window_name,
                'duration_ms': window_ms,
                'model': 'standard_ae',
                'inference_ms': standard_mean,
                'latency_pct': latency_pct,
                'buffer_ms': buffer_time,
                'capable': True
            })
            print(f"  Standard AE: ✓ Capable")
            print(f"    Inference: {standard_mean:.2f} ms ({latency_pct:.1f}% of window)")
            print(f"    Buffer: {buffer_time:.2f} ms")
        else:
            scenarios.append({
                'window': window_name,
                'duration_ms': window_ms,
                'model': 'standard_ae',
                'inference_ms': standard_mean,
                'latency_pct': 100,
                'buffer_ms': 0,
                'capable': False
            })
            print(f"  Standard AE: ✗ Not capable")
            print(f"    Inference: {standard_mean:.2f} ms (exceeds window)")
        
        # LSTM AE analysis
        if lstm_mean < window_ms:
            latency_pct = (lstm_mean / window_ms) * 100
            buffer_time = window_ms - lstm_mean
            scenarios.append({
                'window': window_name,
                'duration_ms': window_ms,
                'model': 'lstm_ae',
                'inference_ms': lstm_mean,
                'latency_pct': latency_pct,
                'buffer_ms': buffer_time,
                'capable': True
            })
            print(f"  LSTM AE: ✓ Capable")
            print(f"    Inference: {lstm_mean:.2f} ms ({latency_pct:.1f}% of window)")
            print(f"    Buffer: {buffer_time:.2f} ms")
        else:
            scenarios.append({
                'window': window_name,
                'duration_ms': window_ms,
                'model': 'lstm_ae',
                'inference_ms': lstm_mean,
                'latency_pct': 100,
                'buffer_ms': 0,
                'capable': False
            })
            print(f"  LSTM AE: ✗ Not capable")
            print(f"    Inference: {lstm_mean:.2f} ms (exceeds window)")
    
    # Attack detection timing analysis
    print("\n" + "=" * 60)
    print("ATTACK DETECTION TIMING ANALYSIS")
    print("=" * 60)
    
    # Different attack scenarios
    attack_scenarios = {
        "Fast DDoS (1-5 seconds)": {
            "duration_range": (1000, 5000),
            "description": "Quick burst attacks that need immediate detection"
        },
        "Medium DDoS (10-30 seconds)": {
            "duration_range": (10000, 30000),
            "description": "Sustained attacks with moderate duration"
        },
        "Long DDoS (1-5 minutes)": {
            "duration_range": (60000, 300000),
            "description": "Extended attacks requiring continuous monitoring"
        }
    }
    
    for scenario_name, scenario_info in attack_scenarios.items():
        min_duration, max_duration = scenario_info["duration_range"]
        description = scenario_info["description"]
        
        print(f"\n{scenario_name}:")
        print(f"  {description}")
        print(f"  Duration range: {min_duration/1000:.1f}-{max_duration/1000:.1f} seconds")
        
        # Check if models can detect within attack duration
        standard_capable = standard_mean < min_duration
        lstm_capable = lstm_mean < min_duration
        
        if standard_capable:
            detection_time = standard_mean
            response_time = f"{detection_time:.2f} ms"
            print(f"  Standard AE: ✓ Can detect within {response_time}")
        else:
            print(f"  Standard AE: ✗ Too slow for this scenario")
        
        if lstm_capable:
            detection_time = lstm_mean
            response_time = f"{detection_time:.2f} ms"
            print(f"  LSTM AE: ✓ Can detect within {response_time}")
        else:
            print(f"  LSTM AE: ✗ Too slow for this scenario")
    
    # Real-time deployment recommendations
    print("\n" + "=" * 60)
    print("REAL-TIME DEPLOYMENT RECOMMENDATIONS")
    print("=" * 60)
    
    # Find best performing model
    best_model = 'lstm_ae' if lstm_mean < standard_mean else 'standard_ae'
    best_time = min(lstm_mean, standard_mean)
    
    print(f"Recommended model: {best_model}")
    print(f"Best inference time: {best_time:.2f} ms")
    
    # Calculate maximum detection frequency
    max_frequency = 1000 / best_time  # detections per second
    print(f"Maximum detection frequency: {max_frequency:.1f} detections/second")
    
    # Recommended time windows
    print(f"\nRecommended time windows for real-time deployment:")
    
    if best_time < 1000:  # Less than 1 second
        print(f"  ✓ 1-second windows: Excellent ({(best_time/1000)*100:.1f}% utilization)")
    
    if best_time < 5000:  # Less than 5 seconds
        print(f"  ✓ 5-second windows: Good ({(best_time/5000)*100:.1f}% utilization)")
    
    if best_time < 10000:  # Less than 10 seconds
        print(f"  ✓ 10-second windows: Good ({(best_time/10000)*100:.1f}% utilization)")
    
    if best_time < 60000:  # Less than 1 minute
        print(f"  ✓ 1-minute windows: Excellent ({(best_time/60000)*100:.1f}% utilization)")
    
    # System requirements
    print(f"\nSystem requirements for real-time deployment:")
    print(f"  - CPU: Multi-core recommended for parallel processing")
    print(f"  - Memory: Sufficient for model loading and data processing")
    print(f"  - Storage: Fast I/O for real-time data ingestion")
    print(f"  - Network: High bandwidth for real-time monitoring")
    
    # Optimization opportunities
    print(f"\nOptimization opportunities:")
    if best_time > 100:  # If inference takes more than 100ms
        print(f"  - Model quantization for faster inference")
        print(f"  - Hardware acceleration (GPU/TPU)")
        print(f"  - Model pruning for smaller architecture")
        print(f"  - Batch processing for efficiency")
    else:
        print(f"  - Current performance is excellent for real-time deployment")
        print(f"  - Consider additional features without performance impact")
    
    return {
        'standard_ae_time': standard_mean,
        'lstm_ae_time': lstm_mean,
        'best_model': best_model,
        'best_time': best_time,
        'max_frequency': max_frequency,
        'scenarios': scenarios
    }

if __name__ == "__main__":
    analyze_realtime_capabilities()
