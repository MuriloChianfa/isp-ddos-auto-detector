#!/usr/bin/env python3
"""
Benchmark runner for DDoS Detection System
"""

import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_benchmarks():
    """Run all benchmarks"""
    print("Running DDoS Detection System Benchmarks")
    print("=" * 60)
    
    # Import benchmark modules
    from performance_test import measure_inference_performance
    from realtime_analysis import analyze_realtime_capabilities
    
    # Run performance test
    print("\n1. PERFORMANCE TEST")
    print("-" * 30)
    start_time = time.time()
    measure_inference_performance()
    perf_time = time.time() - start_time
    print(f"\nPerformance test completed in {perf_time:.2f} seconds")
    
    # Run real-time analysis
    print("\n\n2. REAL-TIME ANALYSIS")
    print("-" * 30)
    start_time = time.time()
    analyze_realtime_capabilities()
    analysis_time = time.time() - start_time
    print(f"\nReal-time analysis completed in {analysis_time:.2f} seconds")
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Performance test: {perf_time:.2f} seconds")
    print(f"Real-time analysis: {analysis_time:.2f} seconds")
    print(f"Total benchmark time: {perf_time + analysis_time:.2f} seconds")
    print("\nBenchmarks completed successfully!")

if __name__ == '__main__':
    run_benchmarks()
