"""
Real-time performance evaluation for DDoS detection models.

This module provides comprehensive performance metrics for evaluating model throughput,
latency, and resource usage in real-time scenarios.
"""

import time
import numpy as np
import pandas as pd
import psutil
import queue
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance evaluation metrics."""
    
    # Throughput metrics
    throughput_samples_per_sec: float
    
    # Latency metrics (in milliseconds)
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    
    # Resource usage metrics
    memory_usage_mb: float
    cpu_usage_percent: float
    
    # Test metadata
    total_samples_processed: int
    total_time_seconds: float
    test_type: str
    model_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for easy serialization."""
        return {
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'avg_latency_ms': self.avg_latency_ms,
            'min_latency_ms': self.min_latency_ms,
            'max_latency_ms': self.max_latency_ms,
            'p95_latency_ms': self.p95_latency_ms,
            'p99_latency_ms': self.p99_latency_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'total_samples_processed': self.total_samples_processed,
            'total_time_seconds': self.total_time_seconds,
            'test_type': self.test_type,
            'model_name': self.model_name,
            'timestamp': pd.Timestamp.now().isoformat()
        }


class RealTimePerformanceEvaluator:
    """Evaluates model performance in real-time scenarios."""
    
    def __init__(self, model: Any, preprocessor: Optional[Any] = None):
        """
        Initialize the performance evaluator.
        
        Args:
            model: Trained model for evaluation
            preprocessor: Optional preprocessor for input data
        """
        self.model = model
        self.preprocessor = preprocessor
        self.process = psutil.Process()
        
        # Metrics storage
        self.latencies: List[float] = []
        self.memory_readings: List[float] = []
        self.cpu_readings: List[float] = []
    
    def _warmup_model(self, sample_data: np.ndarray, iterations: int = 50) -> None:
        """
        Warm up the model to ensure stable performance measurements.
        
        Args:
            sample_data: Sample data for warmup
            iterations: Number of warmup iterations
        """
        logger.info(f"Warming up model with {iterations} iterations...")
        
        for _ in range(iterations):
            processed_data = self._preprocess_data(sample_data[:1])
            _ = self.model.predict(processed_data)
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        Preprocess input data if preprocessor is available.
        
        Args:
            data: Input data to preprocess
            
        Returns:
            Preprocessed data
        """
        if self.preprocessor is not None:
            return self.preprocessor.transform(data)
        elif hasattr(self.model, 'transform_data'):
            # Model has its own data transformation method
            return self.model.transform_data(data)
        return data
    
    def _record_system_metrics(self) -> None:
        """Record current system resource usage."""
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        cpu_percent = self.process.cpu_percent()
        
        self.memory_readings.append(memory_mb)
        self.cpu_readings.append(cpu_percent)
    
    def _calculate_metrics(self, total_time: float, total_samples: int, 
                          test_type: str, model_name: str) -> PerformanceMetrics:
        """
        Calculate performance metrics from recorded data.
        
        Args:
            total_time: Total execution time in seconds
            total_samples: Total number of samples processed
            test_type: Type of performance test
            model_name: Name of the model being evaluated
            
        Returns:
            PerformanceMetrics object
        """
        latencies_array = np.array(self.latencies)
        
        return PerformanceMetrics(
            throughput_samples_per_sec=total_samples / total_time,
            avg_latency_ms=np.mean(latencies_array),
            min_latency_ms=np.min(latencies_array),
            max_latency_ms=np.max(latencies_array),
            p95_latency_ms=np.percentile(latencies_array, 95),
            p99_latency_ms=np.percentile(latencies_array, 99),
            memory_usage_mb=np.mean(self.memory_readings) if self.memory_readings else 0.0,
            cpu_usage_percent=np.mean(self.cpu_readings) if self.cpu_readings else 0.0,
            total_samples_processed=total_samples,
            total_time_seconds=total_time,
            test_type=test_type,
            model_name=model_name
        )
    
    def _reset_metrics(self) -> None:
        """Reset all metric storage lists."""
        self.latencies.clear()
        self.memory_readings.clear()
        self.cpu_readings.clear()
    
    def evaluate_batch_performance(self, test_data: np.ndarray, 
                                 batch_size: int = 1,
                                 num_iterations: int = 1000,
                                 model_name: str = "unknown") -> PerformanceMetrics:
        """
        Evaluate model performance with batch processing.
        
        Args:
            test_data: Test dataset
            batch_size: Number of samples per batch
            num_iterations: Number of iterations to run
            model_name: Name of the model being evaluated
            
        Returns:
            PerformanceMetrics object
        """
        logger.info(f"Running batch performance evaluation ({num_iterations} iterations, batch_size={batch_size})")
        
        # Reset metrics and warmup
        self._reset_metrics()
        self._warmup_model(test_data)
        
        # Prepare random batches
        num_samples = len(test_data)
        batches = []
        
        for _ in range(num_iterations):
            start_idx = np.random.randint(0, max(1, num_samples - batch_size))
            end_idx = min(start_idx + batch_size, num_samples)
            batch = test_data[start_idx:end_idx]
            batches.append(batch)
        
        # Run benchmark
        total_start_time = time.time()
        
        for batch in batches:
            # Record system metrics before prediction
            self._record_system_metrics()
            
            # Preprocess batch
            processed_batch = self._preprocess_data(batch)
            
            # Time the prediction
            prediction_start = time.perf_counter()
            _ = self.model.predict(processed_batch)
            prediction_end = time.perf_counter()
            
            # Record latency
            latency_ms = (prediction_end - prediction_start) * 1000
            self.latencies.append(latency_ms)
        
        total_time = time.time() - total_start_time
        total_samples = num_iterations * batch_size
        
        return self._calculate_metrics(total_time, total_samples, "batch", model_name)
    
    def evaluate_streaming_performance(self, test_data: np.ndarray,
                                     stream_rate_hz: float = 100,
                                     duration_seconds: int = 60,
                                     model_name: str = "unknown") -> PerformanceMetrics:
        """
        Evaluate model performance in streaming scenario.
        
        Args:
            test_data: Test dataset
            stream_rate_hz: Rate of incoming data (samples per second)
            duration_seconds: Duration of the streaming test
            model_name: Name of the model being evaluated
            
        Returns:
            PerformanceMetrics object
        """
        logger.info(f"Running streaming performance evaluation ({stream_rate_hz} Hz, {duration_seconds}s)")
        
        # Reset metrics and warmup
        self._reset_metrics()
        self._warmup_model(test_data)
        
        # Setup streaming simulation
        data_queue = queue.Queue(maxsize=1000)  # Prevent memory issues
        
        def data_producer():
            """Producer thread that generates data at specified rate."""
            interval = 1.0 / stream_rate_hz
            start_time = time.time()
            sample_idx = 0
            
            while time.time() - start_time < duration_seconds:
                sample = test_data[sample_idx % len(test_data)]
                try:
                    data_queue.put((time.time(), sample), timeout=0.1)
                except queue.Full:
                    logger.warning("Data queue full, dropping sample")
                
                sample_idx += 1
                time.sleep(max(0, interval - 0.001))  # Small adjustment for processing time
            
            # Signal end
            data_queue.put(None)
        
        # Start producer thread
        producer_thread = threading.Thread(target=data_producer, daemon=True)
        producer_thread.start()
        
        # Process streaming data
        processed_samples = 0
        start_time = time.time()
        
        while True:
            try:
                item = data_queue.get(timeout=2.0)
                if item is None:  # End signal
                    break
                
                timestamp, sample = item
                
                # Record system metrics
                self._record_system_metrics()
                
                # Process sample
                processed_sample = self._preprocess_data(sample.reshape(1, -1))
                
                prediction_start = time.perf_counter()
                _ = self.model.predict(processed_sample)
                prediction_end = time.perf_counter()
                
                # Record latency
                latency_ms = (prediction_end - prediction_start) * 1000
                self.latencies.append(latency_ms)
                processed_samples += 1
                
            except queue.Empty:
                logger.warning("Data queue timeout, ending streaming test")
                break
        
        producer_thread.join(timeout=1.0)
        total_time = time.time() - start_time
        
        return self._calculate_metrics(total_time, processed_samples, "streaming", model_name)
    
    def evaluate_single_sample_performance(self, test_data: np.ndarray,
                                         num_samples: int = 1000,
                                         model_name: str = "unknown") -> PerformanceMetrics:
        """
        Evaluate model performance for single sample predictions.
        
        Args:
            test_data: Test dataset
            num_samples: Number of individual samples to test
            model_name: Name of the model being evaluated
            
        Returns:
            PerformanceMetrics object
        """
        logger.info(f"Running single sample performance evaluation ({num_samples} samples)")
        
        # Reset metrics and warmup
        self._reset_metrics()
        self._warmup_model(test_data)
        
        # Select random samples
        sample_indices = np.random.choice(len(test_data), num_samples, replace=True)
        
        total_start_time = time.time()
        
        for idx in sample_indices:
            sample = test_data[idx:idx+1]  # Keep as 2D array
            
            # Record system metrics
            self._record_system_metrics()
            
            # Preprocess sample
            processed_sample = self._preprocess_data(sample)
            
            # Time the prediction
            prediction_start = time.perf_counter()
            _ = self.model.predict(processed_sample)
            prediction_end = time.perf_counter()
            
            # Record latency
            latency_ms = (prediction_end - prediction_start) * 1000
            self.latencies.append(latency_ms)
        
        total_time = time.time() - total_start_time
        
        return self._calculate_metrics(total_time, num_samples, "single_sample", model_name)


class PerformanceReporter:
    """Handles reporting and saving of performance metrics."""
    
    @staticmethod
    def print_metrics(metrics: PerformanceMetrics) -> None:
        """Print formatted performance metrics to console."""
        print("\n" + "="*70)
        print(f"REAL-TIME PERFORMANCE METRICS - {metrics.test_type.upper()}")
        print("="*70)
        print(f"Model:                {metrics.model_name}")
        print(f"Test Type:            {metrics.test_type}")
        print(f"Total Samples:        {metrics.total_samples_processed:,}")
        print(f"Total Time:           {metrics.total_time_seconds:.2f} seconds")
        print("-"*70)
        print(f"Throughput:           {metrics.throughput_samples_per_sec:.2f} samples/sec")
        print(f"Average Latency:      {metrics.avg_latency_ms:.3f} ms")
        print(f"Min Latency:          {metrics.min_latency_ms:.3f} ms")
        print(f"Max Latency:          {metrics.max_latency_ms:.3f} ms")
        print(f"95th Percentile:      {metrics.p95_latency_ms:.3f} ms")
        print(f"99th Percentile:      {metrics.p99_latency_ms:.3f} ms")
        print("-"*70)
        print(f"Memory Usage:         {metrics.memory_usage_mb:.2f} MB")
        print(f"CPU Usage:            {metrics.cpu_usage_percent:.2f}%")
        print("="*70)
    
    @staticmethod
    def save_metrics(metrics: PerformanceMetrics, filepath: str) -> None:
        """
        Save performance metrics to CSV file.
        
        Args:
            metrics: Performance metrics to save
            filepath: Path to save the CSV file
        """
        df = pd.DataFrame([metrics.to_dict()])
        
        # Append to existing file or create new one
        try:
            existing_df = pd.read_csv(filepath)
            df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        df.to_csv(filepath, index=False)
        logger.info(f"Performance metrics saved to: {filepath}")
    
    @staticmethod
    def compare_metrics(metrics_list: List[PerformanceMetrics]) -> pd.DataFrame:
        """
        Compare multiple performance metrics.
        
        Args:
            metrics_list: List of performance metrics to compare
            
        Returns:
            DataFrame with comparison data
        """
        comparison_data = []
        
        for metrics in metrics_list:
            comparison_data.append({
                'Model': metrics.model_name,
                'Test Type': metrics.test_type,
                'Throughput (samples/sec)': metrics.throughput_samples_per_sec,
                'Avg Latency (ms)': metrics.avg_latency_ms,
                'P95 Latency (ms)': metrics.p95_latency_ms,
                'P99 Latency (ms)': metrics.p99_latency_ms,
                'Memory (MB)': metrics.memory_usage_mb,
                'CPU (%)': metrics.cpu_usage_percent
            })
        
        return pd.DataFrame(comparison_data)