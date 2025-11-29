"""
Simple performance plotting module.
Creates easy-to-understand charts that actually help with performance analysis.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import csv
from datetime import datetime

from framework.performance import PerformanceMetrics

plt.ioff()  # Disable interactive mode

logger = logging.getLogger(__name__)

class PerformancePlotter:
    """Simple performance plotter that creates useful, easy-to-read charts."""
    
    def __init__(self, output_dir: str):
        """Initialize the performance plotter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Performance plotter initialized with output directory: {self.output_dir}")
    
    def generate_all_charts(self, performance_data, model_name: str) -> List[str]:
        """Generate all performance charts and save CSV."""
        # Convert dict to list if needed
        if isinstance(performance_data, dict):
            performance_samples = list(performance_data.values())
        else:
            performance_samples = performance_data
            
        logger.info(f"Generating simple performance charts for model: {model_name} with {len(performance_samples)} samples")
        
        generated_files = []
        
        try:
            # Generate charts using internal methods
            comparison_path = self._generate_performance_comparison_chart(performance_samples, model_name)
            generated_files.append(comparison_path)
            
            resource_path = self._generate_resource_usage_chart(performance_samples, model_name)
            generated_files.append(resource_path)
            
            summary_path = self._generate_performance_summary_chart(performance_samples, model_name)
            generated_files.append(summary_path)
            
            # Save CSV
            csv_path = self._save_comprehensive_csv(performance_samples, model_name)
            generated_files.append(csv_path)
            
            logger.info(f"Successfully generated {len(generated_files)} performance files")
            return generated_files
            
        except Exception as e:
            logger.error(f"Error generating performance charts: {e}")
            raise
    
    def _generate_performance_comparison_chart(self, performance_samples: List[PerformanceMetrics], 
                                             model_name: str) -> str:
        """Generate simple performance comparison chart."""
        
        # Extract key data
        test_types = []
        throughputs = []
        latencies = []
        
        for sample in performance_samples:
            test_types.append(sample.test_type.replace('_', ' ').title())
            throughputs.append(sample.throughput_samples_per_sec)
            latencies.append(sample.avg_latency_ms)
        
        # Create side-by-side bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Throughput chart
        bars1 = ax1.bar(test_types, throughputs, color='#2E8B57', alpha=0.8)
        ax1.set_title('Throughput Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Samples per Second', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, value in zip(bars1, throughputs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Latency chart
        bars2 = ax2.bar(test_types, latencies, color='#4169E1', alpha=0.8)
        ax2.set_title('Average Latency Comparison', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Latency (ms)', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add values on bars
        for bar, value in zip(bars2, latencies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}ms', ha='center', va='bottom', fontweight='bold')
        
        # Overall title
        fig.suptitle(f'Performance Summary - {model_name.title()}', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = str(self.output_dir / f"performance_comparison_{model_name}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        plt.clf()
        gc.collect()
        
        logger.info(f"Performance comparison chart saved to: {chart_path}")
        return chart_path
    
    def _generate_resource_usage_chart(self, performance_samples: List[PerformanceMetrics], 
                                     model_name: str) -> str:
        """Generate simple resource usage chart."""
        
        # Extract data
        test_types = []
        memory_usage = []
        cpu_usage = []
        
        for sample in performance_samples:
            test_types.append(sample.test_type.replace('_', ' ').title())
            memory_usage.append(sample.memory_usage_mb)
            cpu_usage.append(sample.cpu_usage_percent)
        
        # Create figure with dual y-axis
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(test_types))
        
        # Memory bars
        bars1 = ax1.bar(x_pos - 0.2, memory_usage, 0.4, 
                       color='#FF6B6B', alpha=0.8, label='Memory (MB)')
        ax1.set_xlabel('Test Type', fontweight='bold')
        ax1.set_ylabel('Memory Usage (MB)', color='#FF6B6B', fontweight='bold')
        ax1.tick_params(axis='y', labelcolor='#FF6B6B')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(test_types)
        
        # Add memory values on bars
        for bar, value in zip(bars1, memory_usage):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{value:.0f}MB', ha='center', va='bottom', fontweight='bold',
                    color='#FF6B6B')
        
        # CPU bars (second y-axis)
        ax2 = ax1.twinx()
        bars2 = ax2.bar(x_pos + 0.2, cpu_usage, 0.4, 
                       color='#4ECDC4', alpha=0.8, label='CPU (%)')
        ax2.set_ylabel('CPU Usage (%)', color='#4ECDC4', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='#4ECDC4')
        
        # Add CPU values on bars
        for bar, value in zip(bars2, cpu_usage):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{value:.0f}%', ha='center', va='bottom', fontweight='bold',
                    color='#4ECDC4')
        
        # Title and layout
        ax1.set_title(f'Resource Usage - {model_name.title()}', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = str(self.output_dir / f"resource_usage_{model_name}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        plt.clf()
        gc.collect()
        
        logger.info(f"Resource usage chart saved to: {chart_path}")
        return chart_path
    
    def _generate_performance_summary_chart(self, performance_samples: List[PerformanceMetrics], 
                                          model_name: str) -> str:
        """Generate simple performance summary table."""
        
        # Prepare data for table
        data = []
        for sample in performance_samples:
            data.append({
                'Test Type': sample.test_type.replace('_', ' ').title(),
                'Throughput\n(samples/sec)': f"{sample.throughput_samples_per_sec:.1f}",
                'Avg Latency\n(ms)': f"{sample.avg_latency_ms:.1f}",
                'Memory\n(MB)': f"{sample.memory_usage_mb:.0f}",
                'CPU\n(%)': f"{sample.cpu_usage_percent:.0f}",
                'Status': self._get_performance_status(sample)
            })
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.2, 0.15, 0.15, 0.1, 0.1, 0.15])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)
        
        # Color header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows alternately
        for i in range(1, len(df) + 1):
            color = '#F2F2F2' if i % 2 == 0 else 'white'
            for j in range(len(df.columns)):
                table[(i, j)].set_facecolor(color)
                
        # Color status column based on performance
        for i in range(1, len(df) + 1):
            status = df.iloc[i-1]['Status']
            if status == 'Good':
                table[(i, 5)].set_facecolor('#90EE90')
            elif status == 'OK':
                table[(i, 5)].set_facecolor('#FFE4B5')
            else:
                table[(i, 5)].set_facecolor('#FFB6C1')
        
        # Title
        plt.title(f'Performance Summary - {model_name.title()}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = str(self.output_dir / f"performance_summary_{model_name}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        plt.clf()
        gc.collect()
        
        logger.info(f"Performance summary chart saved to: {chart_path}")
        return chart_path
    
    def _get_performance_status(self, sample: PerformanceMetrics) -> str:
        """Determine performance status based on metrics."""
        # Simple performance assessment
        if sample.avg_latency_ms < 100 and sample.throughput_samples_per_sec > 10:
            return "Good"
        elif sample.avg_latency_ms < 200:
            return "OK"
        else:
            return "Slow"
    
    def _save_comprehensive_csv(self, performance_samples: List, model_name: str) -> str:
        """Save all performance metrics to a comprehensive CSV file."""
        performance_dir = self.output_dir / "performance"
        csv_path = performance_dir / "metrics.csv"
        
        # Prepare data
        data = []
        for i, sample in enumerate(performance_samples, 1):
            row = {
                'run_id': i,
                'test_type': sample.test_type,
                'model_name': model_name,
                'timestamp': sample.timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'),
                'throughput_samples_per_sec': sample.throughput_samples_per_sec,
                'avg_latency_ms': sample.avg_latency_ms,
                'min_latency_ms': sample.min_latency_ms,
                'max_latency_ms': sample.max_latency_ms,
                'p95_latency_ms': sample.p95_latency_ms,
                'p99_latency_ms': sample.p99_latency_ms,
                'memory_usage_mb': sample.memory_usage_mb,
                'cpu_usage_percent': sample.cpu_usage_percent,
                'total_samples_processed': sample.total_samples_processed,
                'total_time_seconds': sample.total_time_seconds,
                'status': self._get_status(sample)
            }
            data.append(row)
        
        # Write CSV
        with open(csv_path, 'w', newline='') as csvfile:
            if data:
                fieldnames = data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
        
        logger.info(f"Performance metrics saved to: {csv_path}")
        return str(csv_path)
    
    def _get_status(self, sample) -> str:
        """Get simple performance status."""
        if sample.avg_latency_ms < 100 and sample.throughput_samples_per_sec > 10:
            return "Good"
        elif sample.avg_latency_ms < 200:
            return "OK"
        else:
            return "Slow"
