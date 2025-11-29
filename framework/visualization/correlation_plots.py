"""
Visualization tools for feature correlation analysis.

This module provides plotting functions to visualize correlation matrices
and correlation coefficient distributions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from typing import Optional

plt.ioff()  # Disable interactive mode


class CorrelationVisualizer:
    """
    Generates visualizations for correlation analysis results.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the correlation visualizer.
        
        Args:
            output_dir: Directory where plots will be saved
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float,
        figsize: tuple = (20, 18)
    ) -> str:
        """
        Generate a correlation heatmap with annotations.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            threshold: Correlation threshold to highlight
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved plot file
        """
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Generate heatmap
        ax = sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=False,  # Don't annotate all cells (too crowded)
            fmt='.2f',
            cmap='RdBu_r',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={
                'label': 'Pearson Correlation Coefficient',
                'shrink': 0.8
            },
            xticklabels=True,
            yticklabels=True
        )
        
        plt.title(
            f'Feature Correlation Matrix\n'
            f'({len(corr_matrix)} Features, Threshold: {threshold})',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        plt.xlabel('Features', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        
        # Explicitly set all tick positions and labels
        ax.set_xticks(np.arange(len(corr_matrix.columns)) + 0.5)
        ax.set_yticks(np.arange(len(corr_matrix.index)) + 0.5)
        ax.set_xticklabels(corr_matrix.columns, rotation=90, ha='right', fontsize=7)
        ax.set_yticklabels(corr_matrix.index, rotation=0, fontsize=7)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        gc.collect()
        
        return output_path
    
    def plot_correlation_distribution(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float,
        figsize: tuple = (12, 6)
    ) -> str:
        """
        Generate a histogram of correlation coefficients.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            threshold: Correlation threshold to mark
            figsize: Figure size (width, height)
            
        Returns:
            Path to saved plot file
        """
        # Extract upper triangle values (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_values = corr_matrix.where(mask).values.flatten()
        corr_values = corr_values[~np.isnan(corr_values)]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Full distribution
        ax1.hist(corr_values, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
        ax1.axvline(-threshold, color='red', linestyle='--', linewidth=2)
        ax1.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Correlation Coefficients\n(All Pairs)', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Focus on high correlations
        high_corr = corr_values[np.abs(corr_values) >= threshold]
        if len(high_corr) > 0:
            ax2.hist(high_corr, bins=20, color='darkred', alpha=0.7, edgecolor='black')
            ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
            ax2.axvline(-threshold, color='red', linestyle='--', linewidth=2)
            ax2.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax2.set_title(f'High Correlations (|r| ≥ {threshold})\n({len(high_corr)} pairs)', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(
                0.5, 0.5,
                f'No correlations\nexceed threshold\n(|r| ≥ {threshold})',
                ha='center', va='center',
                fontsize=14,
                transform=ax2.transAxes
            )
            ax2.set_xlabel('Correlation Coefficient', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax2.set_title(f'High Correlations (|r| ≥ {threshold})', fontsize=12, fontweight='bold')
        
        # Add statistics text
        stats_text = (
            f'Total pairs: {len(corr_values)}\n'
            f'Mean: {np.mean(corr_values):.4f}\n'
            f'Std: {np.std(corr_values):.4f}\n'
            f'Max: {np.max(corr_values):.4f}\n'
            f'Min: {np.min(corr_values):.4f}\n'
            f'|r| ≥ {threshold}: {len(high_corr)}'
        )
        fig.text(
            0.98, 0.02, stats_text,
            fontsize=9,
            ha='right',
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, 'correlation_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        gc.collect()
        
        return output_path
