#!/usr/bin/env python3
"""
Visualization module for DDoS Detection System
Contains all plotting and visualization functions
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
from config import (
    RESULTS_DIR, DPI, FIGURE_SIZE_MAIN, FIGURE_SIZE_TIMELINE,
    ANOMALY_THRESHOLD_PERCENTILE, ATTACK_PERIODS, TRAIN_SPLIT
)
import matplotlib.patches as patches

# Color palette for consistent visualization
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

class DDoSVisualizer:
    """Simplified visualizer focused on Traffic Volume Analysis and Timeline"""
    
    def __init__(self, detection_system):
        self.detection_system = detection_system
    
    def _check_models_ready(self):
        """Check which models are ready for visualization"""
        ready_models = []
        not_ready_models = []
        
        for model_name, model in self.detection_system.models.items():
            # Check if model is trained
            model_trained = hasattr(model, 'model') and model.model is not None
            has_predict = hasattr(model, 'predict') and callable(getattr(model, 'predict'))
            
            if model_trained and has_predict:
                ready_models.append(model_name)
            else:
                not_ready_models.append(model_name)
        
        return ready_models, not_ready_models
    
    def plot_detailed_timeline(self, attack_data, test_data, features_df=None):
        """Create detailed timeline visualization and confusion matrices - saves individual charts"""
        # Store features_df for use in individual timeline methods
        self.features_df = features_df
        
        # Generate individual model timelines
        self.plot_individual_model_timelines(attack_data, test_data)
        
        # Generate confusion matrices alongside timeline charts
        self.plot_confusion_matrices(attack_data, test_data)

    def plot_individual_model_timelines(self, attack_data, test_data):
        """Create separate timeline charts for each model"""
        for idx, (model_name, model) in enumerate(self.detection_system.models.items()):
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))  # Wider figure for individual charts
            self._plot_single_model_timeline(ax, model_name, model, attack_data, test_data, idx)
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend on the right
            filename = f'timeline_{model_name}.png'
            plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"Saved individual timeline: {RESULTS_DIR}/{filename}")
    
    def _plot_single_model_timeline(self, ax, model_name, model, attack_data, test_data, color_idx):
        """Helper function to plot timeline for a single model"""
        # Check if model is trained and has predict method
        # Neural network models have a 'model' attribute
        model_trained = hasattr(model, 'model') and model.model is not None
        
        has_predict = hasattr(model, 'predict') and callable(getattr(model, 'predict'))
        
        if model_trained and has_predict and test_data is not None and attack_data is not None:
            try:
                # Create extended dataset including more of July 16th for better visualization
                # Use the full dataset if available to show more test data  
                if hasattr(self, 'features_df') and self.features_df is not None:
                    # Use the full dataset and filter for test data
                    full_data = self.features_df
                    
                    # Calculate test date (day after train split)
                    train_date = datetime.strptime(TRAIN_SPLIT, "%Y-%m-%d")
                    test_date = train_date + timedelta(days=1)
                    test_date_str = test_date.strftime("%Y-%m-%d")
                    next_date_str = (test_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    
                    # Include the entire test date plus test data for visualization
                    test_day_data = full_data[
                        (full_data['ts_bin'] >= test_date_str) & 
                        (full_data['ts_bin'] < next_date_str)
                    ].copy()
                    
                    # Combine test day data with test data
                    combined_data = pd.concat([test_day_data, test_data], ignore_index=True)
                    
                    # Sort by timestamp to ensure proper order
                    combined_data = combined_data.sort_values('ts_bin').reset_index(drop=True)
                else:
                    # Fallback to original approach if full dataset not available
                    combined_data = pd.concat([test_data, attack_data], ignore_index=True)
                
                # Get numerical features only
                numerical_cols = combined_data.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) == 0:
                    raise ValueError("No numerical features found in data")
                
                # Scale the features
                X_scaled = self.detection_system.feature_engineer.scaler.transform(combined_data[numerical_cols])
                
                # Get predictions from model
                predictions = model.predict(X_scaled)
                if predictions.ndim > 1:
                    predictions = np.mean((X_scaled - predictions) ** 2, axis=1)
                
                # Handle timestamps
                if 'ts_bin' in combined_data.columns:
                    timestamps = pd.to_datetime(combined_data['ts_bin'])
                    
                    # Ensure timestamps and predictions have same length
                    min_length = min(len(timestamps), len(predictions))
                    timestamps = timestamps[:min_length]
                    predictions = predictions[:min_length]
                    
                    # Plot timeline with datetime x-axis
                    ax.plot(timestamps, predictions, color='#1f77b4', 
                           linewidth=2, label='Anomaly Score')
                    
                    # Attack period indicators removed as requested
                    # (Previously showed Flow Attack Start/End and Volume Attack Start/End annotations)
                    
                    # Add threshold line and highlight exceedances
                    if len(predictions) > 0:
                        # Use the actual thresholds from the detection system
                        if model_name in self.detection_system.thresholds:
                            threshold = self.detection_system.thresholds[model_name]
                        else:
                            # Fallback to calculated threshold if not available
                            normal_data_size = max(1, len(predictions) // 4)
                            normal_predictions = predictions[:normal_data_size]
                            if len(normal_predictions) > 0:
                                threshold = np.percentile(normal_predictions, ANOMALY_THRESHOLD_PERCENTILE)
                            else:
                                threshold = np.mean(predictions)  # Fallback
                        
                        # Draw threshold line for all models (moved outside else block)
                        ax.axhline(y=threshold, color='orange', linestyle='--', linewidth=2,
                                  label=f'Threshold ({threshold:.3f})')
                        
                        # Highlight regions where predictions exceed threshold (moved outside else block)
                        # For autoencoders, anomalies are above threshold
                        above_threshold = predictions > threshold
                        
                        # Create highlighted regions for threshold exceedances (moved outside else block)
                        self._highlight_threshold_exceedances(ax, timestamps, predictions, above_threshold, threshold)
                        
                        # Add debug info to title (moved outside else block)
                        num_anomalies = np.sum(above_threshold)
                        total_points = len(predictions)
                        anomaly_percentage = (num_anomalies / total_points) * 100 if total_points > 0 else 0
                        current_title = ax.get_title()
                        ax.set_title(f"{current_title} - {num_anomalies}/{total_points} anomalies ({anomaly_percentage:.1f}%)")
                    
                    # Format x-axis for datetime with hourly ticks
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                    ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=30))
                    
                    # Rotate x-axis labels for better readability
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    ax.set_xlabel('DateTime (MM-DD HH:MM)', fontsize=12, fontweight='bold')
                    
                else:
                    # Fallback to simple time steps if no timestamp column
                    time_steps = np.arange(len(predictions))
                    ax.plot(time_steps, predictions, color='#1f77b4', 
                           linewidth=2, label='Anomaly Score')
                    
                    # Attack period indicator removed as requested
                    
                    ax.set_xlabel('Time Steps', fontsize=12, fontweight='bold')
                
                ax.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
                ax.set_title(f'{model_name.replace("_", " ").title()} - Traffic Timeline', 
                           fontsize=14, fontweight='bold')
                ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                # Enhanced error message with specific details and solution
                error_msg = f'{model_name.replace("_", " ").title()}\nError: {str(e)[:40]}...\n\nSolution: Ensure models are trained and data is properly formatted.'
                ax.text(0.5, 0.5, error_msg, 
                       ha='center', va='center', transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))
                ax.set_title(f'{model_name.replace("_", " ").title()} - Traffic Timeline')
                ax.set_xlabel('DateTime')
                ax.set_ylabel('Anomaly Score')
        else:
            # Enhanced placeholder for untrained models with clear instructions
            if not model_trained:
                status_msg = "Model Not Trained"
                instruction = "Run: detection_system.train_models(train_data, attack_data)"
            else:
                status_msg = "Predict Method Missing"
                instruction = "Check model implementation"
            
            ax.text(0.5, 0.5, f'{model_name.replace("_", " ").title()}\n{status_msg}\n\n{instruction}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
            ax.set_title(f'{model_name.replace("_", " ").title()} - Traffic Timeline')
            ax.set_xlabel('DateTime')
            ax.set_ylabel('Anomaly Score')

    def _highlight_threshold_exceedances(self, ax, timestamps, predictions, above_threshold, threshold):
        """Helper function to highlight regions where predictions exceed threshold"""
        if not any(above_threshold):
            return
        
        # Find continuous regions where threshold is exceeded
        regions = []
        start_idx = None
        
        for i, exceeds in enumerate(above_threshold):
            if exceeds and start_idx is None:
                start_idx = i
            elif not exceeds and start_idx is not None:
                regions.append((start_idx, i-1))
                start_idx = None
        
        # Handle case where exceedance continues to the end
        if start_idx is not None:
            regions.append((start_idx, len(above_threshold)-1))
        
        # Highlight each region
        for start_idx, end_idx in regions:
            if start_idx < len(timestamps) and end_idx < len(timestamps):
                start_time = timestamps.iloc[start_idx] if hasattr(timestamps, 'iloc') else timestamps[start_idx]
                end_time = timestamps.iloc[end_idx] if hasattr(timestamps, 'iloc') else timestamps[end_idx]
                
                # Add shaded region for threshold exceedance with more prominent highlighting
                ax.axvspan(start_time, end_time, alpha=0.4, color='red', 
                          label='Anomaly Detected' if len(regions) == 1 or start_idx == regions[0][0] else "")
                
                # Add a text annotation for the first region
                # if len(regions) == 1 or start_idx == regions[0][0]:
                #     mid_time = start_time + (end_time - start_time) / 2
                #     ax.text(mid_time, ax.get_ylim()[1] * 0.95, 'ANOMALY', 
                #            ha='center', va='top', fontsize=10, fontweight='bold',
                #            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8))

    def plot_confusion_matrices(self, attack_data, test_data, attack_periods=None):
        """
        Create confusion matrix charts for each model showing TP/FP/TN/FN detection metrics
        
        Args:
            attack_data: DataFrame containing attack period data
            test_data: DataFrame containing test data  
            attack_periods: Dict with 'start' and 'end' datetime strings for actual attack periods
                          If None, uses ATTACK_PERIODS from config
        """
        if attack_periods is None:
            # Use attack periods from configuration
            attack_periods = {}
            for period in ATTACK_PERIODS:
                attack_periods[period['name']] = {
                    'start': period['start'],
                    'end': period['end']
                }
        
        ready_models, not_ready_models = self._check_models_ready()
        
        if not ready_models:
            print("No trained models available for confusion matrix analysis")
            return
        
        # Create ground truth labels
        ground_truth, timestamps, predictions_dict = self._prepare_confusion_matrix_data(
            attack_data, test_data, attack_periods, ready_models
        )
        
        if ground_truth is None:
            print("Could not prepare data for confusion matrix analysis")
            return
        
        # Generate confusion matrix for each ready model
        for model_name in ready_models:
            if model_name in predictions_dict:
                self._plot_single_confusion_matrix(
                    model_name, ground_truth, predictions_dict[model_name], 
                    timestamps, attack_periods
                )
    
    def _prepare_confusion_matrix_data(self, attack_data, test_data, attack_periods, ready_models):
        """Prepare ground truth labels and model predictions for confusion matrix analysis"""
        try:
            # Combine datasets
            if hasattr(self, 'features_df') and self.features_df is not None:
                # Use full dataset for comprehensive analysis
                full_data = self.features_df
                
                # Calculate test date (day after train split)
                train_date = datetime.strptime(TRAIN_SPLIT, "%Y-%m-%d")
                test_date = train_date + timedelta(days=1)
                test_date_str = test_date.strftime("%Y-%m-%d")
                next_date_str = (test_date + timedelta(days=1)).strftime("%Y-%m-%d")
                
                test_day_data = full_data[
                    (full_data['ts_bin'] >= test_date_str) & 
                    (full_data['ts_bin'] < next_date_str)
                ].copy()
                combined_data = pd.concat([test_day_data, test_data], ignore_index=True)
            else:
                combined_data = pd.concat([test_data, attack_data], ignore_index=True)
            
            combined_data = combined_data.sort_values('ts_bin').reset_index(drop=True)
            
            # Convert timestamps
            timestamps = pd.to_datetime(combined_data['ts_bin'])
            
            # Create ground truth labels (0 = normal, 1 = attack)
            ground_truth = np.zeros(len(timestamps))
            
            for attack_name, period in attack_periods.items():
                start_time = pd.to_datetime(period['start'])
                end_time = pd.to_datetime(period['end'])
                
                # Mark attack periods as 1 (positive class)
                attack_mask = (timestamps >= start_time) & (timestamps <= end_time)
                ground_truth[attack_mask] = 1
            
            # Get predictions from each model
            numerical_cols = combined_data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) == 0:
                raise ValueError("No numerical features found")
            
            X_scaled = self.detection_system.feature_engineer.scaler.transform(combined_data[numerical_cols])
            
            predictions_dict = {}
            for model_name in ready_models:
                model = self.detection_system.models[model_name]
                
                try:
                    predictions = model.predict(X_scaled)
                    if predictions.ndim > 1:
                        anomaly_scores = np.mean((X_scaled - predictions) ** 2, axis=1)
                    else:
                        anomaly_scores = predictions
                    
                    # Convert to binary predictions using thresholds
                    if model_name in self.detection_system.thresholds:
                        threshold = self.detection_system.thresholds[model_name]
                    else:
                        # Calculate threshold from normal data
                        normal_data_size = max(1, len(anomaly_scores) // 4)
                        normal_scores = anomaly_scores[:normal_data_size]
                        threshold = np.percentile(normal_scores, ANOMALY_THRESHOLD_PERCENTILE)
                    
                    # Convert to binary predictions
                    binary_predictions = (anomaly_scores > threshold).astype(int)
                    
                    predictions_dict[model_name] = {
                        'binary': binary_predictions,
                        'scores': anomaly_scores,
                        'threshold': threshold
                    }
                    
                except Exception as e:
                    print(f"Error getting predictions for {model_name}: {e}")
                    continue
            
            return ground_truth, timestamps, predictions_dict
            
        except Exception as e:
            print(f"Error preparing confusion matrix data: {e}")
            return None, None, None
    
    def _plot_single_confusion_matrix(self, model_name, ground_truth, predictions_data, timestamps, attack_periods):
        """Create and save optimized confusion matrix visualization for a single model"""
        try:
            binary_predictions = predictions_data['binary']
            anomaly_scores = predictions_data['scores']
            threshold = predictions_data['threshold']
            
            # Ensure same length
            min_length = min(len(ground_truth), len(binary_predictions))
            ground_truth = ground_truth[:min_length]
            binary_predictions = binary_predictions[:min_length]
            
            # Calculate confusion matrix
            cm = confusion_matrix(ground_truth, binary_predictions)
            
            # Calculate comprehensive metrics
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Set up enhanced styling
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create figure with improved layout for 2 charts side by side
            fig = plt.figure(figsize=(16, 8))
            gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
            
            # 1. Enhanced Confusion Matrix Heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            
            # Create normalized confusion matrix for better visualization
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Enhanced heatmap with blue tones for scientific appearance
            sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=ax1,
                       xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'],
                       cbar_kws={'label': 'Normalized Count'})
            ax1.set_title('Normalized Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
            ax1.set_ylabel('True Label', fontsize=14, fontweight='semibold')
            ax1.set_xlabel('Predicted Label', fontsize=14, fontweight='semibold')
            
            # Add raw counts as text overlay
            for i in range(2):
                for j in range(2):
                    text = ax1.texts[i*2 + j]
                    text.set_text(f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})')
                    text.set_fontsize(12)
                    text.set_fontweight('bold')
            
            # 2. Enhanced Metrics Bar Chart
            ax3 = fig.add_subplot(gs[0, 1])
            
            metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            metrics_values = [accuracy, precision, recall, f1]
            # Scientific color palette for metrics bars
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            bars = ax3.bar(metrics_names, metrics_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
            ax3.set_title('Detailed Performance Metrics', fontsize=16, fontweight='bold', pad=20)
            ax3.set_ylabel('Score', fontsize=14, fontweight='semibold')
            ax3.set_ylim(0, 1)
            
            # Add value labels on bars with enhanced styling
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save the figure with enhanced quality
            filename = f'confusion_matrix_{model_name}.png'
            filepath = os.path.join(RESULTS_DIR, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
            
            print(f"Saved optimized confusion matrix analysis: {filepath}")
            print(f"Model: {model_name}")
            print(f"   • Accuracy: {accuracy:.3f} | Precision: {precision:.3f}")
            print(f"   • Recall: {recall:.3f} | F1: {f1:.3f}")

            
        except Exception as e:
            print(f"Error creating confusion matrix for {model_name}: {e}")
            import traceback
            traceback.print_exc()
