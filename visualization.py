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
        # Ensure base results directory exists
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def _get_model_results_dir(self, model_name: str) -> str:
        """Return the results directory for a specific model and ensure it exists."""
        # Keep the raw model_name as subfolder to match keys like 'lstm_ae', 'standard_ae'
        model_dir = os.path.join(RESULTS_DIR, str(model_name))
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
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
        
        # Generate daily separated charts
        self.plot_daily_error_metrics(test_data)

    def plot_individual_model_timelines(self, attack_data, test_data):
        """Create separate timeline charts for each model"""
        for idx, (model_name, model) in enumerate(self.detection_system.models.items()):
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))  # Wider figure for individual charts
            self._plot_single_model_timeline(ax, model_name, model, attack_data, test_data, idx)
            
            plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend on the right
            filename = f'timeline_{model_name}.png'
            out_dir = self._get_model_results_dir(model_name)
            plt.savefig(os.path.join(out_dir, filename), dpi=DPI, bbox_inches='tight')
            plt.close()
            print(f"Saved individual timeline: {os.path.join(out_dir, filename)}")
    
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
                    
                    # Calculate extended timeline date range (include last training day + available test period)
                    train_date = datetime.strptime(TRAIN_SPLIT, "%Y-%m-%d")
                    # Start from the last day of training data for comprehensive timeline
                    timeline_start_str = train_date.strftime("%Y-%m-%d")
                    
                    # Use the actual end date of available data instead of fixed extension
                    data_end_date = pd.to_datetime(full_data['ts_bin'].max()).date()
                    timeline_end_str = (data_end_date + timedelta(days=1)).strftime("%Y-%m-%d")
                    
                    # Include last training day + entire available test period for comprehensive visualization
                    test_period_data = full_data[
                        (full_data['ts_bin'] >= timeline_start_str) & 
                        (full_data['ts_bin'] < timeline_end_str)
                    ].copy()
                    
                    # Combine test period data with test data
                    combined_data = pd.concat([test_period_data, test_data], ignore_index=True)
                    
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
                    
                    # Format x-axis for datetime with appropriate ticks for multi-day timeline
                    # Use day locator for major ticks and hour locator for minor ticks
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))
                    
                    # Explicitly set x-axis limits to show all data
                    ax.set_xlim(timestamps.min(), timestamps.max())
                    
                    # Rotate x-axis labels for better readability
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
                    
                    ax.set_xlabel('Date (MM-DD)', fontsize=12, fontweight='bold')
                    
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
                
                # Calculate extended timeline date range (include last training day + available test period)
                train_date = datetime.strptime(TRAIN_SPLIT, "%Y-%m-%d")
                # Start from the last day of training data for comprehensive timeline
                timeline_start_str = train_date.strftime("%Y-%m-%d")
                
                # Use the actual end date of available data instead of fixed extension
                data_end_date = pd.to_datetime(full_data['ts_bin'].max()).date()
                timeline_end_str = (data_end_date + timedelta(days=1)).strftime("%Y-%m-%d")
                
                test_period_data = full_data[
                    (full_data['ts_bin'] >= timeline_start_str) & 
                    (full_data['ts_bin'] < timeline_end_str)
                ].copy()
                combined_data = pd.concat([test_period_data, test_data], ignore_index=True)
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
            out_dir = self._get_model_results_dir(model_name)
            filepath = os.path.join(out_dir, filename)
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

    def plot_model_history(self, model_name, model_history):
        """Plot training history metrics for a model"""
        if not hasattr(model_history, 'history'):
            print(f"No training history available for {model_name}")
            return
            
        history = model_history.history
        epochs = range(1, len(history['loss']) + 1)
            
        plt.figure(figsize=(16, 8))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['loss'], 'bo-', label='Training Loss (MSE)')
        if 'val_loss' in history:
            plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss (MSE)')
        plt.title(f'Training and Validation Loss: {model_name.upper().replace("_", " ")}', 
                fontsize=16, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot MAE
        plt.subplot(1, 2, 2)
        if 'mae' in history:
            plt.plot(epochs, history['mae'], 'bo-', label='Training MAE')
        if 'val_mae' in history:
            plt.plot(epochs, history['val_mae'], 'ro-', label='Validation MAE')
        plt.title(f'Training and Validation MAE: {model_name.upper().replace("_", " ")}', 
                fontsize=16, fontweight='bold')
        plt.xlabel('Epochs', fontsize=12)
        plt.ylabel('Mean Absolute Error', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        filename = f'training_history_{model_name}.png'
        out_dir = self._get_model_results_dir(model_name)
        plt.savefig(os.path.join(out_dir, filename), dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"Saved training history: {os.path.join(out_dir, filename)}")

    def plot_daily_error_metrics(self, test_data):
        """Plot error metrics separated by day for test and validation data"""
        ready_models, _ = self._check_models_ready()
        
        if not ready_models:
            print("No models ready for daily error metrics visualization")
            return
        
        # Get extended data if available
        if not hasattr(self, 'features_df') or self.features_df is None:
            print("No extended features data available for daily analysis")
            return
        
        # Calculate date ranges
        train_date = datetime.strptime(TRAIN_SPLIT, "%Y-%m-%d")
        
        # Define day ranges based on available data
        data_end_date = pd.to_datetime(self.features_df['ts_bin'].max()).date()
        
        day_configs = [
            {
                'name': 'Day_16_Test',
                'start_date': train_date + timedelta(days=1),
                'end_date': train_date + timedelta(days=2),
                'label': 'Day 16 (Test)'
            },
            {
                'name': 'Day_17_Validation', 
                'start_date': train_date + timedelta(days=2),
                'end_date': min(train_date + timedelta(days=3), datetime.combine(data_end_date + timedelta(days=1), datetime.min.time())),
                'label': 'Day 17 (Validation)'
            }
        ]
        
        for model_name in ready_models:
            model = self.detection_system.models[model_name]
            
            try:
                # Get model thresholds
                thresholds = self.detection_system.thresholds
                if isinstance(thresholds, dict) and model_name in thresholds:
                    standard_threshold = thresholds[model_name]
                else:
                    standard_threshold = 0.5
                
                # Calculate overall model metrics from training/validation data for reference lines
                model_metrics = model.get_training_metrics()
                model_mse = model_metrics['mse']
                model_mae = model_metrics['mae'] 
                model_stddev = model_metrics['stddev']
                
                # Create figure with subplots for multiple chart types per day
                fig = plt.figure(figsize=(28, 12))
                fig.suptitle(f'Daily Error Analysis: {model_name.upper().replace("_", " ")}', 
                           fontsize=18, fontweight='bold')
                
                # Create grid: 2x6 layout (2 rows for days, 6 columns: histogram, boxplot, violin, timeline, table, spacer)
                gs = fig.add_gridspec(2, 6, width_ratios=[1.2, 1, 1, 1.5, 1.2, 0.2], hspace=0.3, wspace=0.3)
                
                daily_stats = []
                
                for i, day_config in enumerate(day_configs):
                    # Filter data for this specific day
                    start_str = day_config['start_date'].strftime("%Y-%m-%d")
                    
                    # For the end date, use the actual available data end for the last day
                    if i == len(day_configs) - 1:  # Last day (Day 17)
                        # Use all available data from start of day 17 onwards
                        daily_data = self.features_df[
                            self.features_df['ts_bin'] >= start_str
                        ].copy()
                    else:
                        # Normal day filtering
                        end_str = day_config['end_date'].strftime("%Y-%m-%d")
                        daily_data = self.features_df[
                            (self.features_df['ts_bin'] >= start_str) & 
                            (self.features_df['ts_bin'] < end_str)
                        ].copy()
                    
                    if len(daily_data) == 0:
                        print(f"No data available for {day_config['label']}")
                        continue
                    
                    # Prepare data for prediction
                    normalized_data = self.detection_system.prepare_data_for_prediction(daily_data)
                    errors = model.predict(normalized_data)
                    
                    # Handle LSTM sequence alignment
                    if model_name == 'lstm_ae':
                        sequence_length = getattr(model, 'sequence_length', 10)
                        if len(daily_data) > sequence_length:
                            daily_data_aligned = daily_data.iloc[sequence_length-1:].copy()
                            if len(daily_data_aligned) != len(errors):
                                min_len = min(len(daily_data_aligned), len(errors))
                                daily_data_aligned = daily_data_aligned.iloc[:min_len]
                                errors = errors[:min_len]
                            plot_data = daily_data_aligned
                        else:
                            plot_data = daily_data
                    else:
                        plot_data = daily_data
                    
                    # Calculate statistics
                    mse = np.mean(errors)
                    mae = np.mean(np.abs(errors))
                    std_dev = np.std(errors)
                    anomaly_count = np.sum(errors > standard_threshold)
                    anomaly_percentage = (anomaly_count / len(errors)) * 100 if len(errors) > 0 else 0
                    
                    daily_stats.append({
                        'day': day_config['label'],
                        'mse': mse,
                        'mae': mae,
                        'std_dev': std_dev,
                        'anomalies': anomaly_count,
                        'anomaly_pct': anomaly_percentage,
                        'total_points': len(errors)
                    })
                    
                    # Plot 1: Error Distribution Histogram
                    ax1 = fig.add_subplot(gs[i, 0])
                    sns.histplot(errors, kde=True, ax=ax1, color=COLORS[i])
                    ax1.axvline(standard_threshold, color='red', linestyle='--', 
                              label=f'Threshold: {standard_threshold:.5f}')
                    ax1.axvline(mse, color='green', linestyle='-', 
                              label=f'MSE: {mse:.5f}')
                    ax1.set_title(f'{day_config["label"]} - Histogram', 
                                fontsize=12, fontweight='bold')
                    ax1.set_xlabel('Reconstruction Error', fontsize=10)
                    ax1.set_ylabel('Frequency', fontsize=10)
                    ax1.legend(fontsize=8)
                    
                    # Plot 2: Box Plot
                    ax2 = fig.add_subplot(gs[i, 1])
                    sns.boxplot(y=errors, ax=ax2, color=COLORS[i])
                    ax2.axhline(standard_threshold, color='red', linestyle='--', 
                              label=f'Threshold')
                    ax2.set_title(f'{day_config["label"]} - Box Plot', 
                                fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Reconstruction Error', fontsize=10)
                    ax2.legend(fontsize=8)
                    
                    # Plot 3: Violin Plot
                    ax3 = fig.add_subplot(gs[i, 2])
                    sns.violinplot(y=errors, ax=ax3, color=COLORS[i])
                    ax3.axhline(standard_threshold, color='red', linestyle='--', 
                              label=f'Threshold')
                    ax3.set_title(f'{day_config["label"]} - Violin Plot', 
                                fontsize=12, fontweight='bold')
                    ax3.set_ylabel('Reconstruction Error', fontsize=10)
                    ax3.legend(fontsize=8)
                    
                    # Plot 4: Error Timeline for this day
                    ax4 = fig.add_subplot(gs[i, 3])
                    timestamps = pd.to_datetime(plot_data['ts_bin'])
                    ax4.plot(timestamps, errors, color=COLORS[i], linewidth=1, label='Reconstruction Error')
                    
                    # Add model reference metrics as horizontal lines
                    ax4.axhline(standard_threshold, color='red', linestyle='--', 
                              label=f'Threshold: {standard_threshold:.5f}')
                    
                    # Add overall model training MSE line
                    ax4.axhline(model_mse, color='green', linestyle='-', 
                              label=f'Training MSE: {model_mse:.5f}')
                    
                    # Add overall model training MAE line
                    ax4.axhline(model_mae, color='orange', linestyle='-.', 
                              label=f'Training MAE: {model_mae:.5f}')
                    
                    # Add overall model training Std Dev lines (model_mse ± model_stddev)
                    ax4.axhline(model_mse + model_stddev, color='purple', linestyle=':', alpha=0.7,
                              label=f'Training MSE+σ: {model_mse + model_stddev:.5f}')
                    ax4.axhline(model_mse - model_stddev, color='purple', linestyle=':', alpha=0.7,
                              label=f'Training MSE-σ: {model_mse - model_stddev:.5f}')
                    
                    # Highlight anomalies
                    anomaly_mask = errors > standard_threshold
                    if np.any(anomaly_mask):
                        ax4.scatter(timestamps[anomaly_mask], errors[anomaly_mask], 
                                  color='red', alpha=0.6, s=20, label='Anomalies')
                    
                    ax4.set_title(f'{day_config["label"]} - Timeline', 
                                fontsize=12, fontweight='bold')
                    ax4.set_xlabel('Time', fontsize=10)
                    ax4.set_ylabel('Reconstruction Error', fontsize=10)
                    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, fontsize=8)
                    ax4.legend(fontsize=7, loc='upper left')
                    ax4.grid(True, alpha=0.3)
                    
                    # Plot 5: Individual table for this day
                    table_ax = fig.add_subplot(gs[i, 4])
                    table_ax.axis('off')
                    
                    # Prepare table data for this specific day
                    table_data = [
                        ['MSE', f'{mse:.5f}'],
                        ['MAE', f'{mae:.5f}'],
                        ['Std Dev', f'{std_dev:.5f}'],
                        ['Threshold', f'{standard_threshold:.5f}'],
                        ['Anomalies', f'{anomaly_count}'],
                        ['Anomaly %', f'{anomaly_percentage:.1f}%'],
                        ['Total Points', f'{len(errors)}'],
                        ['Min Error', f'{np.min(errors):.5f}'],
                        ['Max Error', f'{np.max(errors):.5f}']
                    ]
                    
                    headers = ['Metric', day_config['label']]
                    
                    # Create the table
                    table = table_ax.table(cellText=table_data, colLabels=headers,
                                         cellLoc='center', loc='center',
                                         colWidths=[0.5, 0.5])
                    
                    # Style the table
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.8)
                    
                    # Style header row with day-specific color
                    for j in range(len(headers)):
                        table[(0, j)].set_facecolor(COLORS[i])
                        table[(0, j)].set_text_props(weight='bold', color='white')
                    
                    # Style data rows with alternating colors
                    for row_idx in range(1, len(table_data) + 1):
                        for col_idx in range(len(headers)):
                            if row_idx % 2 == 0:
                                table[(row_idx, col_idx)].set_facecolor('#f0f0f0')
                            else:
                                table[(row_idx, col_idx)].set_facecolor('#ffffff')
                    
                    table_ax.set_title(f'{day_config["label"]} Metrics', 
                                     fontsize=12, fontweight='bold', pad=10)
                
                # Adjust layout
                plt.subplots_adjust(top=0.93, bottom=0.05, left=0.05, right=0.95)
                
                # Save figure
                filename = f'daily_error_analysis_{model_name}.png'
                out_dir = self._get_model_results_dir(model_name)
                plt.savefig(os.path.join(out_dir, filename), dpi=DPI, bbox_inches='tight')
                plt.close()
                print(f"Saved daily error analysis: {os.path.join(out_dir, filename)}")
                
            except Exception as e:
                print(f"Error creating daily error analysis for {model_name}: {e}")
                if 'fig' in locals():
                    plt.close(fig)
