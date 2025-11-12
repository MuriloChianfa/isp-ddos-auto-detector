"""
Visualization orchestration and management for DDoS detection pipeline.
Orchestrates all visualization tasks including training plots, feature analysis, and anomaly detection visualizations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any

from framework.visualization.training_plots import TrainingVisualizer
from framework.visualization.anomaly_plots import AnomalyVisualizer
from framework.visualization.evaluation_plots import EvaluationVisualizer
from framework.utils import get_results_path


class VisualizationManager:
    """Orchestrates all visualization tasks"""
    
    def __init__(self, dataset_name: str, model_name: str, time_span: int):
        """
        Initialize VisualizationManager
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            time_span: Time span in seconds
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.time_span = time_span
        self.results_path = get_results_path(dataset_name, model_name, time_span, "models")
    
    def generate_training_visualizations(self, history: Any, model_name: str):
        """
        Generate training-related plots and summaries
        
        Args:
            history: Training history object from model training
            model_name: Name of the model that was trained
        """
        if history is None:
            print("No training history available - skipping training visualizations")
            return
        
        print("\nVisualizing training results...")
        training_viz = TrainingVisualizer(
            dataset_name=self.dataset_name, 
            model_name=self.model_name, 
            time_span=self.time_span
        )
        training_viz.plot_training_history(history, model_name=model_name)
        training_viz.print_training_summary(history)
    
    def generate_feature_analysis(self, model: Any, test_data: np.ndarray, 
                                 feature_names: List[str], feature_errors: np.ndarray,
                                 importance_indices: np.ndarray, importance_analysis: Optional[Dict] = None):
        """
        Generate feature importance and analysis plots
        
        Args:
            model: Trained model object
            test_data: Test dataset
            feature_names: List of feature names
            feature_errors: Array of feature reconstruction errors
            importance_indices: Array of feature importance indices
            importance_analysis: Optional temporal analysis results (planned for future temporal models)
        """
        # Create feature importance visualizations (only for models that support it)
        if self.model_name == 'autoencoder':
            print("Creating feature importance visualizations...")
            eval_viz = EvaluationVisualizer(self.results_path)
            eval_viz.plot_feature_importance(feature_errors, feature_names, importance_indices, top_n=20)
            # eval_viz.plot_feature_importance_detailed(feature_errors, feature_names, importance_indices, top_n=15)
    
    
    def generate_anomaly_visualizations(self, combined_features: pd.DataFrame, threshold: float,
                                       test_scores: np.ndarray, all_thresholds: Dict):
        """
        Generate anomaly detection visualizations
        
        Args:
            combined_features: DataFrame containing all features and anomaly detection results
            threshold: Threshold value used for anomaly detection
            test_scores: Array of test reconstruction scores
            all_thresholds: Dictionary of all calculated thresholds
        """
        print("\nVisualizing anomaly detection results...")
        anomaly_viz = AnomalyVisualizer(
            dataset_name=self.dataset_name, 
            model_name=self.model_name, 
            time_span=self.time_span
        )
        anomaly_viz.print_threshold_comparison(test_scores, all_thresholds)
        anomaly_viz.plot_anomaly_detection(combined_features, threshold)
        anomaly_viz.print_anomaly_statistics(combined_features, threshold)
    
    def generate_feature_reconstruction_error_plots(self, model: Any, processing_list: List[Tuple],
                                                   features_dict: Dict, feature_names: List[str],
                                                   generate_plots: bool = False):
        """
        Generate detailed feature reconstruction error visualizations
        
        Args:
            model: Trained model object
            processing_list: List of (split_name, scaled_data, scores, features_data) tuples
            features_dict: Dictionary containing feature data for each split
            feature_names: List of feature names
            generate_plots: Whether to generate the detailed plots (can create many files)
        """
        if not generate_plots:
            return
            
        print("Generating detailed feature reconstruction error visualizations...")
        try:
            from framework.visualization.feature_error_plots import generate_feature_reconstruction_error_plots
            generate_feature_reconstruction_error_plots(
                model, processing_list, features_dict, feature_names,
                self.dataset_name, self.model_name, self.time_span
            )
        except ImportError as e:
            print(f"Warning: Could not import feature error plots module: {e}")
        except Exception as e:
            print(f"Error generating feature reconstruction error plots: {e}")
    
    def generate_all_visualizations(self, history: Optional[Any], model: Any, 
                                   test_data: np.ndarray, feature_names: List[str],
                                   feature_errors: np.ndarray, importance_indices: np.ndarray,
                                   combined_features: pd.DataFrame, threshold: float,
                                   test_scores: np.ndarray, all_thresholds: Dict,
                                   processing_list: Optional[List[Tuple]] = None,
                                   features_dict: Optional[Dict] = None,
                                   importance_analysis: Optional[Dict] = None,
                                   generate_reconstruction_error: bool = False):
        """
        Generate all visualizations in the correct order
        
        Args:
            history: Training history object
            model: Trained model object
            test_data: Test dataset
            feature_names: List of feature names
            feature_errors: Array of feature reconstruction errors
            importance_indices: Array of feature importance indices
            combined_features: DataFrame with all features and results
            threshold: Anomaly detection threshold
            test_scores: Array of test reconstruction scores
            all_thresholds: Dictionary of all calculated thresholds
            processing_list: Optional list for reconstruction error plots
            features_dict: Optional features dictionary for reconstruction error plots
            importance_analysis: Optional temporal analysis results
            generate_reconstruction_error: Whether to generate detailed reconstruction error plots
        """
        # Training visualizations (if history available)
        if history is not None:
            self.generate_training_visualizations(history, self.model_name)
        
        # Feature importance analysis
        self.generate_feature_analysis(
            model, test_data, feature_names, feature_errors, 
            importance_indices, importance_analysis
        )
        
        # Optional detailed feature reconstruction error plots
        if processing_list is not None and features_dict is not None:
            self.generate_feature_reconstruction_error_plots(
                model, processing_list, features_dict, feature_names, 
                generate_reconstruction_error
            )
        
        # Anomaly detection visualizations
        self.generate_anomaly_visualizations(
            combined_features, threshold, test_scores, all_thresholds
        )
        
        print(f"\nAll visualizations completed. Results saved to: {self.results_path}")
    
    def get_results_path(self) -> str:
        """Get the results directory path"""
        return self.results_path
