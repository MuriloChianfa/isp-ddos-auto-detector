"""
Main pipeline orchestrator for DDoS detection analysis.
Coordinates all components and manages the complete analysis workflow.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List, Any

from framework.settings import SettingsManager
from framework.loader import NetworkDataLoader
from framework.features import NetworkFeatureExtractor
from framework.models.core.manager import ModelManager
from framework.detector import AnomalyDetector
from framework.visualizator import VisualizationManager
from framework.results import ResultsManager
from framework.performance import RealTimePerformanceEvaluator, PerformanceReporter, PerformanceMetrics
from framework.visualization.performance_plots import PerformancePlotter

# Set up logging
logger = logging.getLogger(__name__)


class DDoSDetectorPipeline:
    """Main pipeline orchestrator for DDoS detection"""
    
    def __init__(self, dataset_name: str, model_name: str = 'autoencoder', 
                 time_span: int = 300, **kwargs):
        """
        Initialize the DDoS detection pipeline
        
        Args:
            dataset_name: Name of the dataset to analyze
            model_name: Name of the model to use
            time_span: Time span for feature aggregation in seconds
            **kwargs: Additional configuration parameters
        """
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.time_span = time_span
        
        # Configuration parameters
        self.use_cache = kwargs.get('use_cache', True)
        self.max_processes = kwargs.get('max_processes', None)
        self.use_fixed_threshold = kwargs.get('use_fixed_threshold', False)
        self.force_retrain = kwargs.get('force_retrain', False)
        self.force_regenerate = kwargs.get('force_regenerate', False)
        self.generate_reconstruction_error = kwargs.get('generate_reconstruction_error', False)
        
        # Performance evaluation parameters
        self.evaluate_performance = kwargs.get('evaluate_performance', False)
        self.performance_samples = kwargs.get('performance_samples', 1000)
        
        # Optimization parameters
        self.optimize = kwargs.get('optimize', False)
        self.optimize_n_iter = kwargs.get('optimize_n_iter', 10)
        self.optimize_only = kwargs.get('optimize_only', False)
        
        # Initialize components
        self.settings_manager = SettingsManager()
        self.dataset_config = None
        self.loader = None
        self.feature_extractor = None
        self.model_manager = None
        self.detector = None
        self.visualizer = None
        self.results_manager = None
        
        # Analysis results
        self.features_dict = None
        self.processed_features = None
        self.model = None
        self.training_history = None
        self.combined_features = None
        self.threshold = None
        self.all_thresholds = None
    
    def _get_feature_config_with_hierarchy(self):
        """
        Get feature configuration with proper hierarchy using automatic optimal loading.
        
        Uses the centralized get_feature_config() function which implements:
        1. Config overrides (highest priority)
        2. Optimal features from correlation analysis
        3. All features fallback (lowest priority)
        
        Returns:
            Feature configuration (list or dict)
        """
        from framework.settings import get_feature_config
        
        # Get feature config with auto-loading
        feature_config = get_feature_config(self.dataset_name, self.time_span, self.model_name)
        
        # Handle EMA alpha from window config if needed
        windows = self.dataset_config.get('windows', {})
        window_config = windows.get(str(self.time_span), {})
        
        if 'ema_alpha' in window_config:
            if isinstance(feature_config, dict):
                # If it's a dict, merge EMA alpha if not already present
                if 'ema_alpha' not in feature_config:
                    feature_config = feature_config.copy()
                    feature_config['ema_alpha'] = window_config['ema_alpha']
            elif feature_config:
                # If it's a list, wrap it with EMA alpha
                feature_config = {
                    'features': feature_config,
                    'ema_alpha': window_config['ema_alpha']
                }
        
        return feature_config
    
    def initialize_components(self) -> bool:
        """
        Initialize and validate all pipeline components
        
        Returns:
            True if initialization successful, False otherwise
        """
        # Validate configuration
        is_valid, self.dataset_config = self.settings_manager.validate_all_parameters(
            self.dataset_name, self.model_name, self.time_span
        )
        
        if not is_valid:
            return False
        
        # Print configuration summary
        self.settings_manager.print_configuration_summary(
            self.dataset_name, self.model_name, self.time_span,
            self.use_cache, self.max_processes
        )
        
        # Initialize data loader
        print("Initializing data loader...")
        self.loader = NetworkDataLoader(
            dataset_config=self.dataset_config, 
            use_cache=self.use_cache, 
            max_processes=self.max_processes
        )
        
        # Initialize feature extractor
        print("Initializing feature extractor...")
        feature_config = self._get_feature_config_with_hierarchy()
        
        self.feature_extractor = NetworkFeatureExtractor(
            time_span=self.time_span,
            use_cache=self.use_cache,
            dataset_name=self.dataset_name,
            max_processes=self.max_processes,
            feature_config=feature_config
        )
        
        # Initialize other managers (will be created after model is ready)
        self.results_manager = ResultsManager(
            self.dataset_name, self.model_name, self.time_span
        )
        
        self.visualizer = VisualizationManager(
            self.dataset_name, self.model_name, self.time_span
        )
        
        return True
    
    def extract_and_prepare_features(self) -> bool:
        """
        Extract features and prepare training data
        
        Returns:
            True if successful, False otherwise
        """
        print("Extracting features...")
        self.features_dict = self.feature_extractor.extract_features_to_csv(
            self.loader, force_regenerate=self.force_regenerate
        )
        
        if not self.features_dict:
            print("Error: Feature extraction failed")
            return False
        
        print("Preparing training data...")
        self.processed_features = self.feature_extractor.prepare_training_data(self.features_dict)
        
        return True
    
    def initialize_and_train_model(self) -> bool:
        """
        Initialize model manager and handle model training/loading
        
        Returns:
            True if successful, False otherwise
        """
        # Get training features for model initialization
        train_features = self.processed_features['train']['features']
        val_features = self.processed_features['validation']['features']
        
        # Run hyperparameter optimization if requested
        if self.optimize and self.model_name in ['isolation_forest', 'one_class_svm', 'local_outlier_factor', 'autoencoder']:
            # Check if one_class_svm is using sgd_rbf kernel (not compatible with standard optimization)
            if self.model_name == 'one_class_svm':
                from framework.models.core.factory import get_model_params
                model_params = get_model_params(self.model_name, self.dataset_name, self.time_span)
                if model_params.get('kernel') == 'sgd_rbf':
                    print("\n" + "="*70)
                    print("SKIPPING HYPERPARAMETER OPTIMIZATION")
                    print("="*70)
                    print("Note: Hyperparameter optimization is not supported for sgd_rbf kernel.")
                    print("The sgd_rbf kernel uses SGDOneClassSVM with RBF kernel approximation,")
                    print("which has a different architecture than standard OneClassSVM.")
                    print("Using default parameters from config instead.")
                    print("="*70 + "\n")
                    # Skip optimization for sgd_rbf
                    self.optimize = False
            
            if self.optimize:  # Recheck after potential modification
                print("\n" + "="*70)
                print("RUNNING HYPERPARAMETER OPTIMIZATION")
                print("="*70)
                
                # Create a temporary model manager to get scaled data
                temp_model_manager = ModelManager(
                    self.model_name, self.dataset_name, self.time_span
                )
                temp_model, _ = temp_model_manager.get_or_create_model(
                    train_features, self.use_fixed_threshold, force_retrain=True
                )
                
                # Scale the data
                scaled_train = temp_model.transform_data(train_features)
                scaled_val = temp_model.transform_data(val_features)
                
                # Generate validation labels from attack periods
                # Get attack periods (they apply to test set, but we'll use a portion for validation)
                windows = self.dataset_config.get('windows', {})
                window_config = windows.get(str(self.time_span), {})
                attack_periods_raw = window_config.get('attack_periods', [])
                
                # Get validation timestamps
                val_timestamps = self.features_dict['validation']['timestamp']
                
                # Convert attack periods to datetime if they're tuples
                attack_periods = []
                for period in attack_periods_raw:
                    if isinstance(period, tuple):
                        start_dt = pd.to_datetime(period[0])
                        end_dt = pd.to_datetime(period[1])
                        # Make sure timezone matches val_timestamps
                        if val_timestamps.dt.tz is not None:
                            if start_dt.tz is None:
                                start_dt = start_dt.tz_localize('UTC')
                            if end_dt.tz is None:
                                end_dt = end_dt.tz_localize('UTC')
                        attack_periods.append({
                            'start': start_dt,
                            'end': end_dt
                        })
                    elif isinstance(period, dict):
                        attack_periods.append(period)
                
                # Generate ground truth labels for validation set (1 = attack, 0 = normal)
                val_labels = np.zeros(len(val_timestamps), dtype=int)
                if attack_periods:
                    for period in attack_periods:
                        mask = (val_timestamps >= period['start']) & (val_timestamps <= period['end'])
                        val_labels = val_labels | mask.values
                    # print(f"Validation set: {np.sum(val_labels)} attack samples out of {len(val_labels)} total ({100*np.sum(val_labels)/len(val_labels):.2f}%)")
                # else:
                #     print("Warning: No attack periods defined, using all normal labels for validation")
                
                # Determine scoring method based on validation labels
                num_attack_samples = np.sum(val_labels)
                if num_attack_samples == 0:
                    scoring_method = 'anomaly_score'
                    # print("\nNote: No attack samples in validation set.")
                    # print("Falling back to 'anomaly_score' (unsupervised) for optimization.")
                    # print("This method optimizes for anomaly separation without labeled data.")
                    val_labels = None  # Not needed for anomaly_score
                elif num_attack_samples < 10:
                    scoring_method = 'anomaly_score'
                    # print(f"\nWarning: Very few attack samples ({num_attack_samples}) in validation set.")
                    # print("Falling back to 'anomaly_score' for more stable optimization.")
                    val_labels = None
                else:
                    scoring_method = 'f1_score'
                    print(f"\nUsing F1 score for optimization (balanced precision-recall).")
                    print(f"This will optimize for detecting the {num_attack_samples} attack samples.")
                
                # Create detector for optimization
                from framework.detector import AnomalyDetector
                temp_detector = AnomalyDetector(
                    temp_model, self.dataset_name, self.model_name, self.time_span, False
                )

                # Run optimization
                # Build output directory following the same pattern as other results
                time_span_label = f"{self.time_span}seconds"
                output_dir = f"./results/{self.dataset_name}/{time_span_label}/models/{self.model_name}/optimization/"
                
                best_params = temp_detector.optimize_hyperparameters(
                    scaled_train, scaled_val, 
                    self.model_name,
                    n_iter=self.optimize_n_iter,
                    scoring=scoring_method,
                    y_val=val_labels,
                    output_dir=output_dir,
                    dataset_name=self.dataset_name,
                    time_span=self.time_span
                )
                
                # If optimize_only flag is set, stop here without training
                if getattr(self, 'optimize_only', False):
                    print("\n" + "="*70)
                    print("OPTIMIZATION COMPLETE - Stopping before training")
                    print("="*70)
                    print(f"\nBest parameters have been saved to:")
                    params_file = f"{output_dir}/parameters/optimal_params.py"
                    print(f"  {params_file}")
                    print("\n" + "="*70)
                    
                    # Return special status for batch mode
                    return {
                        'status': 'optimization_complete',
                        'params_file': params_file,
                        'best_params': best_params,
                        'dataset': self.dataset_name,
                        'model': self.model_name,
                        'time_span': self.time_span
                    }
                
                # Update model manager with optimized parameters
                print(f"\nApplying optimized parameters: {best_params}")
                temp_model_manager = None
        
        # Initialize model manager
        self.model_manager = ModelManager(
            self.model_name, self.dataset_name, self.time_span
        )
        
        # Get or create model
        self.model, self.training_history = self.model_manager.get_or_create_model(
            train_features, self.use_fixed_threshold, self.force_retrain
        )
        
        if self.model is None:
            print("Error: Model initialization failed")
            return False
        
        # If model wasn't loaded from artifacts, need to prepare data and train
        if not self.model_manager.is_model_loaded_from_artifacts():
            # Prepare data for training
            val_features = self.processed_features['validation']['features']
            test_features = self.processed_features['test']['features']
            horizon_features = None
            if 'horizon' in self.processed_features:
                horizon_features = self.processed_features['horizon']['features']
            
            scaled_data = self.model_manager.prepare_data_for_training(
                train_features, val_features, test_features, horizon_features
            )
            
            # Train with prepared data
            input_dim = scaled_data['train'].shape[1]
            if self.training_history is None:  # Model wasn't trained in get_or_create_model
                self.training_history = self.model_manager.train_with_prepared_data(
                    scaled_data['train'], scaled_data['validation'], input_dim
                )
        
        return True
    
    def perform_anomaly_detection(self) -> bool:
        """
        Perform anomaly detection across all data splits
        
        Returns:
            True if successful, False otherwise
        """
        # Initialize anomaly detector
        is_loaded = self.model_manager.is_model_loaded_from_artifacts() if self.model_manager else False
        self.detector = AnomalyDetector(
            self.model, self.dataset_name, self.model_name, self.time_span, is_loaded
        )
        
        # Perform anomaly detection
        self.combined_features, self.threshold, self.all_thresholds = self.detector.detect_anomalies_all_splits(
            self.processed_features, self.features_dict, self.use_fixed_threshold
        )
        
        if self.combined_features is None or len(self.combined_features) == 0:
            print("Error: Anomaly detection failed")
            return False
        
        return True
    
    def perform_feature_analysis(self) -> Tuple[np.ndarray, np.ndarray, List[str], Optional[Dict]]:
        """
        Perform feature importance analysis
        
        Returns:
            Tuple of (feature_errors, importance_indices, feature_names, importance_analysis)
        """
        print("\\nAnalyzing feature importance...")
        
        # Get test data for analysis
        test_features = self.processed_features['test']['features']
        scaled_test_data = self.model.transform_data(test_features)
        feature_names = test_features.columns.tolist()
        
        # Get feature importance (all models use the same method now)
        feature_errors, importance_indices = self.model.analyze_feature_importance(
            scaled_test_data, feature_names
        )
        importance_analysis = None
        
        return feature_errors, importance_indices, feature_names, importance_analysis
    
    def generate_visualizations(self, feature_errors: np.ndarray, importance_indices: np.ndarray,
                               feature_names: List[str], importance_analysis: Optional[Dict]):
        """
        Generate all visualizations
        
        Args:
            feature_errors: Array of feature reconstruction errors
            importance_indices: Array of feature importance indices
            feature_names: List of feature names
            importance_analysis: Optional temporal analysis results
        """
        # Get test data for visualization
        test_features = self.processed_features['test']['features']
        scaled_test_data = self.model.transform_data(test_features)
        test_scores = self.detector.model.predict(scaled_test_data)[1]
        
        # Prepare processing list for reconstruction error plots if needed
        processing_list = None
        if self.generate_reconstruction_error:
            val_features = self.processed_features['validation']['features']
            horizon_features = self.processed_features.get('horizon', {}).get('features', None)
            
            processing_list = [
                ('train', self.model.transform_data(self.processed_features['train']['features']), 
                 self.detector.model.predict(self.model.transform_data(self.processed_features['train']['features']))[1], 
                 self.processed_features['train']),
                ('validation', self.model.transform_data(val_features),
                 self.detector.model.predict(self.model.transform_data(val_features))[1],
                 self.processed_features['validation']),
                ('test', scaled_test_data, test_scores, self.processed_features['test'])
            ]
            
            if horizon_features is not None:
                processing_list.append((
                    'horizon', self.model.transform_data(horizon_features),
                    self.detector.model.predict(self.model.transform_data(horizon_features))[1],
                    self.processed_features['horizon']
                ))
        
        # Generate all visualizations
        self.visualizer.generate_all_visualizations(
            history=self.training_history,
            model=self.model,
            test_data=scaled_test_data,
            feature_names=feature_names,
            feature_errors=feature_errors,
            importance_indices=importance_indices,
            combined_features=self.combined_features,
            threshold=self.threshold,
            test_scores=test_scores,
            all_thresholds=self.all_thresholds,
            processing_list=processing_list,
            features_dict=self.features_dict,
            importance_analysis=importance_analysis,
            generate_reconstruction_error=self.generate_reconstruction_error
        )
    
    def save_results_and_evaluate(self):
        """Save results and perform ground truth evaluation"""
        # Save anomalies to CSV
        print("\\nSaving detected anomalies to CSV...")
        anomalies_csv = self.results_manager.save_anomalies_to_csv(
            self.combined_features, self.threshold
        )
        
        # Print comprehensive summary
        self.results_manager.print_summary_statistics(
            self.combined_features, self.threshold
        )
        
        # Perform ground truth evaluation
        evaluation_metrics = self.detector.evaluate_performance(
            self.combined_features, self.threshold, self.dataset_config
        )
        
        # Save analysis metadata
        analysis_results = {
            'model_info': self.model_manager.get_model_info(),
            'threshold_used': self.threshold,
            'total_samples': len(self.combined_features),
            'total_anomalies': len(self.combined_features[self.combined_features['is_anomaly'] == True]),
            'anomalies_csv_path': anomalies_csv,
            'evaluation_metrics': evaluation_metrics
        }
        
        self.results_manager.save_analysis_metadata(analysis_results)
        
        # Print final completion summary
        self.results_manager.print_analysis_completion_summary(self.dataset_config)
        
        return analysis_results
    
    def run_complete_analysis(self) -> Dict:
        """
        Run the complete DDoS detection analysis pipeline
        
        Returns:
            Dictionary containing analysis results and metadata
        """
        try:
            # Step 1: Initialize components
            if not self.initialize_components():
                raise Exception("Component initialization failed")
            
            # Step 2: Extract and prepare features
            if not self.extract_and_prepare_features():
                raise Exception("Feature extraction failed")
            
            # Step 3: Initialize and train model
            result = self.initialize_and_train_model()
            
            # Check if this was optimization-only mode
            if isinstance(result, dict) and result.get('status') == 'optimization_complete':
                print("\n" + "="*60)
                print("OPTIMIZATION-ONLY MODE: Skipping training and detection")
                print("="*60)
                return result
            
            if not result:
                raise Exception("Model initialization/training failed")
            
            # Step 4: Perform anomaly detection
            if not self.perform_anomaly_detection():
                raise Exception("Anomaly detection failed")
            
            # Step 5: Feature importance analysis
            feature_errors, importance_indices, feature_names, importance_analysis = self.perform_feature_analysis()
            
            # Step 6: Generate visualizations
            self.generate_visualizations(feature_errors, importance_indices, feature_names, importance_analysis)
            
            # Step 7: Save results and evaluate
            analysis_results = self.save_results_and_evaluate()
            
            print("\\n" + "="*60)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("="*60)
            
            # Step 8: Evaluate real-time performance if requested
            if self.evaluate_performance:
                performance_metrics = self.evaluate_realtime_performance()
                analysis_results['performance_metrics'] = performance_metrics
            
            return analysis_results
            
        except ValueError as e:
            error_msg = str(e)
            # Check if it's a feature mismatch error (scaler or model expecting different number of features)
            if ("expecting" in error_msg and "features" in error_msg) or \
               ("n_features" in error_msg) or \
               ("feature" in error_msg and "mismatch" in error_msg.lower()):
                # Delete cached model artifacts and retrain
                if self._handle_feature_mismatch_and_retry():
                    print("\\n" + "="*60)
                    print("Analysis completed after retraining")
                    print("="*60)
                    return self._get_results()
                else:
                    print("\\n" + "="*60)
                    print("Unable to complete analysis after retraining")
                    print("="*60)
                    raise Exception(f"Feature mismatch recovery failed: {error_msg}")
            else:
                print(f"\\nPipeline execution failed: {e}")
                print("="*60)
                raise e
        except Exception as e:
            print(f"\\nPipeline execution failed: {e}")
            print("="*60)
            raise e
    
    def evaluate_realtime_performance(self) -> Dict[str, PerformanceMetrics]:
        """
        Evaluate real-time performance of the trained model
        
        Returns:
            Dictionary containing performance metrics for different test scenarios
        """
        if self.model is None:
            raise ValueError("Model must be trained before performance evaluation")
        
        print("\\n" + "="*60)
        print("EVALUATING REAL-TIME PERFORMANCE")
        print("="*60)
        
        # Get test data for performance evaluation
        test_features = self.processed_features['test']['features']
        
        # Limit test data size if needed
        if len(test_features) > self.performance_samples:
            # Randomly sample test data
            sample_indices = np.random.choice(
                len(test_features), self.performance_samples, replace=False
            )
            test_data = test_features.iloc[sample_indices].values
        else:
            test_data = test_features.values
        
        # Create performance evaluator
        # Use the trained model directly for prediction (it includes both transformation and prediction)
        evaluator = RealTimePerformanceEvaluator(
            model=self.model,  # Use the trained model that has transform_data method
            preprocessor=None  # No separate preprocessor needed
        )
        
        performance_results = {}
        
        # Test 1: Single sample performance
        print("\\nRunning single sample performance test...")
        single_sample_metrics = evaluator.evaluate_single_sample_performance(
            test_data=test_data,
            num_samples=min(1000, len(test_data)),
            model_name=self.model_name
        )
        performance_results['single_sample'] = single_sample_metrics
        PerformanceReporter.print_metrics(single_sample_metrics)
        
        # Test 2: Batch performance (small batches)
        print("\\nRunning batch performance test (batch_size=1)...")
        batch_metrics_1 = evaluator.evaluate_batch_performance(
            test_data=test_data,
            batch_size=1,
            num_iterations=min(500, len(test_data)),
            model_name=self.model_name
        )
        performance_results['batch_size_1'] = batch_metrics_1
        PerformanceReporter.print_metrics(batch_metrics_1)
        
        # Test 3: Batch performance (larger batches)
        print("\\nRunning batch performance test (batch_size=10)...")
        batch_metrics_10 = evaluator.evaluate_batch_performance(
            test_data=test_data,
            batch_size=10,
            num_iterations=min(100, len(test_data) // 10),
            model_name=self.model_name
        )
        performance_results['batch_size_10'] = batch_metrics_10
        PerformanceReporter.print_metrics(batch_metrics_10)
        
        # Test 4: Streaming simulation (if we have enough data)
        if len(test_data) >= 100:
            print("\\nRunning streaming performance test (10 Hz, 30 seconds)...")
            streaming_metrics = evaluator.evaluate_streaming_performance(
                test_data=test_data,
                stream_rate_hz=10,  # 10 samples per second
                duration_seconds=30,
                model_name=self.model_name
            )
            performance_results['streaming_10hz'] = streaming_metrics
            PerformanceReporter.print_metrics(streaming_metrics)
        
        # Save performance metrics to file
        self._save_performance_metrics(performance_results)
        
        # Generate performance visualization charts
        self._generate_performance_charts(performance_results)
        
        # Print summary comparison
        self._print_performance_summary(performance_results)
        
        return performance_results
    
    def _save_performance_metrics(self, performance_results: Dict[str, PerformanceMetrics]) -> None:
        """
        Save performance metrics to CSV file
        
        Args:
            performance_results: Dictionary of performance metrics
        """
        if self.results_manager is None:
            return
        
        results_dir = self.results_manager.get_results_directory()
        performance_dir = os.path.join(results_dir, 'performance')
        os.makedirs(performance_dir, exist_ok=True)
        performance_file = os.path.join(performance_dir, 'metrics.csv')
        
        for test_type, metrics in performance_results.items():
            PerformanceReporter.save_metrics(metrics, performance_file)
        
        print(f"\\nPerformance metrics saved to: {performance_file}")
    
    def _generate_performance_charts(self, performance_results: Dict[str, PerformanceMetrics]) -> None:
        """
        Generate performance visualization charts
        
        Args:
            performance_results: Dictionary of performance metrics
        """
        if self.results_manager is None:
            return
        
        try:
            results_dir = self.results_manager.get_results_directory()
            performance_charts_dir = os.path.join(results_dir, 'performance')
            
            # Create performance visualizer
            visualizer = PerformancePlotter(performance_charts_dir)
            
            # Generate all performance charts
            chart_paths = visualizer.generate_all_charts(performance_results, self.model_name)
            
            print(f"\\nPerformance charts generated:")
            for chart_path in chart_paths:
                print(f"  -> {os.path.basename(chart_path)}")
            print(f"Charts saved to: {performance_charts_dir}")
            
        except Exception as e:
            print(f"Warning: Failed to generate performance charts: {e}")
            logger.warning(f"Performance chart generation failed: {e}")
    
    def _print_performance_summary(self, performance_results: Dict[str, PerformanceMetrics]) -> None:
        """
        Print a summary comparison of all performance tests
        
        Args:
            performance_results: Dictionary of performance metrics
        """
        print("\\n" + "="*80)
        print("PERFORMANCE SUMMARY COMPARISON")
        print("="*80)
        
        # Create comparison table
        metrics_list = list(performance_results.values())
        if metrics_list:
            comparison_df = PerformanceReporter.compare_metrics(metrics_list)
            print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        # Print recommendations
        print("\\n" + "-"*80)
        print("RECOMMENDATIONS:")
        
        best_throughput = max(performance_results.items(), 
                             key=lambda x: x[1].throughput_samples_per_sec)
        best_latency = min(performance_results.items(), 
                          key=lambda x: x[1].avg_latency_ms)
        
        print(f"-> Best throughput: {best_throughput[0]} ({best_throughput[1].throughput_samples_per_sec:.2f} samples/sec)")
        print(f"-> Best latency: {best_latency[0]} ({best_latency[1].avg_latency_ms:.3f} ms)")
        
        # Real-time capability assessment
        for test_name, metrics in performance_results.items():
            if metrics.throughput_samples_per_sec >= 100:  # Can handle 100+ samples/sec
                print(f"-> {test_name}: Suitable for high-frequency real-time detection")
            elif metrics.throughput_samples_per_sec >= 10:  # Can handle 10+ samples/sec
                print(f"-> {test_name}: Suitable for medium-frequency real-time detection")
            else:
                print(f"-> {test_name}: May not be suitable for real-time detection")
        
        print("="*80)
    
    def get_results_summary(self) -> Dict:
        """
        Get a summary of analysis results
        
        Returns:
            Dictionary containing results summary
        """
        if self.combined_features is None:
            return {'status': 'analysis_not_completed'}
        
        summary = {
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'time_span': self.time_span,
            'threshold_used': self.threshold,
            'total_samples': len(self.combined_features),
            'total_anomalies': len(self.combined_features[self.combined_features['is_anomaly'] == True]),
            'results_directory': self.results_manager.get_results_directory() if self.results_manager else None,
            'model_loaded_from_artifacts': self.model_manager.is_model_loaded_from_artifacts() if self.model_manager else False
        }
        
        # Add per-dataset breakdown
        summary['per_dataset_stats'] = {}
        for dataset in ['train', 'validation', 'test', 'horizon']:
            subset = self.combined_features[self.combined_features['dataset'] == dataset]
            if len(subset) > 0:
                subset_anomalies = len(subset[subset['is_anomaly'] == True])
                summary['per_dataset_stats'][dataset] = {
                    'total_samples': len(subset),
                    'anomalies': subset_anomalies,
                    'anomaly_rate': (subset_anomalies / len(subset)) * 100 if len(subset) > 0 else 0
                }
        
        return summary

    def _handle_feature_mismatch_and_retry(self) -> bool:
        """
        Handle feature mismatch error by deleting cached model and retraining
        
        Returns:
            True if retry was successful, False otherwise
        """
        import shutil
        
        try:
            # Get the model artifacts directory
            if self.results_manager:
                results_dir = self.results_manager.get_results_directory()
                artifacts_dir = os.path.join(results_dir, 'artifacts')
                
                # Delete the artifacts directory if it exists
                if os.path.exists(artifacts_dir):
                    print(f"\\nDeleting cached model artifacts from: {artifacts_dir}")
                    shutil.rmtree(artifacts_dir)
                    print("Cached model artifacts deleted")
                else:
                    print(f"\\nNo artifacts directory found at: {artifacts_dir}")
            
            # Force retrain flag
            original_force_retrain = self.force_retrain
            self.force_retrain = True
            
            print("\\nRetraining model with current feature configuration...")
            print("-" * 60)
            
            # Reinitialize and train model (this will force a fresh training)
            self.model = None
            self.model_manager = None
            self.training_history = None
            
            if not self.initialize_and_train_model():
                print("Model retraining failed")
                self.force_retrain = original_force_retrain
                return False
            
            print("Model retrained successfully")
            
            # Now try to perform anomaly detection again
            print("\\nRetrying anomaly detection...")
            if not self.perform_anomaly_detection():
                print("Anomaly detection failed after retraining")
                self.force_retrain = original_force_retrain
                return False
            
            print("Anomaly detection successful")
            
            # Continue with feature analysis
            print("\\nPerforming feature analysis...")
            feature_errors, importance_indices, feature_names, importance_analysis = self.perform_feature_analysis()
            print("Feature analysis completed")
            
            # Generate visualizations
            print("\\nGenerating visualizations...")
            self.generate_visualizations(feature_errors, importance_indices, feature_names, importance_analysis)
            print("Visualizations generated")
            
            # Save results and evaluate
            print("\\nSaving results and evaluating...")
            analysis_results = self.save_results_and_evaluate()
            print("Results saved and evaluated")
            
            # Store results for retrieval
            self._recovery_results = analysis_results
            
            # Restore original force_retrain setting
            self.force_retrain = original_force_retrain
            
            return True
            
        except Exception as e:
            print(f"\\nRecovery attempt failed: {e}")
            logger.error(f"Feature mismatch recovery failed: {e}", exc_info=True)
            return False
    
    def _get_results(self) -> Dict:
        """
        Get the analysis results (used after recovery)
        
        Returns:
            Dictionary containing analysis results
        """
        if hasattr(self, '_recovery_results'):
            results = self._recovery_results
        else:
            # Build results from current state
            results = {
                'model_info': self.model_manager.get_model_info() if self.model_manager else {},
                'threshold_used': self.threshold,
                'total_samples': len(self.combined_features) if self.combined_features is not None else 0,
                'total_anomalies': len(self.combined_features[self.combined_features['is_anomaly'] == True]) if self.combined_features is not None else 0,
            }
        
        # Add performance metrics if requested
        if self.evaluate_performance:
            try:
                performance_metrics = self.evaluate_realtime_performance()
                results['performance_metrics'] = performance_metrics
            except Exception as e:
                print(f"Warning: Could not evaluate performance: {e}")
        
        return results
