"""
Model lifecycle management for DDoS detection pipeline.
Handles model creation, loading from artifacts, training, and saving.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any

from framework.models import create_model
from framework.models.core.artifacts import artifacts_exist, get_artifacts_info, load_model_artifacts
from framework.models.core.factory import create_model_with_config, load_model_from_artifacts
from framework.visualization.training_plots import TrainingVisualizer


class ModelManager:
    """Handles model lifecycle: loading, training, saving artifacts"""
    
    def __init__(self, model_name: str, dataset_name: str, time_span: int):
        """
        Initialize ModelManager
        
        Args:
            model_name: Name of the model to manage
            dataset_name: Name of the dataset
            time_span: Time span in seconds
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.time_span = time_span
        self.model = None
        self.is_loaded_from_artifacts = False
    
    def get_or_create_model(self, train_features: pd.DataFrame, 
                           use_fixed_threshold: bool = False,
                           force_retrain: bool = False) -> Tuple[Any, Optional[Any]]:
        """
        Get existing model from artifacts or create and train new one
        
        Args:
            train_features: Training features for model creation/training
            use_fixed_threshold: Whether to use fixed threshold for model
            force_retrain: Whether to force retraining even if artifacts exist
            
        Returns:
            Tuple of (model, training_history)
        """
        print(f"\nInitializing {self.model_name} model...")
        
        # Create model with appropriate parameters (including dataset-specific overrides)
        self.model = create_model_with_config(
            self.model_name, self.time_span, train_features, use_fixed_threshold, self.dataset_name
        )
        
        # Try to load existing model artifacts
        training_history = None
        if self._try_load_existing_model(force_retrain, use_fixed_threshold):
            print("Model loaded from artifacts - skipping training phase")
            training_history = None  # No training history for loaded models
        else:
            # Train new model
            training_history = self._train_new_model(train_features)
        
        return self.model, training_history
    
    def _try_load_existing_model(self, force_retrain: bool, use_fixed_threshold: bool) -> bool:
        """
        Try to load existing model from artifacts
        
        Args:
            force_retrain: Whether to force retraining
            use_fixed_threshold: Whether to use fixed threshold
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        if force_retrain:
            print("\nForce retrain enabled - training new model regardless of existing artifacts")
            return False
        
        if not artifacts_exist(self.dataset_name, self.model_name, self.time_span):
            print("\nNo existing model artifacts found - training new model")
            return False
        
        print(f"\n{'='*60}")
        print("EXISTING MODEL ARTIFACTS FOUND")
        print(f"{'='*60}")
        
        artifacts_info = get_artifacts_info(self.dataset_name, self.model_name, self.time_span)
        if not artifacts_info:
            print("Could not retrieve artifacts info - training new model")
            return False
        
        self._print_artifacts_info(artifacts_info)
        
        try:
            print("\nLoading saved model artifacts...")
            
            # Load the model with its saved configuration using the factory function
            loaded_model = load_model_from_artifacts(
                self.model, self.model_name, self.dataset_name, 
                self.time_span, artifacts_info, use_fixed_threshold
            )
            
            # Replace the created model with the loaded one
            self.model = loaded_model
            self.is_loaded_from_artifacts = True
            
            print("Model artifacts loaded successfully!")
            print(f"Model is trained: {self.model.is_trained}")
            print(f"Model threshold: {self.model.threshold}")
            print("Skipping training phase...\n")
            
            return True
            
        except Exception as e:
            print(f"Failed to load model artifacts: {e}")
            print("Falling back to training a new model...\n")
            self.is_loaded_from_artifacts = False
            return False
    
    def _print_artifacts_info(self, artifacts_info: Dict):
        """
        Print information about existing artifacts
        
        Args:
            artifacts_info: Dictionary containing artifacts information
        """
        print(f"Artifacts saved at: {artifacts_info.get('save_timestamp', 'unknown')}")
        print(f"Model type: {artifacts_info.get('model_type', 'unknown')}")
        print(f"Threshold: {artifacts_info.get('threshold', 'unknown')}")
        print(f"Artifacts directory: {artifacts_info['artifacts_dir']}")
    
    def _train_new_model(self, train_features: pd.DataFrame) -> Any:
        """
        Train a new model from scratch
        
        Args:
            train_features: Training features DataFrame
            
        Returns:
            Training history object
        """
        print("\nPreparing data for training...")
        
        # Fit scaler and transform data
        self.model.fit_scaler(train_features)
        
        # Prepare validation features (assuming they exist in processed_features)
        # This is a simplified approach - in practice, you'd pass val_features separately
        scaled_train_data = self.model.transform_data(train_features)
        
        # For this implementation, we'll use a portion of training data as validation
        # In the actual pipeline, validation data should be passed separately
        split_idx = int(0.8 * len(scaled_train_data))
        scaled_validation_data = scaled_train_data[split_idx:]
        scaled_train_data = scaled_train_data[:split_idx]
        
        input_dim = scaled_train_data.shape[1]
        print(f"\nNumber of input features: {input_dim}")
        
        # Build model with feature dimension
        self.model.build_model(input_dim)
        
        print("\nTraining model...")
        history = self.model.train(scaled_train_data, scaled_validation_data)
        
        # Save model artifacts after successful training
        self._save_model_artifacts(history)
        
        return history
    
    def train_with_prepared_data(self, scaled_train_data: np.ndarray, 
                                scaled_validation_data: np.ndarray,
                                input_dim: int) -> Any:
        """
        Train model with already prepared and scaled data
        
        Args:
            scaled_train_data: Prepared training data
            scaled_validation_data: Prepared validation data
            input_dim: Input dimension for model building
            
        Returns:
            Training history object
        """
        print(f"\nNumber of input features: {input_dim}")
        
        # Build model with feature dimension
        self.model.build_model(input_dim)
        
        print("\nTraining model...")
        history = self.model.train(scaled_train_data, scaled_validation_data)
        
        # Save model artifacts after successful training
        self._save_model_artifacts(history)
        
        return history
    
    def _save_model_artifacts(self, history: Any):
        """
        Save model artifacts after training
        
        Args:
            history: Training history object
        """
        print(f"\nSaving model artifacts...")
        try:
            artifacts_path = self.model.save_artifacts(
                self.dataset_name, 
                self.time_span, 
                history.history if hasattr(history, 'history') else history
            )
            print(f"Model artifacts saved to: {artifacts_path}")
        except Exception as e:
            print(f"Failed to save model artifacts: {e}")
    
    def prepare_data_for_training(self, train_features: pd.DataFrame, 
                                 val_features: pd.DataFrame, 
                                 test_features: pd.DataFrame,
                                 horizon_features: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Prepare all data splits for training and inference
        
        Args:
            train_features: Training features DataFrame
            val_features: Validation features DataFrame  
            test_features: Test features DataFrame
            horizon_features: Optional horizon features DataFrame
            
        Returns:
            Dictionary containing scaled data for each split
        """
        # Only fit scaler if model wasn't loaded from artifacts
        if not self.is_loaded_from_artifacts:
            self.model.fit_scaler(train_features)
        
        # Transform all data splits
        scaled_data = {
            'train': self.model.transform_data(train_features),
            'validation': self.model.transform_data(val_features),
            'test': self.model.transform_data(test_features)
        }
        
        if horizon_features is not None:
            scaled_data['horizon'] = self.model.transform_data(horizon_features)
        
        return scaled_data
    
    def get_model(self) -> Any:
        """Get the managed model instance"""
        return self.model
    
    def is_model_loaded_from_artifacts(self) -> bool:
        """Check if model was loaded from artifacts"""
        return self.is_loaded_from_artifacts
    
    def get_model_info(self) -> Dict:
        """
        Get information about the managed model
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {'status': 'not_initialized'}
        
        info = {
            'model_name': self.model_name,
            'dataset_name': self.dataset_name,
            'time_span': self.time_span,
            'is_trained': getattr(self.model, 'is_trained', False),
            'threshold': getattr(self.model, 'threshold', None),
            'loaded_from_artifacts': self.is_loaded_from_artifacts
        }
        
        # Add model-specific information
        if hasattr(self.model, 'get_model_summary'):
            try:
                summary = self.model.get_model_summary()
                info.update(summary)
            except:
                pass
        
        return info
