#!/usr/bin/env python3
"""
Main DDoS Detection System
"""

# Configure TensorFlow logging at the very beginning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
os.environ['TF_ENABLE_XLA'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import glob
import pickle
import numpy as np
import pandas as pd
import hashlib
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').setLevel(logging.ERROR)

from tensorflow.keras.models import load_model

from config import (
    INPUT_DIR, ARTEFACTS_DIR, AGG_PERIOD, TRAIN_SPLIT, 
    ATTACK_PERIODS, EPOCHS, REQUIRED_MODEL_FILES,
    THRESHOLD_PERCENTILES, FLOW_MULTIPLIER_THRESHOLD, THRESHOLD_METHOD
)
from feature_engineering import AdvancedFeatureEngineer
from models import StandardAutoencoder, LSTMAutoencoder
from utils import configure_gpu_for_training, configure_cpu_for_inference


class DDoSDetectionSystem:
    """Main DDoS detection system"""
    
    def __init__(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.flow_thresholds = {}
        
        # Set threshold method from config
        self.threshold_method = THRESHOLD_METHOD
        
        self.cache_dir = os.path.join(ARTEFACTS_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_enabled = True
        
    def _get_data_hash(self):
        """Generate hash of all CSV files to detect changes"""
        csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
        if not csv_files:
            return None
            
        hash_input = ""
        for file in csv_files:
            stat = os.stat(file)
            hash_input += f"{file}:{stat.st_mtime}:{stat.st_size}\n"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cache_path(self, data_hash):
        """Get cache file path for processed data"""
        return os.path.join(self.cache_dir, f"processed_data_{data_hash}.pkl")
    
    def _load_from_cache(self, data_hash):
        """Load processed data from cache if available"""
        if not self.cache_enabled:
            return None
            
        cache_path = self._get_cache_path(data_hash)
        if os.path.exists(cache_path):
            try:
                print(f"Loading processed data from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Cache loaded successfully! Data shape: {cached_data.shape}")
                return cached_data
            except Exception as e:
                print(f"Cache loading failed: {e}")
                return None
        return None
    
    def _save_to_cache(self, data_hash, features_df):
        """Save processed data to cache"""
        if not self.cache_enabled:
            return
            
        cache_path = self._get_cache_path(data_hash)
        try:
            print(f"Saving processed data to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(features_df, f)
            print("Cache saved successfully!")
        except Exception as e:
            print(f"Cache saving failed: {e}")
    
    def _cleanup_old_cache(self, keep_recent=3):
        """Clean up old cache files, keeping only the most recent ones"""
        cache_files = glob.glob(os.path.join(self.cache_dir, "processed_data_*.pkl"))
        if len(cache_files) > keep_recent:
            # Sort by modification time and keep only the most recent
            cache_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            for old_cache in cache_files[keep_recent:]:
                try:
                    os.remove(old_cache)
                    print(f"Removed old cache: {old_cache}")
                except Exception as e:
                    print(f"Failed to remove old cache {old_cache}: {e}")
    
    def load_and_process_data(self, force_reload=False):
        """Load and process all netflow data with intelligent caching"""
        print("Starting data loading and processing...")
        start_time = time.time()
        
        # Generate data hash to detect changes
        data_hash = self._get_data_hash()
        if data_hash is None:
            raise ValueError("No CSV files found in input directory")
        
        # Try to load from cache first (unless force reload is requested)
        if not force_reload:
            cached_data = self._load_from_cache(data_hash)
            if cached_data is not None:
                self.features_df = cached_data
                elapsed = time.time() - start_time
                return cached_data
        
        # If not in cache or force reload, process the data
        print("Processing data from scratch...")
        
        # Get all CSV files
        csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
        print(f"Found {len(csv_files)} CSV files to process")
        
        # Process files in batches for better memory management
        batch_size = 50  # Process 50 files at a time
        all_dfs = []
        
        for i in range(0, len(csv_files), batch_size):
            batch_files = csv_files[i:i + batch_size]
            batch_dfs = []
            
            print(f"Processing batch {i//batch_size + 1}/{(len(csv_files) + batch_size - 1)//batch_size} "
                  f"({len(batch_files)} files)...")
            
            for file in batch_files:
                try:
                    df = pd.read_csv(file, parse_dates=['firstSeen'])
                    batch_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
                    continue
            
            if batch_dfs:
                batch_df = pd.concat(batch_dfs, ignore_index=True)
                all_dfs.append(batch_df)
                print(f"Batch processed: {len(batch_df)} records")
        
        if not all_dfs:
            raise ValueError("No CSV files could be loaded")
        
        # Combine all batches
        print("Combining all data batches...")
        df = pd.concat(all_dfs, ignore_index=True)
        print(f"Total records loaded: {len(df):,}")
        
        # Add time binning
        print("Adding time binning...")
        df['ts_bin'] = df['firstSeen'].dt.floor(AGG_PERIOD)
        
        # Extract features
        print("Extracting features...")
        features_df = self.feature_engineer.extract_features(df)
        print(f"Features extracted: {features_df.shape}")
        
        # Store the full dataset for visualization purposes
        self.features_df = features_df
        
        # Save to cache
        self._save_to_cache(data_hash, features_df)
        
        # Cleanup old cache files
        self._cleanup_old_cache()
        
        elapsed = time.time() - start_time
        print(f"Data processing completed in {elapsed:.2f} seconds")
        
        return features_df
    
    def load_and_process_data_optimized(self, force_reload=False):
        """Alternative optimized version using parallel processing for very large datasets"""
        print("Starting optimized data loading and processing...")
        start_time = time.time()
        
        # Generate data hash to detect changes
        data_hash = self._get_data_hash()
        if data_hash is None:
            raise ValueError("No CSV files found in input directory")
        
        # Try to load from cache first
        if not force_reload:
            cached_data = self._load_from_cache(data_hash)
            if cached_data is not None:
                self.features_df = cached_data
                elapsed = time.time() - start_time
                return cached_data
        
        # For very large datasets, use chunked processing
        csv_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.csv")))
        print(f"Found {len(csv_files)} CSV files to process")
        
        # Process in smaller chunks for memory efficiency
        chunk_size = 20  # Process 20 files at a time
        all_features = []
        
        for i in range(0, len(csv_files), chunk_size):
            chunk_files = csv_files[i:i + chunk_size]
            print(f"Processing chunk {i//chunk_size + 1}/{(len(csv_files) + chunk_size - 1)//chunk_size} "
                  f"({len(chunk_files)} files)...")
            
            chunk_dfs = []
            for file in chunk_files:
                try:
                    df = pd.read_csv(file, parse_dates=['firstSeen'])
                    df['ts_bin'] = df['firstSeen'].dt.floor(AGG_PERIOD)
                    chunk_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not load {file}: {e}")
                    continue
            
            if chunk_dfs:
                chunk_df = pd.concat(chunk_dfs, ignore_index=True)
                chunk_features = self.feature_engineer.extract_features(chunk_df)
                all_features.append(chunk_features)
                print(f"Chunk processed: {len(chunk_features)} time windows")
        
        if not all_features:
            raise ValueError("No data could be processed")
        
        # Combine all feature dataframes
        print("Combining all feature data...")
        features_df = pd.concat(all_features, ignore_index=True)
        
        # Remove duplicates that might occur at chunk boundaries
        features_df = features_df.drop_duplicates(subset=['ts_bin']).reset_index(drop=True)
        
        print(f"Final features shape: {features_df.shape}")
        
        # Store the full dataset
        self.features_df = features_df
        
        # Save to cache
        self._save_to_cache(data_hash, features_df)
        
        # Cleanup old cache files
        self._cleanup_old_cache()
        
        elapsed = time.time() - start_time
        print(f"Optimized data processing completed in {elapsed:.2f} seconds")
        
        return features_df
    
    def clear_cache(self):
        """Clear all cached data"""
        cache_files = glob.glob(os.path.join(self.cache_dir, "*.pkl"))
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                print(f"Removed cache file: {cache_file}")
            except Exception as e:
                print(f"Failed to remove cache file {cache_file}: {e}")
        print("Cache cleared successfully!")
    
    def get_cache_info(self):
        """Get information about cached data"""
        cache_files = glob.glob(os.path.join(self.cache_dir, "processed_data_*.pkl"))
        if not cache_files:
            print("No cached data found")
            return
        
        print(f"Found {len(cache_files)} cached data files:")
        for cache_file in sorted(cache_files, key=lambda x: os.path.getmtime(x), reverse=True):
            size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
            mtime = time.ctime(os.path.getmtime(cache_file))
            print(f"  {os.path.basename(cache_file)}: {size:.1f}MB, modified: {mtime}")
    
    def split_data_enhanced(self, features_df):
        """Enhanced data splitting with multiple attack types and better period handling"""
        print("Splitting data with enhanced attack period handling...")
        
        # Training data (normal traffic before attacks)
        train_data = features_df[features_df['ts_bin'] < "2025-07-16"].copy()
        
        # Attack data
        attack_info = {}
        all_attack_data = []
        
        for attack_period in ATTACK_PERIODS:
            attack_name = attack_period['name']
            attack_data = features_df[
                (features_df['ts_bin'] >= attack_period['start']) & 
                (features_df['ts_bin'] <= attack_period['end'])
            ].copy()
            
            if len(attack_data) > 0:
                attack_data['attack_type'] = attack_period['type']
                attack_info[attack_name] = attack_data
                all_attack_data.append(attack_data)
        
        if all_attack_data:
            combined_attack_data = pd.concat(all_attack_data, ignore_index=True)
        else:
            combined_attack_data = pd.DataFrame()
        
        # Test data (normal traffic after attacks)
        test_data = features_df[features_df['ts_bin'] >= "2025-07-17"].copy()
        
        # Also get normal data from July 16 (excluding attack periods) for additional testing
        additional_normal_data = features_df[features_df['ts_bin'].dt.date == pd.Timestamp('2025-07-16').date()].copy()
        
        # Remove attack periods from additional normal data
        for attack_period in ATTACK_PERIODS:
            additional_normal_data = additional_normal_data[
                ~((additional_normal_data['ts_bin'] >= attack_period['start']) & 
                  (additional_normal_data['ts_bin'] <= attack_period['end']))
            ]
        
        return train_data, combined_attack_data, test_data, attack_info, additional_normal_data
    

    
    def prepare_features(self, data, fit_scaler=True):
        """Prepare features for training"""
        return self.feature_engineer.prepare_features(data, fit_scaler)
    
    def train_models_enhanced(self, train_data, attack_data):
        """Enhanced training with multiple attack types and better thresholds"""
        print("Starting enhanced model training with GPU acceleration...")
        
        # Configure GPU for training
        gpu_available = configure_gpu_for_training()
        
        X_train, feature_cols = self.prepare_features(train_data, fit_scaler=True)
        X_attack, _ = self.prepare_features(attack_data, fit_scaler=False)
        
        X_train_split, X_val_split = train_test_split(X_train, test_size=0.2, random_state=42)
        
        print(f"Training data shape: {X_train_split.shape}")
        print(f"Validation data shape: {X_val_split.shape}")
        print(f"Attack data shape: {X_attack.shape}")
        
        # Standard Autoencoder
        print("Training Standard Autoencoder...")
        ae = StandardAutoencoder(input_dim=len(feature_cols))
        ae.train(X_train_split, X_val_split, epochs=EPOCHS)
        self.models['standard_ae'] = ae
        
        # LSTM Autoencoder
        print("Training LSTM Autoencoder...")
        try:
            lstm_ae = LSTMAutoencoder(sequence_length=10, n_features=len(feature_cols))
            lstm_ae.train(X_train_split, X_val_split, epochs=EPOCHS)
            self.models['lstm_ae'] = lstm_ae
        except Exception as e:
            print(f"LSTM training failed: {e}. Continuing without LSTM model.")
        
        # Calculate thresholds based on selected method
        threshold_method = self._get_threshold_method()
        if threshold_method == 'statistical':
            self._calculate_thresholds_statistical(X_train_split, train_data)
        elif threshold_method == 'percentile':
            self._calculate_thresholds_percentile(X_train_split, train_data)
        else:  # enhanced (default)
            self._calculate_enhanced_thresholds(X_train_split, train_data)
        
        # Switch back to CPU for inference
        configure_cpu_for_inference()
        print("Training completed, switched to CPU for inference")
    
    def train_models(self, train_data, attack_data):
        """Legacy train_models method - now calls enhanced version"""
        return self.train_models_enhanced(train_data, attack_data)
    
    def _calculate_enhanced_thresholds(self, X_train, train_data):
        """Calculate enhanced thresholds including flow-based detection"""
        print("Calculating enhanced detection thresholds...")
        
        # Calculate baseline flow statistics
        baseline_flows = train_data['flows_sum'].median()
        flow_std = train_data['flows_sum'].std()
        
        # Initialize storage for training errors if not exists
        if not hasattr(self, '_training_errors'):
            self._training_errors = {}
        
        # Standard thresholds
        for name, model in self.models.items():
            errors = model.predict(X_train)
            
            # Store training errors for later threshold adjustments
            self._training_errors[name] = errors
            
            # Get model-specific threshold percentiles
            model_percentiles = THRESHOLD_PERCENTILES.get(name, {'standard': 98, 'flow': 95})
            standard_percentile = model_percentiles['standard']
            flow_percentile = model_percentiles['flow']
            
            # Calculate statistical thresholds
            mse = np.mean(errors ** 2)
            mae = np.mean(np.abs(errors))
            std_dev = np.std(errors)
            
            # Statistical threshold: MSE + stddev
            statistical_threshold = mse + std_dev
            
            # Percentile-based threshold (current approach)
            percentile_threshold = np.percentile(errors, standard_percentile)
            
            # Use statistical threshold instead of percentile
            threshold = statistical_threshold
            
            # More sensitive threshold for flow-based detection
            flow_threshold = np.percentile(errors, flow_percentile)
            self.flow_thresholds[name] = flow_threshold
            
            self.thresholds[name] = threshold
            
            # Store statistical metrics for reference
            if not hasattr(self, '_statistical_metrics'):
                self._statistical_metrics = {}
            self._statistical_metrics[name] = {
                'mse': mse,
                'mae': mae,
                'std_dev': std_dev,
                'mse_plus_std': statistical_threshold,
                'percentile_threshold': percentile_threshold
            }
        
        # Store baseline statistics for flow-based detection
        self.baseline_flows = baseline_flows
        self.flow_multiplier_threshold = FLOW_MULTIPLIER_THRESHOLD
        
        print(f"Baseline flows: {baseline_flows:.0f}")
        print(f"Flow multiplier threshold: {FLOW_MULTIPLIER_THRESHOLD}x")
        print("Standard thresholds:", {k: f"{v:.4f}" for k, v in self.thresholds.items()})
        print("Flow-sensitive thresholds:", {k: f"{v:.4f}" for k, v in self.flow_thresholds.items()})
        
        # Print statistical threshold information
        if hasattr(self, '_statistical_metrics'):
            print("\nStatistical threshold breakdown:")
            for model_name, metrics in self._statistical_metrics.items():
                print(f"  {model_name}:")
                print(f"    MSE: {metrics['mse']:.5f}")
                print(f"    MAE: {metrics['mae']:.5f}")
                print(f"    StdDev: {metrics['std_dev']:.5f}")
                print(f"    MSE + StdDev (threshold): {metrics['mse_plus_std']:.5f}")
                print(f"    Percentile threshold: {metrics['percentile_threshold']:.5f}")
    
    def _calculate_thresholds_statistical(self, X_train, train_data):
        """Calculate thresholds using statistical approach (MSE + stddev)"""
        print("Calculating statistical thresholds (MSE + stddev)...")
        
        # Calculate baseline flow statistics
        baseline_flows = train_data['flows_sum'].median()
        flow_std = train_data['flows_sum'].std()
        
        # Initialize storage for training errors if not exists
        if not hasattr(self, '_training_errors'):
            self._training_errors = {}
        
        # Statistical thresholds
        for name, model in self.models.items():
            errors = model.predict(X_train)
            
            # Store training errors for later threshold adjustments
            self._training_errors[name] = errors
            
            # Calculate statistical measures
            mse = np.mean(errors ** 2)
            mae = np.mean(np.abs(errors))
            std_dev = np.std(errors)
            
            # Statistical threshold: MSE + stddev
            threshold = mse + std_dev
            
            # Flow threshold: MAE + stddev (more sensitive)
            flow_threshold = mae + std_dev
            
            self.thresholds[name] = threshold
            self.flow_thresholds[name] = flow_threshold
            
            # Store statistical metrics for reference
            if not hasattr(self, '_statistical_metrics'):
                self._statistical_metrics = {}
            self._statistical_metrics[name] = {
                'mse': mse,
                'mae': mae,
                'std_dev': std_dev,
                'mse_plus_std': threshold,
                'mae_plus_std': flow_threshold
            }
        
        # Store baseline statistics for flow-based detection
        self.baseline_flows = baseline_flows
        self.flow_multiplier_threshold = FLOW_MULTIPLIER_THRESHOLD
        
        print(f"Baseline flows: {baseline_flows:.0f}")
        print(f"Flow multiplier threshold: {FLOW_MULTIPLIER_THRESHOLD}x")
        print("Statistical thresholds (MSE + stddev):", {k: f"{v:.4f}" for k, v in self.thresholds.items()})
        print("Flow thresholds (MAE + stddev):", {k: f"{v:.4f}" for k, v in self.flow_thresholds.items()})
        
        # Print statistical threshold information
        print("\nStatistical threshold breakdown:")
        for model_name, metrics in self._statistical_metrics.items():
            print(f"  {model_name}:")
            print(f"    MSE: {metrics['mse']:.5f}")
            print(f"    MAE: {metrics['mae']:.5f}")
            print(f"    StdDev: {metrics['std_dev']:.5f}")
            print(f"    MSE + StdDev (threshold): {metrics['mse_plus_std']:.5f}")
            print(f"    MAE + StdDev (flow threshold): {metrics['mae_plus_std']:.5f}")
    
    def _calculate_thresholds_percentile(self, X_train, train_data):
        """Calculate thresholds using percentile-based approach (original method)"""
        print("Calculating percentile-based thresholds...")
        
        # Calculate baseline flow statistics
        baseline_flows = train_data['flows_sum'].median()
        flow_std = train_data['flows_sum'].std()
        
        # Initialize storage for training errors if not exists
        if not hasattr(self, '_training_errors'):
            self._training_errors = {}
        
        # Percentile-based thresholds
        for name, model in self.models.items():
            errors = model.predict(X_train)
            
            # Store training errors for later threshold adjustments
            self._training_errors[name] = errors
            
            # Get model-specific threshold percentiles
            model_percentiles = THRESHOLD_PERCENTILES.get(name, {'standard': 98, 'flow': 95})
            standard_percentile = model_percentiles['standard']
            flow_percentile = model_percentiles['flow']
            
            # Percentile-based threshold
            threshold = np.percentile(errors, standard_percentile)
            
            # More sensitive threshold for flow-based detection
            flow_threshold = np.percentile(errors, flow_percentile)
            self.flow_thresholds[name] = flow_threshold
            
            self.thresholds[name] = threshold
            
            # Store statistical metrics for reference
            if not hasattr(self, '_statistical_metrics'):
                self._statistical_metrics = {}
            mse = np.mean(errors ** 2)
            mae = np.mean(np.abs(errors))
            std_dev = np.std(errors)
            self._statistical_metrics[name] = {
                'mse': mse,
                'mae': mae,
                'std_dev': std_dev,
                'mse_plus_std': mse + std_dev,
                'percentile_threshold': threshold
            }
        
        # Store baseline statistics for flow-based detection
        self.baseline_flows = baseline_flows
        self.flow_multiplier_threshold = FLOW_MULTIPLIER_THRESHOLD
        
        print(f"Baseline flows: {baseline_flows:.0f}")
        print(f"Flow multiplier threshold: {FLOW_MULTIPLIER_THRESHOLD}x")
        print("Percentile thresholds:", {k: f"{v:.4f}" for k, v in self.thresholds.items()})
        print("Flow-sensitive thresholds:", {k: f"{v:.4f}" for k, v in self.flow_thresholds.items()})
        
        # Print statistical threshold information
        print("\nThreshold comparison:")
        for model_name, metrics in self._statistical_metrics.items():
            print(f"  {model_name}:")
            print(f"    MSE: {metrics['mse']:.5f}")
            print(f"    MAE: {metrics['mae']:.5f}")
            print(f"    StdDev: {metrics['std_dev']:.5f}")
            print(f"    MSE + StdDev: {metrics['mse_plus_std']:.5f}")
            print(f"    Percentile threshold: {metrics['percentile_threshold']:.5f}")
            print(f"    Difference: {metrics['percentile_threshold'] - metrics['mse_plus_std']:.5f}")
    
    def prepare_data_for_prediction(self, data):
        """Prepare data for prediction with models - returns normalized features"""
        X, _ = self.prepare_features(data, fit_scaler=False)
        return X
        
    def detect_anomalies_enhanced(self, data, model_name='standard_ae', use_flow_threshold=False):
        """Enhanced anomaly detection with flow-based sensitivity - automatically uses MSE + StdDev threshold"""
        X, _ = self.prepare_features(data, fit_scaler=False)
        model = self.models[model_name]
        
        errors = model.predict(X)
        
        # Automatically use MSE + StdDev as threshold for evaluation
        threshold = self._get_mse_plus_stddev_threshold(model_name, use_flow_threshold)
        
        # Display threshold info on first call or when debug mode is enabled
        if not hasattr(self, '_threshold_info_displayed') or getattr(self, 'debug_mode', False):
            threshold_type = "MAE + StdDev (flow)" if use_flow_threshold else "MSE + StdDev"
            print(f"Using {threshold_type} threshold for {model_name}: {threshold:.6f}")
            if not hasattr(self, '_threshold_info_displayed'):
                self._threshold_info_displayed = True
        
        anomalies = errors > threshold
        
        # Additional flow-based detection
        if hasattr(self, 'baseline_flows') and 'flows_sum' in data.columns:
            flow_anomalies = data['flows_sum'] > (self.baseline_flows * self.flow_multiplier_threshold)
            
            # Handle size mismatch for LSTM models (sequences reduce output size)
            if len(anomalies) != len(flow_anomalies):
                if model_name == 'lstm_ae' and len(anomalies) < len(flow_anomalies):
                    # LSTM produces fewer predictions due to sequence creation
                    # Align flow_anomalies with LSTM predictions (skip first sequence_length-1 samples)
                    sequence_length = getattr(self.models[model_name], 'sequence_length', 10)
                    flow_anomalies = flow_anomalies.iloc[sequence_length-1:]
                elif len(flow_anomalies) < len(anomalies):
                    # Truncate anomalies to match flow_anomalies
                    anomalies = anomalies[:len(flow_anomalies)]
            
            # Combine with model-based detection (OR logic)
            anomalies = anomalies | flow_anomalies.values
        
        return anomalies, errors
    
    def detect_anomalies(self, data, model_name='standard_ae'):
        """Legacy detect_anomalies method - now calls enhanced version"""
        return self.detect_anomalies_enhanced(data, model_name, use_flow_threshold=False)
    
    def adjust_threshold(self, model_name, new_threshold, threshold_type='standard'):
        """
        Manually adjust threshold for a specific model
        
        Args:
            model_name (str): Name of the model ('standard_ae' or 'lstm_ae')
            new_threshold (float): New threshold value
            threshold_type (str): Type of threshold ('standard', 'flow', or 'both')
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return False
        
        if threshold_type in ['standard', 'both']:
            if model_name in self.thresholds:
                old_threshold = self.thresholds[model_name]
                self.thresholds[model_name] = new_threshold
                print(f"Standard threshold for {model_name} adjusted from {old_threshold:.5f} to {new_threshold:.5f}")
            else:
                self.thresholds[model_name] = new_threshold
                print(f"Standard threshold for {model_name} set to {new_threshold:.5f}")
        
        if threshold_type in ['flow', 'both']:
            if model_name in self.flow_thresholds:
                old_flow_threshold = self.flow_thresholds[model_name]
                self.flow_thresholds[model_name] = new_threshold
                print(f"Flow threshold for {model_name} adjusted from {old_flow_threshold:.5f} to {new_threshold:.5f}")
            else:
                self.flow_thresholds[model_name] = new_threshold
                print(f"Flow threshold for {model_name} set to {new_threshold:.5f}")
        
        return True
    
    def adjust_threshold_by_percentile(self, model_name, new_percentile, threshold_type='standard'):
        """
        Adjust threshold by changing the percentile used for calculation
        
        Args:
            model_name (str): Name of the model ('standard_ae' or 'lstm_ae')
            new_percentile (float): New percentile value (0-100)
            threshold_type (str): Type of threshold ('standard', 'flow', or 'both')
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return False
        
        if not hasattr(self, '_training_errors') or model_name not in self._training_errors:
            print(f"No training errors available for {model_name}. Train the model first.")
            return False
        
        errors = self._training_errors[model_name]
        new_threshold = np.percentile(errors, new_percentile)
        
        return self.adjust_threshold(model_name, new_threshold, threshold_type)
    
    def optimize_threshold_by_f1_score(self, model_name, validation_data, validation_labels, 
                                     threshold_range=None, threshold_type='standard'):
        """
        Optimize threshold by finding the best F1-score on validation data
        
        Args:
            model_name (str): Name of the model ('standard_ae' or 'lstm_ae')
            validation_data (pd.DataFrame): Validation data
            validation_labels (np.array): True labels (1 for attack, 0 for normal)
            threshold_range (tuple): Range of thresholds to test (min, max, step)
            threshold_type (str): Type of threshold ('standard', 'flow', or 'both')
        
        Returns:
            float: Optimal threshold value
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
        
        if threshold_range is None:
            # Default range: from 0.1 to 2.0 with 0.05 steps
            threshold_range = (0.1, 2.0, 0.05)
        
        min_threshold, max_threshold, step = threshold_range
        
        # Get predictions on validation data
        X_val = self.prepare_data_for_prediction(validation_data)
        errors = self.models[model_name].predict(X_val)
        
        best_f1 = 0
        best_threshold = min_threshold
        
        print(f"Optimizing threshold for {model_name}...")
        print(f"Testing thresholds from {min_threshold} to {max_threshold} with step {step}")
        
        for threshold in np.arange(min_threshold, max_threshold + step, step):
            predictions = (errors > threshold).astype(int)
            
            # Calculate F1 score
            from sklearn.metrics import f1_score
            try:
                f1 = f1_score(validation_labels, predictions, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except Exception as e:
                continue
        
        print(f"Best F1 score: {best_f1:.4f} at threshold: {best_threshold:.5f}")
        
        # Apply the optimal threshold
        self.adjust_threshold(model_name, best_threshold, threshold_type)
        
        return best_threshold
    
    def get_threshold_info(self, model_name):
        """Get detailed threshold information for a model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return None
        
        if not hasattr(self, '_statistical_metrics') or model_name not in self._statistical_metrics:
            print(f"No threshold metrics available for {model_name}")
            return None
        
        metrics = self._statistical_metrics[model_name]
        current_threshold = self.thresholds.get(model_name, 'Not set')
        current_flow_threshold = self.flow_thresholds.get(model_name, 'Not set')
        
        print(f"\nThreshold Information for {model_name}:")
        print(f"  Current method: {self._get_threshold_method()}")
        print(f"  Current threshold: {current_threshold}")
        print(f"  Current flow threshold: {current_flow_threshold}")
        print(f"  Training MSE: {metrics['mse']:.5f}")
        print(f"  Training MAE: {metrics['mae']:.5f}")
        print(f"  Training StdDev: {metrics['std_dev']:.5f}")
        
        if 'mse_plus_std' in metrics:
            print(f"  MSE + StdDev: {metrics['mse_plus_std']:.5f}")
        
        if 'mae_plus_std' in metrics:
            print(f"  MAE + StdDev: {metrics['mae_plus_std']:.5f}")
        
        if 'percentile_threshold' in metrics:
            print(f"  Percentile threshold: {metrics['percentile_threshold']:.5f}")
        
        # Show what the threshold should be based on current method
        current_method = self._get_threshold_method()
        if current_method == 'statistical':
            expected_threshold = metrics['mse'] + metrics['std_dev']
            print(f"  Expected threshold (MSE + StdDev): {expected_threshold:.5f}")
            if current_threshold != 'Not set':
                difference = abs(current_threshold - expected_threshold)
                print(f"  Difference from expected: {difference:.5f}")
                if difference > 0.001:
                    print(f"  WARNING: Threshold mismatch detected!")
        elif current_method == 'percentile' and 'percentile_threshold' in metrics:
            expected_threshold = metrics['percentile_threshold']
            print(f"  Expected threshold (percentile): {expected_threshold:.5f}")
            if current_threshold != 'Not set':
                difference = abs(current_threshold - expected_threshold)
                print(f"  Difference from expected: {difference:.5f}")
                if difference > 0.001:
                    print(f"  WARNING: Threshold mismatch detected!")
        
        return metrics
    
    def get_all_threshold_info(self):
        """Get threshold information for all models"""
        print(f"\nThreshold Method: {self._get_threshold_method()}")
        print("=" * 50)
        
        for model_name in self.models.keys():
            self.get_threshold_info(model_name)
    
    def update_threshold_percentiles(self, model_name, standard_percentile=None, flow_percentile=None):
        """
        Update threshold percentiles for a specific model in config
        
        Args:
            model_name (str): Name of the model ('standard_ae' or 'lstm_ae')
            standard_percentile (int, optional): New standard threshold percentile (0-100)
            flow_percentile (int, optional): New flow threshold percentile (0-100)
        """
        from config import THRESHOLD_PERCENTILES
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return False
        
        # Validate percentiles
        if standard_percentile is not None and not (0 <= standard_percentile <= 100):
            print("Error: Standard percentile must be between 0 and 100")
            return False
        
        if flow_percentile is not None and not (0 <= flow_percentile <= 100):
            print("Error: Flow percentile must be between 0 and 100")
            return False
        
        # Update config (this will persist across sessions)
        if model_name in THRESHOLD_PERCENTILES:
            if standard_percentile is not None:
                THRESHOLD_PERCENTILES[model_name]['standard'] = standard_percentile
            if flow_percentile is not None:
                THRESHOLD_PERCENTILES[model_name]['flow'] = flow_percentile
            
            print(f"Updated {model_name} threshold percentiles:")
            print(f"  Standard: {THRESHOLD_PERCENTILES[model_name]['standard']}th percentile")
            print(f"  Flow: {THRESHOLD_PERCENTILES[model_name]['flow']}th percentile")
            
            # Recalculate thresholds for this model if training errors are available
            if hasattr(self, '_training_errors') and model_name in self._training_errors:
                self._reset_single_model_thresholds(model_name)
            
            return True
        else:
            print(f"Error: {model_name} not found in THRESHOLD_PERCENTILES config")
            return False
    
    def get_current_percentiles(self, model_name):
        """Get current threshold percentiles for a specific model"""
        from config import THRESHOLD_PERCENTILES
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
        
        if model_name in THRESHOLD_PERCENTILES:
            return THRESHOLD_PERCENTILES[model_name].copy()
        else:
            return {'standard': 98, 'flow': 95}  # Default fallback
    
    def reset_thresholds_to_default(self, model_name=None):
        """
        Reset thresholds to their default percentile-based values
        
        Args:
            model_name (str): Name of the model to reset, or None for all models
        """
        if model_name is None:
            # Reset all models
            for name in self.models.keys():
                self._reset_single_model_thresholds(name)
        else:
            if model_name in self.models:
                self._reset_single_model_thresholds(model_name)
            else:
                print(f"Model {model_name} not found.")
    
    def _reset_single_model_thresholds(self, model_name):
        """Helper method to reset thresholds for a single model"""
        from config import THRESHOLD_PERCENTILES
        
        if hasattr(self, '_training_errors') and model_name in self._training_errors:
            errors = self._training_errors[model_name]
            
            # Get model-specific threshold percentiles
            model_percentiles = THRESHOLD_PERCENTILES.get(model_name, {'standard': 98, 'flow': 95})
            standard_percentile = model_percentiles['standard']
            flow_percentile = model_percentiles['flow']
            
            # Reset to default percentiles
            standard_threshold = np.percentile(errors, standard_percentile)
            flow_threshold = np.percentile(errors, flow_percentile)
            
            self.thresholds[model_name] = standard_threshold
            self.flow_thresholds[model_name] = flow_threshold
            
            print(f"Reset {model_name} thresholds to defaults:")
            print(f"  Standard: {standard_threshold:.5f} ({standard_percentile}th percentile)")
            print(f"  Flow: {flow_threshold:.5f} ({flow_percentile}th percentile)")
        else:
            print(f"No training errors available for {model_name}. Cannot reset to defaults.")
    
    def evaluate_models_enhanced(self, attack_info, test_data, additional_normal_data):
        """Enhanced evaluation on multiple attack types"""
        print("Evaluating models using CPU with enhanced metrics...")
        results = {}
        
        for model_name in self.models.keys():
            print(f"\nEvaluating {model_name}...")
            model_results = {
                'attack_detection_by_type': {},
                'false_positive_rate': 0,
                'overall_performance': {}
            }
            
            # Test on each attack type
            for attack_name, attack_data in attack_info.items():
                # Standard detection
                attack_anomalies, attack_scores = self.detect_anomalies_enhanced(
                    attack_data, model_name, use_flow_threshold=False
                )
                standard_detection_rate = np.mean(attack_anomalies)
                
                # Flow-sensitive detection
                flow_anomalies, _ = self.detect_anomalies_enhanced(
                    attack_data, model_name, use_flow_threshold=True
                )
                flow_detection_rate = np.mean(flow_anomalies)
                
                model_results['attack_detection_by_type'][attack_name] = {
                    'standard_detection_rate': standard_detection_rate,
                    'flow_sensitive_rate': flow_detection_rate,
                    'attack_scores': attack_scores,
                    'windows_detected': np.sum(flow_anomalies),
                    'total_windows': len(attack_data)
                }
            
            # Test on normal data for false positives
            normal_anomalies, normal_scores = self.detect_anomalies_enhanced(
                test_data, model_name, use_flow_threshold=True
            )
            false_positive_rate = np.mean(normal_anomalies)
            
            # Test on additional normal data
            additional_anomalies, _ = self.detect_anomalies_enhanced(
                additional_normal_data, model_name, use_flow_threshold=True
            )
            additional_fp_rate = np.mean(additional_anomalies)
            
            model_results['false_positive_rate'] = false_positive_rate
            model_results['additional_fp_rate'] = additional_fp_rate
            model_results['normal_scores'] = normal_scores
            
            results[model_name] = model_results
        
        return results
    
    def evaluate_models(self, attack_data, test_data):
        """Legacy evaluate_models method for backward compatibility"""
        print("Evaluating models using CPU...")
        results = {}
        
        for model_name in self.models.keys():
            # Test on attack data
            attack_anomalies, attack_scores = self.detect_anomalies(attack_data, model_name)
            attack_detection_rate = np.mean(attack_anomalies)
            
            # Test on normal data
            test_anomalies, test_scores = self.detect_anomalies(test_data, model_name)
            false_positive_rate = np.mean(test_anomalies)
            
            results[model_name] = {
                'attack_detection_rate': attack_detection_rate,
                'false_positive_rate': false_positive_rate,
                'attack_scores': attack_scores,
                'test_scores': test_scores
            }
        
        return results
    
    def save_models(self):
        """Save trained models"""
        for name, model in self.models.items():
            if hasattr(model, 'model'):
                model_path = os.path.join(ARTEFACTS_DIR, f"{name}.keras")
                model.model.save(model_path)
        
        # Save scaler and thresholds
        scaler_path = os.path.join(ARTEFACTS_DIR, "scaler.pkl")
        thresholds_path = os.path.join(ARTEFACTS_DIR, "thresholds.pkl")
        flow_thresholds_path = os.path.join(ARTEFACTS_DIR, "flow_thresholds.pkl")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_engineer.scaler, f)
        with open(thresholds_path, 'wb') as f:
            pickle.dump(self.thresholds, f)
        with open(flow_thresholds_path, 'wb') as f:
            pickle.dump(self.flow_thresholds, f)
        
        # Save threshold method and statistical metrics
        threshold_config_path = os.path.join(ARTEFACTS_DIR, "threshold_config.pkl")
        threshold_config = {
            'method': self._get_threshold_method(),
            'statistical_metrics': getattr(self, '_statistical_metrics', {}),
            'training_errors': getattr(self, '_training_errors', {})
        }
        with open(threshold_config_path, 'wb') as f:
            pickle.dump(threshold_config, f)
        
        # Save baseline statistics
        baseline_path = os.path.join(ARTEFACTS_DIR, "baseline_stats.pkl")
        baseline_stats = {
            'baseline_flows': getattr(self, 'baseline_flows', 2000),
            'flow_multiplier_threshold': getattr(self, 'flow_multiplier_threshold', 5.0)
        }
        with open(baseline_path, 'wb') as f:
            pickle.dump(baseline_stats, f)
        
        print(f"Threshold method '{self._get_threshold_method()}' saved")
        print("Training errors saved for threshold adjustments")
    
    def _recalculate_training_metrics(self, train_data):
        """Recalculate training metrics for loaded models"""
        print("Recalculating training metrics for loaded models...")
        
        # Prepare training data
        X_train = self.prepare_data_for_prediction(train_data)
        
        # Initialize storage
        if not hasattr(self, '_training_errors'):
            self._training_errors = {}
        if not hasattr(self, '_statistical_metrics'):
            self._statistical_metrics = {}
        
        # Calculate metrics for each model
        for name, model in self.models.items():
            if name in self.thresholds:  # Only for models with thresholds
                # Get predictions on training data
                errors = model.predict(X_train)
                
                # Store training errors
                self._training_errors[name] = errors
                
                # Calculate statistical metrics
                mse = np.mean(errors ** 2)  # Fixed: MSE should be mean of squared errors
                mae = np.mean(np.abs(errors))
                std_dev = np.std(errors)
                
                self._statistical_metrics[name] = {
                    'mse': mse,
                    'mae': mae,
                    'std_dev': std_dev,
                    'mse_plus_std': mse + std_dev,
                    'mae_plus_std': mae + std_dev
                }
                
                # Update the model instance with training metrics
                if hasattr(model, 'training_mse'):
                    model.training_mse = mse
                    model.training_mae = mae
                    model.training_stddev = std_dev
                    print(f"  {name}: MSE={mse:.5f}, MAE={mae:.5f}, StdDev={std_dev:.5f}")
                else:
                    print(f"  {name}: Model does not support training metrics storage")
        
        # Check if we need to recalculate thresholds due to method mismatch
        if not self.thresholds:  # Thresholds were not loaded due to method mismatch
            print("Recalculating thresholds with current method due to method mismatch...")
            self._recalculate_thresholds_with_current_method(X_train, train_data)
        elif hasattr(self, '_threshold_verification_needed') and self._threshold_verification_needed:
            # Verify threshold consistency and fix if needed
            print("Verifying loaded threshold consistency...")
            if self._verify_and_fix_threshold_consistency(X_train, train_data):
                print("Threshold consistency verified successfully!")
            else:
                print("Threshold inconsistencies detected and fixed automatically!")
            delattr(self, '_threshold_verification_needed')  # Clean up flag
        
        print("Training metrics recalculated successfully!")
    
    def _recalculate_thresholds_with_current_method(self, X_train, train_data):
        """Recalculate thresholds using the current threshold method"""
        current_method = self._get_threshold_method()
        print(f"Recalculating thresholds using method: {current_method}")
        
        if current_method == 'statistical':
            self._calculate_thresholds_statistical(X_train, train_data)
        elif current_method == 'percentile':
            self._calculate_thresholds_percentile(X_train, train_data)
        else:  # enhanced (default)
            self._calculate_enhanced_thresholds(X_train, train_data)
    
    def _verify_and_fix_threshold_consistency(self, X_train, train_data):
        """Verify threshold consistency and fix automatically if needed"""
        current_method = self._get_threshold_method()
        is_consistent = True
        tolerance = 1e-5  # Small tolerance for floating point comparison
        
        for name in self.models.keys():
            if name in self.thresholds and name in self._statistical_metrics:
                current_threshold = self.thresholds[name]
                metrics = self._statistical_metrics[name]
                
                # Calculate expected threshold based on current method
                if current_method == 'statistical':
                    expected_threshold = metrics['mse'] + metrics['std_dev']
                else:
                    # For other methods, we still want MSE + StdDev to be available for comparison
                    expected_threshold = metrics.get('mse_plus_std', current_threshold)
                
                # Check if threshold matches expected value
                difference = abs(current_threshold - expected_threshold)
                if difference > tolerance:
                    print(f"  Threshold inconsistency detected for {name}:")
                    print(f"     Current: {current_threshold:.5f}")
                    print(f"     Expected: {expected_threshold:.5f}")
                    print(f"     Difference: {difference:.5f}")
                    is_consistent = False
        
        if not is_consistent:
            print("  Automatically fixing threshold inconsistencies...")
            self._recalculate_thresholds_with_current_method(X_train, train_data)
            # Save the corrected thresholds
            self.save_models()
        
        return is_consistent
    
    def force_recalculate_thresholds(self, train_data):
        """Force recalculation of thresholds with current method (useful for debugging)"""
        print("Forcing recalculation of thresholds with current method...")
        
        # Prepare training data
        X_train = self.prepare_data_for_prediction(train_data)
        
        # Recalculate thresholds
        self._recalculate_thresholds_with_current_method(X_train, train_data)
        
        # Save the updated thresholds
        self.save_models()
        print("Thresholds recalculated and saved successfully!")
    
    def load_models(self):
        """Load previously saved models (CPU-based)"""
        print("Loading saved models using CPU...")
        
        # Check if models directory exists and has saved models
        if not os.path.exists(ARTEFACTS_DIR):
            print("No saved models found.")
            return False
        
        # Check for required files
        missing_files = []
        for file in REQUIRED_MODEL_FILES:
            file_path = os.path.join(ARTEFACTS_DIR, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"Missing saved model files: {missing_files}")
            return False
        
        try:
            # Load Keras models
            standard_ae = StandardAutoencoder(input_dim=1)  # Will be updated after loading
            standard_ae.model = load_model(os.path.join(ARTEFACTS_DIR, "standard_ae.keras"))
            self.models['standard_ae'] = standard_ae
            
            lstm_ae = LSTMAutoencoder(sequence_length=10, n_features=1)  # Will be updated after loading
            lstm_ae.model = load_model(os.path.join(ARTEFACTS_DIR, "lstm_ae.keras"))
            self.models['lstm_ae'] = lstm_ae
            
            # Load scaler
            with open(os.path.join(ARTEFACTS_DIR, "scaler.pkl"), 'rb') as f:
                self.feature_engineer.scaler = pickle.load(f)
            
            # Load threshold method and statistical metrics first
            threshold_config_path = os.path.join(ARTEFACTS_DIR, "threshold_config.pkl")
            saved_threshold_method = 'enhanced'  # Default fallback
            if os.path.exists(threshold_config_path):
                with open(threshold_config_path, 'rb') as f:
                    threshold_config = pickle.load(f)
                saved_threshold_method = threshold_config.get('method', 'enhanced')
                self._statistical_metrics = threshold_config.get('statistical_metrics', {})
                self._training_errors = threshold_config.get('training_errors', {})
                print(f"Saved threshold method '{saved_threshold_method}' loaded")
            
            # Check if we need to recalculate thresholds based on current config method
            current_method = self._get_threshold_method()
            if saved_threshold_method != current_method:
                print(f"Threshold method mismatch: saved='{saved_threshold_method}', current='{current_method}'")
                print("Will recalculate thresholds with current method when training data is available")
                # Don't load old thresholds - we'll recalculate them
                self.thresholds = {}
                self.flow_thresholds = {}
            else:
                # Load thresholds temporarily to verify consistency
                with open(os.path.join(ARTEFACTS_DIR, "thresholds.pkl"), 'rb') as f:
                    loaded_thresholds = pickle.load(f)
                
                # Load flow thresholds (if available)
                loaded_flow_thresholds = {}
                flow_thresholds_path = os.path.join(ARTEFACTS_DIR, "flow_thresholds.pkl")
                if os.path.exists(flow_thresholds_path):
                    with open(flow_thresholds_path, 'rb') as f:
                        loaded_flow_thresholds = pickle.load(f)
                
                # Mark that we need to verify threshold consistency after loading training metrics
                self.thresholds = loaded_thresholds
                self.flow_thresholds = loaded_flow_thresholds
                self._threshold_verification_needed = True
            
            # Load baseline statistics (if available)
            baseline_path = os.path.join(ARTEFACTS_DIR, "baseline_stats.pkl")
            if os.path.exists(baseline_path):
                with open(baseline_path, 'rb') as f:
                    baseline_stats = pickle.load(f)
                self.baseline_flows = baseline_stats.get('baseline_flows', 2000)
                self.flow_multiplier_threshold = baseline_stats.get('flow_multiplier_threshold', 5.0)
            
            # Load training errors for threshold adjustments (if available)
            training_errors_path = os.path.join(ARTEFACTS_DIR, "training_errors.pkl")
            if os.path.exists(training_errors_path):
                with open(training_errors_path, 'rb') as f:
                    self._training_errors = pickle.load(f)
                print("Training errors loaded for threshold adjustments")
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def check_saved_models(self):
        """Check if saved models exist"""
        if not os.path.exists(ARTEFACTS_DIR):
            return False
        
        for file in REQUIRED_MODEL_FILES:
            file_path = os.path.join(ARTEFACTS_DIR, file)
            if not os.path.exists(file_path):
                return False
        
        return True

    def _calculate_thresholds(self, X_train):
        """Legacy threshold calculation - now calls enhanced version"""
        # Create dummy train_data for backward compatibility
        dummy_train_data = pd.DataFrame({'flows_sum': np.random.normal(2000, 500, len(X_train))})
        return self._calculate_enhanced_thresholds(X_train, dummy_train_data)
    
    def set_threshold_method(self, method='statistical'):
        """
        Set the threshold calculation method
        
        Args:
            method (str): 'statistical' (MSE + stddev), 'percentile', or 'enhanced' (current default)
        """
        self.threshold_method = method
        print(f"Threshold calculation method set to: {method}")
    
    def _get_threshold_method(self):
        """Get the current threshold calculation method"""
        return getattr(self, 'threshold_method', 'enhanced')

    def _get_mse_plus_stddev_threshold(self, model_name, use_flow_threshold=False):
        """
        Get the MSE + StdDev threshold for a model, automatically calculating if needed
        
        Args:
            model_name (str): Name of the model
            use_flow_threshold (bool): Whether to use flow-based threshold (MAE + StdDev)
            
        Returns:
            float: MSE + StdDev threshold (or MAE + StdDev for flow threshold)
        """
        # Check if we have statistical metrics stored
        if hasattr(self, '_statistical_metrics') and model_name in self._statistical_metrics:
            metrics = self._statistical_metrics[model_name]
            
            if use_flow_threshold and 'mae_plus_std' in metrics:
                return metrics['mae_plus_std']
            elif 'mse_plus_std' in metrics:
                return metrics['mse_plus_std']
            elif 'mse' in metrics and 'std_dev' in metrics:
                if use_flow_threshold and 'mae' in metrics:
                    return metrics['mae'] + metrics['std_dev']
                else:
                    return metrics['mse'] + metrics['std_dev']
        
        # Try to calculate from training errors if available
        if hasattr(self, '_training_errors') and model_name in self._training_errors:
            print(f"Calculating MSE + StdDev threshold for {model_name} from training errors...")
            errors = self._training_errors[model_name]
            
            mse = np.mean(errors ** 2)
            mae = np.mean(np.abs(errors))
            std_dev = np.std(errors)
            
            # Store calculated metrics for future use
            if not hasattr(self, '_statistical_metrics'):
                self._statistical_metrics = {}
            
            self._statistical_metrics[model_name] = {
                'mse': mse,
                'mae': mae,
                'std_dev': std_dev,
                'mse_plus_std': mse + std_dev,
                'mae_plus_std': mae + std_dev
            }
            
            if use_flow_threshold:
                return mae + std_dev
            else:
                return mse + std_dev
        
        # Fallback: use stored thresholds if statistical metrics are not available
        print(f"Warning: No statistical metrics or training errors found for {model_name}")
        print(f"   Using stored threshold (may not be MSE + StdDev)")
        
        if use_flow_threshold and hasattr(self, 'flow_thresholds') and model_name in self.flow_thresholds:
            return self.flow_thresholds[model_name]
        elif hasattr(self, 'thresholds') and model_name in self.thresholds:
            return self.thresholds[model_name]
        else:
            # Ultimate fallback: use a default threshold
            print(f"Warning: No threshold found for {model_name}, using default 0.1")
            return 0.1

    def verify_threshold_consistency(self, train_data):
        """Verify that all thresholds are consistent with the current threshold method"""
        print("Verifying threshold consistency...")
        
        current_method = self._get_threshold_method()
        print(f"Current threshold method: {current_method}")
        
        inconsistencies_found = False
        
        for model_name in self.models.keys():
            if model_name in self.thresholds:
                print(f"\nChecking {model_name}...")
                
                # Get current threshold info
                metrics = self.get_threshold_info(model_name)
                if metrics is None:
                    continue
                
                current_threshold = self.thresholds[model_name]
                
                # Check if threshold matches expected value
                if current_method == 'statistical':
                    expected_threshold = metrics['mse'] + metrics['std_dev']
                    if abs(current_threshold - expected_threshold) > 0.001:
                        print(f"  Inconsistency found!")
                        print(f"     Current: {current_threshold:.5f}")
                        print(f"     Expected: {expected_threshold:.5f}")
                        inconsistencies_found = True
                    else:
                        print(f"  Threshold consistent with statistical method")
                
                elif current_method == 'percentile':
                    if 'percentile_threshold' in metrics:
                        expected_threshold = metrics['percentile_threshold']
                        if abs(current_threshold - expected_threshold) > 0.001:
                            print(f"  Inconsistency found!")
                            print(f"     Current: {current_threshold:.5f}")
                            print(f"     Expected: {expected_threshold:.5f}")
                            inconsistencies_found = True
                        else:
                            print(f"  Threshold consistent with percentile method")
                    else:
                        print(f"  Cannot verify percentile threshold (missing data)")
        
        if inconsistencies_found:
            print(f"\nThreshold inconsistencies detected!")
            print("Use force_recalculate_thresholds() to fix them.")
            return False
        else:
            print(f"\nAll thresholds are consistent with {current_method} method!")
            return True

    def enable_debug_mode(self, enabled=True):
        """Enable or disable debug mode for threshold information display"""
        self.debug_mode = enabled
        if enabled:
            print("Debug mode enabled - threshold info will be displayed during detection")
        else:
            print("Debug mode disabled")
    
    def verify_mse_stddev_usage(self):
        """Verify that MSE + StdDev thresholds are being used correctly"""
        print("Verifying MSE + StdDev threshold usage...")
        
        if not self.models:
            print("No models loaded")
            return False
        
        all_verified = True
        
        for model_name in self.models.keys():
            print(f"\nModel: {model_name}")
            
            # Get the threshold that would be used
            standard_threshold = self._get_mse_plus_stddev_threshold(model_name, use_flow_threshold=False)
            flow_threshold = self._get_mse_plus_stddev_threshold(model_name, use_flow_threshold=True)
            
            print(f"   Standard threshold (MSE + StdDev): {standard_threshold:.6f}")
            print(f"   Flow threshold (MAE + StdDev): {flow_threshold:.6f}")
            
            # Check if we have statistical metrics
            if hasattr(self, '_statistical_metrics') and model_name in self._statistical_metrics:
                metrics = self._statistical_metrics[model_name]
                expected_standard = metrics['mse'] + metrics['std_dev']
                expected_flow = metrics['mae'] + metrics['std_dev']
                
                standard_match = abs(standard_threshold - expected_standard) < 1e-6
                flow_match = abs(flow_threshold - expected_flow) < 1e-6
                
                print(f"   Expected standard: {expected_standard:.6f} {'[OK]' if standard_match else '[MISMATCH]'}")
                print(f"   Expected flow: {expected_flow:.6f} {'[OK]' if flow_match else '[MISMATCH]'}")
                
                if not (standard_match and flow_match):
                    all_verified = False
            else:
                print(f"   No statistical metrics available for verification")
                all_verified = False
        
        if all_verified:
            print(f"\nAll models are using correct MSE + StdDev thresholds")
        else:
            print(f"\nSome models may not be using correct MSE + StdDev thresholds")
        
        return all_verified
