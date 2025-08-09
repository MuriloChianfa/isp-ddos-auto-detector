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
    INPUT_DIR, MODELS_DIR, AGG_PERIOD, TRAIN_SPLIT, 
    ATTACK_PERIODS, EPOCHS, REQUIRED_MODEL_FILES,
    ANOMALY_THRESHOLD_PERCENTILE, FLOW_ANOMALY_THRESHOLD_PERCENTILE,
    FLOW_MULTIPLIER_THRESHOLD
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
        
        self.cache_dir = os.path.join(MODELS_DIR, "cache")
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
        
        # Standard thresholds
        for name, model in self.models.items():
            errors = model.predict(X_train)
            threshold = np.percentile(errors, ANOMALY_THRESHOLD_PERCENTILE)
            
            # More sensitive threshold for flow-based detection
            flow_threshold = np.percentile(errors, FLOW_ANOMALY_THRESHOLD_PERCENTILE)
            self.flow_thresholds[name] = flow_threshold
            
            self.thresholds[name] = threshold
        
        # Store baseline statistics for flow-based detection
        self.baseline_flows = baseline_flows
        self.flow_multiplier_threshold = FLOW_MULTIPLIER_THRESHOLD
        
        print(f"Baseline flows: {baseline_flows:.0f}")
        print(f"Flow multiplier threshold: {FLOW_MULTIPLIER_THRESHOLD}x")
        print("Standard thresholds:", {k: f"{v:.4f}" for k, v in self.thresholds.items()})
        print("Flow-sensitive thresholds:", {k: f"{v:.4f}" for k, v in self.flow_thresholds.items()})
    
    def _calculate_thresholds(self, X_train):
        """Legacy threshold calculation - now calls enhanced version"""
        # Create dummy train_data for backward compatibility
        dummy_train_data = pd.DataFrame({'flows_sum': np.random.normal(2000, 500, len(X_train))})
        return self._calculate_enhanced_thresholds(X_train, dummy_train_data)
    
    def prepare_data_for_prediction(self, data):
        """Prepare data for prediction with models - returns normalized features"""
        X, _ = self.prepare_features(data, fit_scaler=False)
        return X
        
    def detect_anomalies_enhanced(self, data, model_name='standard_ae', use_flow_threshold=False):
        """Enhanced anomaly detection with flow-based sensitivity"""
        X, _ = self.prepare_features(data, fit_scaler=False)
        model = self.models[model_name]
        
        # Choose threshold based on detection type
        if use_flow_threshold and model_name in self.flow_thresholds:
            threshold = self.flow_thresholds[model_name]
        else:
            threshold = self.thresholds[model_name]
        
        errors = model.predict(X)
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
                model_path = os.path.join(MODELS_DIR, f"{name}.keras")
                model.model.save(model_path)
        
        # Save scaler and thresholds
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        thresholds_path = os.path.join(MODELS_DIR, "thresholds.pkl")
        flow_thresholds_path = os.path.join(MODELS_DIR, "flow_thresholds.pkl")
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_engineer.scaler, f)
        with open(thresholds_path, 'wb') as f:
            pickle.dump(self.thresholds, f)
        with open(flow_thresholds_path, 'wb') as f:
            pickle.dump(self.flow_thresholds, f)
        
        # Save baseline statistics
        baseline_path = os.path.join(MODELS_DIR, "baseline_stats.pkl")
        baseline_stats = {
            'baseline_flows': getattr(self, 'baseline_flows', 2000),
            'flow_multiplier_threshold': getattr(self, 'flow_multiplier_threshold', 5.0)
        }
        with open(baseline_path, 'wb') as f:
            pickle.dump(baseline_stats, f)
    
    def load_models(self):
        """Load previously saved models (CPU-based)"""
        print("Loading saved models using CPU...")
        
        # Check if models directory exists and has saved models
        if not os.path.exists(MODELS_DIR):
            print("No saved models found.")
            return False
        
        # Check for required files
        missing_files = []
        for file in REQUIRED_MODEL_FILES:
            file_path = os.path.join(MODELS_DIR, file)
            if not os.path.exists(file_path):
                missing_files.append(file)
        
        if missing_files:
            print(f"Missing saved model files: {missing_files}")
            return False
        
        try:
            # Load Keras models
            standard_ae = StandardAutoencoder(input_dim=1)  # Will be updated after loading
            standard_ae.model = load_model(os.path.join(MODELS_DIR, "standard_ae.keras"))
            self.models['standard_ae'] = standard_ae
            
            lstm_ae = LSTMAutoencoder(sequence_length=10, n_features=1)  # Will be updated after loading
            lstm_ae.model = load_model(os.path.join(MODELS_DIR, "lstm_ae.keras"))
            self.models['lstm_ae'] = lstm_ae
            
            # Load scaler
            with open(os.path.join(MODELS_DIR, "scaler.pkl"), 'rb') as f:
                self.feature_engineer.scaler = pickle.load(f)
            
            # Load thresholds
            with open(os.path.join(MODELS_DIR, "thresholds.pkl"), 'rb') as f:
                self.thresholds = pickle.load(f)
            
            # Load flow thresholds (if available)
            flow_thresholds_path = os.path.join(MODELS_DIR, "flow_thresholds.pkl")
            if os.path.exists(flow_thresholds_path):
                with open(flow_thresholds_path, 'rb') as f:
                    self.flow_thresholds = pickle.load(f)
            
            # Load baseline statistics (if available)
            baseline_path = os.path.join(MODELS_DIR, "baseline_stats.pkl")
            if os.path.exists(baseline_path):
                with open(baseline_path, 'rb') as f:
                    baseline_stats = pickle.load(f)
                    self.baseline_flows = baseline_stats.get('baseline_flows', 2000)
                    self.flow_multiplier_threshold = baseline_stats.get('flow_multiplier_threshold', 5.0)
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def check_saved_models(self):
        """Check if saved models exist"""
        if not os.path.exists(MODELS_DIR):
            return False
        
        for file in REQUIRED_MODEL_FILES:
            file_path = os.path.join(MODELS_DIR, file)
            if not os.path.exists(file_path):
                return False
        
        return True
