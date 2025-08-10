#!/usr/bin/env python3
"""
Machine Learning Models for DDoS Detection System
"""

# Configure TensorFlow logging at the very beginning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# GPU acceleration enabled - remove the lines below if you want CPU-only execution
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
# os.environ['TF_ENABLE_XLA'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import numpy as np
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('tensorflow').disabled = True
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    STANDARD_AE_ENCODING_DIM, STANDARD_AE_DROPOUT,
    LSTM_AE_SEQUENCE_LENGTH
)


class StandardAutoencoder:
    """Standard dense autoencoder for anomaly detection"""
    
    def __init__(self, input_dim, encoding_dim=STANDARD_AE_ENCODING_DIM):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = None
        self.scaler = StandardScaler()
        # Training metrics
        self.training_mse = None
        self.training_mae = None
        self.training_stddev = None
        
    def build_model(self):
        """Build the autoencoder architecture with explicit GPU placement"""
        # Use GPU if available, otherwise fallback to CPU
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        
        with tf.device(device):
            input_layer = layers.Input(shape=(self.input_dim,))
            encoded = layers.Dense(64, activation='relu')(input_layer)
            encoded = layers.Dropout(STANDARD_AE_DROPOUT)(encoded)
            encoded = layers.Dense(32, activation='relu')(encoded)
            encoded = layers.Dropout(STANDARD_AE_DROPOUT)(encoded)
            bottleneck = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
            
            decoded = layers.Dense(32, activation='relu')(bottleneck)
            decoded = layers.Dropout(STANDARD_AE_DROPOUT)(decoded)
            decoded = layers.Dense(64, activation='relu')(decoded)
            decoded = layers.Dropout(STANDARD_AE_DROPOUT)(decoded)
            output_layer = layers.Dense(self.input_dim, activation='linear')(decoded)
            
            self.model = Model(input_layer, output_layer)
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='mse',
                metrics=['mae']
            )
        
        print(f"StandardAutoencoder built on device: {device}")
        return self.model
    
    def train(self, X_train, X_val=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the autoencoder with GPU acceleration"""
        if self.model is None:
            self.build_model()
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        validation_data = (X_val, X_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, X_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate and store training metrics
        self._calculate_training_metrics(X_train)
        
        return history
    
    def _calculate_training_metrics(self, X_train):
        """Calculate and store training performance metrics"""
        try:
            # Get actual reconstructions (not MSE)
            reconstructed = self.model.predict(X_train, verbose=0)
            
            # Calculate raw errors (input - reconstruction)
            raw_errors = X_train - reconstructed
            
            # Calculate MSE: mean of all squared errors across all features and samples
            self.training_mse = np.mean(raw_errors ** 2)
            
            # Calculate MAE: mean of all absolute errors across all features and samples
            self.training_mae = np.mean(np.abs(raw_errors))
            
            # Calculate standard deviation using per-sample reconstruction error (as used in predict method)
            per_sample_mse = np.mean((raw_errors) ** 2, axis=1)
            self.training_stddev = np.std(per_sample_mse)
            
            print(f"StandardAE Training Metrics - MSE: {self.training_mse:.5f}, MAE: {self.training_mae:.5f}, StdDev: {self.training_stddev:.5f}")
            
        except Exception as e:
            print(f"Error calculating training metrics for StandardAE: {e}")
            self.training_mse = 0.0
            self.training_mae = 0.0
            self.training_stddev = 0.0
    
    def get_training_metrics(self):
        """Get training performance metrics"""
        return {
            'mse': self.training_mse or 0.0,
            'mae': self.training_mae or 0.0,
            'stddev': self.training_stddev or 0.0
        }
    
    def predict(self, X):
        """Get reconstruction error using CPU"""
        reconstructed = self.model.predict(X, verbose=0)
        mse = np.mean((X - reconstructed) ** 2, axis=1)
        return mse


class LSTMAutoencoder:
    """LSTM-based autoencoder for temporal pattern detection"""
    
    def __init__(self, sequence_length=LSTM_AE_SEQUENCE_LENGTH, n_features=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        # Training metrics
        self.training_mse = None
        self.training_mae = None
        self.training_stddev = None
        
    def build_model(self):
        """Build LSTM autoencoder with explicit GPU placement"""
        # Use GPU if available, otherwise fallback to CPU
        device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
        
        with tf.device(device):
            input_layer = layers.Input(shape=(self.sequence_length, self.n_features))
            encoded = layers.LSTM(32, return_sequences=True)(input_layer)
            encoded = layers.LSTM(16, return_sequences=False)(encoded)
            bottleneck = layers.Dense(8, activation='relu', name='bottleneck')(encoded)
            
            decoded = layers.RepeatVector(self.sequence_length)(bottleneck)
            decoded = layers.LSTM(16, return_sequences=True)(decoded)
            decoded = layers.LSTM(32, return_sequences=True)(decoded)
            output_layer = layers.TimeDistributed(layers.Dense(self.n_features))(decoded)
            
            self.model = Model(input_layer, output_layer)
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='mse',
                metrics=['mae']
            )
        
        print(f"LSTMAutoencoder built on device: {device}")
        return self.model
    
    def prepare_sequences(self, data, sequence_length):
        """Prepare sequences for LSTM"""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)
    
    def train(self, X_train, X_val=None, epochs=EPOCHS, batch_size=BATCH_SIZE):
        """Train the LSTM autoencoder with GPU acceleration"""
        if self.model is None:
            self.build_model()
        
        X_train_seq = self.prepare_sequences(X_train, self.sequence_length)
        X_val_seq = None
        if X_val is not None:
            X_val_seq = self.prepare_sequences(X_val, self.sequence_length)
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
        ]

        validation_data = (X_val_seq, X_val_seq) if X_val_seq is not None else None

        history = self.model.fit(
            X_train_seq, X_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate and store training metrics
        self._calculate_training_metrics(X_train)
        
        return history
    
    def _calculate_training_metrics(self, X_train):
        """Calculate and store training performance metrics"""
        try:
            # Prepare sequences for LSTM
            X_train_seq = self.prepare_sequences(X_train, self.sequence_length)
            
            if len(X_train_seq) == 0:
                print("No sequences available for LSTM training metrics calculation")
                self.training_mse = 0.0
                self.training_mae = 0.0
                self.training_stddev = 0.0
                return
            
            # Get actual reconstructions (not MSE)
            reconstructed = self.model.predict(X_train_seq, verbose=0)
            
            # Calculate raw errors (input - reconstruction)
            raw_errors = X_train_seq - reconstructed
            
            # Calculate MSE: mean of all squared errors across all timesteps, features and samples  
            self.training_mse = np.mean(raw_errors ** 2)
            
            # Calculate MAE: mean of all absolute errors across all timesteps, features and samples
            self.training_mae = np.mean(np.abs(raw_errors))
            
            # Calculate standard deviation using per-sample reconstruction error (as used in predict method)
            per_sample_mse = np.mean((raw_errors) ** 2, axis=(1, 2))
            self.training_stddev = np.std(per_sample_mse)
            
            print(f"LSTM-AE Training Metrics - MSE: {self.training_mse:.5f}, MAE: {self.training_mae:.5f}, StdDev: {self.training_stddev:.5f}")
            
        except Exception as e:
            print(f"Error calculating training metrics for LSTM-AE: {e}")
            self.training_mse = 0.0
            self.training_mae = 0.0
            self.training_stddev = 0.0
    
    def get_training_metrics(self):
        """Get training performance metrics"""
        return {
            'mse': self.training_mse or 0.0,
            'mae': self.training_mae or 0.0,
            'stddev': self.training_stddev or 0.0
        }

    def predict(self, X):
        """Get reconstruction error using CPU"""
        try:
            # Check if we have enough data for sequence creation
            if len(X) < self.sequence_length:
                # Not enough data for sequences, return high anomaly scores
                return np.full(len(X), 1.0)  # High anomaly score for insufficient data
            
            X_seq = self.prepare_sequences(X, self.sequence_length)
            
            # Check if sequences were created successfully
            if len(X_seq) == 0:
                return np.full(len(X), 1.0)  # High anomaly score for no sequences
            
            reconstructed = self.model.predict(X_seq, verbose=0)  # Disable progress bar
            mse = np.mean((X_seq - reconstructed) ** 2, axis=(1, 2))
            return mse
        except Exception as e:
            # Fallback: if sequence preparation fails, try direct prediction
            if len(X.shape) == 3:
                reconstructed = self.model.predict(X, verbose=0)
                mse = np.mean((X - reconstructed) ** 2, axis=(1, 2))
                return mse
            else:
                # Return high anomaly scores as fallback
                print(f"LSTM prediction failed: {str(e)}. Returning fallback scores.")
                return np.full(len(X), 1.0)
