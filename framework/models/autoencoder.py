import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler


class AutoencoderAnomalyDetector:
    def __init__(self, latent_dim=8):
        self.latent_dim = latent_dim
        self.autoencoder = None
        self.scaler = MinMaxScaler()
        self.threshold = None
        self.history = None
        
    def build_model(self, input_dim):
        """Build the autoencoder architecture"""
        self.autoencoder = keras.Sequential([
            keras.layers.Input(shape=(input_dim,)),
            
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            
            keras.layers.Dense(self.latent_dim, activation='relu'),
            
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            
            keras.layers.Dense(input_dim, activation='linear')
        ])
        
        initial_learning_rate = 0.001
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        self.autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(f"Built autoencoder for {input_dim} features")
        
    def fit_scaler(self, training_features):
        """Fit the scaler on training data"""
        self.scaler.fit(training_features)
        
    def transform_data(self, features):
        """Transform features using the fitted scaler"""
        return self.scaler.transform(features)
        
    def train(self, scaled_train_data, scaled_validation_data, epochs=100, batch_size=64):
        """Train the autoencoder"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        print("Training autoencoder...")
        self.history = self.autoencoder.fit(
            scaled_train_data, scaled_train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(scaled_validation_data, scaled_validation_data),
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"Training completed after {len(self.history.history['loss'])} epochs")
        return self.history
        
    def predict(self, scaled_data):
        """Get reconstructions and calculate MSE"""
        reconstructions = self.autoencoder.predict(scaled_data)
        mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
        return reconstructions, mse
        
    def calculate_threshold(self, train_mse, val_mse, strategy='mean_plus_1std'):
        """Calculate anomaly detection threshold"""
        normal_mse_values = np.concatenate([train_mse, val_mse])
        
        normal_mse_mean = np.mean(normal_mse_values)
        normal_mse_std = np.std(normal_mse_values)
        
        strategies = {
            'normal_mse_mean': normal_mse_mean,
            'mean_plus_1std': normal_mse_mean + normal_mse_std,
            'mean_plus_2std': normal_mse_mean + 2 * normal_mse_std,
            'mean_plus_3std': normal_mse_mean + 3 * normal_mse_std
        }
        
        self.threshold = strategies[strategy]
        
        print(f"Normal Traffic Statistics:")
        print(f"  Mean MSE: {normal_mse_mean:.6f}")
        print(f"  Std MSE:  {normal_mse_std:.6f}")
        print(f"  Selected Threshold ({strategy}): {self.threshold:.6f}")
        
        return self.threshold, strategies
        
    def detect_anomalies(self, mse):
        """Detect anomalies based on threshold"""
        if self.threshold is None:
            raise ValueError("Threshold not set. Call calculate_threshold first.")
        return mse > self.threshold
        
    def get_metrics(self, scaled_data, reconstructions, mse):
        """Calculate reconstruction metrics"""
        mae = np.mean(np.abs(scaled_data - reconstructions))
        mse_value = np.mean(mse)
        rmse = np.sqrt(mse_value)
        
        return {
            'mae': mae,
            'mse': mse_value,
            'rmse': rmse
        }
