"""
LSTM Autoencoder for temporal anomaly detection.

This module implements an LSTM-based autoencoder that captures temporal patterns
in network traffic sequences for improved DDoS detection.
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import logging

from tensorflow import keras
from tensorflow.keras import layers

from .temporal.base_temporal_autoencoder import BaseTemporalAutoencoder
from .temporal.components import LSTMEncoder, LSTMDecoder

logger = logging.getLogger(__name__)


class LSTMAutoencoder(BaseTemporalAutoencoder):
    """
    LSTM-based autoencoder for temporal anomaly detection in network traffic.
    
    This model uses LSTM networks to learn temporal patterns in network traffic
    sequences and detect anomalies based on reconstruction error. The LSTM
    architecture is particularly effective at capturing long-term dependencies
    in sequential data, making it ideal for detecting temporal anomalies in
    network traffic patterns.
    
    Attributes:
        encoder_units: List of hidden units for encoder LSTM layers
        decoder_units: List of hidden units for decoder LSTM layers
        dropout_rate: Dropout rate for regularization
    """
    
    def __init__(
        self, 
        sequence_length: int = 60,
        latent_dim: int = 32,
        encoder_units: Optional[List[int]] = None,
        decoder_units: Optional[List[int]] = None,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the LSTM autoencoder.
        
        Args:
            sequence_length: Length of input sequences
            latent_dim: Dimension of the latent representation
            encoder_units: List of hidden units for encoder layers
            decoder_units: List of hidden units for decoder layers  
            dropout_rate: Dropout rate for regularization
        """
        super().__init__(
            model_name="lstm_autoencoder",
            sequence_length=sequence_length,
            latent_dim=latent_dim
        )
        
        # Default architecture if not specified
        self.encoder_units = encoder_units or [128, 64]
        self.decoder_units = decoder_units or [64, 128]
        self.dropout_rate = dropout_rate
        
    def build_model(self, feature_dim: int) -> None:
        """
        Build the LSTM autoencoder model.
        
        Args:
            feature_dim: Number of features per timestep
            
        Raises:
            ValueError: If feature_dim is invalid
        """
        print(f"Building LSTM Autoencoder for sequences of length {self.sequence_length}")
        print(f"Features per timestep: {feature_dim}")
        
        if feature_dim <= 0:
            raise ValueError(f"Invalid feature dimension: {feature_dim}")
        
        self._feature_dim = feature_dim
        
        # Input layer
        inputs = layers.Input(
            shape=(self.sequence_length, feature_dim), 
            name='input_sequences'
        )
        
        # Simple, stable encoder architecture
        encoder = LSTMEncoder(
            hidden_units=self.encoder_units,
            dropout_rate=self.dropout_rate,
            return_sequences=False
        )
        encoded = encoder(inputs)
        
        # Simple bottleneck layer
        bottleneck = layers.Dense(
            self.latent_dim, 
            activation='relu', 
            name='bottleneck'
        )(encoded)
        
        # Decoder
        decoder = LSTMDecoder(
            hidden_units=self.decoder_units,
            output_dim=feature_dim,
            sequence_length=self.sequence_length,
            dropout_rate=self.dropout_rate
        )
        decoded = decoder(bottleneck)
        
        # Create stable model
        self.model = keras.Model(
            inputs=inputs, 
            outputs=decoded, 
            name='stable_lstm_autoencoder'
        )
        
        # Conservative optimizer configuration for stable training
        initial_learning_rate = 0.0005  # Lower learning rate for stability
        optimizer = keras.optimizers.Adam(
            learning_rate=initial_learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        # Log architecture details
        total_params = self.model.count_params()
        trainable_params = sum([keras.backend.count_params(w) for w in self.model.trainable_weights])
        
        logger.info(f"LSTM Autoencoder architecture:")
        logger.info(f"  Input shape: ({self.sequence_length}, {feature_dim})")
        logger.info(f"  Encoder units: {self.encoder_units}")
        logger.info(f"  Latent dimension: {self.latent_dim}")
        logger.info(f"  Decoder units: {self.decoder_units}")
        logger.info(f"  Dropout rate: {self.dropout_rate}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        print(f"LSTM Autoencoder architecture:")
        print(f"  Input shape: ({self.sequence_length}, {feature_dim})")
        print(f"  Encoder units: {self.encoder_units}")
        print(f"  Latent dimension: {self.latent_dim}")
        print(f"  Decoder units: {self.decoder_units}")
        print(f"  Dropout rate: {self.dropout_rate}")
        print(f"  Total parameters: {total_params:,}")
        
    def get_encoder_output(self, scaled_data: np.ndarray) -> np.ndarray:
        """
        Get encoded representations from the encoder.
        
        Args:
            scaled_data: Scaled input data
            
        Returns:
            Encoded representations
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create encoder model
        encoder_model = keras.Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('bottleneck').output
        )
        
        sequences = self.sequence_processor.create_sequences(scaled_data)
        encoded = encoder_model.predict(sequences, verbose=0)
        
        return encoded
        
    def analyze_lstm_states(self, scaled_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Analyze LSTM hidden states for interpretability.
        
        Args:
            scaled_data: Scaled input data
            
        Returns:
            Dictionary containing LSTM state analysis
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        sequences = self.sequence_processor.create_sequences(scaled_data)
        
        # Get intermediate outputs from encoder LSTM layers
        intermediate_outputs = {}
        
        # Find LSTM layers in the model
        lstm_layers = [layer for layer in self.model.layers 
                      if isinstance(layer, (layers.LSTM, LSTMEncoder))]
        
        for i, layer in enumerate(lstm_layers):
            try:
                intermediate_model = keras.Model(
                    inputs=self.model.input,
                    outputs=layer.output
                )
                output = intermediate_model.predict(sequences[:100], verbose=0)  # Sample for efficiency
                intermediate_outputs[f'lstm_layer_{i}'] = output
            except Exception as e:
                logger.warning(f"Could not extract output from layer {layer.name}: {e}")
        
        return intermediate_outputs
        
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.
        
        Returns:
            Dictionary containing model configuration
        """
        config = {
            'model_type': 'lstm_autoencoder',
            'sequence_length': self.sequence_length,
            'latent_dim': self.latent_dim,
            'encoder_units': self.encoder_units,
            'decoder_units': self.decoder_units,
            'dropout_rate': self.dropout_rate,
            'feature_dim': self._feature_dim
        }
        
        if self.is_trained:
            config.update({
                'threshold': self.threshold,
                'is_trained': True
            })
        
        return config
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'LSTMAutoencoder':
        """
        Create model instance from configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            LSTMAutoencoder instance
        """
        model = cls(
            sequence_length=config['sequence_length'],
            latent_dim=config['latent_dim'],
            encoder_units=config['encoder_units'],
            decoder_units=config['decoder_units'],
            dropout_rate=config['dropout_rate']
        )
        
        if 'feature_dim' in config:
            model.build_model(config['feature_dim'])
            
        if config.get('is_trained', False):
            model.is_trained = True
            model.threshold = config.get('threshold')
        
        return model
