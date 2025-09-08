"""
Temporal components for sequence-based anomaly detection models.

This module provides reusable temporal building blocks for LSTM and TCN-based
autoencoder architectures, following best practices for maintainable code.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SequenceProcessor:
    """Utility class for handling sequence creation and processing."""
    
    def __init__(self, sequence_length: int):
        """
        Initialize sequence processor.
        
        Args:
            sequence_length: Length of sequences to create
        """
        self.sequence_length = sequence_length
        
    def create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sliding window sequences from time series data.
        
        Args:
            data: Input time series data of shape (n_samples, n_features)
            
        Returns:
            Array of sequences with shape (n_sequences, sequence_length, n_features)
            
        Raises:
            ValueError: If data length is less than sequence_length
        """
        if len(data) < self.sequence_length:
            raise ValueError(
                f"Data length {len(data)} is less than sequence length {self.sequence_length}"
            )
        
        n_samples, n_features = data.shape
        n_sequences = n_samples - self.sequence_length + 1
        
        sequences = np.zeros((n_sequences, self.sequence_length, n_features))
        
        for i in range(n_sequences):
            sequences[i] = data[i:i + self.sequence_length]
        
        return sequences
    
    def calculate_sequence_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """
        Calculate MSE for each sequence.
        
        Args:
            original: Original sequences
            reconstructed: Reconstructed sequences
            
        Returns:
            MSE values for each sequence
        """
        return np.mean(np.power(original - reconstructed, 2), axis=(1, 2))


class TemporalBlock(layers.Layer):
    """
    Temporal Convolutional Block with residual connections.
    
    This block implements the core building block for TCN architectures,
    featuring dilated convolutions, normalization, and residual connections.
    """
    
    def __init__(
        self, 
        filters: int, 
        kernel_size: int, 
        dilation_rate: int, 
        dropout_rate: float = 0.1,
        activation: str = 'relu',
        **kwargs
    ):
        """
        Initialize temporal block.
        
        Args:
            filters: Number of convolutional filters
            kernel_size: Size of convolutional kernel
            dilation_rate: Dilation rate for dilated convolution
            dropout_rate: Dropout rate for regularization
            activation: Activation function to use
        """
        super(TemporalBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # First convolution path
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation=activation
        )
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dropout_rate)
        
        # Second convolution path
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation=activation
        )
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dropout_rate)
        
        # Residual connection (will be initialized in build)
        self.residual_conv = None
        
    def build(self, input_shape):
        """Build the layer and initialize residual connection if needed."""
        super(TemporalBlock, self).build(input_shape)
        
        # Add residual connection if input and output dimensions differ
        if input_shape[-1] != self.filters:
            self.residual_conv = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding='same'
            )
    
    def call(self, inputs, training=None):
        """Forward pass through the temporal block."""
        # First convolution path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        
        # Second convolution path
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(inputs)
        else:
            residual = inputs
            
        return layers.Add()([x, residual])


class LSTMEncoder(layers.Layer):
    """LSTM-based encoder for sequence encoding."""
    
    def __init__(
        self, 
        hidden_units: list, 
        dropout_rate: float = 0.1, 
        return_sequences: bool = False,
        **kwargs
    ):
        """
        Initialize LSTM encoder.
        
        Args:
            hidden_units: List of hidden units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            return_sequences: Whether to return full sequences or just last output
        """
        super(LSTMEncoder, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.return_sequences = return_sequences
        
        self.lstm_layers = []
        for i, units in enumerate(hidden_units):
            return_seq = True if i < len(hidden_units) - 1 else return_sequences
            self.lstm_layers.append(
                layers.LSTM(
                    units=units,
                    return_sequences=return_seq,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name=f'lstm_encoder_{i}'
                )
            )
    
    def call(self, inputs, training=None):
        """Forward pass through LSTM encoder."""
        x = inputs
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        return x


class LSTMDecoder(layers.Layer):
    """LSTM-based decoder for sequence reconstruction."""
    
    def __init__(
        self, 
        hidden_units: list, 
        output_dim: int, 
        sequence_length: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize LSTM decoder.
        
        Args:
            hidden_units: List of hidden units for each LSTM layer
            output_dim: Dimension of output features
            sequence_length: Length of output sequences
            dropout_rate: Dropout rate for regularization
        """
        super(LSTMDecoder, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        
        # Repeat vector to expand encoded representation
        self.repeat_vector = layers.RepeatVector(sequence_length)
        
        # LSTM layers
        self.lstm_layers = []
        for i, units in enumerate(hidden_units):
            self.lstm_layers.append(
                layers.LSTM(
                    units=units,
                    return_sequences=True,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    name=f'lstm_decoder_{i}'
                )
            )
        
        # Output layer
        self.output_layer = layers.TimeDistributed(
            layers.Dense(output_dim, activation='linear'),
            name='decoder_output'
        )
    
    def call(self, inputs, training=None):
        """Forward pass through LSTM decoder."""
        # Expand encoded representation to sequence
        x = self.repeat_vector(inputs)
        
        # Pass through LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x, training=training)
        
        # Generate output sequence
        return self.output_layer(x, training=training)


class TCNEncoder(layers.Layer):
    """TCN-based encoder using temporal blocks."""
    
    def __init__(
        self, 
        num_blocks: int, 
        filters: int, 
        kernel_size: int, 
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize TCN encoder.
        
        Args:
            num_blocks: Number of temporal blocks
            filters: Number of filters per block
            kernel_size: Kernel size for convolutions
            dropout_rate: Dropout rate for regularization
        """
        super(TCNEncoder, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # Temporal blocks with increasing dilation
        self.temporal_blocks = []
        for i in range(num_blocks):
            dilation_rate = 2 ** i
            self.temporal_blocks.append(
                TemporalBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    dropout_rate=dropout_rate,
                    name=f'temporal_block_{i}'
                )
            )
        
        # Global pooling for final representation
        self.global_pool = layers.GlobalAveragePooling1D()
    
    def call(self, inputs, training=None):
        """Forward pass through TCN encoder."""
        x = inputs
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Global pooling to get fixed-size representation
        return self.global_pool(x)
    
    def get_receptive_field(self) -> int:
        """Calculate the receptive field of the TCN encoder."""
        return 1 + (self.kernel_size - 1) * sum(2**i for i in range(self.num_blocks))


class TCNDecoder(layers.Layer):
    """TCN-based decoder for sequence reconstruction."""
    
    def __init__(
        self, 
        num_blocks: int, 
        filters: int, 
        kernel_size: int, 
        output_dim: int,
        sequence_length: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize TCN decoder.
        
        Args:
            num_blocks: Number of temporal blocks
            filters: Number of filters per block
            kernel_size: Kernel size for convolutions
            output_dim: Dimension of output features
            sequence_length: Length of output sequences
            dropout_rate: Dropout rate for regularization
        """
        super(TCNDecoder, self).__init__(**kwargs)
        self.num_blocks = num_blocks
        self.filters = filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.dropout_rate = dropout_rate
        
        # Expand latent representation
        self.decoder_dense = layers.Dense(sequence_length * filters, activation='relu')
        self.decoder_reshape = layers.Reshape((sequence_length, filters))
        
        # Temporal blocks with decreasing dilation
        self.temporal_blocks = []
        for i in range(num_blocks):
            dilation_rate = 2 ** (num_blocks - 1 - i)
            self.temporal_blocks.append(
                TemporalBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    dropout_rate=dropout_rate,
                    name=f'decoder_block_{i}'
                )
            )
        
        # Output projection
        self.output_conv = layers.Conv1D(
            filters=output_dim,
            kernel_size=1,
            padding='same',
            activation='linear',
            name='output_projection'
        )
    
    def call(self, inputs, training=None):
        """Forward pass through TCN decoder."""
        # Expand latent representation to sequence
        x = self.decoder_dense(inputs, training=training)
        x = self.decoder_reshape(x)
        
        # Pass through temporal blocks
        for block in self.temporal_blocks:
            x = block(x, training=training)
        
        # Project to output dimension
        return self.output_conv(x, training=training)
