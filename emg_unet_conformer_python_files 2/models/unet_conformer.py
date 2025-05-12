import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from typing import Tuple, List, Optional
import numpy as np

class ConformerBlock(layers.Layer):
    def __init__(
        self,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super(ConformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=ff_dim,
            dropout=dropout_rate
        )
        
        # Feed-forward network
        self.ffn1 = layers.Dense(ff_dim, activation='relu')
        self.ffn2 = layers.Dense(ff_dim)
        
        # Layer normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Multi-head attention
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn1(out1)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        
        return self.layernorm2(out1 + ffn_output)

class UNetConformer(tf.keras.Model):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        num_filters: List[int] = [64, 128, 256, 512],
        num_conformer_blocks: int = 2,
        num_heads: int = 8,
        ff_dim: int = 256,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        **kwargs
    ):
        super(UNetConformer, self).__init__(**kwargs)
        self.input_shape_ = input_shape
        self.num_filters = num_filters
        self.num_conformer_blocks = num_conformer_blocks
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Encoder
        self.encoder_blocks = []
        for i, filters in enumerate(num_filters[:-1]):
            self.encoder_blocks.append(self._encoder_block(filters, i == 0))
        
        # Bottleneck with Conformer blocks
        self.bottleneck_conv = layers.Conv1D(
            num_filters[-1], 3, padding='same',
            kernel_regularizer=regularizers.l2(1e-4)
        )
        self.bottleneck_bn = layers.BatchNormalization() if use_batch_norm else None
        
        self.conformer_blocks = [
            ConformerBlock(num_heads, ff_dim, dropout_rate)
            for _ in range(num_conformer_blocks)
        ]
        
        # Decoder
        self.decoder_blocks = []
        for i, filters in enumerate(reversed(num_filters[:-1])):
            self.decoder_blocks.append(self._decoder_block(filters, i == len(num_filters)-2))
        
        # Output
        self.output_conv = layers.Conv1D(1, 1, activation='linear')
        
    def _encoder_block(
        self,
        filters: int,
        is_first: bool
    ) -> tf.keras.Sequential:
        block = tf.keras.Sequential([
            layers.Conv1D(
                filters, 3, padding='same',
                kernel_regularizer=regularizers.l2(1e-4)
            ),
            layers.BatchNormalization() if self.use_batch_norm else layers.Layer(),
            layers.ReLU(),
            layers.Conv1D(
                filters, 3, padding='same',
                kernel_regularizer=regularizers.l2(1e-4)
            ),
            layers.BatchNormalization() if self.use_batch_norm else layers.Layer(),
            layers.ReLU(),
            layers.MaxPooling1D(2)
        ])
        return block
    
    def _decoder_block(
        self,
        filters: int,
        is_last: bool
    ) -> tf.keras.Sequential:
        block = tf.keras.Sequential([
            layers.UpSampling1D(2),
            layers.Conv1D(
                filters, 3, padding='same',
                kernel_regularizer=regularizers.l2(1e-4)
            ),
            layers.BatchNormalization() if self.use_batch_norm else layers.Layer(),
            layers.ReLU(),
            layers.Conv1D(
                filters, 3, padding='same',
                kernel_regularizer=regularizers.l2(1e-4)
            ),
            layers.BatchNormalization() if self.use_batch_norm else layers.Layer(),
            layers.ReLU()
        ])
        return block
    
    def call(
        self,
        inputs: tf.Tensor,
        training: bool = False
    ) -> tf.Tensor:
        # Encoder path
        skip_connections = []
        x = inputs
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, training=training)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck_conv(x)
        if self.bottleneck_bn:
            x = self.bottleneck_bn(x, training=training)
        x = tf.nn.relu(x)
        
        # Conformer blocks
        for conformer_block in self.conformer_blocks:
            x = conformer_block(x, training=training)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, training=training)
            x = layers.Concatenate()([x, skip_connections[i]])
        
        return self.output_conv(x)
    
    def build_model(self) -> tf.keras.Model:
        """Build and return a Keras model with the specified input shape."""
        inputs = layers.Input(shape=self.input_shape_)
        outputs = self.call(inputs)
        return models.Model(inputs=inputs, outputs=outputs, name='unet_conformer')

def create_unet_conformer(
    input_shape: Tuple[int, int],
    num_filters: Optional[List[int]] = None,
    num_conformer_blocks: int = 2,
    num_heads: int = 8,
    ff_dim: int = 256,
    dropout_rate: float = 0.1,
    use_batch_norm: bool = True
) -> tf.keras.Model:
    """
    Factory function to create a UNet-Conformer model.
    
    Args:
        input_shape: Tuple of (sequence_length, num_channels)
        num_filters: List of filter sizes for each encoder/decoder level
        num_conformer_blocks: Number of Conformer blocks in the bottleneck
        num_heads: Number of attention heads in Conformer blocks
        ff_dim: Feed-forward dimension in Conformer blocks
        dropout_rate: Dropout rate for regularization
        use_batch_norm: Whether to use batch normalization
    
    Returns:
        A compiled Keras model
    """
    if num_filters is None:
        num_filters = [64, 128, 256, 512]
    
    model = UNetConformer(
        input_shape=input_shape,
        num_filters=num_filters,
        num_conformer_blocks=num_conformer_blocks,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout_rate=dropout_rate,
        use_batch_norm=use_batch_norm
    )
    
    # Build the model
    keras_model = model.build_model()
    
    # Compile the model
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    return keras_model 