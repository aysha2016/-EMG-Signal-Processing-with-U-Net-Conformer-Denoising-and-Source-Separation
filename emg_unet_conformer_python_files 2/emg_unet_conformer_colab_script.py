# %% [markdown]
# # EMG UNet-Conformer: Colab Script
# 
# This script implements a hybrid UNet-Conformer model for EMG signal processing and analysis.
# Copy and paste this entire script into a Google Colab cell to run.

# %% [markdown]
# ## Setup and Installation

# %%
# Install required packages
!pip install tensorflow>=2.8.0 numpy>=1.19.5 pandas>=1.3.0 scikit-learn>=0.24.2 matplotlib>=3.4.3 scipy>=1.7.1 tensorboard>=2.8.0 tqdm>=4.62.3 pyyaml>=6.0

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import required libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import signal
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import json
from typing import Tuple, List, Optional, Dict, Union
import glob
from datetime import datetime

# Enable GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# %% [markdown]
# ## Configuration

# %%
# Configuration dictionary
config = {
    # Data parameters
    'data_dir': '/content/drive/MyDrive/emg_data',  # Update with your data directory
    'file_pattern': '*.csv',
    'channels': None,  # Use all channels
    
    # Preprocessing parameters
    'sampling_rate': 1000,
    'notch_freq': 50,
    'bandpass_low': 20,
    'bandpass_high': 500,
    'window_size': 1000,
    'overlap': 0.5,
    'scaler_type': 'minmax',
    
    # Model parameters
    'num_filters': [64, 128, 256, 512],
    'num_conformer_blocks': 2,
    'num_heads': 8,
    'ff_dim': 256,
    'dropout_rate': 0.1,
    'use_batch_norm': True,
    
    # Training parameters
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-4,
    'validation_split': 0.2,
    'early_stopping_patience': 10,
    
    # Output parameters
    'output_dir': '/content/drive/MyDrive/emg_unet_conformer_outputs',
    'experiment_name': datetime.now().strftime('%Y%m%d_%H%M%S')
}

# Create output directory
os.makedirs(os.path.join(config['output_dir'], config['experiment_name']), exist_ok=True)

# Save configuration
with open(os.path.join(config['output_dir'], config['experiment_name'], 'config.json'), 'w') as f:
    json.dump(config, f, indent=4)

# %% [markdown]
# ## Model Implementation

# %%
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
    
    def _encoder_block(self, filters: int, is_first: bool) -> tf.keras.Sequential:
        block = tf.keras.Sequential([
            layers.Conv1D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization() if self.use_batch_norm else layers.Layer(),
            layers.ReLU(),
            layers.Conv1D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization() if self.use_batch_norm else layers.Layer(),
            layers.ReLU(),
            layers.MaxPooling1D(2)
        ])
        return block
    
    def _decoder_block(self, filters: int, is_last: bool) -> tf.keras.Sequential:
        block = tf.keras.Sequential([
            layers.UpSampling1D(2),
            layers.Conv1D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization() if self.use_batch_norm else layers.Layer(),
            layers.ReLU(),
            layers.Conv1D(filters, 3, padding='same', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization() if self.use_batch_norm else layers.Layer(),
            layers.ReLU()
        ])
        return block
    
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
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
        inputs = layers.Input(shape=self.input_shape_)
        outputs = self.call(inputs)
        return models.Model(inputs=inputs, outputs=outputs, name='unet_conformer')

# %% [markdown]
# ## Data Preprocessing

# %%
class EMGPreprocessor:
    def __init__(
        self,
        sampling_rate: int = 1000,
        notch_freq: int = 50,
        bandpass_low: int = 20,
        bandpass_high: int = 500,
        window_size: int = 1000,
        overlap: float = 0.5,
        scaler_type: str = 'minmax'
    ):
        self.sampling_rate = sampling_rate
        self.notch_freq = notch_freq
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.window_size = window_size
        self.overlap = overlap
        self.scaler_type = scaler_type
        
        # Initialize scaler
        self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        
        # Calculate filter coefficients
        self._initialize_filters()
    
    def _initialize_filters(self):
        # Notch filter
        nyquist = self.sampling_rate / 2
        notch_freq_norm = self.notch_freq / nyquist
        self.notch_b, self.notch_a = signal.iirnotch(notch_freq_norm, 30)
        
        # Bandpass filter
        low_norm = self.bandpass_low / nyquist
        high_norm = self.bandpass_high / nyquist
        self.bandpass_b, self.bandpass_a = signal.butter(4, [low_norm, high_norm], btype='band')
    
    def apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        return signal.filtfilt(self.notch_b, self.notch_a, data)
    
    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        return signal.filtfilt(self.bandpass_b, self.bandpass_a, data)
    
    def apply_rectification(self, data: np.ndarray) -> np.ndarray:
        return np.abs(data)
    
    def apply_envelope(self, data: np.ndarray) -> np.ndarray:
        nyquist = self.sampling_rate / 2
        cutoff = 10 / nyquist
        b, a = signal.butter(4, cutoff, btype='low')
        return signal.filtfilt(b, a, data)
    
    def segment_signal(self, data: np.ndarray, labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        n_samples = data.shape[0]
        step_size = int(self.window_size * (1 - self.overlap))
        n_segments = (n_samples - self.window_size) // step_size + 1
        
        segmented_data = np.zeros((n_segments, self.window_size, data.shape[1]))
        segmented_labels = None if labels is None else np.zeros(n_segments)
        
        for i in range(n_segments):
            start_idx = i * step_size
            end_idx = start_idx + self.window_size
            segmented_data[i] = data[start_idx:end_idx]
            if labels is not None:
                window_labels = labels[start_idx:end_idx]
                segmented_labels[i] = np.bincount(window_labels.astype(int)).argmax()
        
        return segmented_data, segmented_labels
    
    def preprocess_signal(self, data: np.ndarray, apply_rectification: bool = True, apply_envelope: bool = True) -> np.ndarray:
        data = self.apply_notch_filter(data)
        data = self.apply_bandpass_filter(data)
        
        if apply_rectification:
            data = self.apply_rectification(data)
        
        if apply_envelope:
            data = self.apply_envelope(data)
        
        return data
    
    def fit_scaler(self, data: np.ndarray):
        self.scaler.fit(data.reshape(-1, data.shape[-1]))
    
    def transform_scaler(self, data: np.ndarray) -> np.ndarray:
        original_shape = data.shape
        scaled_data = self.scaler.transform(data.reshape(-1, data.shape[-1]))
        return scaled_data.reshape(original_shape)

# %% [markdown]
# ## Utility Functions

# %%
def load_emg_data(data_dir: str, file_pattern: str = '*.csv', channels: Optional[List[int]] = None) -> Tuple[np.ndarray, np.ndarray]:
    all_data = []
    all_labels = []
    
    for file_path in glob.glob(os.path.join(data_dir, file_pattern)):
        label = os.path.basename(file_path).split('_')[0]
        df = pd.read_csv(file_path)
        data = df.values
        
        if channels is not None:
            data = data[:, channels]
        
        all_data.append(data)
        all_labels.append(np.full(len(data), label))
    
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y

def plot_emg_signals(data: np.ndarray, labels: np.ndarray, n_samples: int = 5):
    unique_labels = np.unique(labels)
    fig, axes = plt.subplots(len(unique_labels), n_samples, figsize=(15, 3*len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        label_indices = np.where(labels == label)[0]
        sample_indices = np.random.choice(label_indices, n_samples, replace=False)
        
        for j, idx in enumerate(sample_indices):
            ax = axes[i, j] if len(unique_labels) > 1 else axes[j]
            ax.plot(data[idx])
            ax.set_title(f'Label: {label}\\nSample {j+1}')
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def create_data_generator(X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True) -> tf.keras.utils.Sequence:
    class EMGDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, X, y, batch_size, shuffle):
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indexes = np.arange(len(X))
            self.on_epoch_end()
        
        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))
        
        def __getitem__(self, idx):
            batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_X = self.X[batch_indexes]
            batch_y = self.y[batch_indexes]
            return batch_X, batch_y
        
        def on_epoch_end(self):
            if self.shuffle:
                np.random.shuffle(self.indexes)
    
    return EMGDataGenerator(X, y, batch_size, shuffle)

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    ax2.plot(history.history['mae'], label='Training MAE')
    ax2.plot(history.history['val_mae'], label='Validation MAE')
    ax2.set_title('Model MAE')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Main Training Loop

# %%
def main():
    # Load and preprocess data
    print("Loading data...")
    X, y = load_emg_data(
        config['data_dir'],
        file_pattern=config['file_pattern'],
        channels=config['channels']
    )
    
    # Visualize raw signals
    print("\nVisualizing raw signals...")
    plot_emg_signals(X, y)
    
    # Create and configure preprocessor
    preprocessor = EMGPreprocessor(
        sampling_rate=config['sampling_rate'],
        notch_freq=config['notch_freq'],
        bandpass_low=config['bandpass_low'],
        bandpass_high=config['bandpass_high'],
        window_size=config['window_size'],
        overlap=config['overlap'],
        scaler_type=config['scaler_type']
    )
    
    # Preprocess data
    print("\nPreprocessing data...")
    X = preprocessor.preprocess_signal(X)
    X, y = preprocessor.segment_signal(X, y)
    
    # Visualize preprocessed signals
    print("\nVisualizing preprocessed signals...")
    plot_emg_signals(X, y)
    
    # Split data
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * config['validation_split'])
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Create data generators
    train_generator = create_data_generator(
        X_train, y_train,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_generator = create_data_generator(
        X_val, y_val,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    # Create model
    print("\nCreating model...")
    model = UNetConformer(
        input_shape=(config['window_size'], X.shape[-1]),
        num_filters=config['num_filters'],
        num_conformer_blocks=config['num_conformer_blocks'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout_rate=config['dropout_rate'],
        use_batch_norm=config['use_batch_norm']
    )
    
    keras_model = model.build_model()
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='mse',
        metrics=['mae']
    )
    
    # Display model summary
    keras_model.summary()
    
    # Set up callbacks
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config['output_dir'], config['experiment_name'], 'logs')
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config['output_dir'], config['experiment_name'], 'checkpoints', 'model_{epoch:02d}_{val_loss:.4f}.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True
        )
    ]
    
    # Create checkpoint directory
    os.makedirs(os.path.join(config['output_dir'], config['experiment_name'], 'checkpoints'), exist_ok=True)
    
    # Train model
    print("\nStarting training...")
    history = keras_model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history)
    
    # Save final model
    print("\nSaving final model...")
    keras_model.save(os.path.join(config['output_dir'], config['experiment_name'], 'final_model.h5'))
    
    # Save training history
    with open(os.path.join(config['output_dir'], config['experiment_name'], 'training_history.json'), 'w') as f:
        json.dump(history.history, f, indent=4)
    
    print("\nTraining completed!")
    
    # Start TensorBoard
    print("\nStarting TensorBoard...")
    %load_ext tensorboard
    %tensorboard --logdir os.path.join(config['output_dir'], config['experiment_name'], 'logs')

if __name__ == '__main__':
    main() 