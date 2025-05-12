import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import signal
import os
import json

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
        """
        Initialize EMG signal preprocessor.
        
        Args:
            sampling_rate: Sampling rate of the EMG signals in Hz
            notch_freq: Notch filter frequency (usually 50/60 Hz for power line noise)
            bandpass_low: Lower cutoff frequency for bandpass filter
            bandpass_high: Upper cutoff frequency for bandpass filter
            window_size: Size of the sliding window for segmentation
            overlap: Overlap between consecutive windows (0 to 1)
            scaler_type: Type of scaling to apply ('minmax' or 'standard')
        """
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
        """Initialize filter coefficients for signal processing."""
        # Notch filter
        nyquist = self.sampling_rate / 2
        notch_freq_norm = self.notch_freq / nyquist
        self.notch_b, self.notch_a = signal.iirnotch(notch_freq_norm, 30)
        
        # Bandpass filter
        low_norm = self.bandpass_low / nyquist
        high_norm = self.bandpass_high / nyquist
        self.bandpass_b, self.bandpass_a = signal.butter(4, [low_norm, high_norm], btype='band')
    
    def apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove power line noise."""
        return signal.filtfilt(self.notch_b, self.notch_a, data)
    
    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to remove unwanted frequencies."""
        return signal.filtfilt(self.bandpass_b, self.bandpass_a, data)
    
    def apply_rectification(self, data: np.ndarray) -> np.ndarray:
        """Apply full-wave rectification to the signal."""
        return np.abs(data)
    
    def apply_envelope(self, data: np.ndarray) -> np.ndarray:
        """Extract signal envelope using low-pass filter."""
        # Use a low-pass filter to get the envelope
        nyquist = self.sampling_rate / 2
        cutoff = 10 / nyquist  # 10 Hz cutoff for envelope
        b, a = signal.butter(4, cutoff, btype='low')
        return signal.filtfilt(b, a, data)
    
    def segment_signal(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Segment the signal into overlapping windows.
        
        Args:
            data: Input signal of shape (n_samples, n_channels)
            labels: Optional labels for each sample
            
        Returns:
            Tuple of (segmented_data, segmented_labels)
        """
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
                # Use majority voting for window label
                window_labels = labels[start_idx:end_idx]
                segmented_labels[i] = np.bincount(window_labels.astype(int)).argmax()
        
        return segmented_data, segmented_labels
    
    def preprocess_signal(
        self,
        data: np.ndarray,
        apply_rectification: bool = True,
        apply_envelope: bool = True
    ) -> np.ndarray:
        """
        Apply complete preprocessing pipeline to the signal.
        
        Args:
            data: Raw EMG signal of shape (n_samples, n_channels)
            apply_rectification: Whether to apply rectification
            apply_envelope: Whether to extract signal envelope
            
        Returns:
            Preprocessed signal
        """
        # Apply filters
        data = self.apply_notch_filter(data)
        data = self.apply_bandpass_filter(data)
        
        if apply_rectification:
            data = self.apply_rectification(data)
        
        if apply_envelope:
            data = self.apply_envelope(data)
        
        return data
    
    def fit_scaler(self, data: np.ndarray):
        """Fit the scaler on the data."""
        self.scaler.fit(data.reshape(-1, data.shape[-1]))
    
    def transform_scaler(self, data: np.ndarray) -> np.ndarray:
        """Apply scaling to the data."""
        original_shape = data.shape
        scaled_data = self.scaler.transform(data.reshape(-1, data.shape[-1]))
        return scaled_data.reshape(original_shape)
    
    def save_preprocessing_config(self, save_path: str):
        """Save preprocessing configuration to a JSON file."""
        config = {
            'sampling_rate': self.sampling_rate,
            'notch_freq': self.notch_freq,
            'bandpass_low': self.bandpass_low,
            'bandpass_high': self.bandpass_high,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'scaler_type': self.scaler_type
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=4)
    
    @classmethod
    def load_preprocessing_config(cls, config_path: str) -> 'EMGPreprocessor':
        """Load preprocessing configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(**config)

def load_emg_data(
    data_dir: str,
    file_pattern: str = '*.csv',
    channels: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EMG data from CSV files.
    
    Args:
        data_dir: Directory containing the data files
        file_pattern: Pattern to match data files
        channels: List of channel indices to load (None for all channels)
        
    Returns:
        Tuple of (data, labels)
    """
    import glob
    
    all_data = []
    all_labels = []
    
    for file_path in glob.glob(os.path.join(data_dir, file_pattern)):
        # Extract label from filename (assuming format: label_*.csv)
        label = os.path.basename(file_path).split('_')[0]
        
        # Load data
        df = pd.read_csv(file_path)
        data = df.values
        
        if channels is not None:
            data = data[:, channels]
        
        all_data.append(data)
        all_labels.append(np.full(len(data), label))
    
    # Concatenate all data
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y

def create_data_generator(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True
) -> tf.keras.utils.Sequence:
    """
    Create a data generator for training.
    
    Args:
        X: Input data of shape (n_samples, sequence_length, n_channels)
        y: Labels of shape (n_samples,)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        
    Returns:
        A Keras Sequence object for data generation
    """
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