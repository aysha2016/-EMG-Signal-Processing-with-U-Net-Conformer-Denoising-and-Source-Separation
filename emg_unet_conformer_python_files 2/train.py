import os
import argparse
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any

from models.unet_conformer import create_unet_conformer
from data.preprocessing import EMGPreprocessor, load_emg_data, create_data_generator

def parse_args():
    parser = argparse.ArgumentParser(description='Train UNet-Conformer model for EMG signal processing')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the EMG data files')
    parser.add_argument('--file_pattern', type=str, default='*.csv',
                      help='Pattern to match data files')
    parser.add_argument('--channels', type=int, nargs='+', default=None,
                      help='List of channel indices to use (None for all channels)')
    
    # Preprocessing parameters
    parser.add_argument('--sampling_rate', type=int, default=1000,
                      help='Sampling rate of the EMG signals in Hz')
    parser.add_argument('--notch_freq', type=int, default=50,
                      help='Notch filter frequency')
    parser.add_argument('--bandpass_low', type=int, default=20,
                      help='Lower cutoff frequency for bandpass filter')
    parser.add_argument('--bandpass_high', type=int, default=500,
                      help='Upper cutoff frequency for bandpass filter')
    parser.add_argument('--window_size', type=int, default=1000,
                      help='Size of the sliding window for segmentation')
    parser.add_argument('--overlap', type=float, default=0.5,
                      help='Overlap between consecutive windows')
    parser.add_argument('--scaler_type', type=str, default='minmax',
                      choices=['minmax', 'standard'],
                      help='Type of scaling to apply')
    
    # Model parameters
    parser.add_argument('--num_filters', type=int, nargs='+',
                      default=[64, 128, 256, 512],
                      help='List of filter sizes for each encoder/decoder level')
    parser.add_argument('--num_conformer_blocks', type=int, default=2,
                      help='Number of Conformer blocks in the bottleneck')
    parser.add_argument('--num_heads', type=int, default=8,
                      help='Number of attention heads in Conformer blocks')
    parser.add_argument('--ff_dim', type=int, default=256,
                      help='Feed-forward dimension in Conformer blocks')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                      help='Dropout rate for regularization')
    parser.add_argument('--use_batch_norm', action='store_true',
                      help='Whether to use batch normalization')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate for training')
    parser.add_argument('--validation_split', type=float, default=0.2,
                      help='Fraction of data to use for validation')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                      help='Patience for early stopping')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='Directory to save model and training artifacts')
    parser.add_argument('--experiment_name', type=str,
                      default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                      help='Name of the experiment')
    
    return parser.parse_args()

def setup_training(args: argparse.Namespace) -> Dict[str, Any]:
    """Set up training environment and create output directories."""
    # Create output directory
    experiment_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save training configuration
    config = vars(args)
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    # Set up TensorBoard
    log_dir = os.path.join(experiment_dir, 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )
    
    # Set up model checkpoint
    checkpoint_path = os.path.join(experiment_dir, 'checkpoints')
    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, 'model_{epoch:02d}_{val_loss:.4f}.h5'),
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min'
    )
    
    # Set up early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        restore_best_weights=True
    )
    
    return {
        'experiment_dir': experiment_dir,
        'callbacks': [
            tensorboard_callback,
            checkpoint_callback,
            early_stopping_callback
        ]
    }

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up training environment
    training_setup = setup_training(args)
    
    # Load and preprocess data
    print("Loading data...")
    X, y = load_emg_data(
        args.data_dir,
        file_pattern=args.file_pattern,
        channels=args.channels
    )
    
    # Create and configure preprocessor
    preprocessor = EMGPreprocessor(
        sampling_rate=args.sampling_rate,
        notch_freq=args.notch_freq,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        window_size=args.window_size,
        overlap=args.overlap,
        scaler_type=args.scaler_type
    )
    
    # Save preprocessing configuration
    preprocessor.save_preprocessing_config(
        os.path.join(training_setup['experiment_dir'], 'preprocessing_config.json')
    )
    
    # Preprocess data
    print("Preprocessing data...")
    X = preprocessor.preprocess_signal(X)
    X, y = preprocessor.segment_signal(X, y)
    
    # Split data
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    val_size = int(n_samples * args.validation_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Create data generators
    train_generator = create_data_generator(
        X_train, y_train,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_generator = create_data_generator(
        X_val, y_val,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Create model
    print("Creating model...")
    model = create_unet_conformer(
        input_shape=(args.window_size, X.shape[-1]),
        num_filters=args.num_filters,
        num_conformer_blocks=args.num_conformer_blocks,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout_rate=args.dropout_rate,
        use_batch_norm=args.use_batch_norm
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    print("Starting training...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=training_setup['callbacks'],
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(training_setup['experiment_dir'], 'final_model.h5'))
    
    # Save training history
    history_path = os.path.join(training_setup['experiment_dir'], 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f, indent=4)
    
    print(f"Training completed. Results saved to {training_setup['experiment_dir']}")

if __name__ == '__main__':
    main() 