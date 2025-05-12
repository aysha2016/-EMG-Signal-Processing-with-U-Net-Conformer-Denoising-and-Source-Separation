# EMG UNet-Conformer

A hybrid deep learning model combining UNet and Conformer architectures for electromyography (EMG) signal processing and analysis. This implementation provides a powerful framework for tasks such as denoising, segmentation, and feature extraction from EMG signals.

## Features

- **Hybrid Architecture**: Combines the spatial encoding power of UNet with the sequence modeling capabilities of Conformers
- **Comprehensive Preprocessing**: Includes notch filtering, bandpass filtering, rectification, and envelope extraction
- **Flexible Training Pipeline**: Supports various training configurations with TensorBoard integration
- **Data Augmentation**: Built-in data generation with customizable windowing and overlap
- **Model Checkpointing**: Automatic saving of best models during training
- **Early Stopping**: Prevents overfitting with configurable patience

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emg_unet_conformer.git
cd emg_unet_conformer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
emg_unet_conformer/
├── data/
│   └── preprocessing.py    # EMG signal preprocessing utilities
├── models/
│   └── unet_conformer.py   # UNet-Conformer model implementation
├── train.py               # Training script
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Usage

### Data Preparation

Your EMG data should be organized as follows:
```
data_directory/
├── activity1_*.csv
├── activity2_*.csv
└── activity3_*.csv
```

Each CSV file should contain EMG signals with samples in rows and channels in columns.

### Training

To train the model, use the `train.py` script with appropriate arguments:

```bash
python train.py \
    --data_dir /path/to/emg/data \
    --file_pattern "*.csv" \
    --sampling_rate 1000 \
    --window_size 1000 \
    --batch_size 32 \
    --epochs 100 \
    --output_dir outputs \
    --experiment_name my_experiment
```

### Key Parameters

#### Data Parameters
- `--data_dir`: Directory containing EMG data files
- `--file_pattern`: Pattern to match data files (default: "*.csv")
- `--channels`: List of channel indices to use (default: all channels)

#### Preprocessing Parameters
- `--sampling_rate`: Sampling rate in Hz (default: 1000)
- `--notch_freq`: Notch filter frequency (default: 50)
- `--bandpass_low`: Lower cutoff frequency (default: 20)
- `--bandpass_high`: Upper cutoff frequency (default: 500)
- `--window_size`: Size of sliding window (default: 1000)
- `--overlap`: Window overlap (default: 0.5)
- `--scaler_type`: Scaling type ("minmax" or "standard")

#### Model Parameters
- `--num_filters`: Filter sizes for each level (default: [64, 128, 256, 512])
- `--num_conformer_blocks`: Number of Conformer blocks (default: 2)
- `--num_heads`: Number of attention heads (default: 8)
- `--ff_dim`: Feed-forward dimension (default: 256)
- `--dropout_rate`: Dropout rate (default: 0.1)
- `--use_batch_norm`: Enable batch normalization

#### Training Parameters
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--validation_split`: Validation split ratio (default: 0.2)
- `--early_stopping_patience`: Early stopping patience (default: 10)

### Output

The training script creates an experiment directory with:
- Model checkpoints
- Training configuration
- Preprocessing configuration
- TensorBoard logs
- Training history
- Final model

## Model Architecture

The UNet-Conformer architecture combines:

1. **UNet Encoder**:
   - Multiple convolutional layers with increasing filters
   - Max pooling for downsampling
   - Skip connections for feature preservation

2. **Conformer Blocks**:
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization
   - Residual connections

3. **UNet Decoder**:
   - Upsampling layers
   - Skip connections from encoder
   - Convolutional layers for feature refinement

## Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Filtering**:
   - Notch filter for power line noise removal
   - Bandpass filter for frequency band selection

2. **Signal Processing**:
   - Full-wave rectification
   - Envelope extraction

3. **Data Preparation**:
   - Signal segmentation with overlapping windows
   - Normalization (MinMax or Standard scaling)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
 

 
