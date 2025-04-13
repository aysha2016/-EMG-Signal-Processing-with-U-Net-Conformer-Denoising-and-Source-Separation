# EMG-Signal-Processing-with-U-Net-Conformer-Denoising-and-Source-Separation

 
  EMG U-Net + Conformer Model for Source Separation

This repository contains a pipeline for processing EMG (Electromyography) data with a focus on denoising and source separation using a U-Net + Conformer model.

## Steps Included:

### 1. Mount Google Drive
Mounts the user's Google Drive to access EMG data stored there.

### 2. Load and Preprocess EMG Data
Loads CSV files, normalizes the data using MinMaxScaler, and prepares the EMG data for training.

### 3. Autoencoder for Denoising
Builds and trains an autoencoder model for denoising the EMG signals to improve the quality of the input data for the U-Net + Conformer model.

### 4. U-Net + Conformer Model for Source Separation
Combines a U-Net architecture with a Conformer block to improve the separation of sources from the EMG signals.

### 5. Tracking Training with TensorBoard
Integrates TensorBoard to monitor the training process of both models.

### 6. Evaluation
Uses standard metrics like Mean Squared Error (MSE) and R-squared to evaluate the model's performance.

### 7. Export to ONNX and TFLite
Exports the trained model to both ONNX and TFLite formats, enabling deployment across different platforms.

Additionally, the script includes options to download or move the model to Google Drive and integrates a simple TensorBoard setup to monitor the training process.

## How to Run

1. Upload your EMG data to Google Drive.
2. Modify the `data_path` variable in the script to point to your EMG data directory.
3. Run the script in a Colab notebook.
4. Monitor the training process using TensorBoard.
5. The trained model will be saved in both ONNX and TFLite formats.

## Requirements
- TensorFlow
- tf2onnx
- onnxruntime
- Google Colab
