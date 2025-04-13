# EMG Data Processing with U-Net + Conformer

This script trains an Autoencoder for denoising and a U-Net + Conformer model for source separation on EMG data. The trained model is then exported to ONNX and TFLite formats for deployment. Follow the steps below to use the notebook:

### Steps:
1. **Setup**: Install required libraries (`tf2onnx`, `onnxruntime`).
2. **Mount Google Drive**: Mount your Google Drive to access the EMG data.
3. **Load and Preprocess EMG Data**: The data is loaded from CSV files, normalized, and prepared for training.
4. **Autoencoder for Denoising**: An Autoencoder is trained to denoise the EMG signals.
5. **U-Net + Conformer Model**: A U-Net model with a Conformer block is trained to separate sources in the EMG signals.
6. **Evaluation**: Model performance is evaluated using MSE and R-squared.
7. **Export to ONNX and TFLite**: The trained model is exported to ONNX and TFLite formats.
8. **Download**: The models are saved to Google Drive for easy access.

### TensorBoard:
To view the training process, use the following magic command in Colab:
```
%load_ext tensorboard
%tensorboard --logdir /content/logs
```

### Notes:
- Make sure to update the `data_path` to the correct location of your EMG data in Google Drive.
- The EMG data is expected to be in CSV files with each file corresponding to a different activity (e.g., sitting, standing, walking).

Enjoy working with the EMG data! 🚀
