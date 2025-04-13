
import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2, padding='same')(x)
    x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)

    x = layers.Conv1D(32, 3, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(x)

    return models.Model(inputs, decoded)

def train_autoencoder(X):
    autoencoder = build_autoencoder((X.shape[1], 1))
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=30, batch_size=64, shuffle=True)

    # Return the denoised output
    return autoencoder.predict(X)
