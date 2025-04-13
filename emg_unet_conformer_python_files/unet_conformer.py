
import tensorflow as tf
from tensorflow.keras import layers, models

# Conformer Block
def conformer_block(x, num_heads, ff_dim):
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ffn = layers.Dense(ff_dim, activation='relu')(x)
    ffn = layers.Dense(x.shape[-1])(ffn)
    return layers.LayerNormalization()(layers.Add()([x, ffn]))

def build_unet_conformer(input_shape):
    inputs = layers.Input(shape=input_shape)
    c1 = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling1D(2)(c1)

    c2 = layers.Conv1D(128, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling1D(2)(c2)

    b = conformer_block(p2, num_heads=4, ff_dim=128)

    u1 = layers.UpSampling1D(2)(b)
    u1 = layers.Concatenate()([u1, c2])
    c3 = layers.Conv1D(128, 3, activation='relu', padding='same')(u1)

    u2 = layers.UpSampling1D(2)(c3)
    u2 = layers.Concatenate()([u2, c1])
    c4 = layers.Conv1D(64, 3, activation='relu', padding='same')(u2)

    output = layers.Conv1D(1, 1, activation='linear')(c4)

    return models.Model(inputs, output)

def train_unet_conformer(denoised_X, X):
    unet_conformer = build_unet_conformer((X.shape[1], 1))
    unet_conformer.compile(optimizer='adam', loss='mse')
    unet_conformer.fit(denoised_X, X, epochs=30, batch_size=64, shuffle=True)
    return unet_conformer
