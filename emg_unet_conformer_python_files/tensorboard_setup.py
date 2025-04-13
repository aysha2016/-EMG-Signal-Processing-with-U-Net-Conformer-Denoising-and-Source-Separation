
import tensorflow as tf

def setup_tensorboard(log_dir):
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    return tensorboard_cb
