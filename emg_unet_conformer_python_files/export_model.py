
import tf2onnx
import tensorflow as tf

def export_to_onnx(model, X, output_path):
    spec = (tf.TensorSpec((None, X.shape[1], 1), tf.float32),)
    model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)
    print(f"✅ ONNX model saved to {output_path}")

def export_to_tflite(model, output_path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    print(f"✅ TFLite model saved to {output_path}")
