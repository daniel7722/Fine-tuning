
import tensorflow as tf
import os

# Inspect operations in a TensorFlow Lite model
mobilenetv3 = "/home/dan/Desktop/TinyML/coralmicro/Fine-tuning/models/mobileNetV3/mobilenetv3_cats_vs_dogs_edgetpu.tflite"
posenet = "/home/dan/Desktop/TinyML/coralmicro/models/posenet_mobilenet_v1_075_324_324_16_quant_decoder_edgetpu.tflite"

def inspect_op(model_path): 
  interpreter = tf.lite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  ops = interpreter._get_ops_details()
  op_types = set(op['op_name'] for op in ops)
  return op_types

def inspect_size(model_path):
  model_size_bytes = os.path.getsize(model_path)
  return model_size_bytes / 1024  # Return size in KB

# Inspect MobileNetV3 model
mobilenetv3_ops = inspect_op(mobilenetv3)
print("MobileNetV3 Operations:")
for op in mobilenetv3_ops:
    print(f" - {op}")

# # Inspect PoseNet model
# posenet_ops = inspect_op(posenet)
# print("\nPoseNet Operations:")
# for op in posenet_ops:
#     print(f" - {op}")
# The above code is commented out because it has custom operations that may not be supported in the current environment.

# Print model sizes
print("\nModel Sizes:")
mobilenetv3_size = inspect_size(mobilenetv3)
print(f"MobileNetV3 size: {mobilenetv3_size:.2f} KB")
posenet_size = inspect_size(posenet)
print(f"PoseNet size: {posenet_size:.2f} KB")




