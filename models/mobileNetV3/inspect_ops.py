import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="/home/dan/Desktop/TinyML/coralmicro/Fine-tuning/models/mobileNetV3/mobilenetv3_cats_vs_dogs_edgetpu.tflite")
interpreter.allocate_tensors()
ops = interpreter._get_ops_details()
op_types = set(op['op_name'] for op in ops)
print(op_types)
