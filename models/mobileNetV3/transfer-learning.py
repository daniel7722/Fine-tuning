import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import tensorflow_datasets as tfds

import numpy as np
import os
from PIL import Image


"""
This script demonstrates transfer learning with MobileNetV2 model on the Cats vs Dogs dataset, 
including preprocessing, model training, and post-traning quantisation for deployment on edge
devices such as the Coral Edge TPU. 
""" 


# MobileNetV2 is loaded without its classification head, and its weight are frozen to leverage
# pretrained feature from ImageNet, allowing transfer learning on a new dataset. 
base_model = MobileNetV2(
    input_shape=(324, 324, 3), # Customised input shape to match EdgeTPU camera spec
    include_top=False,
    weights='imagenet'         # Load pretrained weights from ImageNet
)
base_model.trainable = False  # Freeze base


# The classification head consists of a global acverage pooling layer (to reduce spatial dimensions), 
# a dense intermediate layer (for learning new representations), and a final dense layer with softmax 
# activation for binary classification.
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),  # Intermediate dense layer to learn new representations
    layers.Dense(2, activation='softmax')  # Output layer for 2-class classification (cats vs dogs)
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)



# The dataset is split into training and validation sets (80/20), and images are resized and normalised 
# to match the EdgeTPU camera spec.
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

# Define image preprocessing for both training and validation: 
#  - Resize image to 324x324 to mathch customised input shape
#  - Cast to float32 for model compatibility.
#  - Use MobileNetV2 preprocessing (scaling to [-1, 1])
def preprocess(image, label):
    image = tf.image.resize(image, (324, 324))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

# Prepare the training and validation datasets:
# - .map(preprocess): Applies our preprocessing function to each sample.
# - .shuffle(1000): Randomizes the order of training samples for better generalization.
# - .batch(32): Combines samples into batches for efficient training.
# - .prefetch(tf.data.AUTOTUNE): Overlaps data preprocessing and model execution for performance.
# The order is important: map first (preprocesses each image), shuffle (randomizes), batch (groups),
# then prefetch (pipelines batches to the GPU/CPU efficiently).
train_ds = ds_train.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
)

# Save the Keras model
model.save("mobilenetv2_catdog.h5")

# Define a representative dataset generator for post-training quantisation:
# - Provides sample input data for the converter to calibrate activations and weights.
# - Ensures quantised model accuracy by matching input distribution.
def representative_data_gen(): 
    for image, _ in ds_train.take(100):
        image = tf.image.resize(image, (324, 324))
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        yield [tf.expand_dims(image, axis=0)]

# Convert to a quantizsd TFLite model for EdgeTPU:
# - optimisations=[Optimize.DEFAULT]: Enables post-training quantisation:.
# - representative_dataset: Supplies sample data for calibration.
# - target_spec.supported_ops: Restricts ops to INT8 only for full quantisation:.
# - inference_input_type/inference_output_type: Set to uint8 for EdgeTPU compatibility.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

# Save the TFLite model
with open('mobilenetv2_catdog_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("✅ TFLite model generated!")
