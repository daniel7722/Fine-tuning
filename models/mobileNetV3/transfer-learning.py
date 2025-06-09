import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
import tensorflow_datasets as tfds

import numpy as np
import os
from PIL import Image


"""
This file demonstrates the use of transfer learning with MobileNetV2
model on the Cats vs Dogs dataset.

This script achieve 0.9868 accuracy and 0.9785 validation accuracy with 0.0610 loss with 5 epochs.
"""

# Load MobileNetV3 without top
base_model = MobileNetV2(
    input_shape=(324, 324, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

# Add new classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(3, activation='softmax') 
])

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# Load Cats vs Dogs dataset
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (324, 324))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

train_ds = ds_train.map(preprocess)
val_ds = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# Load unrelated "none" images from a different dataset
ds_none_raw = tfds.load(
    'cifar10',
    split='train[:10%]',
    as_supervised=True
)

# Relable flower images as class 2 for 'none'
def filter_none_classes(image, label): 
    return tf.logical_and(tf.not_equal(label, 3), tf.not_equal(label, 5))

def relabel_and_preprocess(image, label):
    image = tf.image.resize(image, (324, 324))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, tf.constant(2, dtype=tf.int64)

ds_none = ds_none_raw.filter(filter_none_classes).map(relabel_and_preprocess)
train_ds_combined = train_ds.concatenate(ds_none).repeat().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Estimate total number of training samples
num_train_samples = 18640  # 80% of cats_vs_dogs (23200 total)
num_none_samples = 5000    # 10% of cifar10 (50000 total)
total_samples = num_train_samples + num_none_samples
batch_size = 32
steps_per_epoch = total_samples // batch_size

# 5. Train model
history = model.fit(
    train_ds_combined,
    validation_data=val_ds,
    epochs=5,
    steps_per_epoch=steps_per_epoch
)

# Save the Keras model
model.save("mobilenetv2_catdog.h5")

# Convert to TFLite model with quantization
def representative_data_gen(): 
    for image, _ in ds_train.take(100):
        image = tf.image.resize(image, (324, 324))
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        yield [tf.expand_dims(image, axis=0)]

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
