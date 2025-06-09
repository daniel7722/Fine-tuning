import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow_datasets as tfds


"""
This file demonstrates the use of transfer learning with MobileNetV3Small
model on the Cats vs Dogs dataset.

This script achieve 0.9868 accuracy and 0.9785 validation accuracy with 0.0610 loss with 5 epochs.
"""

# Load MobileNetV3 without top
base_model = MobileNetV3Small(
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
    layers.Dense(2, activation='softmax') 
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
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    return image, label

train_ds = ds_train.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

# 5. Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Save the Keras model
model.save("mobilenetv3_cats_vs_dogs.h5")

# Convert to TFLite model with quantization
def representative_data_gen(): 
    for image, _ in ds_train.take(100):
        image = tf.image.resize(image, (324, 324))
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
        yield [tf.expand_dims(image, axis=0)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_quant_model = converter.convert()

# Save the TFLite model
with open('mobilenetv3_cats_vs_dogs_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("✅ TFLite model generated!")
