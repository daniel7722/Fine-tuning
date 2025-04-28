import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
import tensorflow_datasets as tfds


"""
This file demonstrates the use of transfer learning with MobileNetV3Small
model on the Cats vs Dogs dataset.

This script achieve 0.9868 accuracy and 0.9748 validation accuracy with 0.0720 loss with only 5 epochs.


 
"""

# 1. Load MobileNetV3 without top
base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze base

# 2. Add new classification head
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# 3. Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 4. Load Cats vs Dogs dataset
(ds_train, ds_val), ds_info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
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