import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

"""
This file demonstrates the use of a pretrained MobileNetV3Small model
on the Imagenette dataset.
"""

# Load MobileNetV3Small with ImageNet pretrained weights
model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    weights='imagenet',
    include_top=True  # 1000-class classifier head
)

# Print model summary
model.summary()
print(f"\nTotal parameters: {model.count_params():,}")
print(f"Output layer shape: {model.output.shape}")

# Load Imagenette dataset
raw_ds, ds_info = tfds.load(
    'imagenette/160px',
    split='train',
    as_supervised=True,
    with_info=True
)

# Preprocessing function (but also return original for plotting)
def preprocess_image(image, label):
    original_image = tf.image.resize(image, (224, 224))
    preprocessed_image = tf.cast(original_image, tf.float32)
    preprocessed_image = preprocess_input(preprocessed_image)
    return preprocessed_image, label, original_image

# Prepare dataset
ds = raw_ds.map(preprocess_image).shuffle(1000).batch(8).prefetch(tf.data.AUTOTUNE)

# Take a small batch and predict
for batch in ds.take(1):
    images, labels, originals = batch
    predictions = model.predict(images, verbose=0)
    decoded = decode_predictions(predictions, top=3)

    plt.figure(figsize=(16, 8))
    for i in range(len(images)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(originals[i].numpy().astype("uint8"))
        plt.axis('off')
        top1 = decoded[i][0]
        top2 = decoded[i][1]
        plt.title(f"{top1[1]}: {top1[2]*100:.1f}%\n{top2[1]}: {top2[2]*100:.1f}%")
    plt.tight_layout()
    plt.show()