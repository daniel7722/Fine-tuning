import tensorflow as tf
import tensorflow_datasets as tfds
from dataset import class_to_idx

def model_factory(): 
    """
    Constructs a MobileNetV2-based classification model for the number of classes
    defined in dataset.class_to_idx.
    """
    num_classes = len(class_to_idx)

    base_model = tf.keras.applications.MobileNetV2(
        weights='imagenet',
        input_shape=(160, 160, 3),
        include_top=False,
        pooling='avg'
    )
    base_model.trainable = False # Freeze the base model

    inputs = base_model.input
    x = base_model.output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model

model = model_factory()

(ds_train, ds_val), ds_info = tfds.load(
    'imagenette/160px',
    split=['train[:80%]', 'train[80%:]'],
    as_supervised=True,
    with_info=True
)

def preprocess(image, label):
    image = tf.image.resize(image, (160, 160))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

ds_train = ds_train.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
ds_val = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.fit(ds_train, validation_data=ds_val, epochs=3)

# Fine-tuning the model by unfreezing the last 20 layers
for layer in model.layers[-20:]:
    layer.trainable = True  # Unfreeze the last 20 layers for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
model.fit(ds_train, validation_data=ds_val, epochs=2)
"""
The original imagenet has 1000 classes but the imagenette dataset has only 10 classes. Therefore, the model is frozen 
for the first 3 epochs to avoid overfitting or catastrophic forgetting on smaller dataset. Then, the last 20 layers are
unfrozen for fine-tuning the model with much smaller learning rate.
Results of this model training is: 

Epoch 1/3
2025-06-16 15:25:42.721550: I tensorflow/core/kernels/data/tf_record_dataset_op.cc:387] The default buffer size is 262144, which is overridden by the user specified `buffer_size` of 8388608
323/323 ━━━━━━━━━━━━━━━━━━━━ 35s 103ms/step - accuracy: 0.8374 - loss: 0.5859 - val_accuracy: 0.9794 - val_loss: 0.0757
Epoch 2/3
323/323 ━━━━━━━━━━━━━━━━━━━━ 35s 108ms/step - accuracy: 0.9870 - loss: 0.0523 - val_accuracy: 0.9818 - val_loss: 0.0659
Epoch 3/3
323/323 ━━━━━━━━━━━━━━━━━━━━ 36s 111ms/step - accuracy: 0.9942 - loss: 0.0294 - val_accuracy: 0.9794 - val_loss: 0.0598
Epoch 1/2
323/323 ━━━━━━━━━━━━━━━━━━━━ 45s 133ms/step - accuracy: 0.9780 - loss: 0.0805 - val_accuracy: 0.9814 - val_loss: 0.0571
Epoch 2/2
323/323 ━━━━━━━━━━━━━━━━━━━━ 42s 131ms/step - accuracy: 0.9912 - loss: 0.0395 - val_accuracy: 0.9822 - val_loss: 0.0566
"""
