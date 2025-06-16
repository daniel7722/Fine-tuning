import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

ds_train, ds_info = tfds.load(
    'imagenette/160px',
    split='train',
    as_supervised=True,
    with_info=True
)

ds_val = tfds.load(
    'imagenette/160px',
    split='validation',
    as_supervised=True
)

def preprocess(image, label):
    image = tf.image.resize(image, (160, 160))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

ds_train = (
    ds_train
      .map(preprocess)
      .map(augment)
      .shuffle(1000)
      .batch(32)
      .prefetch(tf.data.AUTOTUNE)
)

ds_val = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)
]

model.fit(ds_train, validation_data=ds_val, epochs=5, callbacks=callbacks)

# Fine-tuning the model by unfreezing the last 20 layers
fine_tuned_callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
]
for layer in model.layers[-20:]:
    layer.trainable = True  # Unfreeze the last 20 layers for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
model.fit(ds_train, validation_data=ds_val, epochs=3, callbacks=fine_tuned_callbacks)

"""
The original imagenet has 1000 classes but the imagenette dataset has only 10 classes. Therefore, the model is frozen 
for the first 3 epochs to avoid overfitting or catastrophic forgetting on smaller dataset. Then, the last 20 layers are
unfrozen for fine-tuning the model with much smaller learning rate. Since imagenette is seriously small compared to imagenet,
data augmentation is applied to the training dataset to improve generalization.
Results of this model training is: 

Epoch 1/5
2025-06-16 15:46:01.722261: I tensorflow/core/kernels/data/tf_record_dataset_op.cc:387] The default buffer size is 262144, which is overridden by the user specified `buffer_size` of 8388608
403/403 ━━━━━━━━━━━━━━━━━━━━ 35s 83ms/step - accuracy: 0.8800 - loss: 0.4280 - val_accuracy: 0.9680 - val_loss: 0.0968 - learning_rate: 0.0010
Epoch 2/5
403/403 ━━━━━━━━━━━━━━━━━━━━ 33s 82ms/step - accuracy: 0.9867 - loss: 0.0504 - val_accuracy: 0.9680 - val_loss: 0.0992 - learning_rate: 0.0010
Epoch 3/5
403/403 ━━━━━━━━━━━━━━━━━━━━ 33s 82ms/step - accuracy: 0.9919 - loss: 0.0308 - val_accuracy: 0.9680 - val_loss: 0.1024 - learning_rate: 5.0000e-04
Epoch 1/3
403/403 ━━━━━━━━━━━━━━━━━━━━ 43s 102ms/step - accuracy: 0.9713 - loss: 0.1061 - val_accuracy: 0.9700 - val_loss: 0.0892 - learning_rate: 1.0000e-05
Epoch 2/3
403/403 ━━━━━━━━━━━━━━━━━━━━ 41s 101ms/step - accuracy: 0.9802 - loss: 0.0742 - val_accuracy: 0.9740 - val_loss: 0.0845 - learning_rate: 1.0000e-05
Epoch 3/3
403/403 ━━━━━━━━━━━━━━━━━━━━ 41s 103ms/step - accuracy: 0.9825 - loss: 0.0622 - val_accuracy: 0.9740 - val_loss: 0.0849 - learning_rate: 1.0000e-05
"""
