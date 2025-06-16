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
unfrozen for fine-tuning the model with much smaller learning rate. Since imagenette is seriously small compared to imagenet,
data augmentation is applied to the training dataset to improve generalization.
Results of this model training is: 

Epoch 1/3
2025-06-16 15:33:46.615598: I tensorflow/core/kernels/data/tf_record_dataset_op.cc:387] The default buffer size is 262144, which is overridden by the user specified `buffer_size` of 8388608
403/403 ━━━━━━━━━━━━━━━━━━━━ 36s 86ms/step - accuracy: 0.8585 - loss: 0.4991 - val_accuracy: 0.9640 - val_loss: 0.1187
Epoch 2/3
403/403 ━━━━━━━━━━━━━━━━━━━━ 34s 85ms/step - accuracy: 0.9855 - loss: 0.0541 - val_accuracy: 0.9640 - val_loss: 0.1327
Epoch 3/3
403/403 ━━━━━━━━━━━━━━━━━━━━ 37s 92ms/step - accuracy: 0.9916 - loss: 0.0313 - val_accuracy: 0.9640 - val_loss: 0.1191
Epoch 1/2
403/403 ━━━━━━━━━━━━━━━━━━━━ 44s 105ms/step - accuracy: 0.9824 - loss: 0.0629 - val_accuracy: 0.9660 - val_loss: 0.1118
Epoch 2/2
403/403 ━━━━━━━━━━━━━━━━━━━━ 41s 102ms/step - accuracy: 0.9854 - loss: 0.0516 - val_accuracy: 0.9680 - val_loss: 0.1093
"""
