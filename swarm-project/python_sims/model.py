from transformers import TFViTModel
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class_names = [
    "n01440764",
    "n02102040",
    "n02979186",
    "n03000684",
    "n03028079",
    "n03394916",
    "n03417042",
    "n03425413",
    "n03445777",
    "n03888257",
]
class_to_idx = {
    "n01440764": 0,
    "n02102040": 1,
    "n02979186": 2,
    "n03000684": 3,
    "n03028079": 4,
    "n03394916": 5,
    "n03417042": 6,
    "n03425413": 7,
    "n03445777": 8,
    "n03888257": 9,
}

class ViTClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        # Load and freeze the TFViT backbone
        self.backbone = TFViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.backbone.trainable = False
        # Classification head
        self.classifier = tf.keras.layers.Dense(num_classes, name="classifier")
        # Metrics trackers
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    def call(self, inputs, training=False):
        # Forward through backbone, get last hidden state
        outputs = self.backbone(pixel_values=inputs, training=training).last_hidden_state
        # Take the [CLS] token (first token)
        cls_token = outputs[:, 0, :]
        return self.classifier(cls_token)

    def train_step(self, data):
        images, labels = data
        with tf.GradientTape() as tape:
            logits = self(images, training=True)
            # Use the loss function passed during compile for flexibility
            loss = self.loss(labels, logits)
        grads = tape.gradient(loss, self.classifier.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.classifier.trainable_variables))
        # Update our metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    def test_step(self, data):
        images, labels = data
        # Forward pass
        logits = self(images, training=False)
        # Compute loss
        loss = self.loss(labels, logits)
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits)
        # Return a dict of metric results
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}

    @property
    def metrics(self):
        # Return our custom metrics for reset at each epoch
        return [self.loss_tracker, self.accuracy_tracker]


def mobilenet_factory():
    """
    Constructs a MobileNetV2-based classification model for the number of classes
    defined in dataset.class_to_idx.
    """
    num_classes = len(class_to_idx)

    base_model = MobileNetV2(
        weights="imagenet", input_shape=(160, 160, 3), include_top=False, pooling="avg"
    )
    base_model.trainable = False  # Freeze the base model

    inputs = base_model.input
    x = base_model.output
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inputs, outputs=outputs)

    return model


def train_vit():
    """
    This train_accuracy < val_accuracy and train_loss > val_loss is due to the regularization applied in the training process, such as drop out.
    This hurts the training accuracy but improves robustness and generalisation, which is the reason why the validation process, without regularization
    has better accuracy and lower loss. 
    Epoch 1/5
    403/403 ━━━━━━━━━━━━━━━━━━━━ 250s 605ms/step - train_accuracy: 0.5267 - train_loss: 1.9424 - val_val_accuracy: 0.9740 - val_val_loss: 0.9560
    Epoch 2/5
    403/403 ━━━━━━━━━━━━━━━━━━━━ 234s 579ms/step - train_accuracy: 0.9833 - train_loss: 0.7728 - val_val_accuracy: 0.9880 - val_val_loss: 0.4423
    Epoch 3/5
    403/403 ━━━━━━━━━━━━━━━━━━━━ 231s 573ms/step - train_accuracy: 0.9907 - train_loss: 0.3700 - val_val_accuracy: 0.9920 - val_val_loss: 0.2662
    Epoch 4/5
    403/403 ━━━━━━━━━━━━━━━━━━━━ 228s 565ms/step - train_accuracy: 0.9931 - train_loss: 0.2268 - val_val_accuracy: 0.9920 - val_val_loss: 0.1874
    Epoch 5/5
    403/403 ━━━━━━━━━━━━━━━━━━━━ 232s 574ms/step - train_accuracy: 0.9933 - train_loss: 0.1610 - val_val_accuracy: 0.9920 - val_val_loss: 0.1444
    """

    num_classes = len(class_to_idx)
    model = ViTClassifier(num_classes)

    # Load datasets
    ds_train, ds_info = tfds.load(
        "imagenette/160px", split="train", as_supervised=True, with_info=True
    )
    ds_val = tfds.load("imagenette/160px", split="validation", as_supervised=True)

    def preprocess(image, label):
        # Resize and normalize for ViT
        image = tf.image.resize(image, (224, 224))
        image = tf.cast(image, tf.float32) / 255.0
        mean = tf.constant([0.5, 0.5, 0.5], shape=[1, 1, 3], dtype=tf.float32)
        std = tf.constant([0.5, 0.5, 0.5], shape=[1, 1, 3], dtype=tf.float32)
        image = (image - mean) / std
        image = tf.transpose(image, perm=[2, 0, 1])  # Change to [C, H, W] format
        return image, label

    ds_train = (
        ds_train
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(1000)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )
    ds_val = (
        ds_val
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.fit(ds_train, validation_data=ds_val, epochs=5)


def train_mobilenet():
    """
    The original imagenet has 1000 classes but the imagenette dataset has only 10
    classes. Therefore, the model is frozen for the first 3 epochs to avoid overfitting or catastrophic forgetting on smaller dataset. Then, the last 20 layers are
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

    model = mobilenet_factory()

    ds_train, ds_info = tfds.load(
        "imagenette/160px", split="train", as_supervised=True, with_info=True
    )

    ds_val = tfds.load("imagenette/160px", split="validation", as_supervised=True)

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
        ds_train.map(preprocess)
        .map(augment)
        .shuffle(1000)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    ds_val = ds_val.map(preprocess).batch(32).prefetch(tf.data.AUTOTUNE)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6),
    ]

    model.fit(ds_train, validation_data=ds_val, epochs=5, callbacks=callbacks)

    # Fine-tuning the model by unfreezing the last 20 layers
    fine_tuned_callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]
    for layer in model.layers[-20:]:
        layer.trainable = True  # Unfreeze the last 20 layers for fine-tuning
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(
        ds_train, validation_data=ds_val, epochs=3, callbacks=fine_tuned_callbacks
    )


def main():
    # Uncomment the model you want to train
    # train_mobilenet()
    train_vit()


if __name__ == "__main__":
    print("GPUs:", tf.config.list_physical_devices("GPU"))
    main()