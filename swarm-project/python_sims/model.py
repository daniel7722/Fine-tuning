import tensorflow as tf
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

    inputs = base_model.input
    x = base_model.output
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model