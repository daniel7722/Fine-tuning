import glob, random
import os
import tensorflow as tf
import numpy as np

BATCH_SIZE = 32
VALIDATION_LIMIT = 200  # maximum number of validation images to load
VALIDATION_PER_CLASS = 50
TRAIN_DIR = "data/imagenette/train"

# Build a mapping from class names to integer labels based on Imagenette folder names
class_names = sorted(
    [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
)
class_to_idx = {name: idx for idx, name in enumerate(class_names)}


def load_and_preprocess(image_path):
    """
    Load a JPEG image from disk and preprocess it:
    - Decode JPEG
    - Resize to 160x160
    - Cast to float32
    - Normalize to [-1, 1] range
    """
    image_data = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.image.resize(image, (160, 160))
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image.numpy()


def load_label(image_path):
    """
    Derive the class label from the parent directory name of the image_path.
    """
    basename = os.path.basename(image_path)
    class_name = basename.split("_", 1)[0]
    return class_to_idx.get(class_name, 0)  # Return -1 if class not found


def data_loader_for(agent_id):
    """
    Returns a Python generator or iterable that yields
    (imgs, targets) tuples for the give agent's training split.
    Each call to next() should give you one batch.
    """
    img_paths = sorted(glob.glob(f"data/uniform_split/agent_{agent_id}/train/*.JPEG"))
    random.shuffle(img_paths)

    while True:
        batch_paths = img_paths[:BATCH_SIZE]
        img_paths = img_paths[BATCH_SIZE:] + batch_paths
        images = [load_and_preprocess(p) for p in batch_paths]
        targets = [load_label(p) for p in batch_paths]

        yield np.stack(images), np.array(targets)


def get_validation_set():
    """
    Returns an iterator over the shared validation set.
    """
    val_dir = "data/imagenette/val"
    img_paths = []
    for class_name in class_to_idx.keys():
        class_dir = os.path.join(val_dir, class_name)
        files = sorted(glob.glob(f"{class_dir}/*.JPEG"))
        img_paths.extend(files[:VALIDATION_PER_CLASS])

    for i in range(0, len(img_paths), BATCH_SIZE):
        batch_paths = img_paths[i : i + BATCH_SIZE]
        images = [load_and_preprocess(p) for p in batch_paths]
        targets = [load_label(p) for p in batch_paths]
        yield np.stack(images), np.array(targets)
