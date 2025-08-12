
import tensorflow as tf

# --- Audio normalization helper (global) ---
_DEF_EPS = 1e-6

def _normalize_waveform(waveform):
    """Normalize a 1-D float waveform per-example to [-1, 1] using peak norm."""
    wf = tf.cast(waveform, tf.float32)
    max_abs = tf.reduce_max(tf.abs(wf))
    denom = tf.maximum(max_abs, tf.constant(_DEF_EPS, dtype=tf.float32))
    return wf / denom

def extract_vision_dataset(data):
    return data.map(lambda x: (x["vision_data"], x["label"]), num_parallel_calls=tf.data.AUTOTUNE)

def extract_audio_dataset(data):
    return data.map(
        lambda x: (_normalize_waveform(x["audio_waveform"]), x["label"]),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

def filter_valid_audio(data):
    return data.filter(lambda x: tf.equal(tf.cast(x["audio_valid"], tf.int32), 1))

def parse_tfrecord(example_proto):
    feature_description = {
        "video_id": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "vision_data": tf.io.FixedLenFeature([224 * 224 * 3], tf.float32),
        "audio_waveform": tf.io.FixedLenFeature([160000], tf.float32),
        "audio_valid": tf.io.FixedLenFeature([], tf.int64),
        "start_time": tf.io.FixedLenFeature([], tf.float32),
        "end_time": tf.io.FixedLenFeature([], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    wav_clip = tf.ensure_shape(parsed_example["audio_waveform"], [160000])
    vision_data = tf.ensure_shape(tf.reshape(parsed_example["vision_data"], [224, 224, 3]), [224, 224, 3])
    return {
        "video_id": parsed_example["video_id"],
        "vision_data": vision_data,
        "audio_waveform": wav_clip,
        "audio_valid": parsed_example["audio_valid"],
        "start_time": parsed_example["start_time"],
        "end_time": parsed_example["end_time"],
        "label": parsed_example["label"]
    }

def load_data():
    def load_split(file_path, is_train):
        dataset = tf.data.TFRecordDataset(file_path)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        if is_train:
            dataset = dataset.shuffle(1000)
        return dataset.prefetch(tf.data.AUTOTUNE)

    train_dataset = load_split("./data/AVE_Dataset/processed/train.tfrecord", is_train=True)
    val_dataset = load_split("./data/AVE_Dataset/processed/val.tfrecord", is_train=False)
    test_dataset = load_split("./data/AVE_Dataset/processed/test.tfrecord", is_train=False)
    return train_dataset, val_dataset, test_dataset