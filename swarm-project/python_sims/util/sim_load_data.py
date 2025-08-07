import tensorflow as tf

def extract_vision_dataset(data):
    return data.map(lambda x: (x["vision_data"], x["label"]), num_parallel_calls=tf.data.AUTOTUNE)

def extract_audio_dataset(data):
    return data.map(lambda x: (x["audio_data"], x["label"]), num_parallel_calls=tf.data.AUTOTUNE)

def parse_tfrecord(example_proto):
    feature_description = {
        "video_id": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "vision_data": tf.io.FixedLenFeature([224 * 224 * 3], tf.float32),
        "audio_data": tf.io.FixedLenFeature([100 * 64], tf.float32),
    }
    parsed_example = tf.io.parse_single_example(example_proto, feature_description)
    vision_data = tf.reshape(parsed_example["vision_data"], [224, 224, 3])
    audio_data = tf.reshape(parsed_example["audio_data"], [100, 64])
    return {
        "video_id": parsed_example["video_id"],
        "vision_data": vision_data,
        "audio_data": audio_data,
        "label": parsed_example["label"]
    }

def load_data(batch_size=32):
    def load_split(file_path, is_train):
        dataset = tf.data.TFRecordDataset(file_path)
        dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        if is_train:
            dataset = dataset.shuffle(1000)
        return dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    train_dataset = load_split("./data/AVE_Dataset/processed/train.tfrecord", is_train=True)
    val_dataset = load_split("./data/AVE_Dataset/processed/val.tfrecord", is_train=False)
    test_dataset = load_split("./data/AVE_Dataset/processed/test.tfrecord", is_train=False)
    return train_dataset, val_dataset, test_dataset