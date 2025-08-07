import os
import csv
import cv2
import librosa
import numpy as np
from tqdm import tqdm
from moviepy import VideoFileClip
from pathlib import Path
import tensorflow as tf



DATA_DIR = "./data/AVE_Dataset/AVE"
SPLIT_DIR = "./data/AVE_Dataset/splits"
OUTPUT_DIR = "./data/AVE_Dataset/processed"
VISION_SIZE = (224, 224)
AUDIO_SAMPLE_RATE = 16000
AUDIO_FEATURE_DIM = 64
AUDIO_TIMESTEPS = 100
label_to_index = {
    "Church bell": 0, 
    "Male speech, man speaking": 1,
    "Bark": 2, 
    "Fixed-wing aircraft, airplane": 3, 
    "Race car, auto racing": 4, 
    "Female speech, woman speaking": 5, 
    "Helicopter": 6, 
    "Violin, fiddle": 7, 
    "Flute": 8, 
    "Ukulele": 9, 
    "Frying (food)": 10,
    "Truck": 11,
    "Shofar": 12,
    "Motorcycle": 13,
    "Acoustic guitar": 14,
    "Train horn": 15,
    "Clock": 16,
    "Banjo": 17,
    "Goat": 18,
    "Baby cry, infant cry": 19,
    "Bus": 20,
    "Chainsaw": 21,
    "Cat": 22,
    "Horse": 23,
    "Toilet flush": 24,
    "Rodents, rats, mice": 25,
    "Accordion": 26,
    "Mandolin": 27
}

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return None
    mid_frame_idx = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        frame = cv2.resize(frame, VISION_SIZE)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame.astype(np.uint8)
    else:
        return None

def extract_audio_features(video_path):
    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio.to_soundarray(fps=AUDIO_SAMPLE_RATE)
        clip.close()
        if audio is None:
            return None
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono
        audio = audio.astype(np.float32)
        # librosa expects 1D float array
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=AUDIO_SAMPLE_RATE, n_mels=AUDIO_FEATURE_DIM)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Pad or truncate to fixed timesteps
        if mel_spec_db.shape[1] > AUDIO_TIMESTEPS:
            mel_spec_db = mel_spec_db[:, :AUDIO_TIMESTEPS]
        else:
            pad_width = AUDIO_TIMESTEPS - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        return mel_spec_db.T.astype(np.float32)
    except Exception as e:
        print(f"Error processing audio for {video_path}: {e}")
        return None
    
def serialize_example(video_id, label, vision_data, audio_data): 
    feature = {
        "video_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id.encode('utf-8')])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        "vision_data": tf.train.Feature(float_list=tf.train.FloatList(value=vision_data.flatten())),
        "audio_data": tf.train.Feature(float_list=tf.train.FloatList(value=audio_data.flatten()))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def process_split(split_name):
    split_file = os.path.join(SPLIT_DIR, f"{split_name}.csv")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    tfrecord_path = os.path.join(OUTPUT_DIR, f"{split_name}.tfrecord")

    with open(split_file, "r") as f, tf.io.TFRecordWriter(tfrecord_path) as writer:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"Processing {split_name}"):
            video_id = row["VideoID"]
            label = row["Category"]
            label = label_to_index.get(label, -1)
            if label == -1:
                print(f"Unknown label {row['Category']} for video {video_id}, skipping.")
                continue
            video_path = os.path.join(DATA_DIR, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                print(f"Video {video_path} does not exist, skipping.")
                continue

            vision_frame = extract_middle_frame(video_path)
            audio_feat = extract_audio_features(video_path)
            if vision_frame is None or audio_feat is None:
                continue

            serialized = serialize_example(
                video_id=video_id,
                label=int(label),
                vision_data=vision_frame,
                audio_data=audio_feat
            )
            writer.write(serialized)

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)