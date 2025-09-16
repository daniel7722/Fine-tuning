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
ANNOTATIONS_PATH = "./data/AVE_Dataset/annotations.txt"  # ampersand-separated
EVENT_CONTEXT_SEC = 0.5  # pad this much context on each side of the event window
MIN_EVENT_SEC = 0.96     # at least one VGGish frame worth of audio
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

def load_annotations_map(path=ANNOTATIONS_PATH):
    """Load AVE annotations mapping VideoID -> (category, start_sec, end_sec).
    The annotations file is expected to be ampersand-separated with a header:
    Category&VideoID&Quality&StartTime&EndTime
    """
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path, "r") as f:
        header = f.readline().strip()  # skip header
        for line in f:
            parts = [p.strip() for p in line.strip().split("&")]
            if len(parts) < 5:
                continue
            category, vid, _quality, start_s, end_s = parts[:5]
            try:
                start = float(start_s)
                end = float(end_s)
            except ValueError:
                continue
            mapping[vid] = (category, start, end)
    return mapping

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


# New extract_audio_waveform implementation
def extract_audio_waveform(video_path, start_sec=None, end_sec=None, sample_rate=AUDIO_SAMPLE_RATE):
    """Extract mono waveform at `sample_rate` and crop to [start_sec, end_sec]
    with EVENT_CONTEXT_SEC padding. Ensures at least MIN_EVENT_SEC seconds output.
    Falls back to center 10s if no start/end provided.
    Returns (wave, audio_valid) where audio_valid is 1 if audio is valid, 0 if silent/empty.
    Decoding prefers librosa; if it fails, we fall back to MoviePy.
    """
    # Helper: decode full audio with librosa (preferred)
    def _decode_full_with_librosa(path, sr):
        # load full file (mono), let librosa resample
        y, sr_ret = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32), sr_ret

    # Helper: decode full audio with MoviePy (fallback)
    def _decode_full_with_moviepy(path, sr):
        clip = VideoFileClip(path)
        audio_array = clip.audio.to_soundarray(fps=sr)
        clip.close()
        y = np.mean(audio_array, axis=1).astype(np.float32) if audio_array.ndim == 2 else audio_array.astype(np.float32)
        return y, sr

    try:
        # Try librosa first
        try:
            audio_mono, _ = _decode_full_with_librosa(video_path, sample_rate)
        except Exception:
            audio_mono, _ = _decode_full_with_moviepy(video_path, sample_rate)

        # Normalize to [-1, 1] by peak
        peak = float(np.max(np.abs(audio_mono))) if audio_mono.size else 0.0
        if peak <= 1e-6:
            # silent or nearly so
            return np.full(sample_rate * 10, -1.0, dtype=np.float32), 0  # explicit canary for silence

        audio_mono = (audio_mono / peak).astype(np.float32)

        total_len = audio_mono.shape[0]
        sr = sample_rate

        if start_sec is None or end_sec is None:
            # Fallback: center crop 10s if available
            desired = sr * 10
            if total_len <= desired:
                wave = audio_mono
            else:
                mid = total_len // 2
                half = desired // 2
                wave = audio_mono[max(0, mid - half): mid + half]
        else:
            # Crop to event with padding on both sides
            s = int(round(max(0.0, start_sec - EVENT_CONTEXT_SEC) * sr))
            e = int(round((end_sec + EVENT_CONTEXT_SEC) * sr))
            s = max(0, s)
            e = min(total_len, e)
            if e <= s:
                # degenerate window → small slice around the nominal start
                center = int(round(start_sec * sr))
                s = max(0, center - int(0.5 * sr))
                e = min(total_len, center + int(0.5 * sr))
            wave = audio_mono[s:e]

            # Ensure at least MIN_EVENT_SEC seconds
            min_len = int(round(MIN_EVENT_SEC * sr))
            if wave.shape[0] < min_len:
                pad = min_len - wave.shape[0]
                wave = np.pad(wave, (0, pad), mode="constant")

        # Hard cap / pad to exactly 10 seconds for consistent TFRecord shape
        max_len = sr * 10
        if wave.shape[0] > max_len:
            wave = wave[:max_len]
        elif wave.shape[0] < max_len:
            wave = np.pad(wave, (0, max_len - wave.shape[0]), mode="constant")

        return wave.astype(np.float32), 1
    except Exception as e:
        print(f"Error processing audio for {video_path}: {e}")
        # return explicit silent canary and invalid flag
        return np.full(sample_rate * 10, -1.0, dtype=np.float32), 0

def serialize_example(video_id, label, vision_data, audio_waveform, start_time, end_time, audio_valid):
    feature = {
        "video_id": tf.train.Feature(bytes_list=tf.train.BytesList(value=[video_id.encode('utf-8')])),
        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label)])),
        "vision_data": tf.train.Feature(float_list=tf.train.FloatList(value=vision_data.flatten())),
        "audio_waveform": tf.train.Feature(float_list=tf.train.FloatList(value=audio_waveform.flatten())),
        "start_time": tf.train.Feature(float_list=tf.train.FloatList(value=[float(start_time)])),
        "end_time": tf.train.Feature(float_list=tf.train.FloatList(value=[float(end_time)])),
        "audio_valid": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(audio_valid)])),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def process_split(split_name):
    split_file = os.path.join(SPLIT_DIR, f"{split_name}.csv")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    tfrecord_path = os.path.join(OUTPUT_DIR, f"{split_name}.tfrecord")

    # Load the annotations map once
    ann_map = load_annotations_map(ANNOTATIONS_PATH)

    with open(split_file, "r") as f, tf.io.TFRecordWriter(tfrecord_path) as writer:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc=f"Processing {split_name}"):
            video_id = row.get("VideoID") or row.get("video_id")
            category = row.get("Category") or row.get("category")
            if video_id is None or category is None:
                print(f"Missing fields in split row: {row}")
                continue

            # Try to read start/end from the split CSV first (if present)
            start_s = row.get("StartTime") or row.get("start_time")
            end_s = row.get("EndTime") or row.get("end_time")
            start_sec = float(start_s) if start_s is not None and start_s != "" else None
            end_sec = float(end_s) if end_s is not None and end_s != "" else None

            # Otherwise, look up from the annotations map
            if (start_sec is None or end_sec is None) and video_id in ann_map:
                _cat_ann, s_ann, e_ann = ann_map[video_id]
                start_sec = s_ann if start_sec is None else start_sec
                end_sec = e_ann if end_sec is None else end_sec

            # Label index
            label = label_to_index.get(category, -1)
            if label == -1:
                print(f"Unknown label {category} for video {video_id}, skipping.")
                continue

            video_path = os.path.join(DATA_DIR, f"{video_id}.mp4")
            if not os.path.exists(video_path):
                print(f"Video {video_path} does not exist, skipping.")
                continue

            vision_frame = extract_middle_frame(video_path)
            audio_wave, audio_valid = extract_audio_waveform(video_path, start_sec=start_sec, end_sec=end_sec)
            if vision_frame is None:
                continue
            if audio_wave is None:
                audio_wave = np.zeros(AUDIO_SAMPLE_RATE*10, dtype=np.float32)
                audio_valid = 0

            # If start/end are still None (no annotations found), store sensible defaults
            st_out = 0.0 if start_sec is None else float(start_sec)
            en_out = 10.0 if end_sec is None else float(end_sec)

            serialized = serialize_example(
                video_id=video_id,
                label=int(label),
                vision_data=vision_frame,
                audio_waveform=audio_wave,
                start_time=st_out,
                end_time=en_out,
                audio_valid=audio_valid,
            )
            writer.write(serialized)

    print(f"Finished {split_name} → wrote {tfrecord_path}")

if __name__ == "__main__":
    for split in ["train_pre", "val_pre", "train_fuse", "val", "test"]:
        process_split(split)
    print("TFRecord export complete →", OUTPUT_DIR)