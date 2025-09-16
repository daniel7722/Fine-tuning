import os
import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
import argparse
import librosa as lr

from util.sim_load_data import load_data, filter_valid_audio

from moviepy import VideoFileClip

DATA_ROOT = "./data/AVE_Dataset"   
VIDEOS_DIR = os.path.join(DATA_ROOT, "AVE") 
SR = 16000

def load_mp4_audio_librosa(video_path, target_sr=SR):
    """Load audio from MP4 using librosa (audioread/ffmpeg), resampled to target_sr.
    Returns float32 mono waveform in [-1, 1] after centering and peak-normalization.
    """
    y, sr = lr.load(video_path, sr=target_sr, mono=True)
    y = y.astype(np.float32)
    y = y - np.mean(y)
    peak = np.max(np.abs(y)) + 1e-8
    y = y / peak
    return y

def load_mp4_audio_16k(video_path, target_sr=SR):
    """
    Load audio from an MP4 using MoviePy, resampled to target_sr.
    Returns float32 waveform in [-1, 1].
    """
    with VideoFileClip(video_path) as clip:
        audio = clip.audio
        if audio is None:
            raise ValueError(f"No audio track found in {video_path}")
        # MoviePy resamples for us here
        arr = audio.to_soundarray(fps=target_sr)  # shape (T, C) or (T,)
    # Stereo -> mono
    if arr.ndim == 2:
        samples = arr.mean(axis=1).astype(np.float32)
    else:
        samples = arr.astype(np.float32)
    # Center & peak-normalize (debug-friendly)
    samples = samples - np.mean(samples)
    peak = np.max(np.abs(samples)) + 1e-8
    samples = samples / peak
    return samples

def plot_wave_and_spec(x, sr, title, ax_wave, ax_spec):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        ax_wave.set_title(title + " (empty)")
        ax_wave.set_xlabel("sec"); ax_wave.set_ylabel("amp")
        ax_spec.set_title(title + " (empty)")
        ax_spec.set_xlabel("sec"); ax_spec.set_ylabel("freq (arb)")
        return
    t = np.arange(len(x)) / sr
    ax_wave.plot(t, x)
    ax_wave.set_title(title + " (waveform)")
    ax_wave.set_xlabel("sec"); ax_wave.set_ylabel("amp")

    # simple log-mel-ish spec using STFT magnitude (debug purposes)
    stft = tf.signal.stft(x, frame_length=512, frame_step=160, fft_length=512)
    mag = tf.abs(stft).numpy() + 1e-8
    logmag = np.log(mag)
    ax_spec.imshow(logmag.T, origin="lower", aspect="auto",
                   extent=[0, len(x)/sr, 0, sr/2/ (512/2)])
    ax_spec.set_title(title + " (log |STFT|)")
    ax_spec.set_xlabel("sec"); ax_spec.set_ylabel("freq (arb)")


# Helper for consistent slicing and padding
def slice_pad(y, start, end, sr=SR, max_len=SR*10):
    s_i = max(0, int(round(start * sr)))
    e_i = int(round(end * sr)) if end and end > 0 else len(y)
    clip = y[s_i:e_i]
    if len(clip) < max_len:
        clip = np.pad(clip, (0, max_len - len(clip)))
    return clip[:max_len]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vid", type=str, default=None, help="Optional video_id to inspect")
    parser.add_argument("--split", type=str, default="train", choices=["train","val","test"], help="Dataset split to draw from")
    args = parser.parse_args()

    train_data, val_data, test_data = load_data()
    ds = {"train": train_data, "val": val_data, "test": test_data}[args.split]
    ds = filter_valid_audio(ds)

    # Pick a single sample (optionally by id)
    sample = None
    if args.vid:
        for ex in ds:
            if ex["video_id"].numpy().decode("utf-8") == args.vid:
                sample = ex
                break
        if sample is None:
            raise ValueError(f"video_id {args.vid} not found in {args.split} split")
    else:
        sample = next(iter(ds.take(1)))

    vid = sample["video_id"].numpy().decode("utf-8")
    label = int(sample["label"].numpy())
    start = float(sample["start_time"].numpy())
    end   = float(sample["end_time"].numpy())
    y_tf  = sample["audio_waveform"].numpy()  # already 10s window, float32

    mp4_path = os.path.join(VIDEOS_DIR, f"{vid}.mp4")
    if not os.path.exists(mp4_path):
        raise FileNotFoundError(f"MP4 not found at {mp4_path}. Adjust VIDEOS_DIR.")

    # Decode from two independent stacks
    y_mv_full = load_mp4_audio_16k(mp4_path, target_sr=SR)
    y_lr_full = load_mp4_audio_librosa(mp4_path, target_sr=SR)

    # Slice by (start,end) and pad to 10s for apples-to-apples
    y_mv = slice_pad(y_mv_full, start, end)
    y_lr = slice_pad(y_lr_full, start, end)

    # Quick sanity: shapes must match
    assert y_tf.shape == y_mv.shape == y_lr.shape == (SR * 10,)

    def stats(name, y):
        print(f"{name:>8} | mean {y.mean(): .4f}  std {y.std(): .4f}  min {y.min(): .4f}  max {y.max(): .4f}")
    def mae(a,b): return float(np.mean(np.abs(a-b)))
    def snr_db(ref, est):
        num = np.linalg.norm(ref) + 1e-8
        den = np.linalg.norm(ref - est) + 1e-8
        return float(20 * np.log10(num / den))
    def corr(a,b):
        a0 = a - a.mean(); b0 = b - b.mean()
        den = (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-8)
        return float(np.dot(a0, b0) / den)

    print(f"Picked sample: video_id={vid}, label={label}, start={start:.3f}s, end={end:.3f}s  (split={args.split})")
    stats("TFRecord", y_tf)
    stats("MoviePy",  y_mv)
    stats("librosa",  y_lr)
    print("\nPairwise deltas (on the same 10s window):")
    print("  TF vs MoviePy : MAE=%.6f  SNR=%.2f dB  corr=%.5f" % (mae(y_tf,y_mv), snr_db(y_mv,y_tf), corr(y_tf,y_mv)))
    print("  TF vs librosa : MAE=%.6f  SNR=%.2f dB  corr=%.5f" % (mae(y_tf,y_lr), snr_db(y_lr,y_tf), corr(y_tf,y_lr)))
    print("  MP vs librosa : MAE=%.6f  SNR=%.2f dB  corr=%.5f" % (mae(y_mv,y_lr), snr_db(y_lr,y_mv), corr(y_mv,y_lr)))

    # Save audio snippets for listening
    sf.write("mp_moviepy.wav", y_mv, SR)
    sf.write("mp_librosa.wav", y_lr, SR)
    sf.write("mp_tfrecord.wav", y_tf, SR)
    print("Wrote: mp_moviepy.wav, mp_librosa.wav, mp_tfrecord.wav")

    # Plots: 3 rows x 2 cols (waveform, STFT)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    plot_wave_and_spec(y_mv, SR, "MoviePy clip", axes[0,0], axes[0,1])
    plot_wave_and_spec(y_lr, SR, "librosa clip", axes[1,0], axes[1,1])
    plot_wave_and_spec(y_tf, SR, "TFRecord clip", axes[2,0], axes[2,1])
    plt.tight_layout()
    plt.savefig("audio_compare.png", dpi=150)
    print("Saved audio_compare.png")

if __name__ == "__main__":
    main()