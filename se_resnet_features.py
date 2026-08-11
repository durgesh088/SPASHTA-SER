"""
3-Channel Spectrogram Feature Extraction for SE-ResNet + BiGRU + Attention

Produces a tensor of shape (3, 64, 501) per audio clip:
  Channel 0 : Log-Mel Spectrogram   (64 mel bins x 501 time frames)
  Channel 1 : Delta                  (1st-order derivative of log-mel)
  Channel 2 : Delta-Delta            (2nd-order derivative of log-mel)

Spectrogram parameters
----------------------
  sr        = 16000
  duration  = 4 seconds  (64,000 samples)
  n_mels    = 64
  n_fft     = 1024
  hop_length= 128        → T = ceil(64000 / 128) + 1 = 501
"""

import os
import numpy as np
import librosa

SAMPLE_RATE  = 16000
DURATION     = 4          # seconds
N_MELS       = 64
N_FFT        = 1024
HOP_LENGTH   = 128
TARGET_T     = 501        # expected time frames


# ════════════════════════════════════════════════════════════
#  Core feature extraction
# ════════════════════════════════════════════════════════════

def extract_spectrogram_features(audio, sr, n_mels=N_MELS, n_fft=N_FFT,
                                  hop_length=HOP_LENGTH, target_t=None):
    """
    Extract a 3-channel spectrogram from a raw waveform.

    Parameters
    ----------
    audio : np.ndarray
        Raw audio waveform (1-D, mono).
    sr : int
        Sample rate.
    n_mels : int
        Number of mel filter banks.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop length between STFT frames.
    target_t : int
        Target number of time frames; spectrograms are padded/cropped
        to exactly this width.

    Returns
    -------
    np.ndarray
        Shape (3, n_mels, target_t) — float32.
    """
    min_samples = sr * 4
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))
    
    target_samples = len(audio)
    if target_t is None:
        target_t = int(target_samples // hop_length) + 1
        if target_t <= 0:
            target_t = 1

    # Channel 0: Log-Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)        # (64, ~501)

    # Pad/crop temporal axis to target_t
    log_mel = _match_time(log_mel, target_t)               # (64, 501)

    # Channel 1: Delta (1st-order)
    delta   = librosa.feature.delta(log_mel)               # (64, 501)

    # Channel 2: Delta-Delta (2nd-order)
    delta2  = librosa.feature.delta(log_mel, order=2)      # (64, 501)

    # Stack into 3-channel image
    spec = np.stack([log_mel, delta, delta2], axis=0)      # (3, 64, 501)
    return spec.astype(np.float32)


def _match_time(matrix, target_t):
    """Pad or crop a 2-D matrix along axis=1 to exactly target_t columns."""
    _, T = matrix.shape
    if T < target_t:
        pad_width = target_t - T
        matrix = np.pad(matrix, ((0, 0), (0, pad_width)), mode='constant')
    elif T > target_t:
        matrix = matrix[:, :target_t]
    return matrix


# ════════════════════════════════════════════════════════════
#  Audio augmentation (same as other pipelines)
# ════════════════════════════════════════════════════════════

def augment_audio(audio, sr):
    """Randomly apply one of: pitch shift, time stretch, or additive noise."""
    import random
    method = random.choice([None, "pitch", "stretch", "noise"])

    if method == "pitch":
        steps = random.choice([-2, -1, 1, 2])
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)

    elif method == "stretch":
        rate = random.uniform(0.85, 1.15)
        orig_len = len(audio)
        audio = librosa.effects.time_stretch(y=audio, rate=rate)
        if len(audio) < orig_len:
            audio = np.pad(audio, (0, orig_len - len(audio)))
        else:
            audio = audio[:orig_len]

    elif method == "noise":
        amp = 0.005 * np.random.uniform() * np.amax(np.abs(audio) + 1e-10)
        audio = audio + amp * np.random.normal(size=audio.shape)

    return audio


# ════════════════════════════════════════════════════════════
#  Single-file processor (for parallel extraction)
# ════════════════════════════════════════════════════════════

def process_single_file(fpath, emotion, augment):
    """
    Load a WAV file, peak-normalise, and extract 3-channel spectrograms.

    Returns a list of (feature, label) tuples (1 or 2 depending on augment).
    """
    try:
        audio, sr = librosa.load(fpath, sr=SAMPLE_RATE)

        # Peak-normalise
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        feat = extract_spectrogram_features(audio, sr)
        results = [(feat, emotion)]

        if augment:
            aug = augment_audio(audio.copy(), sr)
            feat_aug = extract_spectrogram_features(aug, sr)
            results.append((feat_aug, emotion))

        return results
    except Exception as e:
        print(f"\n[ERROR] {os.path.basename(fpath)}: {e}")
        return []


# ════════════════════════════════════════════════════════════
#  CLI: Full dataset extraction → se_resnet_seq.pkl
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split
    from joblib import Parallel, delayed

    DATASET_DIR = "hindi_dataset"
    OUTPUT_PKL  = "se_resnet_seq.pkl"
    AUGMENT     = True

    print("=" * 60)
    print(f"  3-CHANNEL SPECTROGRAM EXTRACTION (3x{N_MELS}x{TARGET_T})")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Output  : {OUTPUT_PKL}")
    print("=" * 60)

    # ── Discover files ──
    VALID_EMOTIONS = {
        'anger', 'disgust', 'fear', 'happy',
        'neutral', 'sad', 'sarcastic', 'surprise',
    }
    emotion_to_files = {emo: [] for emo in VALID_EMOTIONS}

    for root, dirs, files in os.walk(DATASET_DIR):
        dir_name = os.path.basename(root).lower()
        if dir_name in VALID_EMOTIONS:
            for f in files:
                if f.lower().endswith(".wav"):
                    emotion_to_files[dir_name].append(os.path.join(root, f))

    all_tasks = []
    for emotion in sorted(emotion_to_files.keys()):
        for fpath in emotion_to_files[emotion]:
            all_tasks.append((fpath, emotion))

    if len(all_tasks) == 0:
        print(f"[ERROR] No .wav files found in '{DATASET_DIR}'.")
    else:
        fpaths   = [t[0] for t in all_tasks]
        emotions = [t[1] for t in all_tasks]

        # ── 80/20 stratified split ──
        train_paths, val_paths, train_y, val_y = train_test_split(
            fpaths, emotions, test_size=0.2, random_state=42, stratify=emotions
        )
        print(f"[INFO] Train files (original) : {len(train_paths)}")
        print(f"[INFO] Val files              : {len(val_paths)}")

        # ── Train set (with augmentation) ──
        print(f"\n[INFO] Extracting train spectrograms (parallel, augment={AUGMENT})...")
        raw_train = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_single_file)(fp, emo, AUGMENT)
            for fp, emo in zip(train_paths, train_y)
        )

        X_train, y_train = [], []
        for res_list in raw_train:
            for feat, emo in res_list:
                X_train.append(feat)
                y_train.append(emo)
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # ── Val set (no augmentation) ──
        print(f"\n[INFO] Extracting val spectrograms (parallel, no augmentation)...")
        raw_val = Parallel(n_jobs=-1, verbose=5)(
            delayed(process_single_file)(fp, emo, False)
            for fp, emo in zip(val_paths, val_y)
        )

        X_val, y_val = [], []
        for res_list in raw_val:
            for feat, emo in res_list:
                X_val.append(feat)
                y_val.append(emo)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        print(f"\n[OK] Train : X={X_train.shape}, y={y_train.shape}")
        print(f"[OK] Val   : X={X_val.shape},   y={y_val.shape}")

        print(f"[OK] Saving to '{OUTPUT_PKL}' ...")
        joblib.dump((X_train, y_train, X_val, y_val), OUTPUT_PKL, compress=0)

        # Verify
        X_tr_c, y_tr_c, X_va_c, y_va_c = joblib.load(OUTPUT_PKL)
        assert X_tr_c.shape == X_train.shape
        assert X_va_c.shape == X_val.shape
        print("[OK] Feature extraction complete!")
