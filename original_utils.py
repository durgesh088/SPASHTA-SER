import os
import librosa
import numpy as np
from tqdm import tqdm
import random

SAMPLE_RATE = 16000
DURATION = 10  # seconds

# ✅ Augment audio with pitch, stretch, or noise
def augment_audio(audio, sr):
    aug_methods = [None, "pitch", "stretch", "noise"]
    method = random.choice(aug_methods)

    if method == "pitch":
        steps = random.choice([-2, -1, 1, 2])
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)

    elif method == "stretch":
        rate = random.uniform(0.85, 1.15)
        input_length = len(audio)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=int(sr * rate))
        if len(audio) < input_length:
            audio = np.pad(audio, (0, input_length - len(audio)))
        else:
            audio = audio[:input_length]

    elif method == "noise":
        noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
        audio = audio + noise_amp * np.random.normal(size=audio.shape)

    return audio


# ✅ Feature extraction (374D)
def extract_features(audio, sr, pool=True):
    if len(audio) < sr * DURATION:
        audio = np.pad(audio, (0, sr * DURATION - len(audio)))
    else:
        audio = audio[:sr * DURATION]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    mfcc_all = np.vstack([mfcc, delta_mfcc, delta2_mfcc])

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

       # Safe Tonnetz extraction
    try:
        harmonic = librosa.effects.harmonic(audio)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
    except Exception:
        tonnetz = np.zeros((6, mfcc.shape[1]))  # fallback



    # Pitch and energy
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
    pitch = pitches[pitches > 0]
    energy = magnitudes[magnitudes > 0]

    pitch_feat = np.array([np.mean(pitch), np.std(pitch)]) if len(pitch) > 0 else np.zeros(2)
    energy_feat = np.array([np.mean(energy), np.std(energy)]) if len(energy) > 0 else np.zeros(2)

    if not pool:
        return {
            'mfcc_all': mfcc_all,
            'mel': mel,
            'chroma': chroma,
            'contrast': contrast,
            'tonnetz': tonnetz,
            'pitches': pitches,
            'magnitudes': magnitudes
        }

    def pool_func(x): return np.hstack([np.mean(x, axis=1), np.std(x, axis=1)])

    return np.hstack([
        pool_func(mfcc_all),
        pool_func(mel),
        pool_func(chroma),
        pool_func(contrast),
        pool_func(tonnetz),
        pitch_feat,
        energy_feat
    ])

# ✅ Dataset loader with original + augmented
def load_data(dataset_path):
    X, y = [], []
    emotions = sorted(os.listdir(os.path.join(dataset_path, "1/session1")))

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path): continue
        for session in os.listdir(folder_path):
            session_path = os.path.join(folder_path, session)
            for emotion in emotions:
                emotion_path = os.path.join(session_path, emotion)
                for file in os.listdir(emotion_path):
                    if file.endswith(".wav"):
                        file_path = os.path.join(emotion_path, file)
                        try:
                            # Original
                            audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                            features = extract_features(audio, sr)
                            X.append(features)
                            y.append(emotion)

                            # Augmented
                            audio_aug = augment_audio(audio.copy(), sr)
                            features_aug = extract_features(audio_aug, sr)
                            X.append(features_aug)
                            y.append(emotion)

                        except Exception as e:
                            print(f"❌ Error in {file_path}: {e}")
    return np.array(X), np.array(y)

