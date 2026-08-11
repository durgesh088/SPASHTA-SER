import os
import time
import streamlit as st
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import librosa
import librosa.display
import joblib
import soundfile as sf
from sklearn.metrics import confusion_matrix, accuracy_score
from original_model import CNN_GRU_Attention
from original_utils import extract_features
from cnn_lstm_model import HybridCNN_LSTM
from cnn_lstm_features import extract_sequential_features
from se_resnet_model import SEResNet_BiGRU_Attention
from se_resnet_features import extract_spectrogram_features

# ============================================================
#  Global Setup and Configuration Constants
#  Directory creation and hardware device selection (GPU/CPU).
# ============================================================
SAVE_DIR = "outputs"
MY_DATASET_DIR = "hindi_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs("same_output", exist_ok=True)
os.makedirs("misclassified_output", exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
#  Runtime CSV Logging Helper
#  Logs per-sample explanation latencies to results/runtime_results.csv.
# ============================================================
def log_runtime_to_csv(filename, predicted_emotion, runtime):
    import csv
    runtime_csv = os.path.join("results", "runtime_results.csv")
    file_exists = os.path.exists(runtime_csv)
    with open(runtime_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "predicted_emotion", "explanation_runtime"])
        writer.writerow([filename, predicted_emotion, runtime])

if "explanation_runtimes" not in st.session_state:
    st.session_state.explanation_runtimes = []


# ============================================================
#  Cached Model, Scaler, and Label Encoder Loaders
#  Loads and caches trained PyTorch weights and preprocessing artifacts.
# ============================================================
@st.cache_resource
def load_model_bigru():
    encoder = joblib.load("original_label_encoder.pkl")
    scaler = joblib.load("original_scaler.pkl")
    model = CNN_GRU_Attention(input_dim=374, num_classes=len(encoder.classes_))
    model.load_state_dict(torch.load("original_best_model.pt", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, scaler, encoder

@st.cache_resource
def load_model_cnn_lstm():
    encoder = joblib.load("cnn_lstm_label_encoder.pkl")
    scaler = joblib.load("cnn_lstm_scaler.pkl")
    input_dim = scaler.mean_.shape[0]
    model = HybridCNN_LSTM(input_dim=input_dim,
                           num_classes=len(encoder.classes_),
                           hidden_dim=128,
                           dropout=0.3)
    model.load_state_dict(torch.load("cnn_lstm_best_model.pt", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, scaler, encoder

# ============================================================
#  Spectrogram Scaler Class
#  Custom 3D tensor channel-wise normalization class for SE-ResNet.
# ============================================================
class SpectrogramScaler:
    def __init__(self):
        self.means = None
        self.stds  = None

    def fit(self, X):
        self.means = X.mean(axis=(0, 2, 3))
        self.stds  = X.std(axis=(0, 2, 3))
        self.stds[self.stds < 1e-8] = 1.0
        return self

    def transform(self, X):
        if X.ndim == 3:
            return ((X - self.means[:, None, None])
                    / self.stds[:, None, None]).astype(np.float32)
        return ((X - self.means[None, :, None, None])
                / self.stds[None, :, None, None]).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


@st.cache_resource
def load_model_se_resnet():
    encoder = joblib.load("se_resnet_label_encoder.pkl")
    scaler = joblib.load("se_resnet_scaler.pkl")  # SpectrogramScaler instance
    model = SEResNet_BiGRU_Attention(num_classes=len(encoder.classes_),
                                     gru_hidden=128, dropout=0.4)
    model.load_state_dict(torch.load("se_resnet_best_model.pt", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, scaler, encoder

model_bigru, scaler_bigru, encoder_bigru = load_model_bigru()
model_cnn_lstm, scaler_cnn_lstm, encoder_cnn_lstm = load_model_cnn_lstm()
model_se_resnet, scaler_se_resnet, encoder_se_resnet = load_model_se_resnet()

# ============================================================
#  Emotion Prediction Functions across Architectures
#  Extracts features, normalizes inputs, and runs PyTorch inference.
# ============================================================
def predict_emotion_bigru(audio, sr):
    x = extract_features(audio, sr).reshape(1, -1)
    x = scaler_bigru.transform(x)
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        output, _ = model_bigru(x)
        prob = torch.softmax(output[0], dim=0).cpu().numpy()
        pred = encoder_bigru.inverse_transform([np.argmax(prob)])[0]
    return pred, prob

def predict_emotion_cnn_lstm(audio, sr):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    feat = extract_sequential_features(audio, sr)  # (161, 125)
    
    feat_flat = feat.T
    feat_sc_flat = scaler_cnn_lstm.transform(feat_flat)
    feat_sc = feat_sc_flat.T
    feat_sc_batch = np.expand_dims(feat_sc, axis=0)  # (1, 161, 125)

    x = torch.tensor(feat_sc_batch, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits, _ = model_cnn_lstm(x)
        prob = torch.softmax(logits[0], dim=0).cpu().numpy()
        pred = encoder_cnn_lstm.inverse_transform([np.argmax(prob)])[0]
    return pred, prob

def predict_emotion_se_resnet(audio, sr):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    feat = extract_spectrogram_features(audio, sr)        # (3, 64, 501)
    feat_sc = scaler_se_resnet.transform(feat)             # (3, 64, 501)
    x = torch.tensor(feat_sc[np.newaxis], dtype=torch.float32).to(DEVICE)  # (1, 3, 64, 501)
    with torch.no_grad():
        logits, _ = model_se_resnet(x)
        prob = torch.softmax(logits[0], dim=0).cpu().numpy()
        pred = encoder_se_resnet.inverse_transform([np.argmax(prob)])[0]
    return pred, prob

# ============================================================
#  SPASHTA Explanation & Salient Audio Reconstruction Engines
#  Implements batched temporal sliding window perturbation logic.
# ============================================================
def reconstruct_audio(y, importance, sr=16000, threshold_ratio=0.6):
    if len(importance) == 0:
        return y.copy()
    max_imp = np.max(importance)
    if max_imp <= 0:
        return y.copy()
    thresh = threshold_ratio * max_imp
    mask_steps = (importance >= thresh).astype(float)
    stride = len(y) // len(importance) if len(importance) > 0 else int(0.1 * sr)
    if stride <= 0:
        stride = 1
    mask = np.zeros_like(y)
    for i, val in enumerate(mask_steps):
        s = i * stride
        e = min(s + stride, len(mask))
        mask[s:e] = val
    mask = np.convolve(mask, np.ones(1000) / 1000, mode="same")
    return y * mask

def reconstruct_audio_from_mask(y, mask_steps, sr=16000):
    if len(mask_steps) == 0:
        return y.copy()
    stride = len(y) // len(mask_steps) if len(mask_steps) > 0 else int(0.1 * sr)
    if stride <= 0:
        stride = 1
    mask = np.zeros_like(y)
    for i, val in enumerate(mask_steps):
        s = i * stride
        e = min(s + stride, len(mask))
        mask[s:e] = val
    mask = np.convolve(mask, np.ones(1000) / 1000, mode="same")
    return y * mask

def calculate_auc(y_values, x_values):
    return sum((y_values[i] + y_values[i+1]) / 2.0 * (x_values[i+1] - x_values[i]) for i in range(len(x_values)-1))

def spashta_explain_bigru(y, baseline_prob, win_len_sec, sr=16000, filename="unknown"):
    import time
    t_start = time.time()

    DURATION = 10
    target_len = sr * DURATION
    if len(y) < target_len:
        y_processed = np.pad(y, (0, target_len - len(y)))
    else:
        y_processed = y[:target_len]

    # Extract unpooled features once
    unpooled = extract_features(y_processed, sr, pool=False)

    stride = int(0.1 * sr)
    win = int(win_len_sec * sr)
    pred_idx = np.argmax(baseline_prob)
    hop_length = 512

    # >>> START OF PERTURBATION TIMING BLOCK <<<
    # We time the entire perturbation loop including feature masking and batch inference
    t_perf_start = time.perf_counter()

    features_batch = []
    t_loop_start = time.time()
    for start in range(0, len(y_processed), stride):
        end = min(start + win, len(y_processed))
        start_frame = int(start // hop_length)
        end_frame = int(end // hop_length)

        # Copy and apply mask
        mfcc_m = unpooled['mfcc_all'].copy(); mfcc_m[:, start_frame:end_frame] = 0.0
        mel_m = unpooled['mel'].copy(); mel_m[:, start_frame:end_frame] = 0.0
        chroma_m = unpooled['chroma'].copy(); chroma_m[:, start_frame:end_frame] = 0.0
        contrast_m = unpooled['contrast'].copy(); contrast_m[:, start_frame:end_frame] = 0.0
        tonnetz_m = unpooled['tonnetz'].copy(); tonnetz_m[:, start_frame:end_frame] = 0.0

        pitches_m = unpooled['pitches'].copy(); pitches_m[:, start_frame:end_frame] = 0.0
        magnitudes_m = unpooled['magnitudes'].copy(); magnitudes_m[:, start_frame:end_frame] = 0.0

        # Pool
        def pool(x): return np.hstack([np.mean(x, axis=1), np.std(x, axis=1)])

        pitch_val = pitches_m[pitches_m > 0]
        energy_val = magnitudes_m[magnitudes_m > 0]

        pitch_feat = np.array([np.mean(pitch_val), np.std(pitch_val)]) if len(pitch_val) > 0 else np.zeros(2)
        energy_feat = np.array([np.mean(energy_val), np.std(energy_val)]) if len(energy_val) > 0 else np.zeros(2)

        feat_approx = np.hstack([
            pool(mfcc_m),
            pool(mel_m),
            pool(chroma_m),
            pool(contrast_m),
            pool(tonnetz_m),
            pitch_feat,
            energy_feat
        ])
        features_batch.append(feat_approx)
    t_loop_end = time.time()

    # Scale and predict in batch
    X_batch = np.vstack(features_batch)
    X_scaled = scaler_bigru.transform(X_batch)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        output, _ = model_bigru(X_tensor)
        probs_masked = torch.softmax(output, dim=1).cpu().numpy()

    t_perf_end = time.perf_counter()
    explanation_runtime = t_perf_end - t_perf_start
    # >>> END OF PERTURBATION TIMING BLOCK <<<

    # Log explanation runtime to CSV and store in session state
    pred_emotion = encoder_bigru.inverse_transform([pred_idx])[0]
    log_runtime_to_csv(filename, pred_emotion, explanation_runtime)
    if "explanation_runtimes" in st.session_state:
        st.session_state.explanation_runtimes.append(explanation_runtime)

    importance = []
    for prob_masked in probs_masked:
        diff = baseline_prob[pred_idx] - prob_masked[pred_idx]
        importance.append(diff)

    importance = np.array(importance)
    total = np.sum(importance)
    norm_importance = (importance / total) * 100 if total > 0 else importance

    total_time = time.time() - t_start
    avg_loop = (t_loop_end - t_loop_start) / len(features_batch) if len(features_batch) > 0 else 0.0

    return norm_importance, total_time, avg_loop

def spashta_explain_cnn_lstm(y, baseline_prob, win_len_sec, sr=16000, filename="unknown"):
    import time
    t_start = time.time()

    DURATION = max(len(y) / sr, 4.0)
    target_len_samples = int(sr * DURATION)
    if len(y) < target_len_samples:
        y_proc = np.pad(y, (0, target_len_samples - len(y)))
    else:
        y_proc = y

    max_val = np.max(np.abs(y_proc))
    if max_val > 0:
        y_proc = y_proc / max_val

    # Extract sequential features once
    feat = extract_sequential_features(y_proc, sr)  # (161, 125)

    stride = int(0.1 * sr)
    win    = int(win_len_sec * sr)
    pred_idx = np.argmax(baseline_prob)
    hop_length = 512

    # >>> START OF PERTURBATION TIMING BLOCK <<<
    # We time the entire perturbation loop including feature masking and batch inference
    t_perf_start = time.perf_counter()

    features_batch = []
    t_loop_start = time.time()

    for start in range(0, len(y_proc), stride):
        end = min(start + win, len(y_proc))
        sf_idx = int(start // hop_length)
        ef_idx = int(end   // hop_length)

        # Copy and mask along columns
        masked = feat.copy()
        masked[:, sf_idx:ef_idx] = 0.0

        # Scale
        masked_flat = masked.T
        masked_sc_flat = scaler_cnn_lstm.transform(masked_flat)
        masked_sc = masked_sc_flat.T

        features_batch.append(masked_sc)

    t_loop_end = time.time()

    # Batch prediction
    X_batch = np.array(features_batch)  # (B, 161, 125)
    X_t     = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits, _ = model_cnn_lstm(X_t)
        probs_masked = torch.softmax(logits, dim=1).cpu().numpy()

    t_perf_end = time.perf_counter()
    explanation_runtime = t_perf_end - t_perf_start
    # >>> END OF PERTURBATION TIMING BLOCK <<<

    # Log explanation runtime to CSV and store in session state
    pred_emotion = encoder_cnn_lstm.inverse_transform([pred_idx])[0]
    log_runtime_to_csv(filename, pred_emotion, explanation_runtime)
    if "explanation_runtimes" in st.session_state:
        st.session_state.explanation_runtimes.append(explanation_runtime)

    importance = np.array([
        baseline_prob[pred_idx] - pm[pred_idx]
        for pm in probs_masked
    ])
    total = np.sum(importance)
    norm_imp = (importance / total) * 100 if total > 0 else importance

    total_time = time.time() - t_start
    avg_loop = ((t_loop_end - t_loop_start) / len(features_batch)
                if features_batch else 0.0)

    return norm_imp, total_time, avg_loop

def spashta_explain_se_resnet(y, baseline_prob, win_len_sec, sr=16000, filename="unknown"):
    import time
    t_start = time.time()

    DURATION = max(len(y) / sr, 4.0)
    target_len_samples = int(sr * DURATION)
    if len(y) < target_len_samples:
        y_proc = np.pad(y, (0, target_len_samples - len(y)))
    else:
        y_proc = y

    max_val = np.max(np.abs(y_proc))
    if max_val > 0:
        y_proc = y_proc / max_val

    # Extract full spectrogram once
    full_spec = extract_spectrogram_features(y_proc, sr)   # (3, 64, 501)

    stride   = int(0.1 * sr)
    win      = int(win_len_sec * sr)
    pred_idx = np.argmax(baseline_prob)
    hop_length = 128  # SE-ResNet uses hop=128

    # >>> START OF PERTURBATION TIMING BLOCK <<<
    # We time the entire perturbation loop including feature masking and batch inference
    t_perf_start = time.perf_counter()

    features_batch = []
    t_loop_start = time.time()

    for start in range(0, len(y_proc), stride):
        end = min(start + win, len(y_proc))
        sf_idx = int(start // hop_length)
        ef_idx = int(end   // hop_length)

        # Mask temporal columns across all 3 channels
        masked = full_spec.copy()
        masked[:, :, sf_idx:ef_idx] = 0.0

        # Scale
        masked_sc = scaler_se_resnet.transform(masked)
        features_batch.append(masked_sc)

    t_loop_end = time.time()

    # Batch prediction
    X_batch = np.array(features_batch)                     # (B, 3, 64, 501)
    X_t     = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits, _ = model_se_resnet(X_t)
        probs_masked = torch.softmax(logits, dim=1).cpu().numpy()

    t_perf_end = time.perf_counter()
    explanation_runtime = t_perf_end - t_perf_start
    # >>> END OF PERTURBATION TIMING BLOCK <<<

    # Log explanation runtime to CSV and store in session state
    pred_emotion = encoder_se_resnet.inverse_transform([pred_idx])[0]
    log_runtime_to_csv(filename, pred_emotion, explanation_runtime)
    if "explanation_runtimes" in st.session_state:
        st.session_state.explanation_runtimes.append(explanation_runtime)

    importance = np.array([
        baseline_prob[pred_idx] - pm[pred_idx]
        for pm in probs_masked
    ])
    total = np.sum(importance)
    norm_imp = (importance / total) * 100 if total > 0 else importance

    total_time = time.time() - t_start
    avg_loop = ((t_loop_end - t_loop_start) / len(features_batch)
                if features_batch else 0.0)

    return norm_imp, total_time, avg_loop

def spashta_explain(y, baseline_prob, win_len_sec, arch_choice, sr=16000, filename="unknown"):
    if arch_choice == "CNN-BiGRU-Attention":
        return spashta_explain_bigru(y, baseline_prob, win_len_sec, sr, filename)
    elif arch_choice == "SE-ResNet-BiGRU":
        return spashta_explain_se_resnet(y, baseline_prob, win_len_sec, sr, filename)
    else:
        return spashta_explain_cnn_lstm(y, baseline_prob, win_len_sec, sr, filename)

def sparsha_importance(y, baseline_prob, arch_choice, sr=16000, filename="unknown"):
    if arch_choice == "CNN-BiGRU-Attention":
        return spashta_explain_bigru(y, baseline_prob, 0.5, sr, filename)[0]
    elif arch_choice == "SE-ResNet-BiGRU":
        return spashta_explain_se_resnet(y, baseline_prob, 0.5, sr, filename)[0]
    else:
        return spashta_explain_cnn_lstm(y, baseline_prob, 0.5, sr, filename)[0]

# ============================================================
#  Visualization and Artifact Saving Helpers
#  Renders audio waveforms, spectrograms, and SPASHTA bar charts.
# ============================================================
def plot_and_save_all(y, sr, pred, importance, true_label, filename, arch_choice, render_to_streamlit=False):
    is_correct = (pred.lower() == true_label.lower())
    output_dir = "same_output" if is_correct else "misclassified_output"
    os.makedirs(output_dir, exist_ok=True)
    
    arch_prefix = "bigru" if arch_choice == "CNN-BiGRU-Attention" else ("seresnet" if arch_choice == "SE-ResNet-BiGRU" else "cnnlstm")
    fname = f"{arch_prefix}_{true_label}_{pred}_{filename}"

    # Waveform
    fig1, ax1 = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    plt.tight_layout()
    path1 = os.path.join(output_dir, f"{fname}_waveform.png")
    plt.savefig(path1)
    if render_to_streamlit:
        st.image(path1)
    plt.close()

    # Spectrogram
    fig2, ax2 = plt.subplots()
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", ax=ax2)
    ax2.set_title("Spectrogram")
    plt.tight_layout()
    path2 = os.path.join(output_dir, f"{fname}_spectrogram.png")
    plt.savefig(path2)
    if render_to_streamlit:
        st.image(path2)
    plt.close()

    # SPASHTA Importance Bar Plot
    fig3, ax3 = plt.subplots()
    t = np.linspace(0, len(importance) * 0.1, len(importance))
    ax3.bar(t, importance, width=0.09, color='red')
    ttl = f"Actual: {true_label} Predicted: {pred} Importance (%)"
    ax3.set_title(ttl)
    ax3.set_xlabel("Time (s)")
    plt.tight_layout()
    path3 = os.path.join(output_dir, f"{fname}_explanation.png")
    plt.savefig(path3)
    if render_to_streamlit:
        st.image(path3)
    plt.close()

    # Masked audio
    masked_audio = reconstruct_audio(y, importance, sr=sr)
    wav_path = os.path.join(output_dir, f"{fname}_masked.wav")
    sf.write(wav_path, masked_audio, sr)

def display_audio_visuals(y, sr):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Waveform")
        fig1, ax1 = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax1)
        ax1.set_title("Waveform")
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
    with col2:
        st.subheader("Spectrogram")
        fig2, ax2 = plt.subplots()
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", ax=ax2)
        ax2.set_title("Spectrogram")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

def run_grid_search(y, prob, arch_choice, sr=16000, filename="unknown"):
    window_sizes = [0.2, 0.4, 0.5, 0.6, 0.8]
    duration = len(y) / sr
    for w in window_sizes:
        st.subheader(f"SPASHTA (Window = {w}s)")
        importance, total_time, avg_loop = spashta_explain(y, prob, w, arch_choice, sr=sr, filename=filename)
        
        st.write(f"Total Time: {total_time:.3f} sec")
        st.write(f"Avg Loop Time: {avg_loop:.5f} sec")
        
        explained_audio = reconstruct_audio(y, importance, sr=sr)
        output_path = os.path.join(SAVE_DIR, f"explained_{w}.wav")
        sf.write(output_path, explained_audio, sr)
        st.audio(output_path)
        
        current_duration = min(len(y) / sr, duration)
        time_axis = np.linspace(0, current_duration, len(importance))
        fig2, ax2 = plt.subplots()
        
        max_bars = 500
        if len(time_axis) > max_bars:
            chunk_size = len(importance) // max_bars
            if chunk_size == 0:
                importance_plot = importance
                time_plot = time_axis
                width = current_duration / len(time_axis) * 0.9
            else:
                importance_plot = importance[: chunk_size * max_bars].reshape(max_bars, chunk_size).mean(axis=1)
                time_plot = np.linspace(0, current_duration, max_bars)
                width = current_duration / max_bars * 0.9
        else:
            importance_plot = importance
            time_plot = time_axis
            width = current_duration / len(time_axis) * 0.9 if len(time_axis) > 0 else 0.01

        ax2.bar(time_plot, importance_plot, color="red", width=width, align="center")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Importance (%)")
        ax2.set_title(f"Importance (Window={w})")
        st.pyplot(fig2)
        plt.close()


# ============================================================
#  Streamlit User Interface Dashboard
#  Manages architecture selection, file inputs, and benchmarking.
# ============================================================
st.title("🎤 Real-Time Emotion Detection App with SPASHTA")

# Model dropdown selection
arch_choice_label = st.selectbox(
    "Choose Architecture",
    ["1. CNN-BiGRU-Attention", "2. CNN-LSTM Model", "3. SE-ResNet-BiGRU-Attention"]
)

if "SE-ResNet" in arch_choice_label:
    arch_key = "SE-ResNet-BiGRU"
elif "BiGRU" in arch_choice_label:
    arch_key = "CNN-BiGRU-Attention"
else:
    arch_key = "CNN-LSTM"

if arch_key == "CNN-BiGRU-Attention":
    encoder = encoder_bigru
    predict_fn = predict_emotion_bigru
elif arch_key == "SE-ResNet-BiGRU":
    encoder = encoder_se_resnet
    predict_fn = predict_emotion_se_resnet
else:
    encoder = encoder_cnn_lstm
    predict_fn = predict_emotion_cnn_lstm

option = st.radio("Select Input Method", ["Upload File", "Record Audio", "Run Test Dataset"])

# Upload File
if option == "Upload File":
    uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
    if uploaded_file:
        path = "uploaded.wav"
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        try:
            y, sr = librosa.load(path, sr=16000)
            pred, prob = predict_fn(y, sr)
            st.success(f"🎯 Predicted Emotion: `{pred}`")
            display_audio_visuals(y, sr)
            run_grid_search(y, prob, arch_key, sr, filename="uploaded.wav")
        except Exception as e:
            st.error(f"❌ Error loading or processing audio file: {e}")

# Record Audio
elif option == "Record Audio":
    import sounddevice as sd
    
    # Query input devices in the background
    selected_device_id = None
    input_devices_found = False
    try:
        devices = sd.query_devices()
        input_devices = [
            i for i, d in enumerate(devices)
            if d['max_input_channels'] > 0
        ]
        if input_devices:
            input_devices_found = True
            default_device_idx = sd.default.device[0]
            if default_device_idx in input_devices:
                selected_device_id = default_device_idx
            else:
                selected_device_id = input_devices[0]
    except Exception as e:
        pass
        
    duration = st.selectbox("🎚️ Select Duration (seconds)", options=[10], index=0)
    
    if st.button("Start Recording"):
        if selected_device_id is None and not input_devices_found:
            st.error("❌ Cannot record: No working input device detected.")
        else:
            status_placeholder = st.empty()
            status_placeholder.info("🎙️ Recording...")
            try:
                audio = sd.rec(
                    int(duration * 16000), 
                    samplerate=16000, 
                    channels=1, 
                    dtype='float32',
                    device=selected_device_id
                )
                sd.wait()
                status_placeholder.success("✅ Recorded!")
                y = audio.flatten()
                sr = 16000
                pred, prob = predict_fn(y, sr)
                st.success(f"🎯 Predicted Emotion: `{pred}`")
                display_audio_visuals(y, sr)
                run_grid_search(y, prob, arch_key, sr, filename="recorded_audio.wav")
            except Exception as e:
                status_placeholder.error(f"❌ Could not record audio: {e}")
                st.warning("Tips: Try selecting a different microphone from the list above, verify that the device is not being blocked by another application, or check system privacy permissions for the microphone.")

# Run Test Dataset
elif option == "Run Test Dataset":
    all_files = []
    for root, _, files in os.walk(MY_DATASET_DIR):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                true_label = os.path.basename(os.path.dirname(path)).lower()
                all_files.append((path, true_label))

    if not all_files:
        st.error(f"No .wav files found in '{MY_DATASET_DIR}' directory.")
    else:
        st.markdown(f"Total test files available: `{len(all_files)}`")
        
        max_val = max(5, len(all_files))
        step_val = 5 if max_val % 5 == 0 else 1
        num_eval = st.slider("Number of files to evaluate", min_value=5, max_value=max_val, value=min(20, max_val), step=step_val)
        generate_expl = st.checkbox("Generate SPASHTA explanation plots (slows down evaluation significantly)", value=False)
        render_plots_ui = st.checkbox("Display explanation plots in browser (caution: slow)", value=False) if generate_expl else False

        
        if st.button("Run Evaluation"):
            import random
            random.seed(42)
            st.session_state.explanation_runtimes = []
            
            # Group by class to ensure a balanced / stratified sample
            from collections import defaultdict
            class_to_files = defaultdict(list)
            for path, true in all_files:
                class_to_files[true].append((path, true))
            
            sampled_files = []
            classes = sorted(class_to_files.keys())
            files_per_class = max(1, num_eval // len(classes))
            
            for cls in classes:
                cls_files = class_to_files[cls]
                sample_size = min(len(cls_files), files_per_class)
                sampled_files.extend(random.sample(cls_files, sample_size))
                
            # Sample remainder if needed
            remaining = [f for f in all_files if f not in sampled_files]
            if len(sampled_files) < num_eval and remaining:
                extra_needed = num_eval - len(sampled_files)
                sampled_files.extend(random.sample(remaining, min(len(remaining), extra_needed)))
                
            random.shuffle(sampled_files)
            
            actual, predicted = [], []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            detail_container = st.expander("📄 Show Detailed Predictions", expanded=True)
            
            for idx, (path, true) in enumerate(sampled_files):
                status_text.text(f"Processing file {idx+1}/{len(sampled_files)}: {os.path.basename(path)}...")
                
                try:
                    y, sr = librosa.load(path, sr=16000)
                    pred, prob = predict_fn(y, sr)
                    actual.append(true)
                    predicted.append(pred)
                    
                    is_correct = (pred.lower() == true.lower())
                    icon = "✅" if is_correct else "⚠️"
                    
                    if generate_expl:
                        importance = sparsha_importance(y, prob, arch_key, sr=sr, filename=os.path.basename(path))
                        plot_and_save_all(y, sr, pred, importance, true, 
                                         os.path.basename(path).replace(".wav", ""),
                                         arch_key,
                                         render_to_streamlit=render_plots_ui)
                        
                    with detail_container:
                        if is_correct:
                            st.markdown(f"{icon} Correct: `{os.path.basename(path)}` | Emotion: `{true}`")
                        else:
                            st.markdown(f"{icon} Mismatch: `{os.path.basename(path)}` | True: `{true}` | Pred: `{pred}`")
                            
                except Exception as e:
                    with detail_container:
                        st.error(f"Error processing {os.path.basename(path)}: {e}")
                        
                progress_bar.progress((idx + 1) / len(sampled_files))
                
            status_text.text("Evaluation complete!")
            
            # Confusion matrix & Accuracy summary
            if actual and predicted:
                acc = accuracy_score(actual, predicted)
                st.subheader("📊 Evaluation Summary")
                st.success(f"Overall Evaluation Accuracy: **{acc * 100:.2f}%** ({sum(np.array(actual) == np.array(predicted))}/{len(actual)})")
                
                cm = confusion_matrix(actual, predicted, labels=encoder.classes_)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d",
                            xticklabels=encoder.classes_,
                            yticklabels=encoder.classes_, cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                plt.tight_layout()
                cm_path = os.path.join(SAVE_DIR, "confusion_matrix.png")
                plt.savefig(cm_path)
                st.image(cm_path)
                plt.close()
                
                # Print and Display Explanation Runtime Statistics
                if generate_expl and "explanation_runtimes" in st.session_state and st.session_state.explanation_runtimes:
                    runtimes = st.session_state.explanation_runtimes
                    total_rt = sum(runtimes)
                    avg_rt = np.mean(runtimes)
                    std_rt = np.std(runtimes)
                    min_rt = np.min(runtimes)
                    max_rt = np.max(runtimes)
                    
                    # Print summary to console (stdout)
                    print("\n" + "=" * 40, flush=True)
                    print("  SPASHTA Explanation Runtime Summary", flush=True)
                    print("=" * 40, flush=True)
                    print(f"Average Runtime: {avg_rt:.4f} s", flush=True)
                    print(f"Std Runtime: {std_rt:.4f} s", flush=True)
                    print(f"Min Runtime: {min_rt:.4f} s", flush=True)
                    print(f"Max Runtime: {max_rt:.4f} s", flush=True)
                    print("=" * 40 + "\n", flush=True)
                    
                    # Render research-paper-ready table in Streamlit
                    st.subheader("⏱️ SPASHTA Computational Overhead Analysis")
                    st.markdown("The table below summarizes the runtime overhead of generating SPASHTA explanations across the processed samples:")
                    
                    import pandas as pd
                    df_summary = pd.DataFrame({
                        "Metric": [
                            "Total Explanation Runtime",
                            "Average Runtime per Sample",
                            "Minimum Runtime",
                            "Maximum Runtime",
                            "Total Samples Processed"
                        ],
                        "Value (seconds)": [
                            f"{total_rt:.4f} s",
                            f"{avg_rt:.4f} s",
                            f"{min_rt:.4f} s",
                            f"{max_rt:.4f} s",
                            f"{len(runtimes)}"
                        ]
                    })
                    st.table(df_summary)



