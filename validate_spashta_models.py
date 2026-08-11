import os
import time
import csv
import joblib
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import model definitions
from original_model import CNN_GRU_Attention
from original_utils import extract_features
from cnn_lstm_model import HybridCNN_LSTM
from cnn_lstm_features import extract_sequential_features
from se_resnet_model import SEResNet_BiGRU_Attention
from se_resnet_features import extract_spectrogram_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ════════════════════════════════════════════════════════════
#  SpectrogramScaler Class (Required for SE-ResNet scaler)
# ════════════════════════════════════════════════════════════
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

# ════════════════════════════════════════════════════════════
#  Model Loaders
# ════════════════════════════════════════════════════════════
def load_bigru_model():
    encoder = joblib.load("original_label_encoder.pkl")
    scaler = joblib.load("original_scaler.pkl")
    model = CNN_GRU_Attention(input_dim=374, num_classes=len(encoder.classes_))
    model.load_state_dict(torch.load("original_best_model.pt", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, scaler, encoder

def load_cnn_lstm_model():
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

def load_se_resnet_model():
    encoder = joblib.load("se_resnet_label_encoder.pkl")
    scaler = joblib.load("se_resnet_scaler.pkl")
    model = SEResNet_BiGRU_Attention(num_classes=len(encoder.classes_),
                                     gru_hidden=128, dropout=0.4)
    model.load_state_dict(torch.load("se_resnet_best_model.pt", map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
    return model, scaler, encoder

# ════════════════════════════════════════════════════════════
#  Predict Functions
# ════════════════════════════════════════════════════════════
def predict_bigru(audio, sr, model, scaler, encoder):
    x = extract_features(audio, sr).reshape(1, -1)
    x = scaler.transform(x)
    x = torch.tensor(x, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        output, _ = model(x)
        prob = torch.softmax(output[0], dim=0).cpu().numpy()
        pred = encoder.inverse_transform([np.argmax(prob)])[0]
    return pred, prob

def predict_cnn_lstm(audio, sr, model, scaler, encoder):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    feat = extract_sequential_features(audio, sr)
    feat_flat = feat.T
    feat_sc_flat = scaler.transform(feat_flat)
    feat_sc = feat_sc_flat.T
    feat_sc_batch = np.expand_dims(feat_sc, axis=0)
    x = torch.tensor(feat_sc_batch, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(x)
        prob = torch.softmax(logits[0], dim=0).cpu().numpy()
        pred = encoder.inverse_transform([np.argmax(prob)])[0]
    return pred, prob

def predict_se_resnet(audio, sr, model, scaler, encoder):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    feat = extract_spectrogram_features(audio, sr)
    feat_sc = scaler.transform(feat)
    x = torch.tensor(feat_sc[np.newaxis], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(x)
        prob = torch.softmax(logits[0], dim=0).cpu().numpy()
        pred = encoder.inverse_transform([np.argmax(prob)])[0]
    return pred, prob

# ════════════════════════════════════════════════════════════
#  Reconstruction Salient Helpers
# ════════════════════════════════════════════════════════════
def reconstruct_audio(y, importance, sr=16000, threshold_ratio=0.5):
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

def reconstruct_audio_inverse(y, importance, sr=16000, threshold_ratio=0.5):
    if len(importance) == 0:
        return np.zeros_like(y)
    max_imp = np.max(importance)
    if max_imp <= 0:
        return np.zeros_like(y)
    thresh = threshold_ratio * max_imp
    mask_steps = (importance < thresh).astype(float)
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

# ════════════════════════════════════════════════════════════
#  SPASHTA Explanations
# ════════════════════════════════════════════════════════════
def spashta_bigru_explain(y, baseline_prob, model, scaler, encoder, win_len_sec=0.5, sr=16000):
    DURATION = 10
    target_len = sr * DURATION
    if len(y) < target_len:
        y_processed = np.pad(y, (0, target_len - len(y)))
    else:
        y_processed = y[:target_len]

    unpooled = extract_features(y_processed, sr, pool=False)
    stride = int(0.1 * sr)
    win = int(win_len_sec * sr)
    pred_idx = np.argmax(baseline_prob)
    hop_length = 512

    t_start = time.perf_counter()
    features_batch = []
    for start in range(0, len(y_processed), stride):
        end = min(start + win, len(y_processed))
        start_frame = int(start // hop_length)
        end_frame = int(end // hop_length)

        mfcc_m = unpooled['mfcc_all'].copy(); mfcc_m[:, start_frame:end_frame] = 0.0
        mel_m = unpooled['mel'].copy(); mel_m[:, start_frame:end_frame] = 0.0
        chroma_m = unpooled['chroma'].copy(); chroma_m[:, start_frame:end_frame] = 0.0
        contrast_m = unpooled['contrast'].copy(); contrast_m[:, start_frame:end_frame] = 0.0
        tonnetz_m = unpooled['tonnetz'].copy(); tonnetz_m[:, start_frame:end_frame] = 0.0
        pitches_m = unpooled['pitches'].copy(); pitches_m[:, start_frame:end_frame] = 0.0
        magnitudes_m = unpooled['magnitudes'].copy(); magnitudes_m[:, start_frame:end_frame] = 0.0

        def pool(x): return np.hstack([np.mean(x, axis=1), np.std(x, axis=1)])

        pitch_val = pitches_m[pitches_m > 0]
        energy_val = magnitudes_m[magnitudes_m > 0]
        pitch_feat = np.array([np.mean(pitch_val), np.std(pitch_val)]) if len(pitch_val) > 0 else np.zeros(2)
        energy_feat = np.array([np.mean(energy_val), np.std(energy_val)]) if len(energy_val) > 0 else np.zeros(2)

        feat_approx = np.hstack([
            pool(mfcc_m), pool(mel_m), pool(chroma_m), pool(contrast_m), pool(tonnetz_m), pitch_feat, energy_feat
        ])
        features_batch.append(feat_approx)

    X_batch = np.vstack(features_batch)
    X_scaled = scaler.transform(X_batch)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        output, _ = model(X_tensor)
        probs_masked = torch.softmax(output, dim=1).cpu().numpy()
    t_end = time.perf_counter()

    importance = np.array([baseline_prob[pred_idx] - pm[pred_idx] for pm in probs_masked])
    return importance, t_end - t_start

def spashta_cnn_lstm_explain(y, baseline_prob, model, scaler, encoder, win_len_sec=0.5, sr=16000):
    DURATION = max(len(y) / sr, 4.0)
    target_len_samples = int(sr * DURATION)
    if len(y) < target_len_samples:
        y_proc = np.pad(y, (0, target_len_samples - len(y)))
    else:
        y_proc = y

    max_val = np.max(np.abs(y_proc))
    if max_val > 0:
        y_proc = y_proc / max_val

    feat = extract_sequential_features(y_proc, sr)
    stride = int(0.1 * sr)
    win    = int(win_len_sec * sr)
    pred_idx = np.argmax(baseline_prob)
    hop_length = 512

    t_start = time.perf_counter()
    features_batch = []
    for start in range(0, len(y_proc), stride):
        end = min(start + win, len(y_proc))
        sf_idx = int(start // hop_length)
        ef_idx = int(end   // hop_length)

        masked = feat.copy()
        masked[:, sf_idx:ef_idx] = 0.0

        masked_flat = masked.T
        masked_sc_flat = scaler.transform(masked_flat)
        masked_sc = masked_sc_flat.T
        features_batch.append(masked_sc)

    X_batch = np.array(features_batch)
    X_t     = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(X_t)
        probs_masked = torch.softmax(logits, dim=1).cpu().numpy()
    t_end = time.perf_counter()

    importance = np.array([baseline_prob[pred_idx] - pm[pred_idx] for pm in probs_masked])
    return importance, t_end - t_start

def spashta_se_resnet_explain(y, baseline_prob, model, scaler, encoder, win_len_sec=0.5, sr=16000):
    DURATION = max(len(y) / sr, 4.0)
    target_len_samples = int(sr * DURATION)
    if len(y) < target_len_samples:
        y_proc = np.pad(y, (0, target_len_samples - len(y)))
    else:
        y_proc = y

    max_val = np.max(np.abs(y_proc))
    if max_val > 0:
        y_proc = y_proc / max_val

    full_spec = extract_spectrogram_features(y_proc, sr)
    stride   = int(0.1 * sr)
    win      = int(win_len_sec * sr)
    pred_idx = np.argmax(baseline_prob)
    hop_length = 128

    t_start = time.perf_counter()
    features_batch = []
    for start in range(0, len(y_proc), stride):
        end = min(start + win, len(y_proc))
        sf_idx = int(start // hop_length)
        ef_idx = int(end   // hop_length)

        masked = full_spec.copy()
        masked[:, :, sf_idx:ef_idx] = 0.0

        masked_sc = scaler.transform(masked)
        features_batch.append(masked_sc)

    X_batch = np.array(features_batch)
    X_t     = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(X_t)
        probs_masked = torch.softmax(logits, dim=1).cpu().numpy()
    t_end = time.perf_counter()

    importance = np.array([baseline_prob[pred_idx] - pm[pred_idx] for pm in probs_masked])
    return importance, t_end - t_start

# ════════════════════════════════════════════════════════════
#  Execution and Evaluation Loop
# ════════════════════════════════════════════════════════════
def run_validation():
    # Setup dataset
    DATASET_DIR = "hindi_dataset"
    VALID_EMOTIONS = {'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'sarcastic', 'surprise'}
    
    emotion_to_files = {emo: [] for emo in VALID_EMOTIONS}
    for root, _, files in os.walk(DATASET_DIR):
        dir_name = os.path.basename(root).lower()
        if dir_name in VALID_EMOTIONS:
            for f in files:
                if f.lower().endswith(".wav"):
                    emotion_to_files[dir_name].append(os.path.join(root, f))
    
    all_files = []
    for emotion in sorted(emotion_to_files.keys()):
        for fpath in emotion_to_files[emotion]:
            all_files.append((fpath, emotion))
            
    fpaths = [t[0] for t in all_files]
    emotions = [t[1] for t in all_files]
    
    _, val_paths, _, val_y = train_test_split(
        fpaths, emotions, test_size=0.2, random_state=42, stratify=emotions
    )
    
    print(f"[INFO] Loaded {len(val_paths)} validation samples.")

    # Load architectures
    print("[INFO] Loading architectures and checkpoints...")
    models = {
        "CNN + BiLSTM": load_cnn_lstm_model(),
        "CNN + BiGRU + Attention": load_bigru_model(),
        "SE-ResNet + BiGRU + Attention": load_se_resnet_model()
    }
    
    predict_fns = {
        "CNN + BiLSTM": predict_cnn_lstm,
        "CNN + BiGRU + Attention": predict_bigru,
        "SE-ResNet + BiGRU + Attention": predict_se_resnet
    }
    
    spashta_fns = {
        "CNN + BiLSTM": spashta_cnn_lstm_explain,
        "CNN + BiGRU + Attention": spashta_bigru_explain,
        "SE-ResNet + BiGRU + Attention": spashta_se_resnet_explain
    }

    results = {}
    
    for arch_name, (model, scaler, encoder) in models.items():
        print(f"\n[INFO] Evaluating classification metrics for: {arch_name}...")
        pred_fn = predict_fns[arch_name]
        
        actual, predicted = [], []
        
        for idx, (path, true) in enumerate(zip(val_paths, val_y)):
            try:
                audio, sr = librosa.load(path, sr=16000)
                pred_emo, _ = pred_fn(audio, sr, model, scaler, encoder)
                actual.append(true)
                predicted.append(pred_emo)
            except Exception as e:
                print(f"[ERROR] Failed predicting {path}: {e}")
                
        # Compute metrics
        acc = accuracy_score(actual, predicted)
        prec = precision_score(actual, predicted, average='weighted', zero_division=0)
        rec = recall_score(actual, predicted, average='weighted', zero_division=0)
        f1 = f1_score(actual, predicted, average='weighted', zero_division=0)
        
        print(f"[OK] {arch_name} classification summary: Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
        
        # Explainability Metrics on a subset of 50 samples
        print(f"[INFO] Evaluating SPASHTA explainability on a subset of 50 samples for {arch_name}...")
        xai_paths = val_paths[:50]
        xai_y = val_y[:50]
        
        runtimes, gains, drops = [], [], []
        explain_fn = spashta_fns[arch_name]
        
        for idx, (path, true) in enumerate(zip(xai_paths, xai_y)):
            try:
                audio, sr = librosa.load(path, sr=16000)
                target_duration = 4 if "BiLSTM" in arch_name else 10
                target_len_samples = int(sr * target_duration)
                if len(audio) < target_len_samples:
                    audio = np.pad(audio, (0, target_len_samples - len(audio)))
                else:
                    audio = audio[:target_len_samples]
                pred_emo, prob_orig = pred_fn(audio, sr, model, scaler, encoder)
                pred_idx = np.argmax(prob_orig)
                
                # Run SPASHTA
                importance, rt = explain_fn(audio, prob_orig, model, scaler, encoder, win_len_sec=0.5, sr=sr)
                runtimes.append(rt)
                
                # Gain & Drop
                y_empty = np.zeros_like(audio)
                _, prob_empty = pred_fn(y_empty, sr, model, scaler, encoder)
                
                y_delete = reconstruct_audio_inverse(audio, importance, sr, threshold_ratio=0.5)
                _, prob_delete = pred_fn(y_delete, sr, model, scaler, encoder)
                
                y_retain = reconstruct_audio(audio, importance, sr, threshold_ratio=0.5)
                _, prob_retain = pred_fn(y_retain, sr, model, scaler, encoder)
                
                drop = prob_orig[pred_idx] - prob_delete[pred_idx]
                gain = prob_retain[pred_idx] - prob_empty[pred_idx]
                
                gains.append(gain)
                drops.append(drop)
            except Exception as e:
                print(f"[ERROR] Failed explaining {path} with SPASHTA: {e}")
                
        avg_rt = np.mean(runtimes)
        avg_gain = np.mean(gains)
        avg_drop = np.mean(drops)
        
        print(f"[OK] {arch_name} SPASHTA summary: Gain: {avg_gain:.4f}, Drop: {avg_drop:.4f}, Runtime: {avg_rt:.4f} s")
        
        results[arch_name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "Gain": avg_gain,
            "Drop": avg_drop,
            "Explanation_Runtime": avg_rt
        }

    # ════════════════════════════════════════════════════════════
    #  Save Results to CSV
    # ════════════════════════════════════════════════════════════
    csv_file = os.path.join("results", "architecture_comparison.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Architecture", "Accuracy", "Precision", "Recall", "F1_Score", "Gain", "Drop", "Explanation_Runtime"])
        for arch_name, metrics in results.items():
            writer.writerow([
                arch_name,
                f"{metrics['Accuracy']:.4f}",
                f"{metrics['Precision']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['F1_Score']:.4f}",
                f"{metrics['Gain']:.4f}",
                f"{metrics['Drop']:.4f}",
                f"{metrics['Explanation_Runtime']:.4f}"
            ])
            
    print(f"\n[OK] CSV comparison logs successfully saved to '{csv_file}'.")

    # ════════════════════════════════════════════════════════════
    #  Output Paper-Ready Markdown Table
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  SPASHTA CROSS-ARCHITECTURE VALIDATION SUMMARY")
    print("=" * 100)
    print("| Architecture | Accuracy | Precision | Recall | F1 Score | Gain | Drop | Explanation Runtime (s) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for arch_name, metrics in results.items():
        print(f"| {arch_name} | {metrics['Accuracy']*100:.2f}% | {metrics['Precision']:.4f} | {metrics['Recall']:.4f} | {metrics['F1_Score']:.4f} | {metrics['Gain']:.4f} | {metrics['Drop']:.4f} | {metrics['Explanation_Runtime']:.4f} s |")
    print("=" * 100 + "\n")

    # ════════════════════════════════════════════════════════════
    #  Generate Publication-Quality Figures
    # ════════════════════════════════════════════════════════════
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    archs = list(results.keys())
    short_archs = ["CNN+BiLSTM", "CNN+BiGRU+Attn", "SE-ResNet+BiGRU+Attn"]
    
    # Figure 1: Classification Performance
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(short_archs))
    width = 0.18
    
    rects1 = ax.bar(x - 1.5*width, [results[arch]["Accuracy"] for arch in archs], width, label='Accuracy', color='#2ca02c', edgecolor='black', alpha=0.85)
    rects2 = ax.bar(x - 0.5*width, [results[arch]["Precision"] for arch in archs], width, label='Precision', color='#1f77b4', edgecolor='black', alpha=0.85)
    rects3 = ax.bar(x + 0.5*width, [results[arch]["Recall"] for arch in archs], width, label='Recall', color='#ff7f0e', edgecolor='black', alpha=0.85)
    rects4 = ax.bar(x + 1.5*width, [results[arch]["F1_Score"] for arch in archs], width, label='F1 Score', color='#d62728', edgecolor='black', alpha=0.85)
    
    ax.set_ylabel('Score / Ratio', fontweight='bold')
    ax.set_title('SER Architecture Classification Performance Metrics', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_archs, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.set_ylim(0, 1.1)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)
    
    plt.tight_layout()
    plt.savefig("plots/architecture_performance.png", dpi=300)
    plt.close()
    
    # Figure 2: SPASHTA Fidelity Metrics
    fig, ax = plt.subplots(figsize=(6, 4))
    width = 0.25
    rects_gain = ax.bar(x - width/2, [results[arch]["Gain"] for arch in archs], width, label='Gain (Higher is better)', color='#4c72b0', edgecolor='black', alpha=0.85)
    rects_drop = ax.bar(x + width/2, [results[arch]["Drop"] for arch in archs], width, label='Drop (Higher is better)', color='#c44e52', edgecolor='black', alpha=0.85)
    
    ax.set_ylabel('Fidelity Metric Value', fontweight='bold')
    ax.set_title('SPASHTA Fidelity Metrics (Gain vs. Drop) Across Models', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_archs, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', edgecolor='gray')
    ax.set_ylim(0, max([results[arch]["Drop"] for arch in archs] + [results[arch]["Gain"] for arch in archs]) * 1.25)
    
    autolabel(rects_gain)
    autolabel(rects_drop)
    
    plt.tight_layout()
    plt.savefig("plots/spashta_fidelity_comparison.png", dpi=300)
    plt.close()

    # Figure 3: Explanation Runtime
    fig, ax = plt.subplots(figsize=(6, 4))
    rects_rt = ax.bar(x, [results[arch]["Explanation_Runtime"] for arch in archs], 0.4, label='Explanation Runtime', color='#8172b3', edgecolor='black', alpha=0.85)
    
    ax.set_ylabel('Time (Seconds)', fontweight='bold')
    ax.set_title('SPASHTA Explanation Generation Latency Across Models', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(short_archs, fontweight='bold')
    ax.set_ylim(0, max([results[arch]["Explanation_Runtime"] for arch in archs]) * 1.2)
    
    for rect in rects_rt:
        height = rect.get_height()
        ax.annotate(f'{height:.3f} s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig("plots/spashta_runtime_comparison.png", dpi=300)
    plt.close()
    
    print("[OK] Generated publication-quality figures in plots/: 'architecture_performance.png', 'spashta_fidelity_comparison.png', 'spashta_runtime_comparison.png'.")

if __name__ == "__main__":
    run_validation()
