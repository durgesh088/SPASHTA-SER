import os
import time
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
import joblib
from se_resnet_model import SEResNet_BiGRU_Attention
from se_resnet_features import extract_spectrogram_features

# ════════════════════════════════════════════════════════════
#  Setup & Constants
# ════════════════════════════════════════════════════════════
DATASET_DIR = "hindi_dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "se_resnet_best_model.pt"
ENCODER_PATH = "se_resnet_label_encoder.pkl"
SCALER_PATH = "se_resnet_scaler.pkl"

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
            return ((X - self.means[:, None, None]) / self.stds[:, None, None]).astype(np.float32)
        return ((X - self.means[None, :, None, None]) / self.stds[None, :, None, None]).astype(np.float32)
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ════════════════════════════════════════════════════════════
#  Masking Helpers
# ════════════════════════════════════════════════════════════
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

def reconstruct_audio_inverse(y, importance, sr=16000, threshold_ratio=0.6):
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
#  Model Inference and Explanation Generation
# ════════════════════════════════════════════════════════════
def predict_raw(y, sr, model, scaler):
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    feat = extract_spectrogram_features(y, sr)
    feat_sc = scaler.transform(feat)
    x = torch.tensor(feat_sc[np.newaxis], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
            logits, _ = model(x)
            prob = torch.softmax(logits[0], dim=0).cpu().numpy()
    return prob

def run_analysis(num_samples=16):
    # Load assets
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Trained SE-ResNet model assets are missing! Please train the model first.")

    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    model = SEResNet_BiGRU_Attention(num_classes=len(encoder.classes_), gru_hidden=128, dropout=0.5)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)

    # Sample files from dataset
    valid_emotions = set(encoder.classes_)
    class_to_files = {emo: [] for emo in valid_emotions}
    for root, _, files in os.walk(DATASET_DIR):
        dir_name = os.path.basename(root).lower()
        if dir_name in valid_emotions:
            for f in files:
                if f.lower().endswith(".wav"):
                    class_to_files[dir_name].append(os.path.join(root, f))

    sampled_files = []
    files_per_class = max(1, num_samples // len(valid_emotions))
    random.seed(42)
    for emo in sorted(class_to_files.keys()):
        files = class_to_files[emo]
        if files:
            sampled_files.extend(random.sample(files, min(len(files), files_per_class)))

    # Fallback to general files if needed
    if len(sampled_files) < num_samples:
        all_wavs = []
        for emo, files in class_to_files.items():
            all_wavs.extend(files)
        sampled_files.extend(random.sample(all_wavs, min(len(all_wavs), num_samples - len(sampled_files))))

    print(f"[INFO] Running systematic threshold analysis on {len(sampled_files)} files...")

    thresholds = [0.2, 0.4, 0.5, 0.6, 0.8]
    
    # Store aggregated results per threshold
    results_by_thresh = {t: {"gain": [], "drop": [], "runtime": [], "selected_frames": [], "pct_retained": []} for t in thresholds}

    # Evaluate each file
    for idx, fpath in enumerate(sampled_files):
        print(f"  [{idx+1}/{len(sampled_files)}] Processing {os.path.basename(fpath)}...")
        try:
            y, sr = librosa.load(fpath, sr=16000)
            DURATION = 4
            target_len_samples = sr * DURATION
            if len(y) < target_len_samples:
                y = np.pad(y, (0, target_len_samples - len(y)))
            else:
                y = y[:target_len_samples]
            
            # Normalise
            max_val = np.max(np.abs(y))
            if max_val > 0:
                y = y / max_val

            # Baseline Prediction
            prob_orig = predict_raw(y, sr, model, scaler)
            pred_idx = np.argmax(prob_orig)

            # Pre-extract spectrogram for explanation to optimize perturbation loop
            y_proc = y
            
            full_spec = extract_spectrogram_features(y_proc, sr)
            stride = int(0.1 * sr)
            hop_length = 128

            # Run analysis for each threshold
            for t in thresholds:
                # ── START TIMING PERTURBATION LOOP ──
                # This measures the entire perturbation loop (masking + batch prediction) for ratio t
                t_start = time.perf_counter()
                
                win = int(t * sr) # window duration scales with threshold ratio
                features_batch = []
                
                # Loop generating masks
                for start in range(0, len(y_proc), stride):
                    end = min(start + win, len(y_proc))
                    sf_idx = int(start // hop_length)
                    ef_idx = int(end // hop_length)
                    
                    masked = full_spec.copy()
                    masked[:, :, sf_idx:ef_idx] = 0.0
                    
                    masked_sc = scaler.transform(masked)
                    features_batch.append(masked_sc)
                    
                X_batch = np.array(features_batch)
                X_t = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
                        logits, _ = model(X_t)
                        probs_masked = torch.softmax(logits, dim=1).cpu().numpy()
                
                t_end = time.perf_counter()
                runtime = t_end - t_start
                # ── END TIMING PERTURBATION LOOP ──

                # Compute Saliency
                importance = np.array([prob_orig[pred_idx] - pm[pred_idx] for pm in probs_masked])
                max_imp = np.max(importance)
                
                # Metrics computation
                thresh_val = t * max_imp
                selected_steps = (importance >= thresh_val).astype(float) if max_imp > 0 else np.zeros_like(importance)
                n_selected = int(np.sum(selected_steps))
                pct_retained = (n_selected / len(importance)) * 100 if len(importance) > 0 else 0.0

                # Compute Drop (Delete Saliency)
                y_delete = reconstruct_audio_inverse(y, importance, sr, threshold_ratio=t)
                prob_delete = predict_raw(y_delete, sr, model, scaler)
                drop = prob_orig[pred_idx] - prob_delete[pred_idx]

                # Compute Gain (Keep Saliency)
                y_retain = reconstruct_audio(y, importance, sr, threshold_ratio=t)
                prob_retain = predict_raw(y_retain, sr, model, scaler)
                
                # Compute Empty Baseline (all-zero audio)
                y_empty = np.zeros_like(y)
                prob_empty = predict_raw(y_empty, sr, model, scaler)
                gain = prob_retain[pred_idx] - prob_empty[pred_idx]

                # Store metrics
                results_by_thresh[t]["gain"].append(gain)
                results_by_thresh[t]["drop"].append(drop)
                results_by_thresh[t]["runtime"].append(runtime)
                results_by_thresh[t]["selected_frames"].append(n_selected)
                results_by_thresh[t]["pct_retained"].append(pct_retained)

        except Exception as e:
            print(f"  [ERROR] Processing failed for {os.path.basename(fpath)}: {e}")

    # Compute Averages
    final_summary = []
    for t in thresholds:
        summary = {
            "threshold": t,
            "avg_gain": np.mean(results_by_thresh[t]["gain"]),
            "avg_drop": np.mean(results_by_thresh[t]["drop"]),
            "avg_runtime": np.mean(results_by_thresh[t]["runtime"]),
            "avg_frames": np.mean(results_by_thresh[t]["selected_frames"]),
            "avg_retained": np.mean(results_by_thresh[t]["pct_retained"])
        }
        final_summary.append(summary)

    # Save to CSV
    csv_path = "results/threshold_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Threshold", "Average Gain", "Average Drop", "Average Runtime (s)", "Selected Frames", "Audio Retained (%)"])
        for s in final_summary:
            writer.writerow([s["threshold"], s["avg_gain"], s["avg_drop"], s["avg_runtime"], s["avg_frames"], s["avg_retained"]])
    print(f"\n[OK] Saved results to '{csv_path}'.")

    # Generate Line Plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    t_vals = [s["threshold"] for s in final_summary]
    
    # 1. Threshold vs Gain
    plt.figure(figsize=(6, 4))
    plt.plot(t_vals, [s["avg_gain"] for s in final_summary], marker='o', color='forestgreen', lw=2)
    plt.title("Threshold Ratio vs Average Gain")
    plt.xlabel("Threshold Ratio")
    plt.ylabel("Average Gain")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/threshold_vs_gain.png", dpi=150)
    plt.close()

    # 2. Threshold vs Drop
    plt.figure(figsize=(6, 4))
    plt.plot(t_vals, [s["avg_drop"] for s in final_summary], marker='s', color='crimson', lw=2)
    plt.title("Threshold Ratio vs Average Drop")
    plt.xlabel("Threshold Ratio")
    plt.ylabel("Average Drop")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/threshold_vs_drop.png", dpi=150)
    plt.close()

    # 3. Threshold vs Runtime
    plt.figure(figsize=(6, 4))
    plt.plot(t_vals, [s["avg_runtime"] for s in final_summary], marker='^', color='royalblue', lw=2)
    plt.title("Threshold Ratio vs Average Explanation Runtime")
    plt.xlabel("Threshold Ratio")
    plt.ylabel("Runtime (seconds)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/threshold_vs_runtime.png", dpi=150)
    plt.close()
    
    print("[OK] Generated line plots in plots/: 'threshold_vs_gain.png', 'threshold_vs_drop.png', 'threshold_vs_runtime.png'.")

    # Output Paper-Ready Table
    print("\n" + "=" * 80)
    print("  SPASHTA Threshold Sensitivity Analysis — Paper-Ready Table")
    print("=" * 80)
    print("| Threshold | Average Gain | Average Drop | Runtime (s) | Selected Frames | Audio Retained (%) |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for s in final_summary:
        t = s["threshold"]
        g = f"{s['avg_gain']:.4f}"
        d = f"{s['avg_drop']:.4f}"
        r = f"{s['avg_runtime']:.4f}"
        f_cnt = f"{s['avg_frames']:.2f}"
        ret = f"{s['avg_retained']:.2f}%"
        print(f"| {t:.1f} | {g} | {d} | {r} | {f_cnt} | {ret} |")
    print("=" * 80 + "\n")

    return final_summary

if __name__ == "__main__":
    run_analysis()
