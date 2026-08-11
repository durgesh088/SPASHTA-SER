import os
import time
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

# Import model definitions
from cnn_lstm_model import HybridCNN_LSTM
from cnn_lstm_features import extract_sequential_features
from lime.lime_image import LimeImageExplainer  # type: ignore
import shap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════
#  Model Loader & Wrapper
# ════════════════════════════════════════════════════════════
class SHAPModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits, _ = self.model(x)
        return logits

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

# ════════════════════════════════════════════════════════════
#  Reconstruction Helpers (Fidelity Metrics)
# ════════════════════════════════════════════════════════════
def predict_emotion_cnn_lstm(audio, sr, model, scaler, encoder):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    feat = extract_sequential_features(audio, sr, target_len=125)
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

# ════════════════════════════════════════════════════════════
#  Explainer Attributions
# ════════════════════════════════════════════════════════════
def run_spashta(y, sr, model, scaler, encoder, run_idx=0):
    prob_orig = predict_emotion_cnn_lstm(y, sr, model, scaler, encoder)[1]
    pred_idx = np.argmax(prob_orig)
    
    DURATION = max(len(y) / sr, 4.0)
    target_len_samples = int(sr * DURATION)
    if len(y) < target_len_samples:
        y_proc = np.pad(y, (0, target_len_samples - len(y)))
    else:
        y_proc = y
        
    feat = extract_sequential_features(y_proc, sr, target_len=125)
    stride = int(0.1 * sr)
    win = int(0.5 * sr)
    hop_length = 512
    
    features_batch = []
    for start in range(0, len(y_proc), stride):
        end = min(start + win, len(y_proc))
        sf_idx = int(start // hop_length)
        ef_idx = int(end // hop_length)
        
        masked = feat.copy()
        masked[:, sf_idx:ef_idx] = 0.0
        
        masked_flat = masked.T
        masked_sc_flat = scaler.transform(masked_flat)
        masked_sc = masked_sc_flat.T
        features_batch.append(masked_sc)
        
    X_batch = np.array(features_batch)
    X_t = torch.tensor(X_batch, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits, _ = model(X_t)
        probs_masked = torch.softmax(logits, dim=1).cpu().numpy()
        
    importance = np.array([prob_orig[pred_idx] - pm[pred_idx] for pm in probs_masked])
    return importance

def run_lime(y, sr, model, scaler, encoder, lime_expl, run_idx=0):
    feat = extract_sequential_features(y, sr, target_len=125)
    img = np.repeat(feat[:, :, np.newaxis], 3, axis=2)
    prob_orig = predict_emotion_cnn_lstm(y, sr, model, scaler, encoder)[1]
    pred_idx = np.argmax(prob_orig)
    
    def lime_predict(images):
        specs = images[:, :, :, 0]
        scaled_batch = []
        for spec in specs:
            spec_flat = spec.T
            spec_sc_flat = scaler.transform(spec_flat)
            spec_sc = spec_sc_flat.T
            scaled_batch.append(spec_sc)
        scaled_batch = np.array(scaled_batch)
        x = torch.tensor(scaled_batch, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs
        
    # We pass run_idx as random_seed to ensure different perturbation paths across iterations
    explanation = lime_expl.explain_instance(
        img,
        lime_predict,
        top_labels=1,
        num_samples=250,
        random_seed=run_idx
    )
    
    segments = explanation.segments
    local_exp = explanation.local_exp[explanation.top_labels[0]]
    heatmap = np.zeros_like(segments, dtype=float)
    for seg_id, weight in local_exp:
        heatmap[segments == seg_id] = weight
        
    importance = np.sum(np.abs(heatmap), axis=0) # shape (125,)
    return importance

def run_shap(y, sr, model, scaler, encoder, wrapper_model, shap_expl, run_idx=0):
    # Set seed to randomize path points in GradientSHAP
    np.random.seed(run_idx)
    torch.manual_seed(run_idx)
    
    feat = extract_sequential_features(y, sr, target_len=125)
    feat_flat = feat.T
    feat_sc_flat = scaler.transform(feat_flat)
    feat_sc = feat_sc_flat.T
    
    x = torch.tensor(feat_sc[np.newaxis], dtype=torch.float32).to(DEVICE)
    prob_orig = predict_emotion_cnn_lstm(y, sr, model, scaler, encoder)[1]
    pred_idx = np.argmax(prob_orig)
    
    shap_values = shap_expl.shap_values(x)
    if isinstance(shap_values, list):
        shap_for_pred = shap_values[pred_idx][0]
    else:
        shap_for_pred = shap_values[0, :, :, pred_idx]
        
    importance = np.sum(np.abs(shap_for_pred), axis=0) # shape (125,)
    return importance

# ════════════════════════════════════════════════════════════
#  Similarity Metrics
# ════════════════════════════════════════════════════════════
def compute_cosine_similarity(e1, e2):
    norm1 = np.linalg.norm(e1)
    norm2 = np.linalg.norm(e2)
    if norm1 == 0 or norm2 == 0:
        return 1.0 if np.array_equal(e1, e2) else 0.0
    return np.dot(e1, e2) / (norm1 * norm2)

def compute_pearson_correlation(e1, e2):
    if np.array_equal(e1, e2):
        return 1.0
    std1 = np.std(e1)
    std2 = np.std(e2)
    if std1 == 0 or std2 == 0:
        return 1.0 if np.array_equal(e1, e2) else 0.0
    corr = np.corrcoef(e1, e2)[0, 1]
    return corr if np.isfinite(corr) else 1.0

def compute_pairwise_metrics(explanations):
    # explanations: list of 5 vectors
    num_runs = len(explanations)
    cosines, correlations = [], []
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            e1, e2 = explanations[i], explanations[j]
            cosines.append(compute_cosine_similarity(e1, e2))
            correlations.append(compute_pearson_correlation(e1, e2))
    return np.mean(cosines), np.mean(correlations)

# ════════════════════════════════════════════════════════════
#  Process Single Sample (for Parallel Job)
# ════════════════════════════════════════════════════════════
def process_single_sample(path, true_emo, model, scaler, encoder, wrapper_model, lime_expl, shap_expl):
    try:
        audio, sr = librosa.load(path, sr=16000)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        spashta_runs = []
        lime_runs = []
        shap_runs = []
        
        # Run each 5 times
        for run_idx in range(5):
            spashta_runs.append(run_spashta(audio, sr, model, scaler, encoder, run_idx))
            lime_runs.append(run_lime(audio, sr, model, scaler, encoder, lime_expl, run_idx))
            shap_runs.append(run_shap(audio, sr, model, scaler, encoder, wrapper_model, shap_expl, run_idx))
            
        sp_cos, sp_corr = compute_pairwise_metrics(spashta_runs)
        li_cos, li_corr = compute_pairwise_metrics(lime_runs)
        sh_cos, sh_corr = compute_pairwise_metrics(shap_runs)
        
        return {
            "SPASHTA": {"cosine": sp_cos, "pearson": sp_corr},
            "LIME": {"cosine": li_cos, "pearson": li_corr},
            "SHAP": {"cosine": sh_cos, "pearson": sh_corr}
        }
    except Exception as e:
        print(f"[ERROR] Failed processing sample {path}: {e}")
        return None

# ════════════════════════════════════════════════════════════
#  Main Stability Runner
# ════════════════════════════════════════════════════════════
def run_stability_analysis():
    # Setup dataset split
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
    
    eval_paths = val_paths[:50]
    eval_y = val_y[:50]
    bg_files = val_paths[100:115]
    
    # Load Model assets
    print("[INFO] Loading CNN-LSTM SER model components...")
    model, scaler, encoder = load_cnn_lstm_model()
    
    # Load explainers
    print("[INFO] Preparing explainers baseline assets...")
    lime_expl = LimeImageExplainer(random_state=42)
    
    # Prepare background dataset for SHAP
    bg_feats = []
    for bg_fpath in bg_files:
        bg_audio, bg_sr = librosa.load(bg_fpath, sr=16000)
        max_val = np.max(np.abs(bg_audio))
        if max_val > 0:
            bg_audio = bg_audio / max_val
        bg_feat = extract_sequential_features(bg_audio, bg_sr, target_len=125)
        bg_feat_flat = bg_feat.T
        bg_feat_sc_flat = scaler.transform(bg_feat_flat)
        bg_feat_sc = bg_feat_sc_flat.T
        bg_feats.append(bg_feat_sc)
    bg_tensor = torch.tensor(np.array(bg_feats), dtype=torch.float32).to(DEVICE)
    
    wrapper_model = SHAPModelWrapper(model)
    shap_expl = shap.GradientExplainer(wrapper_model, bg_tensor)
    
    # Run evaluation in parallel to save time on CPU
    print(f"[INFO] Computing stability metrics over 50 samples in parallel (5 runs per sample)...")
    raw_results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_sample)(path, true, model, scaler, encoder, wrapper_model, lime_expl, shap_expl)
        for path, true in zip(eval_paths, eval_y)
    )
    
    # Filter out failed runs
    valid_results = [r for r in raw_results if r is not None]
    print(f"[INFO] Completed successfully for {len(valid_results)}/50 samples.")
    
    # Aggregate stats
    methods = ["SPASHTA", "LIME", "SHAP"]
    metrics = ["cosine", "pearson"]
    
    stats = {m: {met: [] for met in metrics} for m in methods}
    for res in valid_results:
        for m in methods:
            for met in metrics:
                stats[m][met].append(res[m][met])
                
    summary = {}
    for m in methods:
        summary[m] = {}
        for met in metrics:
            summary[m][f"{met}_mean"] = np.mean(stats[m][met])
            summary[m][f"{met}_std"] = np.std(stats[m][met])
            
    # Save to CSV
    csv_file = os.path.join("results", "stability_results.csv")
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Cosine_Mean", "Cosine_Std", "Pearson_Mean", "Pearson_Std"])
        for m in methods:
            writer.writerow([
                m,
                f"{summary[m]['cosine_mean']:.4f}",
                f"{summary[m]['cosine_std']:.4f}",
                f"{summary[m]['pearson_mean']:.4f}",
                f"{summary[m]['pearson_std']:.4f}"
            ])
            
    print(f"\n[OK] CSV stability results saved to '{csv_file}'.")
    
    # Print Paper-Ready Table
    print("\n" + "=" * 90)
    print("  EXPLAINABILITY METHOD STABILITY ANALYSIS SUMMARY")
    print("=" * 90)
    print("| Method | Cosine Mean | Cosine Std | Pearson Mean | Pearson Std |")
    print("| :--- | :---: | :---: | :---: | :---: |")
    for m in methods:
        print(f"| **{m}** | {summary[m]['cosine_mean']:.4f} | {summary[m]['cosine_std']:.4f} | {summary[m]['pearson_mean']:.4f} | {summary[m]['pearson_std']:.4f} |")
    print("=" * 90 + "\n")
    
    # ════════════════════════════════════════════════════════════
    #  Generate Publication-Quality Comparison Figure
    # ════════════════════════════════════════════════════════════
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True)
    
    # Compile raw data for boxplots
    for ax_idx, met in enumerate(metrics):
        ax = axes[ax_idx]
        data_to_plot = [stats[m][met] for m in methods]
        
        # Draw boxplots
        bp = ax.boxplot(data_to_plot, tick_labels=methods, patch_artist=True,
                        boxprops=dict(facecolor='#eaeaf2', color='black', alpha=0.8),
                        medianprops=dict(color='crimson', linewidth=1.5),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))
        
        # Color individual boxes
        colors = ['#2ca02c', '#1f77b4', '#d62728'] # Green, Blue, Red
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_title(f"{met.upper()} Similarity", fontweight='bold', fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
    fig.suptitle('Comparative Explanation Stability Across 5 Independent Runs', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig("plots/stability_comparison.png", dpi=300)
    plt.close()
    
    print("[OK] Generated stability comparison figure: 'stability_comparison.png'.")

if __name__ == "__main__":
    run_stability_analysis()

