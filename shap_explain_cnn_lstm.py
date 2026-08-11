import os
import time
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
from sklearn.model_selection import train_test_split
import shap
from cnn_lstm_model import HybridCNN_LSTM
from cnn_lstm_features import extract_sequential_features

# ════════════════════════════════════════════════════════════
#  Setup & Configuration
# ════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

DATASET_DIR = "hindi_dataset"
ENCODER_PATH = "cnn_lstm_label_encoder.pkl"
SCALER_PATH = "cnn_lstm_scaler.pkl"
MODEL_PATH = "cnn_lstm_best_model.pt"
OUTPUT_CSV = "results/shap_results.csv"
PLOT_DIR = "shap_plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
#  Load Assets
# ════════════════════════════════════════════════════════════
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Missing CNN-LSTM assets! Please ensure model, label encoder, and scaler exist.")

encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# Initialize model
raw_model = HybridCNN_LSTM(input_dim=161, num_classes=len(encoder.classes_), hidden_dim=128, dropout=0.3)
raw_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
raw_model.eval()
raw_model.to(DEVICE)
print("[OK] Model loaded successfully.")

# Model Wrapper to ensure single tensor output (logits) for SHAP
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        logits, _ = self.model(x)
        return logits

model = ModelWrapper(raw_model)

# ════════════════════════════════════════════════════════════
#  Deterministic Train/Val Split (to identify validation paths)
# ════════════════════════════════════════════════════════════
VALID_EMOTIONS = {'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'sarcastic', 'surprise'}
emotion_to_files = {emo: [] for emo in VALID_EMOTIONS}

for root, _, files in os.walk(DATASET_DIR):
    dir_name = os.path.basename(root).lower()
    if dir_name in VALID_EMOTIONS:
        for f in files:
            if f.lower().endswith(".wav"):
                emotion_to_files[dir_name].append(os.path.join(root, f))

all_tasks = []
for emotion in sorted(emotion_to_files.keys()):
    for fpath in emotion_to_files[emotion]:
        all_tasks.append((fpath, emotion))

fpaths = [t[0] for t in all_tasks]
emotions = [t[1] for t in all_tasks]

train_paths, val_paths, train_y, val_y = train_test_split(
    fpaths, emotions, test_size=0.2, random_state=42, stratify=emotions
)

print(f"[INFO] Total validation files available: {len(val_paths)}")
eval_paths = val_paths[:100]
eval_y = val_y[:100]
print(f"[INFO] Selected {len(eval_paths)} validation samples for evaluation.")

# ════════════════════════════════════════════════════════════
#  Prepare SHAP Background Dataset
# ════════════════════════════════════════════════════════════
print("[INFO] Preparing background dataset of 20 samples...")
background_files = val_paths[100:120]  # Use different validation files for background set
background_feats = []
for fpath in background_files:
    try:
        audio, sr = librosa.load(fpath, sr=16000)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        feat = extract_sequential_features(audio, sr, target_len=125)
        # Scale
        feat_flat = feat.T
        feat_sc_flat = scaler.transform(feat_flat)
        feat_sc = feat_sc_flat.T
        background_feats.append(feat_sc)
    except Exception as e:
        print(f"[WARNING] Failed to load background sample {os.path.basename(fpath)}: {e}")

background_feats = np.array(background_feats)
background_tensor = torch.tensor(background_feats, dtype=torch.float32).to(DEVICE)
print(f"[OK] Background tensor shape: {background_tensor.shape}")

# ════════════════════════════════════════════════════════════
#  Initialize SHAP Explainer (DeepSHAP with fallback to GradientSHAP)
# ════════════════════════════════════════════════════════════
explainer_type = "DeepSHAP"
try:
    print("[INFO] Attempting to initialize DeepExplainer (DeepSHAP)...")
    explainer = shap.DeepExplainer(model, background_tensor)
    # Verify execution (LSTMs fail additivity check during shap_values call)
    test_run = explainer.shap_values(background_tensor[0:1])
    print("[OK] DeepExplainer initialized and verified.")
except Exception as e:
    print(f"[WARNING] DeepExplainer failed verification: {e}")
    print("[INFO] Falling back to GradientExplainer (GradientSHAP)...")
    explainer = shap.GradientExplainer(model, background_tensor)
    explainer_type = "GradientSHAP"
    print("[OK] GradientExplainer initialized.")

# ════════════════════════════════════════════════════════════
#  Evaluation Loop
# ════════════════════════════════════════════════════════════
runtimes = []
sparsities = []

# Open CSV for writing
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample id", "predicted emotion", "confidence", "runtime", "sparsity"])

print(f"\n[INFO] Starting SHAP explainability loop using {explainer_type}...")

for idx, (fpath, true_emo) in enumerate(zip(eval_paths, eval_y)):
    sample_id = os.path.basename(fpath)
    print(f"  [{idx+1}/100] Processing: {sample_id}...", end="", flush=True)
    
    try:
        # 1. Load and Normalise Audio
        audio, sr = librosa.load(fpath, sr=16000)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
        # 2. Extract Sequential Features
        feat = extract_sequential_features(audio, sr, target_len=125) # (161, 125)
        
        # 3. Scale sequential features
        feat_flat = feat.T
        feat_sc_flat = scaler.transform(feat_flat)
        feat_sc = feat_sc_flat.T # (161, 125)
        
        # 4. Model Prediction
        x = torch.tensor(feat_sc[np.newaxis], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_idx = np.argmax(probs[0])
        pred_emo = encoder.inverse_transform([pred_idx])[0]
        confidence = probs[0, pred_idx]
        
        # 5. Generate SHAP Values and Measure Runtime
        t_start = time.perf_counter()
        shap_values = explainer.shap_values(x)
        t_end = time.perf_counter()
        runtime = t_end - t_start
        runtimes.append(runtime)
        
        # Extract SHAP values for the predicted class
        if isinstance(shap_values, list):
            # If it's a list, the class index selects the list element, shape (1, 161, 125)
            shap_for_pred = shap_values[pred_idx][0] # (161, 125)
        else:
            # If it's a numpy array of shape (B, 161, 125, 8)
            shap_for_pred = shap_values[0, :, :, pred_idx] # (161, 125)
        
        # 6. Calculate Explanation Sparsity
        # Proportion of values below 1% of the maximum absolute SHAP value
        max_abs = np.max(np.abs(shap_for_pred))
        if max_abs > 0:
            sparsity = np.mean(np.abs(shap_for_pred) < 0.01 * max_abs)
        else:
            sparsity = 1.0
        sparsities.append(sparsity)
        
        # 7. Save Publication-Quality Side-by-Side Figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left Panel: Original sequential features
        im1 = axes[0].imshow(feat, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f"Original Feature Map (Shape: 161x125)\nTrue: {true_emo} | Pred: {pred_emo} ({confidence:.2%})", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Time Frames", fontsize=9)
        axes[0].set_ylabel("Feature Dimensions", fontsize=9)
        fig.colorbar(im1, ax=axes[0], label="Magnitude")
        
        # Right Panel: SHAP Heatmap
        # Use a diverging colormap where warm color indicates positive impact
        vmax = max(abs(shap_for_pred.min()), abs(shap_for_pred.max())) if shap_for_pred.any() else 1.0
        im2 = axes[1].imshow(shap_for_pred, aspect='auto', origin='lower', cmap='coolwarm', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f"SHAP Heatmap ({explainer_type})\nSparsity: {sparsity:.2%}", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Time Frames", fontsize=9)
        axes[1].set_ylabel("Feature Dimensions", fontsize=9)
        fig.colorbar(im2, ax=axes[1], label="SHAP Value")
        
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"sample_{idx+1}_{sample_id.replace('.wav', '')}_shap.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        
        # 8. Save results to CSV
        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([sample_id, pred_emo, f"{confidence:.4f}", f"{runtime:.4f}", f"{sparsity:.4f}"])
            
        print(f" Pred: {pred_emo} ({confidence:.2%}) | SHAP Time: {runtime:.2f}s | Sparsity: {sparsity:.2%}")
        
    except Exception as e:
        print(f" [ERROR] Failed processing {sample_id}: {e}")

# ════════════════════════════════════════════════════════════
#  Summarize Metrics
# ════════════════════════════════════════════════════════════
if runtimes:
    avg_rt = np.mean(runtimes)
    std_rt = np.std(runtimes)
    avg_sp = np.mean(sparsities)
    print("\n" + "=" * 50)
    print("  SHAP Explainability Evaluation Complete!")
    print("=" * 50)
    print(f"Total Samples Processed : {len(runtimes)}")
    print(f"Average Runtime/Sample  : {avg_rt:.4f} seconds")
    print(f"Runtime Std Dev         : {std_rt:.4f} seconds")
    print(f"Average Sparsity        : {avg_sp:.2%}")
    print(f"Results Saved To        : {OUTPUT_CSV}")
    print(f"Plots Saved To          : {PLOT_DIR}/")
    print("=" * 50 + "\n")
else:
    print("[ERROR] No samples were successfully processed.")
