import os
import time
import csv
import joblib
import numpy as np
import matplotlib.pyplot as plt
import torch
import librosa
from sklearn.model_selection import train_test_split
from lime.lime_image import LimeImageExplainer  # type: ignore
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
OUTPUT_CSV = "results/lime_results.csv"
PLOT_DIR = "lime_plots"

os.makedirs(PLOT_DIR, exist_ok=True)

# ════════════════════════════════════════════════════════════
#  Load Assets
# ════════════════════════════════════════════════════════════
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError("Missing CNN-LSTM assets! Please ensure model, label encoder, and scaler exist.")

encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# Initialize model
model = HybridCNN_LSTM(input_dim=161, num_classes=len(encoder.classes_), hidden_dim=128, dropout=0.3)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
model.to(DEVICE)
print("[OK] Model loaded successfully.")

# ════════════════════════════════════════════════════════════
#  LIME Prediction Wrapper
# ════════════════════════════════════════════════════════════
def lime_predict(images):
    # images shape: (N, 161, 125, 3)
    # Extract first channel (N, 161, 125)
    specs = images[:, :, :, 0]
    
    scaled_batch = []
    for spec in specs:
        # Scale frame-by-frame
        spec_flat = spec.T # (125, 161)
        spec_sc_flat = scaler.transform(spec_flat)
        spec_sc = spec_sc_flat.T # (161, 125)
        scaled_batch.append(spec_sc)
        
    scaled_batch = np.array(scaled_batch)
    x = torch.tensor(scaled_batch, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == "cuda")):
            logits, _ = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs

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
# Take first 100 validation files
eval_paths = val_paths[:100]
eval_y = val_y[:100]
print(f"[INFO] Selected {len(eval_paths)} validation samples for evaluation.")

# ════════════════════════════════════════════════════════════
#  Initialize LIME Explainer
# ════════════════════════════════════════════════════════════
explainer = LimeImageExplainer(random_state=42)

# ════════════════════════════════════════════════════════════
#  Evaluation Loop
# ════════════════════════════════════════════════════════════
results = []
runtimes = []

# Open CSV for writing
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["sample id", "predicted emotion", "confidence", "runtime"])

print("\n[INFO] Starting LIME evaluation loop...")

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
        
        # 3. Predict Emotion and Confidence
        img = np.repeat(feat[:, :, np.newaxis], 3, axis=2) # (161, 125, 3)
        probs = lime_predict(img[np.newaxis])
        pred_idx = np.argmax(probs[0])
        pred_emo = encoder.inverse_transform([pred_idx])[0]
        confidence = probs[0, pred_idx]
        
        # 4. Generate LIME Explanation and Measure Runtime
        t_start = time.perf_counter()
        explanation = explainer.explain_instance(
            img,
            lime_predict,
            top_labels=1,
            num_samples=250, # High fidelity perturbations
            random_seed=42
        )
        t_end = time.perf_counter()
        runtime = t_end - t_start
        runtimes.append(runtime)
        
        # 5. Extract Feature Attributions
        segments = explanation.segments
        local_exp = explanation.local_exp[explanation.top_labels[0]]
        
        # Reconstruct attribution heatmap
        heatmap = np.zeros_like(segments, dtype=float)
        for seg_id, weight in local_exp:
            heatmap[segments == seg_id] = weight
            
        # 6. Save Publication-Quality Visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left Panel: Original Spectrogram-like Feature Map
        im1 = axes[0].imshow(feat, aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title(f"Original Feature Map (Shape: 161x125)\nTrue: {true_emo} | Pred: {pred_emo} ({confidence:.2%})", fontsize=11, fontweight='bold')
        axes[0].set_xlabel("Time Frames", fontsize=9)
        axes[0].set_ylabel("Feature Dimensions", fontsize=9)
        fig.colorbar(im1, ax=axes[0], label="Magnitude")
        
        # Right Panel: LIME Heatmap
        # Use a diverging colormap where red = positive attribution and blue = negative
        vmax = max(abs(heatmap.min()), abs(heatmap.max())) if heatmap.any() else 1.0
        im2 = axes[1].imshow(heatmap, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[1].set_title("LIME Feature Attributions\nPositive (Red) vs Negative (Blue) Contributions", fontsize=11, fontweight='bold')
        axes[1].set_xlabel("Time Frames", fontsize=9)
        axes[1].set_ylabel("Feature Dimensions", fontsize=9)
        fig.colorbar(im2, ax=axes[1], label="Attribution Weight")
        
        plt.tight_layout()
        plot_path = os.path.join(PLOT_DIR, f"sample_{idx+1}_{sample_id.replace('.wav', '')}_lime.png")
        plt.savefig(plot_path, dpi=200)
        plt.close()
        
        # 7. Log to CSV
        with open(OUTPUT_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([sample_id, pred_emo, f"{confidence:.4f}", f"{runtime:.4f}"])
            
        print(f" Predicted: {pred_emo} ({confidence:.2%}) | LIME Time: {runtime:.2f}s")
        
    except Exception as e:
        print(f" [ERROR] Failed: {e}")

# ════════════════════════════════════════════════════════════
#  Summarize Results
# ════════════════════════════════════════════════════════════
if runtimes:
    avg_rt = np.mean(runtimes)
    std_rt = np.std(runtimes)
    print("\n" + "=" * 50)
    print("  LIME Explainability Evaluation Complete!")
    print("=" * 50)
    print(f"Total Samples Processed : {len(runtimes)}")
    print(f"Average Runtime/Sample  : {avg_rt:.4f} seconds")
    print(f"Std Dev of Runtimes    : {std_rt:.4f} seconds")
    print(f"Results Saved To        : {OUTPUT_CSV}")
    print(f"Plots Saved To          : {PLOT_DIR}/")
    print("=" * 50 + "\n")
else:
    print("[ERROR] No samples were successfully processed.")
