"""
Training Pipeline for the Hybrid 1D-CNN + LSTM Hindi SER model

Stages
------
1. Walk  hindi_dataset/<emotion>/*.wav
2. Preprocess each file (Kalman → NLMS → Wiener)
3. Extract 523-D features  (+  augmented copy)
4. LASSO feature selection  → ~360 features
5. Scale with StandardScaler
6. Train Hybrid CNN-LSTM for 60 epochs
7. Save model, encoder, scaler, selected-feature indices
8. Generate evaluation plots
"""

import os
import sys
import random
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from cnn_lstm_preprocess import preprocess_audio, SAMPLE_RATE
from cnn_lstm_features import extract_features_cnn_lstm, DURATION
from cnn_lstm_model import HybridCNN_LSTM


#  Configuration
# ════════════════════════════════════════════════════════════
DATASET_DIR  = "hindi_dataset"
EPOCHS       = 30
BATCH_SIZE   = 16
LR           = 0.0005
TRAIN_SPLIT  = 0.8
AUGMENT      = True          # double data with augmentation
LASSO_C      = 0.05          # LASSO regularisation (smaller = sparser)

# Output artefacts
MODEL_PATH   = "cnn_lstm_best_model.pt"
ENCODER_PATH = "cnn_lstm_label_encoder.pkl"
SCALER_PATH  = "cnn_lstm_scaler.pkl"
PLOTS_DIR    = "cnn_lstm_plots"


# ════════════════════════════════════════════════════════════
#  Data augmentation  (reuses logic from existing utils.py)
# ════════════════════════════════════════════════════════════
def augment_audio(audio, sr):
    """Randomly apply one of: pitch shift, stretch, or noise."""
    method = random.choice([None, "pitch", "stretch", "noise"])

    if method == "pitch":
        steps = random.choice([-2, -1, 1, 2])
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=steps)

    elif method == "stretch":
        rate = random.uniform(0.85, 1.15)
        orig_len = len(audio)
        audio = librosa.resample(audio, orig_sr=sr,
                                 target_sr=int(sr * rate))
        if len(audio) < orig_len:
            audio = np.pad(audio, (0, orig_len - len(audio)))
        else:
            audio = audio[:orig_len]

    elif method == "noise":
        amp = 0.005 * np.random.uniform() * np.amax(np.abs(audio) + 1e-10)
        audio = audio + amp * np.random.normal(size=audio.shape)

    return audio


# ════════════════════════════════════════════════════════════
#  Dataset loading
# ════════════════════════════════════════════════════════════
def load_dataset(dataset_dir):
    """
    Walk dataset_dir recursively to find all wav files, group by emotion, and extract features.
    Supports both nested (speaker/session/emotion) and flat (emotion) structures.
    """
    import librosa as _lr          # local import to avoid top-level issue
    from tqdm import tqdm          # local import for visual progress tracking

    VALID_EMOTIONS = {'anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'sarcastic', 'surprise'}
    
    # Group found files by emotion
    emotion_to_files = {emo: [] for emo in VALID_EMOTIONS}
    
    for root, dirs, files in os.walk(dataset_dir):
        dir_name = os.path.basename(root).lower()
        if dir_name in VALID_EMOTIONS:
            for f in files:
                if f.lower().endswith(".wav"):
                    emotion_to_files[dir_name].append(os.path.join(root, f))

    X, y = [], []
    total = 0

    for emotion in sorted(emotion_to_files.keys()):
        wav_files = emotion_to_files[emotion]
        if not wav_files:
            continue
            
        print(f"\n📂 Processing '{emotion}' ({len(wav_files)} files)…")
        for fpath in tqdm(wav_files, desc=f"  {emotion}", unit="file"):
            try:
                audio, sr = _lr.load(fpath, sr=SAMPLE_RATE)

                # ── Preprocess ──
                audio = preprocess_audio(audio, sr)

                # ── Original features ──
                feat = extract_features_cnn_lstm(audio, sr, pool=True)
                X.append(feat)
                y.append(emotion)
                total += 1

                # ── Augmented features ──
                if AUGMENT:
                    aug = augment_audio(audio.copy(), sr)
                    feat_aug = extract_features_cnn_lstm(aug, sr, pool=True)
                    X.append(feat_aug)
                    y.append(emotion)
                    total += 1

            except Exception as e:
                print(f"\n❌ Error processing {os.path.basename(fpath)}: {e}")

    if total == 0:
        raise ValueError(
            f"No .wav files found in any valid emotion folders under '{dataset_dir}'. "
            f"Please check your dataset directory structure."
        )

    print(f"\n📦 Total samples: {total}  |  Feature dim: {X[0].shape[0]}")
    return np.array(X), np.array(y)


# ════════════════════════════════════════════════════════════
#  Plot helpers
# ════════════════════════════════════════════════════════════
def _save_loss_comparison(train_losses, val_losses, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="royalblue", lw=2)
    plt.plot(val_losses, label="Validation Loss", color="darkorange", lw=2)
    plt.title("Loss Comparison (Train vs Val)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def _save_accuracy_comparison(train_accs, val_accs, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy", color="royalblue", lw=2)
    plt.plot(val_accs, label="Validation Accuracy", color="darkorange", lw=2)
    plt.title("Accuracy Comparison (Train vs Val)")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def _save_class_metrics(y_true, y_pred, classes, filename):
    classes_str = [str(c) for c in classes]
    report = classification_report(y_true, y_pred, target_names=classes_str, output_dict=True)
    precisions = [report[c]["precision"] for c in classes_str]
    recalls = [report[c]["recall"] for c in classes_str]
    f1_scores = [report[c]["f1-score"] for c in classes_str]
    
    x = np.arange(len(classes_str))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="skyblue")
    ax.bar(x, recalls, width, label="Recall", color="lightgreen")
    ax.bar(x + width, f1_scores, width, label="F1-Score", color="salmon")
    
    ax.set_title("Emotion-wise Classification Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(classes_str, rotation=45)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def train_model(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, num_classes, F,
                lr=0.0005, batch_size=16, hidden_dim=128, dropout=0.3, label_smoothing=0.005, epochs=60,
                model_path=MODEL_PATH, plots_dir=PLOTS_DIR, save_plots=True, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = TensorDataset(X_train_tensor, y_train_tensor)
    val_set = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size)

    model = HybridCNN_LSTM(
        input_dim=F,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    # Class-weighted loss
    y_train_np = y_train_tensor.numpy()
    counts = np.bincount(y_train_np)
    weights = 1.0 / counts
    cw = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    patience_limit = 10
    epochs_no_improve = 0

    for epoch in range(epochs):
        # — train —
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        t_acc = correct / total
        train_losses.append(run_loss / len(train_loader))
        train_accs.append(t_acc)

        # — validate —
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                v_loss += loss.item()
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total += yb.size(0)

        v_acc = v_correct / v_total
        val_losses.append(v_loss / len(val_loader))
        val_accs.append(v_acc)
        scheduler.step(v_acc)
        current_lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f"  Epoch {epoch + 1:3d}/{epochs}  "
                  f"Train Acc: {t_acc:.4f}  Val Acc: {v_acc:.4f}  LR: {current_lr:.6f}", flush=True)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            if model_path:
                torch.save(model.state_dict(), model_path)
                if verbose:
                    print(f"  >> Best model saved  (val acc = {v_acc:.4f})", flush=True)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_limit:
                if verbose:
                    print(f"\n[INFO] Early stopping triggered after {epoch + 1} epochs. No improvement in validation accuracy for {patience_limit} consecutive epochs.", flush=True)
                train_losses = train_losses[:epoch+1]
                train_accs = train_accs[:epoch+1]
                val_losses = val_losses[:epoch+1]
                val_accs = val_accs[:epoch+1]
                break

    if save_plots and plots_dir:
        os.makedirs(plots_dir, exist_ok=True)
        # Load the best model weights for evaluation before plotting
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Get final predictions on validation loader using the best model
        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits, _ = model(xb)
                preds = logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        # Save comparison curves
        _save_loss_comparison(train_losses, val_losses, os.path.join(plots_dir, "loss_comparison.png"))
        _save_loss_comparison(train_losses, val_losses, os.path.join(plots_dir, "train_val_loss.png"))
        _save_accuracy_comparison(train_accs, val_accs, os.path.join(plots_dir, "accuracy_comparison.png"))
        _save_accuracy_comparison(train_accs, val_accs, os.path.join(plots_dir, "train_val_acc.png"))

        # Save individual curves
        plt.figure(); plt.plot(train_losses); plt.title("Train Loss"); plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "train_loss.png")); plt.close()
        plt.figure(); plt.plot(val_losses); plt.title("Validation Loss"); plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "val_loss.png")); plt.close()
        plt.figure(); plt.plot(train_accs); plt.title("Train Accuracy"); plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "train_acc.png")); plt.close()
        plt.figure(); plt.plot(val_accs); plt.title("Validation Accuracy"); plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "val_acc.png")); plt.close()

        # Class-wise metrics
        le = joblib.load(ENCODER_PATH)
        _save_class_metrics(all_targets, all_preds, le.classes_, os.path.join(plots_dir, "class_metrics.png"))
        _save_class_metrics(all_targets, all_preds, le.classes_, os.path.join(plots_dir, "emotion_metrics.png"))

        # Generate and print Markdown Table of metrics
        from sklearn.metrics import accuracy_score
        report = classification_report(all_targets, all_preds, target_names=le.classes_, output_dict=True)
        acc = accuracy_score(all_targets, all_preds)
        print("\n" + "=" * 60, flush=True)
        print("  Performance Metrics per Emotion Class", flush=True)
        print("=" * 60, flush=True)
        print("| Class | Precision (%) | Recall (%) | F1 Score (%) |", flush=True)
        print("| :--- | :---: | :---: | :---: |", flush=True)
        for cls_name in le.classes_:
            p = report[str(cls_name)]['precision'] * 100
            r = report[str(cls_name)]['recall'] * 100
            f1 = report[str(cls_name)]['f1-score'] * 100
            print(f"| {cls_name} | {p:.2f} | {r:.2f} | {f1:.2f} |", flush=True)
        print(f"| **Overall Accuracy** | **{acc * 100:.2f}%** | | |", flush=True)
        print("=" * 60 + "\n", flush=True)

        # Confusion matrix
        classes_str = [str(c) for c in le.classes_]
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes_str, yticklabels=classes_str, cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
        plt.close()

    return best_val_acc


def main():
    import librosa               # ensure available at runtime
    import joblib
    import torch
    torch.set_num_threads(4)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 1. Load Pre-extracted Features ─────────────────────
    print("=" * 60)
    print("Stage 1/5 : Loading pre-extracted features from 'cnn_lstm_seq.pkl' ...")
    print("=" * 60)
    
    features_pkl_path = "cnn_lstm_seq.pkl"
    if not os.path.exists(features_pkl_path):
        print(f"\n[ERROR] '{features_pkl_path}' not found!")
        print("Please extract sequential features first by running:")
        print("  python cnn_lstm_features.py")
        sys.exit(1)

    X_train, y_train, X_val, y_val = joblib.load(features_pkl_path)
    print(f"[OK] Loaded train features shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"[OK] Loaded val features shape: X_val={X_val.shape}, y_val={y_val.shape}")

    # ── 2. Encode & scale ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 2/5 : Encoding labels & scaling sequential features ...")
    print("=" * 60)
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc = le.transform(y_val)
    joblib.dump(le, ENCODER_PATH)
    print(f"  Classes: {list(le.classes_)}")

    # Scaling 3D sequential data: (N, F, T) -> fit scaler on flattened (N*T, F)
    N_tr, F, T = X_train.shape
    X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, F)
    
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    joblib.dump(scaler, SCALER_PATH)

    # Scale Train
    X_train_scaled = scaler.transform(X_train_flat).reshape(N_tr, T, F).transpose(0, 2, 1)

    # Scale Val
    N_va, _, _ = X_val.shape
    X_val_flat = X_val.transpose(0, 2, 1).reshape(-1, F)
    X_val_scaled = scaler.transform(X_val_flat).reshape(N_va, T, F).transpose(0, 2, 1)

    print(f"  Scaled feature shape: Train={X_train_scaled.shape}, Val={X_val_scaled.shape}")

    # ── 3. Convert to tensors ──────────────────────────────
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_enc, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val_enc, dtype=torch.long)

    # ── 4. Train ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Stage 3/5 : Building and training model ...")
    print("=" * 60)

    best_val_acc = train_model(
        X_train_tensor=X_train_tensor,
        y_train_tensor=y_train_tensor,
        X_val_tensor=X_val_tensor,
        y_val_tensor=y_val_tensor,
        num_classes=len(le.classes_),
        F=F,
        lr=LR,
        batch_size=BATCH_SIZE,
        hidden_dim=128,
        dropout=0.3,
        label_smoothing=0.005,
        epochs=EPOCHS,
        model_path=MODEL_PATH,
        plots_dir=PLOTS_DIR,
        save_plots=True,
        verbose=True
    )

    print(f"\n--- Training complete. ---")
    print(f"   Best validation accuracy : {best_val_acc:.4f}")
    print(f"   Model saved to           : {MODEL_PATH}")
    print(f"   Plots saved to           : {PLOTS_DIR}/")


if __name__ == "__main__":
    import librosa               # top-level guard
    main()
