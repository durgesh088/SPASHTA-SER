"""
Training Pipeline for the SE-ResNet + BiGRU + Attention Hindi SER model

Stages
------
1. Load pre-extracted 3-channel spectrograms from  se_resnet_seq.pkl
2. Encode labels  +  per-channel standardisation
3. Train  SEResNet_BiGRU_Attention  for 60 epochs (early stopping patience=10)
4. Save model, encoder, scaler, plots
"""

import os
import sys
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from se_resnet_model import SEResNet_BiGRU_Attention


# ════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════
DATASET_PKL  = "se_resnet_seq.pkl"
EPOCHS       = 60
BATCH_SIZE   = 32
LR           = 0.0005
LABEL_SMOOTH = 0.005

MODEL_PATH   = "se_resnet_best_model.pt"
ENCODER_PATH = "se_resnet_label_encoder.pkl"
SCALER_PATH  = "se_resnet_scaler.pkl"
PLOTS_DIR    = "se_resnet_plots"


# ════════════════════════════════════════════════════════════
#  Plot helpers
# ════════════════════════════════════════════════════════════
def _save_loss_comparison(train_losses, val_losses, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="royalblue", lw=2)
    plt.plot(val_losses,   label="Validation Loss", color="darkorange", lw=2)
    plt.title("Loss Comparison (Train vs Val)")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def _save_accuracy_comparison(train_accs, val_accs, filename):
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label="Train Accuracy", color="royalblue", lw=2)
    plt.plot(val_accs,   label="Validation Accuracy", color="darkorange", lw=2)
    plt.title("Accuracy Comparison (Train vs Val)")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


def _save_class_metrics(y_true, y_pred, classes, filename):
    classes_str = [str(c) for c in classes]
    report = classification_report(y_true, y_pred,
                                   target_names=classes_str, output_dict=True)
    precisions = [report[c]["precision"] for c in classes_str]
    recalls    = [report[c]["recall"]    for c in classes_str]
    f1_scores  = [report[c]["f1-score"]  for c in classes_str]

    x = np.arange(len(classes_str))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precisions, width, label="Precision", color="skyblue")
    ax.bar(x,         recalls,    width, label="Recall",    color="lightgreen")
    ax.bar(x + width, f1_scores,  width, label="F1-Score",  color="salmon")
    ax.set_title("Emotion-wise Classification Metrics")
    ax.set_xticks(x); ax.set_xticklabels(classes_str, rotation=45)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout(); plt.savefig(filename, dpi=150); plt.close()


# ════════════════════════════════════════════════════════════
#  Per-channel scaler for 4-D spectrograms
# ════════════════════════════════════════════════════════════
class SpectrogramScaler:
    """
    Standardises each of the 3 spectrogram channels independently.

    For a dataset of shape (N, 3, 64, 501), computes per-channel
    mean and std across all pixels and all samples, then applies
    z-normalisation.
    """

    def __init__(self):
        self.means = None    # shape (3,)
        self.stds  = None    # shape (3,)

    def fit(self, X):
        """X : np.ndarray, shape (N, 3, H, W)"""
        # Mean/std across N, H, W — one value per channel
        self.means = X.mean(axis=(0, 2, 3))               # (3,)
        self.stds  = X.std(axis=(0, 2, 3))                 # (3,)
        self.stds[self.stds < 1e-8] = 1.0                  # safety
        return self

    def transform(self, X):
        """X : np.ndarray, shape (N, 3, H, W) or (3, H, W)"""
        if X.ndim == 3:
            # single sample
            return ((X - self.means[:, None, None])
                    / self.stds[:, None, None]).astype(np.float32)
        # batch
        return ((X - self.means[None, :, None, None])
                / self.stds[None, :, None, None]).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ════════════════════════════════════════════════════════════
#  Training loop
# ════════════════════════════════════════════════════════════
def train_model(X_train_t, y_train_t, X_val_t, y_val_t, num_classes,
                lr=LR, batch_size=BATCH_SIZE, gru_hidden=128,
                dropout=0.5, label_smoothing=LABEL_SMOOTH, weight_decay=1e-4,
                epochs=EPOCHS, model_path=MODEL_PATH,
                plots_dir=PLOTS_DIR, save_plots=True, verbose=True):
    """
    Train the SE-ResNet + BiGRU + Attention model.

    Parameters
    ----------
    X_train_t, y_train_t : torch.Tensor
        Training data and labels.
    X_val_t, y_val_t : torch.Tensor
        Validation data and labels.
    num_classes : int
    lr, batch_size, gru_hidden, dropout, label_smoothing, epochs : various
        Hyperparameters.
    model_path : str
        Where to save best weights.
    plots_dir : str
        Where to save evaluation plots.
    save_plots : bool
    verbose : bool

    Returns
    -------
    float
        Best validation accuracy achieved.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size, pin_memory=True)

    model = SEResNet_BiGRU_Attention(
        num_classes=num_classes,
        gru_hidden=gru_hidden,
        dropout=dropout,
    ).to(device)

    # Class-weighted loss
    y_np   = y_train_t.numpy()
    counts = np.bincount(y_np)
    w      = 1.0 / counts
    cw     = torch.tensor(w, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw,
                                     label_smoothing=label_smoothing)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []
    best_val_acc = 0.0
    patience_limit = 10
    epochs_no_improve = 0

    scaler_amp = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, _ = model(xb)
                loss = criterion(logits, yb)
                
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            run_loss += loss.item()
            correct  += (logits.argmax(1) == yb).sum().item()
            total    += yb.size(0)

        t_acc = correct / total
        train_losses.append(run_loss / len(train_loader))
        train_accs.append(t_acc)

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, _ = model(xb)
                    loss = criterion(logits, yb)
                v_loss    += loss.item()
                v_correct += (logits.argmax(1) == yb).sum().item()
                v_total   += yb.size(0)

        v_acc = v_correct / v_total
        val_losses.append(v_loss / len(val_loader))
        val_accs.append(v_acc)
        scheduler.step(v_acc)
        cur_lr = optimizer.param_groups[0]['lr']

        if verbose:
            print(f"  Epoch {epoch+1:3d}/{epochs}  "
                  f"Train Acc: {t_acc:.4f}  Val Acc: {v_acc:.4f}  "
                  f"LR: {cur_lr:.6f}", flush=True)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            if model_path:
                torch.save(model.state_dict(), model_path)
                if verbose:
                    print(f"  >> Best model saved  (val acc = {v_acc:.4f})",
                          flush=True)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_limit:
                if verbose:
                    print(f"\n[INFO] Early stopping after {epoch+1} epochs "
                          f"(no improvement for {patience_limit} epochs).",
                          flush=True)
                break

    # ── Evaluation & Plots ──
    if save_plots and plots_dir:
        os.makedirs(plots_dir, exist_ok=True)

        # Reload best weights
        if model_path and os.path.exists(model_path):
            model.load_state_dict(
                torch.load(model_path, map_location=device))
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits, _ = model(xb)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_targets.extend(yb.cpu().numpy())

        # Curves
        _save_loss_comparison(train_losses, val_losses,
                              os.path.join(plots_dir, "loss_comparison.png"))
        _save_accuracy_comparison(train_accs, val_accs,
                                  os.path.join(plots_dir, "accuracy_comparison.png"))

        # Individual curves
        plt.figure(); plt.plot(train_losses); plt.title("Train Loss")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "train_loss.png")); plt.close()
        plt.figure(); plt.plot(val_losses); plt.title("Validation Loss")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "val_loss.png")); plt.close()
        plt.figure(); plt.plot(train_accs); plt.title("Train Accuracy")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "train_acc.png")); plt.close()
        plt.figure(); plt.plot(val_accs); plt.title("Validation Accuracy")
        plt.tight_layout(); plt.savefig(os.path.join(plots_dir, "val_acc.png")); plt.close()

        # Class metrics
        le = joblib.load(ENCODER_PATH)
        _save_class_metrics(all_targets, all_preds, le.classes_,
                            os.path.join(plots_dir, "class_metrics.png"))

        # Markdown table
        report = classification_report(all_targets, all_preds,
                                       target_names=le.classes_,
                                       output_dict=True)
        acc = accuracy_score(all_targets, all_preds)
        print("\n" + "=" * 60, flush=True)
        print("  Performance Metrics per Emotion Class", flush=True)
        print("=" * 60, flush=True)
        print("| Class | Precision (%) | Recall (%) | F1 Score (%) |",
              flush=True)
        print("| :--- | :---: | :---: | :---: |", flush=True)
        for cls in le.classes_:
            p  = report[str(cls)]['precision'] * 100
            r  = report[str(cls)]['recall']    * 100
            f1 = report[str(cls)]['f1-score']  * 100
            print(f"| {cls} | {p:.2f} | {r:.2f} | {f1:.2f} |", flush=True)
        print(f"| **Overall Accuracy** | **{acc*100:.2f}%** | | |",
              flush=True)
        print("=" * 60 + "\n", flush=True)

        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d",
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
        plt.close()

    return best_val_acc


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════
def main():
    torch.set_num_threads(4)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 1. Load features ──
    print("=" * 60)
    print(f"Stage 1/4 : Loading features from '{DATASET_PKL}' ...")
    print("=" * 60)

    if not os.path.exists(DATASET_PKL):
        print(f"\n[ERROR] '{DATASET_PKL}' not found!")
        print("Extract features first:  python se_resnet_features.py")
        sys.exit(1)

    X_train, y_train, X_val, y_val = joblib.load(DATASET_PKL)
    print(f"[OK] Train : X={X_train.shape}, y={y_train.shape}")
    print(f"[OK] Val   : X={X_val.shape},   y={y_val.shape}")

    # ── 2. Encode & scale ──
    print("\n" + "=" * 60)
    print("Stage 2/4 : Encoding labels & scaling spectrograms ...")
    print("=" * 60)

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_val_enc   = le.transform(y_val)
    joblib.dump(le, ENCODER_PATH)
    print(f"  Classes: {list(le.classes_)}")

    scaler = SpectrogramScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Scaler channel means : {scaler.means}")
    print(f"  Scaler channel stds  : {scaler.stds}")

    # ── 3. Tensors ──
    X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_enc, dtype=torch.long)
    X_val_t   = torch.tensor(X_val_sc,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val_enc,  dtype=torch.long)

    # ── 4. Train ──
    print("\n" + "=" * 60)
    print("Stage 3/4 : Training SE-ResNet + BiGRU + Attention ...")
    print("=" * 60)

    best = train_model(
        X_train_t, y_train_t, X_val_t, y_val_t,
        num_classes=len(le.classes_),
        lr=LR, batch_size=BATCH_SIZE,
        gru_hidden=128, dropout=0.5,
        label_smoothing=LABEL_SMOOTH,
        weight_decay=1e-4,
        epochs=EPOCHS,
        model_path=MODEL_PATH,
        plots_dir=PLOTS_DIR,
        save_plots=True, verbose=True,
    )

    print(f"\n--- Training complete. ---")
    print(f"   Best validation accuracy : {best:.4f}")
    print(f"   Model saved to           : {MODEL_PATH}")
    print(f"   Plots saved to           : {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
