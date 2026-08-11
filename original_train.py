"""
Training Pipeline for the CNN-BiGRU-Attention (Original) Hindi SER model

Saves all training artifacts and plots to original_plots/ directory,
matching the same output structure as the CNN-LSTM training pipeline.
"""

import os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from original_model import CNN_GRU_Attention  # Make sure your model file is updated

# ════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════
EPOCHS       = 60
BATCH_SIZE   = 32
LR           = 0.001
TRAIN_SPLIT  = 0.8

# Output artefacts
MODEL_PATH   = "original_best_model.pt"
ENCODER_PATH = "original_label_encoder.pkl"
SCALER_PATH  = "original_scaler.pkl"
PLOTS_DIR    = "original_plots"


# ════════════════════════════════════════════════════════════
#  Plot helpers  (same style as CNN-LSTM)
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


def _save_class_precision(y_true, y_pred, classes, filename):
    classes_str = [str(c) for c in classes]
    report = classification_report(y_true, y_pred, target_names=classes_str, output_dict=True)
    precision_per_class = [report[c]['precision'] for c in classes_str]

    plt.figure(figsize=(10, 6))
    plt.bar(classes_str, precision_per_class, color="steelblue")
    plt.title("Class-wise Precision")
    plt.ylabel("Precision")
    plt.xticks(rotation=45)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════
#  Main Training
# ════════════════════════════════════════════════════════════
def main():
    torch.set_num_threads(4)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ── 1. Load features ───────────────────────────────────
    print("=" * 60)
    print("Stage 1/4 : Loading pre-extracted features ...")
    print("=" * 60)

    X, y = joblib.load("original_features.pkl")
    print(f"Loaded data: {X.shape}, labels: {len(y)}")

    # ── 2. Encode & Scale ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 2/4 : Encoding labels & scaling features ...")
    print("=" * 60)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"  Classes: {list(le.classes_)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Convert to tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    # Dataset and split
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    print(f"  Train: {train_size} samples | Val: {val_size} samples")

    # ── 3. Build & Train ───────────────────────────────────
    print("\n" + "=" * 60)
    print("Stage 3/4 : Building and training model ...")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN_GRU_Attention(input_dim=374, num_classes=len(le.classes_)).to(device)

    # Weighted loss for imbalance
    class_counts = np.bincount(y_encoded)
    weights = 1. / class_counts
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    # Training loop
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    patience_limit = 10
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs, _ = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_acc = correct / total
        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs, _ = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item()

                preds = outputs.argmax(1)
                val_correct += (preds == yb).sum().item()
                val_total += yb.size(0)

        val_acc = val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}  LR: {current_lr:.6f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  >> Best model saved  (val acc = {val_acc:.4f})", flush=True)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience_limit:
                print(f"\n[INFO] Early stopping triggered after {epoch + 1} epochs. "
                      f"No improvement in validation accuracy for {patience_limit} consecutive epochs.", flush=True)
                train_losses = train_losses[:epoch+1]
                train_accuracies = train_accuracies[:epoch+1]
                val_losses = val_losses[:epoch+1]
                val_accuracies = val_accuracies[:epoch+1]
                break

    # Save encoder and scaler
    joblib.dump(le, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # ── 4. Generate Plots & Metrics ────────────────────────
    print("\n" + "=" * 60)
    print("Stage 4/4 : Generating evaluation plots ...")
    print("=" * 60)

    # Load the best model weights for evaluation
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Get final predictions on validation set using the best model
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs, _ = model(xb)
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(yb.cpu().numpy())

    # ── Comparison curves ──
    _save_loss_comparison(train_losses, val_losses,
                          os.path.join(PLOTS_DIR, "loss_comparison.png"))
    _save_loss_comparison(train_losses, val_losses,
                          os.path.join(PLOTS_DIR, "train_val_loss.png"))
    _save_accuracy_comparison(train_accuracies, val_accuracies,
                              os.path.join(PLOTS_DIR, "accuracy_comparison.png"))
    _save_accuracy_comparison(train_accuracies, val_accuracies,
                              os.path.join(PLOTS_DIR, "train_val_acc.png"))

    # ── Individual curves ──
    plt.figure(); plt.plot(train_losses); plt.title("Train Loss"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "train_loss.png"), dpi=150); plt.close()
    plt.figure(); plt.plot(val_losses); plt.title("Validation Loss"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "val_loss.png"), dpi=150); plt.close()
    plt.figure(); plt.plot(train_accuracies); plt.title("Train Accuracy"); plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "train_acc.png"), dpi=150); plt.close()
    plt.figure(); plt.plot(val_accuracies); plt.title("Validation Accuracy"); plt.xlabel("Epochs"); plt.ylabel("Accuracy"); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "val_acc.png"), dpi=150); plt.close()

    # ── Class-wise metrics ──
    _save_class_metrics(all_targets, all_preds, le.classes_,
                        os.path.join(PLOTS_DIR, "class_metrics.png"))
    _save_class_metrics(all_targets, all_preds, le.classes_,
                        os.path.join(PLOTS_DIR, "emotion_metrics.png"))
    _save_class_precision(all_targets, all_preds, le.classes_,
                          os.path.join(PLOTS_DIR, "class_precision.png"))

    # ── Confusion Matrix ──
    classes_str = [str(c) for c in le.classes_]
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes_str,
                yticklabels=classes_str, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── Print Metrics Table ──
    report = classification_report(all_targets, all_preds,
                                   target_names=classes_str, output_dict=True)
    acc = accuracy_score(all_targets, all_preds)

    print("\n" + "=" * 60)
    print("  Performance Metrics per Emotion Class")
    print("=" * 60)
    print("| Class | Precision (%) | Recall (%) | F1 Score (%) |")
    print("| :--- | :---: | :---: | :---: |")
    for cls_name in le.classes_:
        p = report[str(cls_name)]['precision'] * 100
        r = report[str(cls_name)]['recall'] * 100
        f1 = report[str(cls_name)]['f1-score'] * 100
        print(f"| {cls_name} | {p:.2f} | {r:.2f} | {f1:.2f} |")
    print(f"| **Overall Accuracy** | **{acc * 100:.2f}%** | | |")
    print("=" * 60 + "\n")

    print(f"\n--- Training complete. ---")
    print(f"   Best validation accuracy : {best_val_acc:.4f}")
    print(f"   Model saved to           : {MODEL_PATH}")
    print(f"   Plots saved to           : {PLOTS_DIR}/")
    print(f"\n[SUCCESS] All graphs and model files saved to '{PLOTS_DIR}/'.")


if __name__ == "__main__":
    main()
