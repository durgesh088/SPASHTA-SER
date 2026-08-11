"""
Hybrid 1D-CNN + LSTM Model for Hindi SER

Architecture
------------
Input (B, selected_dim)
  → Unsqueeze → (B, 1, selected_dim)
  → Conv1D(1→64,  k=5, p=2) + BN + ReLU
  → Conv1D(64→128, k=5, p=2) + BN + ReLU
  → MaxPool1d(2)
  → Conv1D(128→128, k=3, p=1) + BN + ReLU
  → AdaptiveAvgPool1d(50)        → (B, 128, 50)
  → Permute                       → (B, 50, 128)
  → BiLSTM(128→64, layers=2)      → (B, 50, 128)
  → Last hidden state concat      → (B, 128)
  → FC(128→64) + ReLU + Dropout
  → FC(64→num_classes)
"""

import torch
import torch.nn as nn


class HybridCNN_LSTM(nn.Module):
    """
    Hybrid 1D-CNN + Bidirectional LSTM classifier.

    The CNN front-end extracts local spectral-temporal patterns while
    the BiLSTM backend captures long-range sequential dependencies.
    This is intentionally a *different* architecture from the existing
    CNN_GRU_Attention to prove SPASHTA's model-agnostic claim.

    Parameters
    ----------
    input_dim : int
        Number of input features (after LASSO selection, ~360).
    hidden_dim : int
        LSTM hidden state size per direction.
    num_classes : int
        Number of emotion classes (8 for the Hindi dataset).
    """

    def __init__(self, input_dim=161, hidden_dim=128, num_classes=8, dropout=0.3):
        super().__init__()

        # ── CNN front-end ──
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Down-sample
            nn.MaxPool1d(kernel_size=2),

            # Block 3
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Fixed-length output
            nn.AdaptiveAvgPool1d(50),         # → (B, 128, 50)
        )

        # ── BiLSTM backend ──
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        # ── Classifier head ──
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),                  # Regularize BiLSTM output
            nn.Linear(hidden_dim * 4, 64),    # Concatenated average + max pooling (hidden_dim * 2 * 2)
            nn.ReLU(),
            nn.Dropout(dropout),                  # Increased classifier dropout
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, input_dim, T).

        Returns
        -------
        logits : torch.Tensor
            Shape (B, num_classes).
        lstm_out : torch.Tensor
            Full LSTM output (B, 50, 2*hidden_dim) — kept for
            potential attention / interpretability hooks.
        """
        x = self.cnn(x)                      # (B, 128, 50)
        x = x.permute(0, 2, 1)               # (B, 50, 128)

        lstm_out, _ = self.lstm(x)            # lstm_out: (B, 50, 2*hidden_dim)

        # Global Average Pooling and Global Max Pooling over temporal frames
        avg_pool = torch.mean(lstm_out, dim=1)  # (B, 2*hidden_dim)
        max_pool, _ = torch.max(lstm_out, dim=1)  # (B, 2*hidden_dim)
        hidden = torch.cat([avg_pool, max_pool], dim=1)  # (B, 4*hidden_dim)

        logits = self.classifier(hidden)      # (B, num_classes)
        return logits, lstm_out
