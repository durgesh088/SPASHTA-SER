import torch
import torch.nn as nn

class CNN_GRU_Attention(nn.Module):
    def __init__(self, input_dim=374, hidden_dim=128, num_classes=8):
        super().__init__()

        # CNN for initial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(50)  # Output shape: (B, 64, 50)
        )

        # GRU for sequence modeling
        self.gru = nn.GRU(
            input_size=64, hidden_size=hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True
        )

        # Attention layer
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)             # (B, 1, 374)
        x = self.cnn(x)                # (B, 64, 50)
        x = x.permute(0, 2, 1)         # (B, 50, 64)

        gru_out, _ = self.gru(x)       # (B, 50, 2*hidden_dim)

        # Attention
        attn_scores = self.attn(gru_out)   # (B, 50, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, 50, 1)
        context = torch.sum(attn_weights * gru_out, dim=1)  # (B, 2*hidden_dim)

        out = self.classifier(context)  # (B, num_classes)
        return out, attn_weights.squeeze(-1)  # attention weights optional

