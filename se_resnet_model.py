"""
SE-ResNet + BiGRU + Attention Model for Hindi SER

Architecture (from Chinese SER paper, adapted for Hindi dataset)
-----------------------------------------------------------------
Input (B, 3, 64, 501)                     ← 3-channel spectrogram
  → Conv2D(3→64, k=5, p=2) + BN + ReLU   ← Initial feature extraction
  → SE-ResNet Block (64→64)   + MaxPool2D(2)
  → SE-ResNet Block (64→128)  + MaxPool2D(2)
  → SE-ResNet Block (128→256) + MaxPool2D(2)
  → SE-ResNet Block (256→256) + MaxPool2D(2)
  → Spatial Flatten: (B, C*H, W) → (B, W, C*H)
  → BiGRU(input=C*H, hidden=128, layers=2) → (B, T, 256)
  → Attention                               → (B, 256)
  → Linear(256,128) + ReLU + Dropout(0.4)
  → Linear(128, num_classes)
  → Output: (logits, attention_weights)

Modules
-------
  SEModule              – Squeeze-and-Excitation channel attention
  SEResNetBlock         – Residual block with SE and MaxPool
  AttentionLayer        – Learned soft attention over temporal steps
  SEResNet_BiGRU_Attention – Complete end-to-end classifier
"""

import torch
import torch.nn as nn
import torch.nn.init as init


# ════════════════════════════════════════════════════════════
#  Squeeze-and-Excitation Module
# ════════════════════════════════════════════════════════════
class SEModule(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Adaptively recalibrates channel-wise feature responses by
    modelling inter-channel dependencies.

    Parameters
    ----------
    channels : int
        Number of input channels.
    reduction : int
        Reduction ratio for the bottleneck FC layers (default 16).
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool2d(1)          # (B, C, 1, 1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),                    # FC(C → C//16)
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),                    # FC(C//16 → C)
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor   Shape (B, C, H, W)

        Returns
        -------
        torch.Tensor        Shape (B, C, H, W) — channel-recalibrated
        """
        B, C, _, _ = x.shape
        scale = self.squeeze(x).view(B, C)               # (B, C)
        scale = self.excitation(scale).view(B, C, 1, 1)   # (B, C, 1, 1)
        return x * scale                                  # channel-wise scaling


# ════════════════════════════════════════════════════════════
#  SE-ResNet Block
# ════════════════════════════════════════════════════════════
class SEResNetBlock(nn.Module):
    """
    Residual block with two 3×3 convolutions, Squeeze-and-Excitation
    channel attention, and 2×2 max-pooling for spatial downsampling.

    If ``in_channels != out_channels`` a 1×1 projection convolution is
    added to the skip connection so dimensions match for the residual add.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    reduction : int
        SE reduction ratio (default 16).
    """

    def __init__(self, in_channels, out_channels, reduction=16):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.se    = SEModule(out_channels, reduction)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool2d(kernel_size=2)

        # Projection shortcut when channel dimensions change
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor   Shape (B, C_in, H, W)

        Returns
        -------
        torch.Tensor        Shape (B, C_out, H//2, W//2)
        """
        identity = self.shortcut(x)                      # (B, C_out, H, W)

        out = self.relu(self.bn1(self.conv1(x)))          # (B, C_out, H, W)
        out = self.bn2(self.conv2(out))                   # (B, C_out, H, W)
        out = self.se(out)                                # (B, C_out, H, W) — SE recalibration

        out = self.relu(out + identity)                   # Residual add + ReLU
        out = self.pool(out)                              # (B, C_out, H//2, W//2)
        return out


# ════════════════════════════════════════════════════════════
#  Attention Layer
# ════════════════════════════════════════════════════════════
class AttentionLayer(nn.Module):
    """
    Learned soft attention over temporal steps.

    For each timestep produces a scalar score; softmax normalises
    scores across timesteps; the output is the weighted sum of
    BiGRU hidden states (context vector).

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of each input timestep vector (= 2 * GRU hidden).
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, gru_out):
        """
        Parameters
        ----------
        gru_out : torch.Tensor   Shape (B, T, hidden_dim)

        Returns
        -------
        context : torch.Tensor   Shape (B, hidden_dim)
        weights : torch.Tensor   Shape (B, T) — attention distribution
        """
        scores  = self.attn(gru_out)                      # (B, T, 1)
        weights = torch.softmax(scores, dim=1)            # (B, T, 1)
        context = torch.sum(weights * gru_out, dim=1)     # (B, hidden_dim)
        return context, weights.squeeze(-1)                # (B, hidden_dim), (B, T)


# ════════════════════════════════════════════════════════════
#  Complete Model
# ════════════════════════════════════════════════════════════
class SEResNet_BiGRU_Attention(nn.Module):
    """
    SE-ResNet + BiGRU + Attention end-to-end Speech Emotion
    Recognition classifier.

    Parameters
    ----------
    num_classes : int
        Number of emotion categories (default 8).
    gru_hidden : int
        Hidden size per direction for the BiGRU (default 128).
    gru_layers : int
        Number of stacked BiGRU layers (default 2).
    dropout : float
        Dropout probability in the classifier head (default 0.4).
    """

    def __init__(self, num_classes=8, gru_hidden=128, gru_layers=2,
                 dropout=0.4):
        super().__init__()

        # ── Initial Conv2D layer ──
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )                                                  # (B, 64, 64, 501)

        # ── 4× SE-ResNet Blocks ──
        self.block1 = SEResNetBlock(64,  64)               # → (B,  64, 32, 250)
        self.block2 = SEResNetBlock(64, 128)               # → (B, 128, 16, 125)
        self.block3 = SEResNetBlock(128, 256)              # → (B, 256,  8,  62)
        self.block4 = SEResNetBlock(256, 256)              # → (B, 256,  4,  31)

        # After block4: shape is (B, 256, 4, 31)
        # Spatial flatten: merge C and H → feature dim = 256 * 4 = 1024
        # Width (31) becomes the temporal sequence length
        self._gru_input_dim = 256 * 4                      # 1024

        # ── BiGRU ──
        self.gru = nn.GRU(
            input_size=self._gru_input_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if gru_layers > 1 else 0.0,
        )                                                  # → (B, 31, 256)

        # ── Attention ──
        self.attention = AttentionLayer(gru_hidden * 2)    # input dim = 256

        # ── Classifier head ──
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden * 2, 128),                # 256 → 128
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),                   # 128 → 8
        )

        # ── Weight initialisation ──
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming init for Conv2d and Xavier init for Linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out',
                                     nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 3, 64, 501) — 3-channel spectrogram.

        Returns
        -------
        logits : torch.Tensor
            Shape (B, num_classes).
        attn_weights : torch.Tensor
            Shape (B, T) — attention distribution over temporal steps.
        """
        # ── CNN backbone ──
        x = self.init_conv(x)                              # (B,  64, 64, 501)
        x = self.block1(x)                                 # (B,  64, 32, 250)
        x = self.block2(x)                                 # (B, 128, 16, 125)
        x = self.block3(x)                                 # (B, 256,  8,  62)
        x = self.block4(x)                                 # (B, 256,  4,  31)

        # ── Spatial flattening ──
        B, C, H, W = x.shape
        x = x.reshape(B, C * H, W)                        # (B, 1024, 31)
        x = x.permute(0, 2, 1)                            # (B, 31, 1024)

        # ── BiGRU ──
        gru_out, _ = self.gru(x)                           # (B, 31, 256)

        # ── Attention ──
        context, attn_weights = self.attention(gru_out)    # (B, 256), (B, 31)

        # ── Classifier ──
        logits = self.classifier(context)                  # (B, num_classes)

        return logits, attn_weights


# ════════════════════════════════════════════════════════════
#  Model summary and inference example
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SEResNet_BiGRU_Attention(num_classes=8).to(device)

    # ── Model summary ──
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print("  SE-ResNet + BiGRU + Attention  —  Model Summary")
    print("=" * 60)
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable:,}")
    print(f"  Device               : {device}")
    print("=" * 60)

    # ── Layer-by-layer breakdown ──
    print("\nLayer breakdown:")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s}  params={n_params:>10,}")

    # ── Inference example ──
    print("\n--- Inference Example ---")
    dummy = torch.randn(2, 3, 64, 501).to(device)
    print(f"  Input shape : {dummy.shape}")

    model.eval()
    with torch.no_grad():
        logits, attn_w = model(dummy)

    print(f"  Logits shape         : {logits.shape}")      # (2, 8)
    print(f"  Attention weights    : {attn_w.shape}")       # (2, 31)
    print(f"  Predicted classes    : {logits.argmax(dim=1).tolist()}")
    print(f"  Attention sum (row)  : {attn_w.sum(dim=1).tolist()}")
    print("\n[OK] Model instantiation and forward pass successful.")
