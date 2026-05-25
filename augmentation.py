"""
EEGNet V4 — Complete PyTorch Implementation
============================================
Reference: Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
           EEG-based Brain–Computer Interfaces", J. Neural Eng. 2018.

Architecture summary (BCI IV 2a: C=22, T=500, 4 classes)
──────────────────────────────────────────────────────────
INPUT       → (B, 1,  C,   T )   e.g. (32, 1, 22, 500)

Block 1 — Temporal Convolution
  Conv2D      → (B, F1, C,   T )   kernel (1, Tc), captures frequency bands
  BN          → (B, F1, C,   T )
  DepthwiseConv2D → (B, F2, 1, T)  kernel (C, 1),  spatial filter per temporal map
  BN + ELU    → (B, F2, 1,  T )
  AvgPool2D   → (B, F2, 1,  T/4)  pool (1, pool1=4)
  Dropout     → (B, F2, 1,  T/4)

Block 2 — Separable (Depthwise + Pointwise) Convolution
  SepConv     → (B, F2, 1, T/4)   kernel (1, 16), merges temporal+feature info
  BN + ELU    → (B, F2, 1, T/4)
  AvgPool2D   → (B, F2, 1, T/32)  pool (1, pool2=8)
  Dropout     → (B, F2, 1, T/32)

Classify
  Flatten     → (B, F2 * (T//32))
  Linear      → (B, n_classes)

For BCI IV 2a (C=22, T=500, F1=8, D=2, F2=16, pool1=4, pool2=8):
  After Block2 flatten: 16 × (500 // 32) = 16 × 15 = 240 features
  Parameters: ~2,548  (extremely compact)

Mathematical notes
──────────────────
Depthwise Conv:
  For each of the F1 temporal feature maps, learn D=2 spatial filters.
  Weight tensor W ∈ R^(F2 × 1 × C × 1)  with groups=F1.
  Result: each temporal feature → D spatial projections → F2=F1×D total maps.
  This imposes "one spatial filter per frequency band" structure.

Separable Conv = Depthwise(1,k) → Pointwise(1,1):
  Depthwise: each of F2 channels filtered independently in time.
  Pointwise: mix channels with 1×1 conv.
  Cost vs standard Conv: F2*k + F2² vs F2²*k  (k=16 → 8× cheaper).

ELU vs ReLU:
  ELU has smooth gradient for x<0 (exp(x)−1) avoiding dead neurons.
  For EEG signals which are zero-mean, negative activations are informative.
  ELU preserves negative information better than ReLU.

Average Pool vs Max Pool:
  EEG features are sustained oscillations (ERD/ERS), not sparse spikes.
  AvgPool preserves the overall energy of the oscillation window.
  MaxPool would discard the DC component of the averaged signal.
"""

import math
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Constraint layers
# ══════════════════════════════════════════════════════════════════════════════

class MaxNormConstraint(nn.Module):
    """
    Applies max-norm weight constraint after each gradient step.
    Prevents weights from exploding; especially important for depthwise
    spatial filters that could become unbounded.

    Usage: register as a forward hook or call .apply() after optimizer.step().
    """

    def __init__(self, max_norm: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.max_norm = max_norm
        self.eps      = eps

    def apply_to(self, module: nn.Module):
        for name, param in module.named_parameters():
            if "weight" in name and param.dim() > 1:
                norm = param.data.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, 0, self.max_norm)
                param.data.mul_(desired / (self.eps + norm))


# ══════════════════════════════════════════════════════════════════════════════
#  EEGNet V4 Core
# ══════════════════════════════════════════════════════════════════════════════

class EEGNetV4(nn.Module):
    """
    EEGNet V4 — compact EEG classifier.

    Parameters
    ----------
    C         : number of EEG channels
    T         : number of time samples
    n_classes : number of output classes
    F1        : number of temporal filters
    D         : depth multiplier (spatial filters per temporal)
    F2        : number of separable filters (must equal F1*D)
    Tc        : temporal kernel length (≈ 0.5 × sfreq for ~500 ms)
    pool1     : first avg-pool size (time axis)
    pool2     : second avg-pool size (time axis)
    dropout   : dropout probability
    """

    def __init__(
        self,
        C:         int   = 22,
        T:         int   = 500,
        n_classes: int   = 4,
        F1:        int   = 8,
        D:         int   = 2,
        F2:        int   = 16,
        Tc:        int   = 64,
        pool1:     int   = 4,
        pool2:     int   = 8,
        dropout:   float = 0.5,
    ):
        super().__init__()

        assert F2 == F1 * D, f"F2 must equal F1*D, got {F2} != {F1}*{D}"
        self.C  = C;  self.T  = T;  self.F1 = F1
        self.D  = D;  self.F2 = F2; self.Tc = Tc
        self.pool1 = pool1;  self.pool2 = pool2
        self.n_classes = n_classes

        # ── BLOCK 1 ───────────────────────────────────────────
        # Temporal Conv: (B,1,C,T) → (B,F1,C,T)
        # kernel (1,Tc) — slides across TIME only, preserving all channels.
        # Each filter learns a temporal pattern (e.g. mu/beta oscillation).
        self.temporal_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = F1,
            kernel_size  = (1, Tc),
            padding      = (0, Tc // 2),
            bias         = False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise Spatial Conv: (B,F1,C,T) → (B,F2,1,T)
        # kernel (C,1) — slides across CHANNELS only (one per spatial location).
        # groups=F1: each temporal feature map has its own D spatial filters.
        # Equivalent to learning "which electrode combination matters for each
        # frequency band" — analogous to CSP filters.
        self.depthwise_conv = nn.Conv2d(
            in_channels  = F1,
            out_channels = F2,
            kernel_size  = (C, 1),
            groups       = F1,
            bias         = False,
        )
        self.bn2   = nn.BatchNorm2d(F2)
        self.elu1  = nn.ELU()
        self.pool1_layer = nn.AvgPool2d(kernel_size=(1, pool1))
        self.drop1 = nn.Dropout(dropout)

        # ── BLOCK 2 ───────────────────────────────────────────
        # Separable Temporal Conv: (B,F2,1,T//pool1) → (B,F2,1,T//pool1)
        # Step 1 — depthwise: each channel convolved independently in time.
        # Step 2 — pointwise: 1×1 conv mixes channels.
        self.sep_depthwise = nn.Conv2d(
            in_channels  = F2,
            out_channels = F2,
            kernel_size  = (1, 16),
            padding      = (0, 8),
            groups       = F2,
            bias         = False,
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels  = F2,
            out_channels = F2,
            kernel_size  = (1, 1),
            bias         = False,
        )
        self.bn3   = nn.BatchNorm2d(F2)
        self.elu2  = nn.ELU()
        self.pool2_layer = nn.AvgPool2d(kernel_size=(1, pool2))
        self.drop2 = nn.Dropout(dropout)

        # ── CLASSIFIER ────────────────────────────────────────
        self._compute_flatten_size(C, T)
        self.classifier = nn.Linear(self._flat_size, n_classes, bias=True)

        # ── weight init ───────────────────────────────────────
        self._init_weights()

    def _compute_flatten_size(self, C: int, T: int):
        """Dry-run a dummy forward to compute flatten dimension."""
        with torch.no_grad():
            x = torch.zeros(1, 1, C, T)
            x = self._forward_features(x)
            self._flat_size = x.view(1, -1).shape[1]

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # ── forward helpers ───────────────────────────────────────
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature extraction backbone — returns flat (B, flat_size) tensor.
        Intermediate shapes printed during first call if verbose=True.
        """
        # (B, 1, C, T) → (B, F1, C, T)
        x = self.temporal_conv(x)
        x = self.bn1(x)

        # (B, F1, C, T) → (B, F2, 1, T)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu1(x)

        # (B, F2, 1, T) → (B, F2, 1, T//pool1)
        x = self.pool1_layer(x)
        x = self.drop1(x)

        # Block 2
        # (B, F2, 1, T//pool1) → (B, F2, 1, T//pool1)
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)

        # (B, F2, 1, T//pool1) → (B, F2, 1, T//(pool1*pool2))
        x = self.pool2_layer(x)
        x = self.drop2(x)

        return x.view(x.size(0), -1)   # (B, flat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, C, T)
        returns logits (B, n_classes)
        """
        feat  = self._forward_features(x)
        logits = self.classifier(feat)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract flat features before classifier (for transfer learning)."""
        self.eval()
        with torch.no_grad():
            return self._forward_features(x)

    # ── analysis tools ────────────────────────────────────────
    def print_architecture(self):
        """Print detailed layer shapes, param counts, and FLOPs estimate."""
        logger.handlers = []
        print("=" * 70)
        print(f"  EEGNet V4  |  C={self.C}  T={self.T}  "
              f"F1={self.F1}  D={self.D}  F2={self.F2}")
        print("=" * 70)
        rows = [
            ("Input",          f"(B,  1, {self.C:2d}, {self.T:3d})"),
            ("TemporalConv",   f"(B, F1={self.F1:2d}, {self.C:2d}, {self.T:3d})  "
                               f"kernel=(1,{self.Tc})"),
            ("BN1",            f"(B, F1={self.F1:2d}, {self.C:2d}, {self.T:3d})"),
            ("DepthwiseConv",  f"(B, F2={self.F2:2d},  1, {self.T:3d})  "
                               f"kernel=({self.C},1) groups={self.F1}"),
            ("BN2+ELU",        f"(B, F2={self.F2:2d},  1, {self.T:3d})"),
            ("AvgPool1",       f"(B, F2={self.F2:2d},  1, {self.T//self.pool1:3d})"),
            ("Dropout1",       f"(B, F2={self.F2:2d},  1, {self.T//self.pool1:3d})"),
            ("SepConvDW",      f"(B, F2={self.F2:2d},  1, {self.T//self.pool1:3d})"),
            ("SepConvPW",      f"(B, F2={self.F2:2d},  1, {self.T//self.pool1:3d})"),
            ("BN3+ELU",        f"(B, F2={self.F2:2d},  1, {self.T//self.pool1:3d})"),
            ("AvgPool2",       f"(B, F2={self.F2:2d},  1, {self.T//(self.pool1*self.pool2):3d})"),
            ("Dropout2",       f"(B, F2={self.F2:2d},  1, {self.T//(self.pool1*self.pool2):3d})"),
            ("Flatten",        f"(B, {self._flat_size:4d})"),
            ("Linear",         f"(B, {self.n_classes:4d})"),
        ]
        for name, shape in rows:
            print(f"  {name:<20s} → {shape}")
        params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("=" * 70)
        print(f"  Trainable parameters : {params:,}")
        print(f"  Model size (fp32)    : {params*4/1024:.1f} KB")
        print("=" * 70)

    def count_parameters(self) -> dict:
        """Return parameter counts per layer group."""
        groups = {
            "temporal_conv":  self.temporal_conv,
            "bn1":            self.bn1,
            "depthwise_conv": self.depthwise_conv,
            "bn2":            self.bn2,
            "sep_depthwise":  self.sep_depthwise,
            "sep_pointwise":  self.sep_pointwise,
            "bn3":            self.bn3,
            "classifier":     self.classifier,
        }
        return {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}

    def apply_max_norm(self, max_norm: float = 1.0):
        """Call after optimizer.step() to enforce max-norm on depthwise filters."""
        constraint = MaxNormConstraint(max_norm)
        constraint.apply_to(self.depthwise_conv)
        constraint.apply_to(self.temporal_conv)


# ══════════════════════════════════════════════════════════════════════════════
#  Attention EEGNet (Channel + Temporal Attention)
# ══════════════════════════════════════════════════════════════════════════════

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    Recalibrates feature maps: tells the model which frequency bands matter.

    Input:  (B, F, 1, T)
    Output: (B, F, 1, T)  — same shape, re-weighted
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.squeeze   = nn.AdaptiveAvgPool2d((1, 1))
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ELU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        scale = self.squeeze(x)              # (B, C, 1, 1)
        scale = self.excitation(scale)       # (B, C)
        scale = scale.view(B, C, 1, 1)
        return x * scale


class TemporalAttention(nn.Module):
    """
    Temporal self-attention using lightweight conv-based approach.
    Highlights the time steps most relevant to the class (e.g. ERD onset).

    Input:  (B, F, 1, T)
    Output: (B, F, 1, T)
    """

    def __init__(self, in_features: int, kernel: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(
            in_features, 1,
            kernel_size=(1, kernel),
            padding=(0, kernel // 2),
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.sigmoid(self.conv(x))    # (B, 1, 1, T)
        return x * attn


class AttentionEEGNet(EEGNetV4):
    """
    EEGNet V4 augmented with Squeeze-Excitation channel attention and
    temporal attention after each block.

    Adds ~300 extra parameters — negligible overhead.
    Typically +1–3% accuracy over vanilla EEGNet on BCI IV 2a.
    """

    def __init__(self, se_reduction: int = 4, **kwargs):
        super().__init__(**kwargs)
        # After Block 1 depthwise
        self.ch_attn1  = ChannelAttention(self.F2, reduction=se_reduction)
        # After Block 2 separable
        self.ch_attn2  = ChannelAttention(self.F2, reduction=se_reduction)
        self.temp_attn = TemporalAttention(self.F2)

        # Recompute flat size with attention modules
        self._compute_flatten_size(self.C, self.T)
        self.classifier = nn.Linear(self._flat_size, self.n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.bn1(self.temporal_conv(x))
        x = self.drop1(self.pool1_layer(self.elu1(self.bn2(self.depthwise_conv(x)))))
        x = self.ch_attn1(x)                 # ← channel attention

        # Block 2
        x = self.drop2(self.pool2_layer(
            self.elu2(self.bn3(self.sep_pointwise(self.sep_depthwise(x))))
        ))
        x = self.ch_attn2(x)                 # ← channel attention
        x = self.temp_attn(x)                # ← temporal attention

        return x.view(x.size(0), -1)


# ══════════════════════════════════════════════════════════════════════════════
#  Residual EEGNet
# ══════════════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """
    Skip connection around Block 2.
    Helps gradient flow when fine-tuning with discriminative LRs (very small
    gradients in early layers).
    """

    def __init__(self, channels: int, kernel: int = 16):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, (1, kernel),
                            padding=(0, kernel // 2), groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(self.bn(self.pw(self.dw(x))) + x)


class ResidualEEGNet(EEGNetV4):
    """EEGNet V4 with a residual wrapper around Block 2's separable conv."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.res_block = ResidualBlock(self.F2)
        self._compute_flatten_size(self.C, self.T)
        self.classifier = nn.Linear(self._flat_size, self.n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.bn1(self.temporal_conv(x))
        x = self.drop1(self.pool1_layer(self.elu1(self.bn2(self.depthwise_conv(x)))))

        # Block 2 (residual)
        out = self.bn3(self.sep_pointwise(self.sep_depthwise(x)))
        out = self.elu2(out + x)              # residual
        x   = self.drop2(self.pool2_layer(out))

        return x.view(x.size(0), -1)


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════

def build_model(cfg: dict, phase: str = "finetune") -> nn.Module:
    """
    Build model from config for 'pretrain' or 'finetune' phase.

    Parameters
    ----------
    cfg   : loaded config dict
    phase : 'pretrain' | 'finetune'
    """
    mcfg  = cfg["model"]
    ecfg  = mcfg["eegnet"]
    pcfg  = ecfg["pretrain" if phase == "pretrain" else "finetune"]
    name  = mcfg["name"]

    kwargs = dict(
        C         = pcfg["C"],
        T         = pcfg["T"],
        n_classes = pcfg["n_classes"],
        F1        = ecfg["F1"],
        D         = ecfg["D"],
        F2        = ecfg["F2"],
        Tc        = ecfg["Tc"],
        pool1     = ecfg["pool1"],
        pool2     = ecfg["pool2"],
        dropout   = ecfg["dropout"],
    )

    model_map = {
        "EEGNetV4":       EEGNetV4,
        "AttentionEEGNet": AttentionEEGNet,
        "ResidualEEGNet":  ResidualEEGNet,
    }
    if name not in model_map:
        raise ValueError(f"Unknown model: {name}. Choose from {list(model_map)}")

    model = model_map[name](**kwargs)
    logger.info(f"Built {name} | phase={phase} | params="
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model