"""
Trainer — Generic Training Engine
===================================
Supports:
  • Mixed-precision training (torch.cuda.amp)
  • Cosine annealing with linear warmup
  • Mixup data augmentation (applied at batch level)
  • Label smoothing cross-entropy
  • Early stopping with best-checkpoint saving
  • TensorBoard integration
  • Gradient clipping
  • Max-norm weight constraint on depthwise filters
"""

import os
import time
import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from preprocessing.augmentation import mixup_data, mixup_criterion

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Loss functions
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy with label smoothing.
    Replaces one-hot targets with (1-ε)·one_hot + ε/K distribution.
    Prevents overconfident predictions — important when fine-tuning on
    small datasets where calibration matters.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        K    = logits.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (K - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = -(smooth_targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean() if self.reduction == "mean" else loss


# ══════════════════════════════════════════════════════════════════════════════
#  LR Schedulers
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs` then cosine annealing to `eta_min`.
    Called step() once per epoch.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        eta_min: float = 1e-6,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        # Store initial LRs
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        e = self.current_epoch
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i] if i < len(self.base_lrs) else self.base_lrs[-1]
            if e < self.warmup_epochs:
                lr = base_lr * (e + 1) / self.warmup_epochs
            else:
                progress = (e - self.warmup_epochs) / max(
                    1, self.total_epochs - self.warmup_epochs
                )
                lr = self.eta_min + 0.5 * (base_lr - self.eta_min) * (
                    1 + np.cos(np.pi * progress)
                )
            group["lr"] = lr
        self.current_epoch += 1

    def get_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]


# ══════════════════════════════════════════════════════════════════════════════
#  Early Stopping
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    """Stops training if val_acc doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = -np.inf
        self.counter    = 0
        self.stop       = False

    def __call__(self, val_acc: float) -> bool:
        if val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ══════════════════════════════════════════════════════════════════════════════
#  Trainer
# ══════════════════════════════════════════════════════════════════════════════

class Trainer:
    """
    Unified trainer for pretrain and finetune phases.

    Parameters
    ----------
    model         : EEGNet model
    train_loader  : training DataLoader
    val_loader    : validation DataLoader
    optimizer     : AdamW optimizer (may have multiple param groups)
    cfg           : full config dict
    phase         : 'pretrain' | 'finetune'
    checkpoint_path: where to save best model
    unfreeze_scheduler: GradualUnfreezeScheduler (finetune only) or None
    device        : 'cuda' | 'cpu'
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        cfg: dict,
        phase: str = "pretrain",
        checkpoint_path: str = "checkpoints/best.pth",
        unfreeze_scheduler=None,
        device: str = "cuda",
    ):
        self.model       = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.optimizer    = optimizer
        self.cfg          = cfg
        self.phase        = phase
        self.ckpt_path    = Path(checkpoint_path)
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.unfreeze_sch = unfreeze_scheduler
        self.device       = device

        pcfg = cfg[phase]
        self.epochs        = pcfg["epochs"]
        self.use_amp       = pcfg["mixed_precision"] and device == "cuda"
        self.mixup_alpha   = pcfg.get("mixup_alpha", 0.0)
        self.label_smooth  = pcfg.get("label_smoothing", 0.0)

        # Loss
        if self.label_smooth > 0:
            self.criterion = LabelSmoothingCrossEntropy(self.label_smooth)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(device)

        # LR scheduler
        self.lr_scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs = pcfg["warmup_epochs"],
            total_epochs  = self.epochs,
        )

        # Early stopping
        es_cfg = pcfg["early_stopping"]
        self.early_stop = EarlyStopping(
            patience  = es_cfg["patience"],
            min_delta = es_cfg["min_delta"],
        )

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # TensorBoard
        log_dir = Path(cfg["paths"]["logs_dir"]) / phase
        self.writer = SummaryWriter(str(log_dir))

        # History
        self.history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        logger.info(f"Trainer ready | phase={phase} | device={device} | "
                    f"AMP={self.use_amp} | mixup={self.mixup_alpha}")

    # ── training loop ─────────────────────────────────────────
    def fit(self) -> Dict[str, list]:
        """Run full training. Returns history dict."""
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()

            # Gradual unfreezing
            if self.unfreeze_sch is not None:
                self.unfreeze_sch.step(epoch - 1)
                self.optimizer = self.unfreeze_sch.get_optimizer()

            train_loss, train_acc = self._train_epoch()
            val_loss,   val_acc   = self._val_epoch()
            self.lr_scheduler.step()

            # Apply max-norm constraint (depthwise conv)
            if hasattr(self.model, "apply_max_norm"):
                self.model.apply_max_norm(max_norm=1.0)

            elapsed = time.time() - t0
            current_lr = self.lr_scheduler.get_lr()[0]

            # Logging
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            self.writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
            self.writer.add_scalar("LR", current_lr, epoch)

            logger.info(
                f"[{self.phase}] Epoch {epoch:3d}/{self.epochs} | "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} | "
                f"lr={current_lr:.2e} | {elapsed:.1f}s"
            )

            # Save best checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc, val_loss)
                logger.info(f"  ★ Best val_acc={val_acc:.4f} → saved checkpoint")

            # Early stopping
            if self.early_stop(val_acc):
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {self.early_stop.patience} epochs)"
                )
                break

        self.writer.close()
        logger.info(f"Training complete. Best val_acc={self.best_val_acc:.4f}")
        return self.history

    # ── single epoch helpers ──────────────────────────────────
    def _train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, n = 0.0, 0, 0

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Mixup
            if self.mixup_alpha > 0:
                x, ya, yb, lam = mixup_data(x, y, self.mixup_alpha, self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    logits = self.model(x)
                    if self.mixup_alpha > 0:
                        loss = mixup_criterion(self.criterion, logits, ya, yb, lam)
                    else:
                        loss = self.criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x)
                if self.mixup_alpha > 0:
                    loss = mixup_criterion(self.criterion, logits, ya, yb, lam)
                else:
                    loss = self.criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == (ya if self.mixup_alpha > 0 else y)).sum().item()
            n       += x.size(0)

        return total_loss / n, correct / n

    @torch.no_grad()
    def _val_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, n = 0.0, 0, 0
        ce = nn.CrossEntropyLoss()   # no smoothing for validation

        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            if self.use_amp:
                with autocast():
                    logits = self.model(x)
                    loss   = ce(logits, y)
            else:
                logits = self.model(x)
                loss   = ce(logits, y)
            total_loss += loss.item() * x.size(0)
            correct    += (logits.argmax(1) == y).sum().item()
            n          += x.size(0)

        return total_loss / n, correct / n

    # ── checkpoint ────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, val_acc: float, val_loss: float):
        torch.save(
            {
                "epoch":            epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state":  self.optimizer.state_dict(),
                "val_acc":          val_acc,
                "val_loss":         val_loss,
                "history":          self.history,
                "cfg":              self.cfg,
            },
            str(self.ckpt_path),
        )

    def load_best(self):
        """Restore best saved weights into model."""
        ckpt = torch.load(str(self.ckpt_path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded best checkpoint (epoch={ckpt['epoch']}, "
                    f"val_acc={ckpt['val_acc']:.4f})")
        return ckpt