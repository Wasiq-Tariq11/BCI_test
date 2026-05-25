"""
EEGMMIDB Dataset Loader
=======================
Loads PhysioNet EEG Motor Movement/Imagery Database directly from local EDF files.
Download: https://physionet.org/content/eegmmidb/1.0.0/

Directory layout expected
─────────────────────────
<eegmmidb_root>/
    S001/
        S001R01.edf   ← baseline eyes-open
        S001R02.edf   ← baseline eyes-closed
        S001R03.edf   ← task 1 run 1  (left/right fist MI)
        S001R04.edf   ← task 2 run 1  (left/right fist ACTUAL)
        S001R05.edf   ← task 2 run 1  (both fists / both feet MI)
        ...
        S001R14.edf
    S002/ ...
    ...
    S109/

Key facts
─────────
  Channels  : 64 EEG (10-10 system), Fc5,Fc3,Fc1,Fcz,...
  Sfreq     : 160 Hz
  Annotation codes
      T0 = rest
      T1 = left fist  (tasks 1,2) | both fists    (tasks 3,4)
      T2 = right fist (tasks 1,2) | both feet     (tasks 3,4)
  MI runs   : 3,7,11 (left/right) and 5,9,13 (hands/feet)
"""

import os
import glob
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import mne
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import signal as sp_signal

mne.set_log_level("WARNING")
logger = logging.getLogger(__name__)


# ─── label mapping per run type ────────────────────────────────────────────────
# Runs 3,7,11 → T1=left hand (1),  T2=right hand (2)
# Runs 5,9,13 → T1=both hands (3), T2=feet      (4)
LEFTRIGHT_RUNS = {3, 7, 11}
HANDSFEET_RUNS = {5, 9, 13}

EVENT_MAP_LR = {"T0": 0, "T1": 1, "T2": 2}   # rest / left / right
EVENT_MAP_HF = {"T0": 0, "T1": 3, "T2": 4}   # rest / both-hands / feet


def _run_number(edf_path: str) -> int:
    """Extract run number from filename, e.g. 'S001R05.edf' → 5."""
    stem = Path(edf_path).stem           # 'S001R05'
    return int(stem[-2:])                # '05' → 5


class EEGMMIDBPreprocessor:
    """
    Bandpass → Notch → CAR → Epoch → Z-score
    Works entirely in numpy/scipy for speed; called inside the Dataset.
    """

    def __init__(self, cfg: dict):
        bp = cfg["preprocessing"]["bandpass"]
        self.bp_low   = bp["low"]
        self.bp_high  = bp["high"]
        self.sfreq    = cfg["eegmmidb"]["sfreq"]
        self.notch_f  = cfg["preprocessing"]["notch"]["freqs"]
        self.do_car   = cfg["preprocessing"]["reference"] == "CAR"
        self.norm     = cfg["preprocessing"]["normalization"]

    # ── filters ───────────────────────────────────────────────
    def bandpass(self, x: np.ndarray) -> np.ndarray:
        """x: (C, T) float64 in Volts"""
        nyq = self.sfreq / 2.0
        lo, hi = self.bp_low / nyq, self.bp_high / nyq
        b, a = sp_signal.butter(5, [lo, hi], btype="bandpass")
        return sp_signal.filtfilt(b, a, x, axis=-1)

    def notch(self, x: np.ndarray) -> np.ndarray:
        for f in self.notch_f:
            b, a = sp_signal.iirnotch(f / (self.sfreq / 2.0), Q=30.0)
            x = sp_signal.filtfilt(b, a, x, axis=-1)
        return x

    def car(self, x: np.ndarray) -> np.ndarray:
        """Common Average Reference — subtract mean across channels."""
        return x - x.mean(axis=0, keepdims=True)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.norm == "zscore":
            mu  = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-8
            return (x - mu) / std
        elif self.norm == "minmax":
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = self.bandpass(x)
        x = self.notch(x)
        if self.do_car:
            x = self.car(x)
        x = self.normalize(x)
        return x.astype(np.float32)


class EEGMMIDBDataset(Dataset):
    """
    PyTorch Dataset for the PhysioNet EEGMMIDB.

    Parameters
    ----------
    root        : path to eegmmidb root (contains S001/, S002/, ...)
    cfg         : loaded config dict
    subjects    : list of subject indices (1–109); None → all
    mi_runs     : run numbers to include (default [3,5,7,9,11,13])
    preprocessor: EEGMMIDBPreprocessor instance
    augment     : EEGAugmentor instance or None
    """

    def __init__(
        self,
        root: str,
        cfg: dict,
        subjects: Optional[List[int]] = None,
        mi_runs: Optional[List[int]] = None,
        preprocessor: Optional[EEGMMIDBPreprocessor] = None,
        augment=None,
    ):
        self.root = Path(root)
        self.cfg  = cfg
        ecfg      = cfg["eegmmidb"]

        if subjects is None:
            subjects = list(range(1, 110))
        if mi_runs is None:
            mi_runs = ecfg["mi_runs"]

        self.mi_runs     = set(mi_runs)
        self.preprocessor = preprocessor or EEGMMIDBPreprocessor(cfg)
        self.augment      = augment

        # epoch window
        self.tmin   = ecfg["tmin"]    # -0.5 s
        self.tmax   = ecfg["tmax"]    #  2.5 s
        self.sfreq  = ecfg["sfreq"]   # 160 Hz
        # samples = (tmax - tmin) * sfreq = 3.0 * 160 = 480
        self.n_times = int((self.tmax - self.tmin) * self.sfreq)

        self.pretrain_classes = set(ecfg["pretrain_classes"])   # {1,2,3,4}

        self.epochs: List[np.ndarray] = []   # (C, T)
        self.labels: List[int]        = []

        self._load_all_subjects(subjects)
        logger.info(
            f"EEGMMIDBDataset: {len(self.epochs)} epochs | "
            f"classes={np.unique(self.labels).tolist()} | "
            f"shape={self.epochs[0].shape}"
        )

    # ── internal loading ──────────────────────────────────────
    def _load_all_subjects(self, subjects: List[int]):
        for subj in subjects:
            subj_dir = self.root / f"S{subj:03d}"
            if not subj_dir.exists():
                logger.warning(f"Subject directory not found: {subj_dir}")
                continue
            for run_no in sorted(self.mi_runs):
                edf_path = subj_dir / f"S{subj:03d}R{run_no:02d}.edf"
                if not edf_path.exists():
                    logger.warning(f"Missing EDF: {edf_path}")
                    continue
                self._load_run(str(edf_path), run_no)

    def _load_run(self, edf_path: str, run_no: int):
        # ── choose label map based on run type ────────────────
        if run_no in LEFTRIGHT_RUNS:
            event_map = EVENT_MAP_LR
        else:
            event_map = EVENT_MAP_HF

        # ── read raw EDF ──────────────────────────────────────
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)          # keep only EEG channels

        # ── parse annotations → events ────────────────────────
        # MNE reads PhysioNet annotations automatically
        events, event_id = mne.events_from_annotations(
            raw,
            event_id={"T0": 0, "T1": 1, "T2": 2},
            verbose=False,
        )
        if events.size == 0:
            return

        # ── epoch ─────────────────────────────────────────────
        epochs = mne.Epochs(
            raw,
            events,
            event_id={"T1": 1, "T2": 2},   # skip rest
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=(self.tmin, 0.0),
            preload=True,
            reject=None,
            verbose=False,
        )

        data   = epochs.get_data()          # (n_ep, C, T_raw)
        labels = epochs.events[:, -1]       # 1 or 2

        for i, (ep, lab) in enumerate(zip(data, labels)):
            # remap annotation code → class index
            # T1 in LR run → 1 (left hand), T1 in HF run → 3 (both hands)
            code = event_map.get("T1" if lab == 1 else "T2", -1)
            if code not in self.pretrain_classes:
                continue

            # ensure fixed length
            if ep.shape[-1] < self.n_times:
                pad = self.n_times - ep.shape[-1]
                ep = np.pad(ep, ((0, 0), (0, pad)))
            else:
                ep = ep[:, :self.n_times]

            ep = self.preprocessor(ep)      # (C, T) float32
            self.epochs.append(ep)
            # remap to 0-indexed: class 1→0, 2→1, 3→2, 4→3
            self.labels.append(code - 1)

    # ── Dataset interface ─────────────────────────────────────
    def __len__(self) -> int:
        return len(self.epochs)

    def __getitem__(self, idx: int):
        x = self.epochs[idx].copy()           # (C, T) float32
        y = self.labels[idx]

        if self.augment is not None:
            x = self.augment(x)

        # EEGNet expects (1, C, T) — add channel dim
        x = torch.from_numpy(x).unsqueeze(0)  # (1, C, T)
        return x, torch.tensor(y, dtype=torch.long)

    # ── helper: split ─────────────────────────────────────────
    def train_val_split(self, val_frac: float = 0.15, seed: int = 42):
        """Return stratified train / val index lists."""
        idx = np.arange(len(self.labels))
        lbl = np.array(self.labels)
        tr_idx, va_idx = train_test_split(
            idx, test_size=val_frac, stratify=lbl, random_state=seed
        )
        return torch.utils.data.Subset(self, tr_idx), \
               torch.utils.data.Subset(self, va_idx)


# ── factory ────────────────────────────────────────────────────────────────────
def build_eegmmidb_loaders(cfg: dict, augment=None) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for EEGMMIDB pretraining.

    Parameters
    ----------
    cfg     : loaded config dict
    augment : augmentation callable (applied to train only)

    Returns
    -------
    train_loader, val_loader
    """
    ecfg     = cfg["eegmmidb"]
    tcfg     = cfg["pretrain"]
    subjects = ecfg["subjects"]   # None → all

    preprocessor = EEGMMIDBPreprocessor(cfg)

    full_ds = EEGMMIDBDataset(
        root         = cfg["paths"]["eegmmidb_root"],
        cfg          = cfg,
        subjects     = subjects,
        preprocessor = preprocessor,
        augment      = None,          # augment only on train subset below
    )
    train_ds, val_ds = full_ds.train_val_split(
        val_frac=tcfg["val_fraction"],
        seed=tcfg["seed"],
    )

    # attach augmentor only to training subset
    if augment is not None:
        train_ds.dataset.augment = augment   # Subset wraps original dataset

    train_loader = DataLoader(
        train_ds,
        batch_size  = tcfg["batch_size"],
        shuffle     = True,
        num_workers = tcfg["num_workers"],
        pin_memory  = tcfg["pin_memory"],
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = tcfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = tcfg["num_workers"],
        pin_memory  = tcfg["pin_memory"],
    )
    logger.info(
        f"EEGMMIDB loaders — train: {len(train_ds)} | val: {len(val_ds)}"
    )
    return train_loader, val_loader