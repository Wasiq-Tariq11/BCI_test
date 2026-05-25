"""
BCI Competition IV Dataset 2a — Local GDF Loader
==================================================
Loads the four-class motor imagery dataset directly from GDF files.
NO MOABB dependency.

Download
--------
  Train GDF : https://www.bbci.de/competition/iv/desc_2a.pdf
  Eval GDF  : same page → A01T.gdf … A09T.gdf  and  A01E.gdf … A09E.gdf
  True labels for evaluation: https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip
                               yields A01E.mat … A09E.mat (field 'classlabel')

Directory layout expected
─────────────────────────
<bciv2a_root>/
    A01T.gdf    ← subject 1 training session
    A01E.gdf    ← subject 1 evaluation session
    A01E.mat    ← true labels for A01E  (from BBCI website)
    A02T.gdf
    A02E.gdf
    A02E.mat
    ...
    A09E.mat

Key facts
─────────
  EEG channels  : 22 (Fz … Oz, standard 10-20)
  EOG channels  : 3  (indices 22,23,24 in raw)
  Sfreq         : 250 Hz
  Classes       : 769=left hand  770=right hand  771=feet  772=tongue
  Cue onset     : annotation 'class'
  Trial window  : [0.5 s, 2.5 s] relative to cue  → 500 samples
  Rejected trials: annotation '1023' (artifacts)
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io as sio
import mne
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

mne.set_log_level("WARNING")
logger = logging.getLogger(__name__)

# ── class mappings ──────────────────────────────────────────────────────────────
# GDF event code → 0-indexed class label
GDF_CLASS_MAP: Dict[int, int] = {769: 0, 770: 1, 771: 2, 772: 3}
CLASS_NAMES = ["Left hand", "Right hand", "Feet", "Tongue"]
ARTIFACT_CODE = 1023          # GDF annotation for rejected trial
CUE_CODE      = 768           # 'start of trial' — not used for epoching


class BCIV2aPreprocessor:
    """
    Signal-processing chain for BCI IV 2a.

    Bandpass 4–40 Hz → Notch 50 Hz → CAR → Z-score per channel per epoch.

    Parameters
    ----------
    cfg : full config dict
    """

    def __init__(self, cfg: dict):
        bp = cfg["preprocessing"]["bandpass"]
        self.sfreq   = cfg["bciv2a"]["sfreq"]   # 250 Hz
        self.bp_low  = bp["low"]                 # 4 Hz
        self.bp_high = bp["high"]                # 40 Hz
        self.notch_f = cfg["preprocessing"]["notch"]["freqs"]
        self.do_car  = cfg["preprocessing"]["reference"] == "CAR"
        self.norm    = cfg["preprocessing"]["normalization"]

    def bandpass(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        return raw.filter(
            self.bp_low, self.bp_high,
            method="fir",
            fir_window="hamming",
            verbose=False,
        )

    def notch(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        for f in self.notch_f:
            raw = raw.notch_filter(f, verbose=False)
        return raw

    def car(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        raw.set_eeg_reference("average", projection=False, verbose=False)
        return raw

    def normalize_epoch(self, x: np.ndarray) -> np.ndarray:
        """x: (C, T)"""
        if self.norm == "zscore":
            mu  = x.mean(axis=-1, keepdims=True)
            std = x.std(axis=-1, keepdims=True) + 1e-8
            return (x - mu) / std
        elif self.norm == "minmax":
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn + 1e-8)
        return x

    def process_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Full raw-level pipeline; epoch normalisation happens per-epoch."""
        raw = self.bandpass(raw)
        raw = self.notch(raw)
        if self.do_car:
            raw = self.car(raw)
        return raw


def _load_true_labels_mat(mat_path: str) -> np.ndarray:
    """
    Load evaluation-set true labels from .mat file.
    The BBCI .mat files contain field 'classlabel' with values 1–4.
    Returns 0-indexed array.
    """
    mat = sio.loadmat(mat_path)
    # field name may vary; try common variants
    for key in ("classlabel", "classLabel", "y", "labels"):
        if key in mat:
            lbl = mat[key].flatten().astype(int) - 1   # 1-indexed → 0-indexed
            return lbl
    raise KeyError(f"Could not find class labels in {mat_path}. "
                   f"Keys: {list(mat.keys())}")


def _load_gdf_raw(gdf_path: str, eog_channels: List[int]) -> mne.io.BaseRaw:
    """
    Read a GDF file.  Returns raw with EEG and EOG channels separated.
    EOG channels are marked but NOT dropped — caller decides.
    """
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)

    # Set channel types: first 22 are EEG, next 3 are EOG
    ch_types = {}
    for i, ch in enumerate(raw.ch_names):
        if i in eog_channels:
            ch_types[ch] = "eog"
        else:
            ch_types[ch] = "eeg"
    raw.set_channel_types(ch_types)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    return raw


def _extract_epochs_from_raw(
    raw: mne.io.BaseRaw,
    tmin: float,
    tmax: float,
    reject_threshold: float,
    true_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract epochs from a preprocessed raw object.

    For TRAINING GDFs: labels are embedded in GDF annotations (event codes 769–772).
    For EVALUATION GDFs: labels come from the .mat true-labels file.

    Returns
    -------
    data   : float32 (n_epochs, C, T)
    labels : int32   (n_epochs,)    0-indexed
    valid  : bool    (n_epochs,)    True = not artifact-rejected
    """
    events, event_id_found = mne.events_from_annotations(raw, verbose=False)

    # ── detect artifact events ─────────────────────────────────
    artifact_rows = set()
    for i, ev in enumerate(events):
        if ev[2] == ARTIFACT_CODE:
            artifact_rows.add(i)

    # ── select only MI-class cue events ───────────────────────
    mi_event_ids = {k: v for k, v in event_id_found.items()
                    if int(k) in GDF_CLASS_MAP}

    if not mi_event_ids and true_labels is not None:
        # Evaluation GDF: use 'cue' event 768 (or first non-artifact event)
        # Find events that correspond to trial starts
        trial_events = events[
            np.isin(events[:, 2], list(GDF_CLASS_MAP.keys()) + [768])
        ]
        # If no class codes, use 768 and attach true labels by order
        if len(trial_events) == 0:
            # fall back: every event that isn't artifact/EOG
            trial_events = events[~np.isin(events[:, 2], [ARTIFACT_CODE])]
        mi_events = trial_events
    else:
        mi_events = mne.pick_events(events, include=list(GDF_CLASS_MAP.keys()))

    if len(mi_events) == 0:
        logger.warning("No MI events found in raw. Skipping.")
        return np.empty(0), np.empty(0), np.empty(0, dtype=bool)

    # ── epoch ──────────────────────────────────────────────────
    epochs_obj = mne.Epochs(
        raw,
        mi_events,
        tmin    = tmin,
        tmax    = tmax,
        baseline= None,           # no baseline after cue
        preload = True,
        picks   = "eeg",          # keep EEG only
        reject  = dict(eeg=reject_threshold),
        verbose = False,
        on_missing="ignore",
    )
    data       = epochs_obj.get_data(copy=True)          # (N, C, T)
    drop_log   = epochs_obj.drop_log                     # list of drop reasons

    # build valid mask from drop_log
    valid = np.array([len(d) == 0 for d in drop_log])

    # ── resolve labels ─────────────────────────────────────────
    if true_labels is not None:
        # evaluation GDF — match by order; trim if lengths differ
        n = min(data.shape[0], len(true_labels))
        labels = true_labels[:n].astype(np.int32)
        data   = data[:n]
        valid  = valid[:n]
    else:
        labels = np.array(
            [GDF_CLASS_MAP[ev[2]] for ev in mi_events], dtype=np.int32
        )
        # align with dropped epochs
        kept_mask = np.array([len(d) == 0 for d in drop_log[:len(mi_events)]])
        labels = labels[kept_mask]

    return data.astype(np.float32), labels, valid


class BCIV2aDataset(Dataset):
    """
    PyTorch Dataset for BCI Competition IV Dataset 2a.

    Supports
    --------
    - Training session  (A0{s}T.gdf)
    - Evaluation session (A0{s}E.gdf  +  A0{s}E.mat true labels)
    - Single subject or all subjects pooled

    Parameters
    ----------
    root        : path to directory containing GDF and MAT files
    cfg         : loaded config dict
    subjects    : list of subject ids 1–9; None → all 9
    session     : 'train' | 'eval' | 'both'
    preprocessor: BCIV2aPreprocessor instance (or None → default)
    augment     : augmentation callable | None
    """

    def __init__(
        self,
        root: str,
        cfg: dict,
        subjects: Optional[List[int]] = None,
        session: str = "train",
        preprocessor: Optional[BCIV2aPreprocessor] = None,
        augment=None,
    ):
        self.root    = Path(root)
        self.cfg     = cfg
        bcfg         = cfg["bciv2a"]

        if subjects is None:
            subjects = bcfg["subjects"]
        self.subjects  = subjects
        self.session   = session
        self.augment   = augment

        self.tmin      = bcfg["tmin"]
        self.tmax      = bcfg["tmax"]
        self.sfreq     = bcfg["sfreq"]
        self.n_times   = int((self.tmax - self.tmin) * self.sfreq)  # 500
        self.reject_th = bcfg["reject_threshold"]
        self.eog_chs   = bcfg["eog_channels"]

        self.preprocessor = preprocessor or BCIV2aPreprocessor(cfg)

        self.data:   List[np.ndarray] = []   # (C, T) each
        self.labels: List[int]        = []
        self.subject_ids: List[int]   = []   # which subject each epoch belongs to

        self._load_all()
        logger.info(
            f"BCIV2aDataset[{session}] — {len(self.data)} epochs | "
            f"subjects={subjects} | shape={self.data[0].shape}"
        )

    # ── loaders ──────────────────────────────────────────────
    def _load_all(self):
        sessions = []
        if self.session in ("train", "both"):
            sessions.append("T")
        if self.session in ("eval", "both"):
            sessions.append("E")

        for subj in self.subjects:
            for sess in sessions:
                gdf_file = self.root / f"A{subj:02d}{sess}.gdf"
                mat_file = self.root / f"A{subj:02d}{sess}.mat"
                if not gdf_file.exists():
                    logger.warning(f"GDF not found: {gdf_file}")
                    continue

                # true labels are needed for evaluation session
                true_labels = None
                if sess == "E":
                    if mat_file.exists():
                        true_labels = _load_true_labels_mat(str(mat_file))
                    else:
                        logger.warning(
                            f"No .mat label file for {gdf_file.name}. "
                            f"Eval labels unavailable — skipping this session."
                        )
                        continue

                self._load_gdf(str(gdf_file), true_labels, subj)

    def _load_gdf(self, gdf_path: str, true_labels, subj: int):
        # 1. Read raw
        raw = _load_gdf_raw(gdf_path, self.eog_chs)

        # 2. Preprocess (filter → CAR) — operates on continuous signal
        raw = self.preprocessor.process_raw(raw)

        # 3. Extract epochs
        data, labels, valid = _extract_epochs_from_raw(
            raw,
            tmin             = self.tmin,
            tmax             = self.tmax,
            reject_threshold = self.reject_th,
            true_labels      = true_labels,
        )
        if data.shape[0] == 0:
            return

        for ep, lab in zip(data, labels):
            # ensure fixed length
            if ep.shape[-1] < self.n_times:
                pad = self.n_times - ep.shape[-1]
                ep = np.pad(ep, ((0, 0), (0, pad)))
            else:
                ep = ep[:, :self.n_times]

            ep = self.preprocessor.normalize_epoch(ep)
            self.data.append(ep)
            self.labels.append(int(lab))
            self.subject_ids.append(subj)

    # ── Dataset interface ─────────────────────────────────────
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        x = self.data[idx].copy()     # (C, T) float32
        y = self.labels[idx]

        if self.augment is not None:
            x = self.augment(x)

        x = torch.from_numpy(x).unsqueeze(0)    # (1, C, T)
        return x, torch.tensor(y, dtype=torch.long)

    # ── split utilities ───────────────────────────────────────
    def loso_splits(self) -> List[Tuple[Subset, Subset]]:
        """Leave-One-Subject-Out cross-validation splits."""
        subj_arr = np.array(self.subject_ids)
        splits = []
        for held_out in self.subjects:
            tr_idx = np.where(subj_arr != held_out)[0]
            va_idx = np.where(subj_arr == held_out)[0]
            splits.append((Subset(self, tr_idx), Subset(self, va_idx)))
        return splits

    def subject_split(
        self, subj: int
    ) -> Tuple[Subset, Subset]:
        """Train on remaining subjects, test on `subj`."""
        subj_arr = np.array(self.subject_ids)
        tr_idx   = np.where(subj_arr != subj)[0]
        va_idx   = np.where(subj_arr == subj)[0]
        return Subset(self, tr_idx), Subset(self, va_idx)

    def kfold_within_subject(
        self, subj: int, n_splits: int = 5, seed: int = 42
    ) -> List[Tuple[Subset, Subset]]:
        """Stratified K-Fold for within-subject evaluation."""
        subj_arr = np.array(self.subject_ids)
        idx      = np.where(subj_arr == subj)[0]
        labels   = np.array(self.labels)[idx]
        skf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [
            (Subset(self, idx[tr]), Subset(self, idx[va]))
            for tr, va in skf.split(idx, labels)
        ]


# ── factory ────────────────────────────────────────────────────────────────────
def build_bciv2a_loaders(
    cfg: dict,
    subject: int,
    session: str = "train",
    augment=None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build subject-independent train / eval DataLoaders for BCI IV 2a.

    Train loader : all subjects EXCEPT `subject`
    Val   loader : only `subject` (evaluation session)
    """
    fcfg = cfg["finetune"]

    full_ds = BCIV2aDataset(
        root    = cfg["paths"]["bciv2a_root"],
        cfg     = cfg,
        session = "both",
        augment = None,
    )

    train_ds, val_ds = full_ds.subject_split(subj=subject)

    # attach augmentor only to train
    if augment is not None:
        # wrap the subset's dataset augmentor via a thin proxy
        class _AugSubset(Subset):
            def __getitem__(self, idx):
                x, y = self.dataset.data[self.indices[idx]], \
                       self.dataset.labels[self.indices[idx]]
                x = augment(x.copy())
                x = torch.from_numpy(x).unsqueeze(0)
                return x, torch.tensor(y, dtype=torch.long)
        train_ds = _AugSubset(full_ds, train_ds.indices)

    train_loader = DataLoader(
        train_ds,
        batch_size  = fcfg["batch_size"],
        shuffle     = True,
        num_workers = fcfg["num_workers"],
        pin_memory  = fcfg["pin_memory"],
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = fcfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = fcfg["num_workers"],
        pin_memory  = fcfg["pin_memory"],
    )
    logger.info(
        f"BCIV2a loaders | Subject {subject} held-out — "
        f"train: {len(train_ds)} | val: {len(val_ds)}"
    )
    return train_loader, val_loader