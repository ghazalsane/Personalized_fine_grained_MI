"""
============================================================================
 Per-subject personalization for fine-grained MI
 — built to be compared, trial for trial, against run_shallow_all.py
============================================================================

 WHAT IS HELD EQUAL TO THE BASELINE (run_shallow_all.py)

   same trials      folds come from subject_splits(), the same function the
                    baseline uses, so train / validation / test trials of
                    every (subject, fold, pair) are identical in both scripts
   same task        one binary ShallowConvNet per pair, per window
   same recipe      z-score on training trials, Gaussian-noise augmentation,
                    Adam, 50 epochs, best-validation-accuracy model selection
   same seeds       augmentation, shuffling and weight init use the baseline's
                    seed formulas

   Proof of this: `--arm fixed --no-ea` (the baseline's configuration, nothing
   added) reproduces run_shallow_all.py's numbers exactly.

 WHAT THIS SCRIPT ADDS  (= the contribution)

   Stage 1  Band screen.    Rank the three preprocessed bands by how well they
                            separate the 8 classes for THIS subject. Keep 2.
   Stage 2  Config search.  ASHA over band x temporal scale x channel set x
                            spatial filters, on the 8-class problem.
   Per-subject alignment.   Euclidean alignment fitted on the subject's own
                            training trials (--no-ea turns it off).
   Stage 3  Pair models.    With the chosen config frozen, train one binary
                            model per pair and window, exactly as the baseline
                            does, and test on the fold's held-out pair trials.

 NO LEAKAGE. For a given fold, Stages 1-2 see only that fold's 8-class
 training and validation trials. A pair's test trials are a subset of the
 fold's 8-class test trials, which no selection step ever touches.

 ARMS
   --arm personalized            the full method                (default)
   --arm fixed                   baseline config + alignment    (what the
                                 alignment alone is worth)
   --arm fixed --no-ea           must equal the baseline        (sanity check)
   --init 8class                 start each pair model from an 8-class model
                                 trained at the chosen config, then fine-tune

 RUNNING  — one process per GPU; nothing is shared between processes.
   for i in 0 1 2 3; do
     CUDA_VISIBLE_DEVICES=$i python run_personalized.py --shard $i --num-shards 4 &
   done
   wait
   python run_personalized.py --merge        # also prints the comparison

 Unit of work = one (subject, fold): 18 x 5 = 90. Each writes its own JSON, so
 a relaunch skips whatever is finished.
============================================================================
"""

# ===========================================================================
# [P1] DETERMINISM BOOTSTRAP — before torch/CUDA
# ===========================================================================
import os

SEED = 42
os.environ.setdefault('PYTHONHASHSEED', str(SEED))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import gc
import re
import glob
import json
import time
import math
import random
import argparse
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import decimate
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')


# ===========================================================================
# CONFIG
# ===========================================================================
# Colab default. On a cluster: export MI_DRIVE_ROOT=/path/to/data
DRIVE = os.environ.get('MI_DRIVE_ROOT', '/content/drive/MyDrive')

# Stage 1 screens over these. Keys are the band labels used everywhere else.
BAND_ROOTS = {
    '0.5-3':  f'{DRIVE}/fixationcrossbaseline_0.5_3hz_MIonly',
    '8-30':   f'{DRIVE}/fixationcrossbaseline_8_30hz_MIonly',
    '4-40':   f'{DRIVE}/datasetpaper_4_40hz_MIonly',
}

BASELINE_ROOT = f'{DRIVE}/mi_cmp_baseline'       # written by run_shallow_all.py

SAMPLING_RATE = 250
WINDOWS_MS = [800, 1500, 3000, 4000]
EARLY_EXIT_WINDOWS = [800, 1500]      # only these may stop the cascade early

N_FOLDS = 5
N_CLASSES = 8

JOINT_NAMES = {0: "HOC", 1: "WFE", 2: "WAA", 3: "EPS",
               4: "EFE", 5: "SPS", 6: "SAA", 7: "SFE"}
ALL_PAIRS = list(itertools.combinations(range(N_CLASSES), 2))

# --- the searched factor space -------------------------------------------
TEMPORAL_SCALES = {
    # name:  (sampling rate, decimation factor, conv kernel, pool, stride)
    'fast': dict(fs=250, decim=1, kernel=25, pool=75, stride=15),   # 0.10 s kernel
    'slow': dict(fs=50,  decim=5, kernel=25, pool=25, stride=5),    # 0.50 s kernel
}
CHANNEL_SETS = ['contra', 'bilateral']
SPAT_FILTERS = [20, 40, 80]

# The baseline's configuration: what every subject gets when nothing is
# personalized. Used by --arm fixed.
FIXED_CONFIG = dict(band='4-40', temporal='fast', channels='bilateral', n_spat=40)

N_KEEP_BANDS = 2               # Stage 1 survivors
ASHA_RUNGS = [8, 16, 32, 50]   # epochs per rung
ASHA_ETA = 2                   # keep 1/eta of configs at each rung
MAX_EPOCHS = 50                # same as the baseline
BATCH_SIZE = 32                # same as the baseline
LR = 1e-3                      # same as the baseline
WEIGHT_DECAY = 1e-4            # same as the baseline
NOISE_LEVEL = 0.15             # same as the baseline

SMOKE = os.environ.get('MI_SMOKE') == '1'
if SMOKE:
    # Tiny version of everything, to check the plumbing in a minute or two.
    # Numbers from a smoke run mean nothing; it writes to its own folder.
    ASHA_RUNGS = [1, 2]
    MAX_EPOCHS = 2
    SPAT_FILTERS = [4, 8]
    ALL_PAIRS = ALL_PAIRS[:3]


# ===========================================================================
# DETERMINISM
# ===========================================================================
def set_global_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_global_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)


def seed_worker(worker_id):
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    random.seed(ws)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'


# ===========================================================================
# SPLITS — shared with run_shallow_all.py. KEEP THE TWO COPIES IDENTICAL.
#
# Folds are drawn once per subject on all 320 trials, stratified by the 8
# classes. A pair's folds are those same folds restricted to its two classes
# (8 test, 8 val, 24 train trials per class).
# ===========================================================================
def subject_splits(y8, subj_id, n_splits=5):
    idx = np.arange(len(y8))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=int(42 + subj_id))
    out = []
    for trainval, test in skf.split(idx, y8):
        tr, va = train_test_split(trainval, test_size=0.25,
                                  random_state=int(42 + subj_id),
                                  stratify=y8[trainval])
        out.append((np.sort(tr), np.sort(va), np.sort(test)))
    return out


# ===========================================================================
# DATA
# ===========================================================================
def subject_number(path):
    m = re.search(r"subject(\d+)", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else 0


def list_subjects(root):
    return sorted({subject_number(p)
                   for p in glob.glob(os.path.join(root, "subject*_eeg_epochs_*.npz"))})


_CACHE = {'subject': None, 'bands': {}}


def _read_band(subject, band):
    root = BAND_ROOTS[band]
    cands = glob.glob(os.path.join(root, f"subject{subject}_eeg_epochs_*.npz"))
    assert len(cands) == 1, f"subject{subject} in {band}: expected 1 .npz, found {cands}"
    with np.load(cands[0], allow_pickle=True) as z:
        X = z["data"].astype(np.float32).copy()
        y = z["labels"].astype(np.int64).copy()
        ch = None
        for k in ("ch_names", "channels", "channel_names", "chanlocs"):
            if k in z.files:
                ch = [str(c) for c in z[k]]
                break
    return X, y, ch


def load_subject_band(subject, band):
    """Return (X, y, ch_names) for one subject in one band.

    All bands of the CURRENT subject stay in memory and are dropped when the
    subject changes.
    """
    if _CACHE['subject'] != subject:
        _CACHE['subject'] = subject
        _CACHE['bands'] = {}
        gc.collect()
    if band not in _CACHE['bands']:
        X, y, ch = _read_band(subject, band)

        # One set of folds is reused in every band, which is only valid if
        # trial i is the same trial in every folder. Check it.
        for other, (Xo, yo, cho) in _CACHE['bands'].items():
            if len(yo) != len(y) or not np.array_equal(yo, y):
                raise RuntimeError(
                    f"subject{subject}: labels in '{band}' and '{other}' differ "
                    f"(n={len(y)} vs {len(yo)}). The band folders do not hold the "
                    f"same trials in the same order, so folds cannot be shared "
                    f"across bands. Re-run preprocessing with one loader.")
            if ch is None and cho is not None and len(cho) == X.shape[1]:
                ch = cho                      # borrow names from a sibling band
        _CACHE['bands'][band] = (X, y, ch)
        if ch is not None:                    # ...and lend them back
            for other, (Xo, yo, cho) in list(_CACHE['bands'].items()):
                if cho is None and Xo.shape[1] == len(ch):
                    _CACHE['bands'][other] = (Xo, yo, ch)
    return _CACHE['bands'][band]


def read_labels_only(subject, band):
    """Labels of one file without loading its data (cheap)."""
    cands = glob.glob(os.path.join(BAND_ROOTS[band], f"subject{subject}_eeg_epochs_*.npz"))
    assert len(cands) == 1, f"subject{subject} in {band}: expected 1 .npz, found {cands}"
    with np.load(cands[0], allow_pickle=True) as z:
        return z["labels"].astype(np.int64)


def verify_band_folders(subjects):
    """Stop before any compute if the band folders do not line up.

    One set of folds is reused in every band, which is only valid if trial i
    is the same trial in every folder. Checked for EVERY subject here, so a
    mismatch cannot surface hours into a run.
    """
    ref_band = FIXED_CONFIG['band']
    problems = []
    for s in subjects:
        ref = read_labels_only(s, ref_band)
        counts = np.bincount(ref, minlength=N_CLASSES)
        if ref.min() < 0 or ref.max() >= N_CLASSES or counts.min() < N_FOLDS:
            problems.append(f"subject{s} [{ref_band}]: labels not usable, "
                            f"per-class counts {counts.tolist()}")
        for band in BAND_ROOTS:
            if band == ref_band:
                continue
            y = read_labels_only(s, band)
            if len(y) != len(ref):
                problems.append(f"subject{s}: {band} has {len(y)} trials, "
                                f"{ref_band} has {len(ref)}")
            elif not np.array_equal(y, ref):
                problems.append(f"subject{s}: {band} and {ref_band} have the same "
                                f"number of trials but in a different order")
    if problems:
        raise RuntimeError(
            "The band folders do not hold the same trials in the same order, so "
            "folds cannot be shared across bands:\n  " + "\n  ".join(problems) +
            "\nRe-create the mismatched folder(s) with the same loader as "
            f"{ref_band}, or remove that band from BAND_ROOTS.")
    print(f"trial-order check passed: {len(subjects)} subjects x {len(BAND_ROOTS)} bands")


def subject_labels(subject):
    """8-class labels. Read from the baseline's band so folds match it exactly."""
    return load_subject_band(subject, FIXED_CONFIG['band'])[1]


def contralateral_mask(ch_names, n_channels):
    """Electrodes over the hemisphere opposite the moving (right) arm.

    Right-arm imagery projects to the LEFT hemisphere: odd-numbered electrodes
    in 10-20 naming. Midline (z) sites are kept.
    """
    if ch_names is None or len(ch_names) != n_channels:
        return None
    keep = []
    for i, name in enumerate(ch_names):
        s = name.strip()
        m = re.search(r"(\d+)$", s)
        if s.lower().endswith('z'):
            keep.append(i)
        elif m and int(m.group(1)) % 2 == 1:
            keep.append(i)
    return np.array(keep, dtype=int) if len(keep) >= 8 else None


def euclidean_alignment_matrix(X_fit):
    """Whitening matrix from the mean covariance of X_fit (training trials).

    Uses no labels. Average-referenced data has a rank-deficient covariance,
    so the inverse square root goes through an eigen-decomposition with a
    floor rather than a plain matrix power.
    """
    covs = np.matmul(X_fit, X_fit.transpose(0, 2, 1)) / X_fit.shape[2]
    R = covs.mean(axis=0).astype(np.float64)
    R = (R + R.T) / 2.0
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, w.max() * 1e-6)
    return ((V * (w ** -0.5)) @ V.T).astype(np.float32)


def shape_for_config(X, cfg, window_ms, ch_idx):
    """Crop to the window, select channels, decimate if the config asks."""
    ts = TEMPORAL_SCALES[cfg['temporal']]
    n_samp = min(int(window_ms * SAMPLING_RATE / 1000), X.shape[2])
    Xw = X[:, :, :n_samp]
    if ch_idx is not None:
        Xw = Xw[:, ch_idx, :]
    if ts['decim'] > 1:
        # cropped first, then filtered: an early window never sees later samples
        Xw = decimate(Xw, ts['decim'], axis=-1, zero_phase=True).astype(np.float32)
    return np.ascontiguousarray(Xw)


def prepare_window(subject, cfg, window_ms, fit_idx, use_ea):
    """All 320 trials of one subject, shaped for cfg and (optionally) aligned."""
    X, _, ch_names = load_subject_band(subject, cfg['band'])
    ch_idx = None
    if cfg['channels'] == 'contra':
        ch_idx = contralateral_mask(ch_names, X.shape[1])
        if ch_idx is None:
            raise RuntimeError(f"subject{subject}, band {cfg['band']}: config asks for "
                               f"contralateral channels but no channel names are available")
    Xw = shape_for_config(X, cfg, window_ms, ch_idx)
    if use_ea:
        P = euclidean_alignment_matrix(Xw[fit_idx])
        Xw = np.matmul(P, Xw).astype(np.float32)
    return Xw


# ---- verbatim from run_shallow_all.py -------------------------------------
def z_score_normalize(X_train, X_val, X_test):
    """Per-channel z-score normalization using training data statistics only."""
    channel_means = np.mean(X_train, axis=(0, 2))
    channel_stds = np.std(X_train, axis=(0, 2))
    channel_stds = np.where(channel_stds == 0, 1.0, channel_stds)

    channel_means = channel_means[np.newaxis, :, np.newaxis]
    channel_stds = channel_stds[np.newaxis, :, np.newaxis]

    X_train_norm = (X_train - channel_means) / channel_stds
    X_val_norm = (X_val - channel_means) / channel_stds
    X_test_norm = (X_test - channel_means) / channel_stds

    return X_train_norm, X_val_norm, X_test_norm


def add_gaussian_noise_augmentation(X, noise_level=0.1, random_seed=None):
    """Add Gaussian noise to EEG data for augmentation. DA = DR + NL*GN"""
    if random_seed is not None:
        np.random.seed(int(random_seed))

    channel_stds = np.std(X, axis=(0, 2), keepdims=True)
    noise = np.random.normal(loc=0.0, scale=noise_level * channel_stds, size=X.shape)
    return X + noise
# ---------------------------------------------------------------------------


# ===========================================================================
# STAGE 1 — band screen
# ===========================================================================
def screen_features(X, n_segments=8):
    """Two views of every channel, per time segment.

      mean amplitude : keeps the SIGN of a slow deflection
      log variance   : band power (what mu/beta desynchronization carries)

    Either one alone rigs the screen: amplitude cannot see anything in
    8-30 Hz, power undersells 0.5-3 Hz.
    """
    n, c, t = X.shape
    seg = t // n_segments
    Xs = X[:, :, :seg * n_segments].reshape(n, c, n_segments, seg)
    amp = Xs.mean(axis=3).reshape(n, c * n_segments)
    logvar = np.log(Xs.var(axis=3) + 1e-12).reshape(n, c * n_segments)
    return np.concatenate([amp, logvar], axis=1)


def separability_score(X, y, seed):
    """Mutual information and Fisher ratio against the 8 labels (top decile)."""
    Fm = screen_features(X)
    Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-8)

    mi = mutual_info_classif(Fm, y, random_state=seed)

    overall = Fm.mean(0)
    between = np.zeros(Fm.shape[1])
    within = np.zeros(Fm.shape[1])
    for cls in np.unique(y):
        m = (y == cls)
        between += m.sum() * (Fm[m].mean(0) - overall) ** 2
        within += m.sum() * Fm[m].var(0)
    fisher = between / (within + 1e-8)

    k = max(1, len(mi) // 10)
    return dict(mi=float(np.sort(mi)[-k:].mean()),
                fisher=float(np.sort(fisher)[-k:].mean()))


def stage1_band_screen(subject, fit_idx, seed):
    """Rank bands for one subject on fit_idx (the fold's train+val trials)."""
    n_max = int(WINDOWS_MS[-1] * SAMPLING_RATE / 1000)
    scores = {}
    for band in BAND_ROOTS:
        X, y, _ = load_subject_band(subject, band)
        scores[band] = separability_score(X[fit_idx][:, :, :n_max], y[fit_idx], seed)

    by_mi = sorted(scores, key=lambda b: scores[b]['mi'], reverse=True)
    by_fisher = sorted(scores, key=lambda b: scores[b]['fisher'], reverse=True)
    return by_mi[:N_KEEP_BANDS], dict(
        scores=scores, mi_rank=by_mi, fisher_rank=by_fisher,
        criteria_agree=bool(set(by_mi[:N_KEEP_BANDS]) == set(by_fisher[:N_KEEP_BANDS])))


# ===========================================================================
# MODEL — ShallowConvNet, parameterized by the searched factors.
# With FIXED_CONFIG it is the baseline's network, layer for layer.
# ===========================================================================
class Square(nn.Module):
    def forward(self, x):
        return x * x


class SafeLog(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowConvNet(nn.Module):
    """temporal conv -> spatial conv -> bn -> square -> avgpool -> log -> fc."""

    def __init__(self, n_channels, n_timepoints, n_classes,
                 kernel=25, pool=75, stride=15, n_spat=40, drop_prob=0.5):
        super().__init__()
        n_time = n_spat          # ShallowConvNet keeps these equal

        # keep the pooling window inside whatever the temporal conv leaves
        t_after = n_timepoints - kernel + 1
        if t_after < 4:
            raise ValueError(f"window too short: {n_timepoints} samples, kernel {kernel}")
        pool = int(min(pool, max(2, t_after // 2)))
        stride = int(max(1, min(stride, pool)))

        self.conv_time = nn.Conv2d(1, n_time, kernel_size=(1, kernel), bias=True)
        self.conv_spat = nn.Conv2d(n_time, n_spat, kernel_size=(n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_spat, momentum=0.1, affine=True, eps=1e-5)
        self.square = Square()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool), stride=(1, stride))
        self.safe_log = SafeLog()
        self.drop = nn.Dropout(p=drop_prob)

        with torch.no_grad():
            n_feat = self._features(torch.zeros(1, 1, n_channels, n_timepoints)).shape[1]
        self.classifier = nn.Linear(n_feat, n_classes)

    def _features(self, x):
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.safe_log(x)
        x = self.drop(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.classifier(self._features(x))


BACKBONE_PREFIXES = ('conv_time.', 'conv_spat.', 'bn.')


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


def build_model(cfg, n_channels, n_timepoints, n_classes):
    ts = TEMPORAL_SCALES[cfg['temporal']]
    model = ShallowConvNet(n_channels, n_timepoints, n_classes,
                           kernel=ts['kernel'], pool=ts['pool'], stride=ts['stride'],
                           n_spat=cfg['n_spat']).to(DEVICE)
    model.apply(init_weights)
    return model


# ===========================================================================
# TRAINING — the baseline's recipe, step for step
# ===========================================================================
class EEGDataset(Dataset):
    def __init__(self, eeg_data, labels):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'eeg': self.eeg_data[idx], 'label': self.labels[idx]}


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test, aug_seed, loader_seed):
    """z-score on train, add one noisy copy of the training set, build loaders."""
    Xtr, Xva, Xte = z_score_normalize(X_train, X_val, X_test)
    Xaug = add_gaussian_noise_augmentation(Xtr, noise_level=NOISE_LEVEL,
                                           random_seed=aug_seed)
    Xtr_final = np.concatenate([Xtr, Xaug], axis=0)
    ytr_final = np.concatenate([y_train, y_train], axis=0)

    g = torch.Generator()
    g.manual_seed(int(loader_seed))
    kw = dict(batch_size=BATCH_SIZE, num_workers=0, worker_init_fn=seed_worker)
    tr = DataLoader(EEGDataset(Xtr_final, ytr_final), shuffle=True, generator=g, **kw)
    va = DataLoader(EEGDataset(Xva, y_val), shuffle=False, **kw)
    te = DataLoader(EEGDataset(Xte, y_test), shuffle=False, **kw)
    return tr, va, te


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    for batch in loader:
        eeg = batch['eeg'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(eeg), labels)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(model, loader):
    """Returns (mean batch loss, accuracy %, logits, labels)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    logits_all, labels_all = [], []
    for batch in loader:
        eeg = batch['eeg'].to(DEVICE)
        labels = batch['label'].to(DEVICE)
        out = model(eeg)
        running_loss += criterion(out, labels).item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
        logits_all.append(out.cpu())
        labels_all.append(labels.cpu())
    return (running_loss / len(loader), 100.0 * correct / total,
            torch.cat(logits_all), torch.cat(labels_all))


def fit(model, tr_loader, va_loader, n_epochs):
    """Train; keep the epoch with best val accuracy (ties: lower val loss)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc, best_val_loss, best_state = -1.0, float('inf'), None
    for _ in range(n_epochs):
        train_epoch(model, tr_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, va_loader)
        scheduler.step(val_loss)
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss = val_acc, val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val_acc, best_val_loss


def seed_torch(s):
    torch.manual_seed(int(s))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(s))


# ===========================================================================
# STAGE 2 — ASHA config search, on the 8-class problem
# ===========================================================================
def stage2_search(subject, survivors, y8, tr8, va8, te8, ch_sets, seed, use_ea):
    """Successive halving over the factor grid. Returns the winning config.

    Selection uses the fold's 8-class train and validation trials only. The
    8-class test accuracy of the finalists is recorded as a diagnostic (it is
    how the winner's-curse gap is measured) and never enters the selection.
    """
    window_ms = WINDOWS_MS[-1]
    grid = [dict(band=b, temporal=t, channels=c, n_spat=s)
            for b in survivors
            for t in TEMPORAL_SCALES
            for c in ch_sets
            for s in SPAT_FILTERS]

    prepared = {}

    def prep(cfg):
        key = (cfg['band'], cfg['temporal'], cfg['channels'])
        if key not in prepared:
            prepared[key] = prepare_window(subject, cfg, window_ms, tr8, use_ea)
        return prepared[key]

    alive = list(range(len(grid)))
    scores = {}          # gi -> (val acc, val loss) at the LATEST rung it ran
    test_acc = {}

    def rank_key(g):
        acc, loss = scores[g]
        return (acc, -loss)

    for rung, n_ep in enumerate(ASHA_RUNGS):
        last = (rung == len(ASHA_RUNGS) - 1)
        for gi in alive:
            cfg = grid[gi]
            Xw = prep(cfg)
            # Same seed at every rung: with deterministic training the longer
            # run reproduces the shorter one and carries on from it.
            s = int(seed + 7000 + gi * 17)
            tr, va, te = make_loaders(Xw[tr8], y8[tr8], Xw[va8], y8[va8],
                                      Xw[te8], y8[te8], aug_seed=s, loader_seed=s)
            seed_torch(s)
            model = build_model(cfg, Xw.shape[1], Xw.shape[2], N_CLASSES)
            scores[gi] = fit(model, tr, va, n_ep)
            if last:
                test_acc[gi] = evaluate(model, te)[1]
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not last:
            keep = max(1, len(alive) // ASHA_ETA)
            alive = sorted(alive, key=rank_key, reverse=True)[:keep]

    # Winner comes from the finalists only; configs cut earlier have a score
    # from a shorter run and are not comparable.
    best = max(alive, key=rank_key)
    return grid[best], dict(inner_val_acc_8class=float(scores[best][0]),
                            test_acc_8class=float(test_acc[best]),
                            n_candidates=len(grid), n_finalists=len(alive))


# ===========================================================================
# CASCADE over the four pair models (extra output; the per-window accuracies
# are what line up against the baseline)
# ===========================================================================
def fit_temperature(logits, labels):
    logT = torch.zeros(1, requires_grad=True)
    opt = optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / torch.exp(logT), labels)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        T = float(torch.exp(logT).item())
        return T if math.isfinite(T) and 0.05 < T < 100 else 1.0
    except Exception:
        return 1.0


def accumulate(probs, weights, upto_ms):
    used = [ms for ms in WINDOWS_MS if ms <= upto_ms]
    return sum(weights[ms] * probs[ms] for ms in used) / sum(weights[ms] for ms in used)


def run_cascade(probs, weights, tau):
    n = probs[WINDOWS_MS[0]].shape[0]
    preds = torch.zeros(n, dtype=torch.long)
    times = torch.zeros(n, dtype=torch.float)
    decided = torch.zeros(n, dtype=torch.bool)
    for ms in WINDOWS_MS:
        p = accumulate(probs, weights, ms)
        top2 = torch.topk(p, 2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        if ms == WINDOWS_MS[-1]:
            fire = ~decided
        elif ms in EARLY_EXIT_WINDOWS:
            fire = (~decided) & (margin >= tau)
        else:
            fire = torch.zeros_like(decided)
        if fire.any():
            preds[fire] = p[fire].argmax(1)
            times[fire] = float(ms)
            decided |= fire
    return preds, times


def cascade_eval(val_logits, val_labels, test_logits, test_labels, n_classes):
    """Calibrate, weight and threshold on validation; score on test."""
    val_p, test_p, weights = {}, {}, {}
    for ms in WINDOWS_MS:
        T = fit_temperature(val_logits[ms], val_labels)
        val_p[ms] = F.softmax(val_logits[ms] / T, dim=1)
        test_p[ms] = F.softmax(test_logits[ms] / T, dim=1)
        acc = (val_p[ms].argmax(1) == val_labels).float().mean().item()
        weights[ms] = max(1e-3, acc - 1.0 / n_classes)      # credit above chance
    tot = sum(weights.values())
    weights = {ms: w / tot for ms, w in weights.items()}

    # smallest margin at which exiting early costs nothing on validation
    full = accumulate(val_p, weights, WINDOWS_MS[-1])
    full_acc = (full.argmax(1) == val_labels).float().mean().item()
    se = math.sqrt(max(full_acc * (1 - full_acc), 1e-6) / max(len(val_labels), 1))
    tau = 1.01                                              # = never exit early
    for t in np.arange(0.02, 1.0, 0.02):
        preds, _ = run_cascade(val_p, weights, float(t))
        if (preds == val_labels).float().mean().item() >= full_acc - se:
            tau = float(t)
            break

    preds, times = run_cascade(test_p, weights, tau)
    full_test = accumulate(test_p, weights, WINDOWS_MS[-1])
    return dict(
        n_correct=int((preds == test_labels).sum().item()),
        n_correct_no_exit=int((full_test.argmax(1) == test_labels).sum().item()),
        n_test=int(len(test_labels)),
        mean_decision_ms=float(times.mean().item()),
        tau=tau)


# ===========================================================================
# ONE UNIT OF WORK = one (subject, fold)
# ===========================================================================
def run_unit(subject, fold, ch_sets, opts):
    t_start = time.time()
    unit_seed = int(SEED + subject * 1000 + fold * 100)

    y8 = subject_labels(subject)
    tr8, va8, te8 = subject_splits(y8, subject, N_FOLDS)[fold]
    trainval8 = np.sort(np.concatenate([tr8, va8]))

    # ---- choose the configuration (8-class, training side of the fold only) --
    if opts.arm == 'fixed':
        cfg, survivors, screen_info, search_info = dict(FIXED_CONFIG), None, None, None
    else:
        survivors, screen_info = stage1_band_screen(subject, trainval8, unit_seed)
        cfg, search_info = stage2_search(subject, survivors, y8, tr8, va8, te8,
                                         ch_sets, unit_seed, opts.ea)

    # ---- Stage 3: binary pair models at the frozen config ---------------------
    pair_out = {f"{JOINT_NAMES[a]}/{JOINT_NAMES[b]}": dict(windows={})
                for a, b in ALL_PAIRS}
    val_logits = {k: {} for k in pair_out}
    test_logits = {k: {} for k in pair_out}
    val_labels, test_labels = {}, {}

    for tw_idx, ms in enumerate(WINDOWS_MS):
        Xw = prepare_window(subject, cfg, ms, tr8, opts.ea)
        n_ch, n_t = Xw.shape[1], Xw.shape[2]

        # the baseline's seed formulas, so nothing but the method differs
        aug_seed = int(SEED + subject + fold + tw_idx * 100)
        run_seed = int(SEED + subject * 1000 + fold * 100 + tw_idx)

        backbone = None
        if opts.init == '8class':
            # one 8-class model at this config and window; its feature
            # extractor is the starting point for every pair model below
            s8 = run_seed + 50000
            tr, va, _ = make_loaders(Xw[tr8], y8[tr8], Xw[va8], y8[va8],
                                     Xw[te8], y8[te8], aug_seed=s8, loader_seed=s8)
            seed_torch(s8)
            m8 = build_model(cfg, n_ch, n_t, N_CLASSES)
            fit(m8, tr, va, MAX_EPOCHS)
            backbone = {k: v.detach().clone() for k, v in m8.state_dict().items()
                        if k.startswith(BACKBONE_PREFIXES)}
            del m8

        for a, b in ALL_PAIRS:
            name = f"{JOINT_NAMES[a]}/{JOINT_NAMES[b]}"
            in_pair = (y8 == a) | (y8 == b)
            y_bin = np.where(y8 == b, 1, 0)
            tr_i, va_i, te_i = tr8[in_pair[tr8]], va8[in_pair[va8]], te8[in_pair[te8]]

            tr, va, te = make_loaders(Xw[tr_i], y_bin[tr_i], Xw[va_i], y_bin[va_i],
                                      Xw[te_i], y_bin[te_i],
                                      aug_seed=aug_seed, loader_seed=run_seed)
            seed_torch(run_seed)
            model = build_model(cfg, n_ch, n_t, 2)
            if backbone is not None:
                model.load_state_dict(backbone, strict=False)

            best_val_acc, _ = fit(model, tr, va, MAX_EPOCHS)
            _, te_acc, te_lg, te_lb = evaluate(model, te)
            _, _, va_lg, va_lb = evaluate(model, va)

            pair_out[name]['windows'][str(ms)] = dict(
                accuracy=float(te_acc), val_accuracy=float(best_val_acc),
                n_correct=int((te_lg.argmax(1) == te_lb).sum().item()),
                n_test=int(len(te_lb)))
            val_logits[name][ms], test_logits[name][ms] = va_lg, te_lg
            val_labels[name], test_labels[name] = va_lb, te_lb
            del model

        del Xw
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for name in pair_out:
        pair_out[name]['cascade'] = cascade_eval(
            val_logits[name], val_labels[name],
            test_logits[name], test_labels[name], n_classes=2)

    return dict(subject=subject, fold=fold, arm=opts.arm, ea=bool(opts.ea),
                init=opts.init, seed=unit_seed, gpu=GPU, torch=torch.__version__,
                config=cfg, survivors=survivors, stage1=screen_info,
                stage2=search_info, pairs=pair_out,
                seconds=round(time.time() - t_start, 1))


# ===========================================================================
# DRIVER
# ===========================================================================
def results_root(opts):
    name = f"mi_cmp_{opts.arm}"
    if not opts.ea:
        name += "_noea"
    if opts.init != 'scratch':
        name += "_ft8"
    if SMOKE:
        name += "_SMOKE"
    return os.path.join(DRIVE, name)


def unit_path(root, s, f):
    return os.path.join(root, 'units', f"S{s:02d}_F{f}.json")


def main_run(opts):
    root = results_root(opts)
    os.makedirs(os.path.join(root, 'units'), exist_ok=True)

    bands_needed = list(BAND_ROOTS) if opts.arm == 'personalized' else [FIXED_CONFIG['band']]
    subjects = sorted(set.intersection(*[set(list_subjects(BAND_ROOTS[b]))
                                         for b in bands_needed]))
    print(f"[shard {opts.shard}/{opts.num_shards}] arm={opts.arm} ea={opts.ea} "
          f"init={opts.init} | device={DEVICE} gpu={GPU}")
    print(f"out: {root}")
    print(f"subjects: {subjects}")

    ch_sets = ['bilateral']
    if opts.arm == 'personalized':
        verify_band_folders(subjects)
        # Load every band for the first subject up front: runs the cross-band
        # trial-order check before any compute is spent, and lets a folder
        # without channel names borrow them from one that has them.
        for band in BAND_ROOTS:
            Xp, yp, ch = load_subject_band(subjects[0], band)
            print(f"  {band:<6}: X={Xp.shape} {Xp.dtype}, "
                  f"classes={np.bincount(yp, minlength=N_CLASSES).tolist()}, "
                  f"ch_names={'yes' if ch is not None else 'no'}")
        masks = [contralateral_mask(load_subject_band(subjects[0], b)[2],
                                    load_subject_band(subjects[0], b)[0].shape[1])
                 for b in BAND_ROOTS]
        if all(m is not None for m in masks):
            ch_sets = CHANNEL_SETS
            print(f"channel sets: bilateral = all, contra = {len(masks[0])} "
                  f"(left hemisphere + midline)")
        else:
            print("NOTE: no usable channel names for at least one band, so the "
                  "channel-set factor is off and every config uses all channels.")

    units = [(s, f) for s in subjects for f in range(N_FOLDS)]
    mine = [u for i, u in enumerate(units) if i % opts.num_shards == opts.shard]
    print(f"{len(units)} units total, {len(mine)} on this shard\n")

    t0 = time.time()
    for k, (s, f) in enumerate(mine, 1):
        out = unit_path(root, s, f)
        if os.path.exists(out):
            print(f"[{k}/{len(mine)}] S{s} F{f}: done, skipping")
            continue
        res = run_unit(s, f, ch_sets, opts)
        with open(out + ".tmp", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(out + ".tmp", out)

        last = str(WINDOWS_MS[-1])
        acc = np.mean([p['windows'][last]['accuracy'] for p in res['pairs'].values()])
        c = res['config']
        print(f"[{k}/{len(mine)}] S{s} F{f}: band={c['band']:<6} scale={c['temporal']:<4} "
              f"ch={c['channels']:<9} spat={c['n_spat']:<3} | "
              f"mean pair acc @{last}ms = {acc:5.2f}% | {res['seconds']:.0f}s "
              f"| elapsed {(time.time() - t0) / 60:.1f}min")
        gc.collect()

    print(f"\n[shard {opts.shard}] finished in {(time.time() - t0) / 60:.1f} min")


# ===========================================================================
# MERGE + COMPARISON AGAINST THE BASELINE
# ===========================================================================
def sign_flip_p(d):
    """Two-sided paired permutation test on per-subject differences."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0:
        return float('nan')
    obs = abs(d.mean())
    if n <= 20:                                   # exact: all 2^n sign patterns
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2 - 1
    else:
        signs = np.random.RandomState(0).choice([-1, 1], size=(100000, n))
    return float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())


def main_merge(opts):
    root = results_root(opts)
    files = sorted(glob.glob(os.path.join(root, 'units', '*.json')))
    if not files:
        print(f"no finished units in {root}")
        return
    units = [json.load(open(f)) for f in files]
    print(f"merging {len(units)} units from {root}\n")

    fold_rows, casc_rows, cfg_rows = [], [], []
    for u in units:
        for pair, pr in u['pairs'].items():
            for ms, w in pr['windows'].items():
                fold_rows.append(dict(pair=pair, subject=u['subject'], window_ms=int(ms),
                                      fold=u['fold'] + 1, accuracy=w['accuracy'],
                                      val_accuracy=w['val_accuracy'],
                                      n_correct=w['n_correct'], n_test=w['n_test']))
            c = pr['cascade']
            casc_rows.append(dict(pair=pair, subject=u['subject'], fold=u['fold'] + 1, **c))
        row = dict(subject=u['subject'], fold=u['fold'] + 1, **u['config'])
        if u.get('stage2'):
            row.update(u['stage2'])
            row['screen_agree'] = u['stage1']['criteria_agree']
        cfg_rows.append(row)

    df_fold = pd.DataFrame(fold_rows)
    # per-subject accuracy pooled over folds, as the baseline computes it
    g = df_fold.groupby(['pair', 'subject', 'window_ms'], as_index=False)[['n_correct', 'n_test']].sum()
    g['accuracy'] = 100.0 * g['n_correct'] / g['n_test']
    df_subj = g[['pair', 'subject', 'window_ms', 'accuracy', 'n_test']]
    df_cfg = pd.DataFrame(cfg_rows)
    df_casc = pd.DataFrame(casc_rows)

    df_fold.to_csv(os.path.join(root, 'per_fold.csv'), index=False)
    df_subj.to_csv(os.path.join(root, 'all_pairs_per_subject.csv'), index=False)
    df_cfg.to_csv(os.path.join(root, 'configs.csv'), index=False)
    df_casc.to_csv(os.path.join(root, 'cascade.csv'), index=False)
    (df_subj.groupby(['pair', 'window_ms'])['accuracy'].mean().unstack('window_ms')
     .to_csv(os.path.join(root, 'pair_x_window_mean.csv')))

    print("=" * 72)
    print(f"ARM: {opts.arm} | ea={opts.ea} | init={opts.init}")
    print("mean pair accuracy by window (over pairs and subjects):")
    print(df_subj.groupby('window_ms')['accuracy'].mean().round(2).to_string())

    cs = df_casc.groupby('subject')[['n_correct', 'n_correct_no_exit', 'n_test']].sum()
    print(f"\ncascade over the four windows:")
    print(f"  accuracy with early exit : {100 * cs.n_correct.sum() / cs.n_test.sum():.2f}%")
    print(f"  accuracy, no early exit  : {100 * cs.n_correct_no_exit.sum() / cs.n_test.sum():.2f}%")
    print(f"  mean decision time       : {df_casc.mean_decision_ms.mean():.0f} ms "
          f"(full window {WINDOWS_MS[-1]} ms)")

    # ---- what was chosen, and how consistently -----------------------------
    if opts.arm == 'personalized':
        stab = []
        for factor in ['band', 'temporal', 'channels', 'n_spat']:
            for subj, gs in df_cfg.groupby('subject'):
                vc = gs[factor].astype(str).value_counts()
                # pre-declared rule: the same level in at least 4 of 5 folds
                stab.append(dict(factor=factor, subject=subj, modal_level=vc.index[0],
                                 n_folds_modal=int(vc.iloc[0]), n_folds=len(gs),
                                 stable=bool(vc.iloc[0] >= 0.8 * len(gs) and len(gs) >= 4)))
        stab = pd.DataFrame(stab)
        stab.to_csv(os.path.join(root, 'stability.csv'), index=False)

        print("\nFACTOR STABILITY (same choice in >= 4 of 5 folds)")
        for factor, gs in stab.groupby('factor', sort=False):
            print(f"  {factor:<9}: stable for {int(gs.stable.sum())}/{len(gs)} subjects "
                  f"| modal levels {gs.modal_level.value_counts().to_dict()}")
        gap = (df_cfg.inner_val_acc_8class - df_cfg.test_acc_8class).mean()
        print(f"\n8-class accuracy of the chosen config: "
              f"val {df_cfg.inner_val_acc_8class.mean():.2f}%, "
              f"test {df_cfg.test_acc_8class.mean():.2f}% (chance 12.5%)")
        print(f"selection gap: {gap:+.2f} pp (val minus test; large = search fitted noise)")
        print(f"band screen: MI and Fisher agreed in "
              f"{100 * df_cfg.screen_agree.mean():.0f}% of units")

    # ---- comparison against run_shallow_all.py -----------------------------
    base_files = glob.glob(os.path.join(BASELINE_ROOT, '*', 'per_subject.csv'))
    print("\n" + "=" * 72)
    if not base_files:
        print(f"No baseline results under {BASELINE_ROOT}. Run run_shallow_all.py, "
              f"then --merge again to get the comparison.")
        return
    base = pd.concat([pd.read_csv(f) for f in base_files], ignore_index=True)
    m = df_subj.merge(base, on=['pair', 'subject', 'window_ms'], suffixes=('_new', '_base'))
    print(f"COMPARISON vs baseline ({BASELINE_ROOT})")
    print(f"matched cells: {len(m)} of {len(df_subj)} here, {len(base)} in baseline "
          f"(pair x subject x window; only matched cells are compared)\n")
    if len(m) == 0:
        return
    m['delta'] = m['accuracy_new'] - m['accuracy_base']
    m.to_csv(os.path.join(root, 'vs_baseline_cells.csv'), index=False)

    rows = []
    for ms, gm in m.groupby('window_ms'):
        per_subj = gm.groupby('subject')[['accuracy_base', 'accuracy_new']].mean()
        d = (per_subj.accuracy_new - per_subj.accuracy_base).values
        per_pair = gm.groupby('pair')['delta'].mean()
        rows.append(dict(window_ms=ms, baseline=per_subj.accuracy_base.mean(),
                         this_arm=per_subj.accuracy_new.mean(), delta_pp=d.mean(),
                         subjects_better=f"{int((d > 0).sum())}/{len(d)}",
                         pairs_better=f"{int((per_pair > 0).sum())}/{len(per_pair)}",
                         p_value=sign_flip_p(d),
                         max_abs_cell_diff=gm.delta.abs().max()))
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(root, 'vs_baseline_summary.csv'), index=False)
    print(summary.round(3).to_string(index=False))
    print("\np_value: paired sign-flip permutation test across subjects, two-sided, "
          "not corrected for the four windows.")
    print(f"Wrote per_fold.csv, all_pairs_per_subject.csv, configs.csv, cascade.csv, "
          f"vs_baseline_*.csv -> {root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['personalized', 'fixed'], default='personalized')
    ap.add_argument('--no-ea', dest='ea', action='store_false',
                    help='turn off per-subject Euclidean alignment')
    ap.add_argument('--init', choices=['scratch', '8class'], default='scratch',
                    help="'8class': fine-tune pair models from an 8-class model")
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--merge', action='store_true')
    opts = ap.parse_args()
    if opts.merge:
        main_merge(opts)
    else:
        main_run(opts)
