"""
============================================================================
 Fine-grained MI EEG — DATASET-PAPER BASELINE (ShallowConvNet)
 ALL 28 JOINT PAIRS x ALL SUBJECTS, in one process.

 Same training recipe as run_fine.py: same z-scoring, same noise
 augmentation, same optimizer, same model-selection rule, same determinism
 patches, same per-pair outputs. Differences:
   - the network is ShallowConvNet (the dataset paper's benchmark)
   - the 28 pairs loop INSIDE this script, so the .npz files are read once
   - folds come from subject_splits(): drawn on the 8 classes and restricted
     to the pair, so run_personalized.py can use the very same trials

 THIS IS THE BASELINE ARM for run_personalized.py. The two scripts share
 splits, seeds, augmentation, optimizer and selection rule; what differs is
 only what the personalized script adds.

 Seeds depend on (subject, fold, window) and not on the pair, exactly as in
 run_fine.py.

 USAGE (Colab):
   1. RESTART RUNTIME (CUBLAS_WORKSPACE_CONFIG must be set before CUDA init)
   2. Mount Drive
   3. Run top to bottom. No environment variables needed.

 Resumable at two levels: a finished pair is skipped, and inside an
 unfinished pair a finished subject is loaded from its checkpoint. Killing
 and restarting the job costs you at most one subject.
============================================================================
"""
try:
    from google.colab import drive
    pass
except ImportError:
    pass  # not running in Google Colab

# ===========================================================================
# [P1] DETERMINISM BOOTSTRAP — must run on a FRESH runtime, before torch/CUDA
# ===========================================================================
import os

# Training seed. Default 42 reproduces the original run exactly.
# For extra seed runs: export MI_SEED=<n> before starting Python.
# MI_SEED changes ONLY training randomness (weight init, batch order, noise
# copy). The folds in subject_splits() are pinned to 42 + subject_id and do
# not depend on it, so every seed is trained and tested on the same trials.
seed = int(os.environ.get('MI_SEED', '42'))
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # [P1] before CUDA init

import gc
import random
import glob
import re
import json
import time
import pickle
import itertools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# [P2] was warn_only=True -> silently fell back to nondeterministic kernels
torch.use_deterministic_algorithms(True, warn_only=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device} | gpu={GPU} | torch={torch.__version__}')
print('NOTE: determinism holds per-GPU-model. Record this GPU name.')


# [P4] deterministic DataLoader workers
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ===========================================================================
# CONFIG
# ===========================================================================
JOINT_NAMES = {0: "HOC", 1: "WFE", 2: "WAA", 3: "EPS",
               4: "EFE", 5: "SPS", 6: "SAA", 7: "SFE"}

# All 28 unordered pairs, in a fixed order so reruns are comparable.
ALL_PAIRS = list(itertools.combinations(range(8), 2))

# 4-40 Hz, 250 Hz — the dataset paper's own preprocessing
# Colab default. On a cluster: export MI_DRIVE_ROOT=/path/to/data
DRIVE = os.environ.get('MI_DRIVE_ROOT', '/content/drive/MyDrive')
DATASET_ROOT = f'{DRIVE}/datasetpaper_4_40hz_MIonly'
# Each non-default seed gets its own output folder. This matters: the resume
# logic skips any pair it finds finished, so a new seed pointed at the seed-42
# folder would silently return the seed-42 results.
RESULTS_ROOT = (f'{DRIVE}/mi_cmp_baseline' if seed == 42
                else f'{DRIVE}/mi_cmp_baseline_seed{seed}')
SAMPLING_RATE = 250

TEST_TIME_WINDOWS_MS = [800, 1500, 3000, 4000]
TEST_TIME_WINDOWS_SAMPLES = [int(t * SAMPLING_RATE / 1000) for t in TEST_TIME_WINDOWS_MS]

batch_size = 32
n_epochs = 50
n_classes = 2
n_folds = 5
NOISE_LEVEL = 0.15

os.makedirs(RESULTS_ROOT, exist_ok=True)

print(f"\nModel: ShallowConvNet (dataset-paper baseline)")
print(f"Data : {DATASET_ROOT}")
print(f"Out  : {RESULTS_ROOT}")
print(f"Pairs: {len(ALL_PAIRS)}")
print(f"Seed : {seed} (training only; folds pinned to 42 + subject_id)")
print(f"Test time windows: {TEST_TIME_WINDOWS_MS} ms "
      f"({TEST_TIME_WINDOWS_SAMPLES} samples)")


# ===========================================================================
# HELPERS (verbatim from run_fine.py, except init_weights covers 2-D layers)
# ===========================================================================
def init_weights(m):
    """Initialize model weights deterministically."""
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(eeg)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            eeg = batch['eeg'].to(device)
            labels = batch['label'].to(device)

            outputs = model(eeg)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc, all_preds, all_labels


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
        np.random.seed(int(random_seed))  # [P5] int cast

    channel_stds = np.std(X, axis=(0, 2), keepdims=True)
    noise = np.random.normal(loc=0.0, scale=noise_level * channel_stds, size=X.shape)
    return X + noise


# ===========================================================================
# SPLITS — shared with run_personalized.py. KEEP THE TWO COPIES IDENTICAL.
#
# Folds are drawn once per subject on all 320 trials, stratified by the 8
# classes. A pair's folds are those same folds restricted to its two classes
# (8 test, 8 val, 24 train trials per class). Because both scripts use this
# function, every model in either script is trained, selected and tested on
# exactly the same trials, and the personalized script can choose its
# configuration on a fold's 8-class training trials without ever seeing a
# trial that a pair is later tested on.
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
# MODEL — ShallowConvNet (Schirrmeister et al. 2017, Hum. Brain Mapp. 38:5391)
#
# temporal conv -> spatial conv -> batchnorm -> square -> average pool
# -> log -> dropout -> linear
#
# The square/log pair makes this a band-power model: squaring turns the
# filtered signal into instantaneous power, average pooling integrates it
# over ~300 ms, the log makes the result roughly Gaussian. Note this discards
# the SIGN of a slow deflection — relevant if this is ever pointed at a
# low-frequency folder.
#
# No max-pooling anywhere and avg_pool2d has a deterministic backward, so it
# runs clean under use_deterministic_algorithms(True).
# ===========================================================================
class Square(nn.Module):
    def forward(self, x):
        return x * x


class SafeLog(nn.Module):
    """log with a floor, so a zero-power pooling window can't produce -inf."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowConvNet(nn.Module):
    """ShallowConvNet.

    Signature matches CNNEarlyClassificationModel so the training loop is
    untouched. `embedding_dim` is accepted and ignored — ShallowConvNet has
    no projection head.
    """

    def __init__(self, n_channels, n_timepoints, n_classes=2, embedding_dim=None,
                 n_filters_time=40, n_filters_spat=40,
                 filter_time_length=25, pool_time_length=75,
                 pool_time_stride=15, drop_prob=0.5):
        super(ShallowConvNet, self).__init__()

        # Temporal convolution: slides along time within each channel.
        self.conv_time = nn.Conv2d(
            1, n_filters_time, kernel_size=(1, filter_time_length), bias=True)

        # Spatial convolution: collapses all n_channels into one row by
        # learning a weighting over electrodes. The CSP-like step.
        self.conv_spat = nn.Conv2d(
            n_filters_time, n_filters_spat,
            kernel_size=(n_channels, 1), bias=False)

        self.bn = nn.BatchNorm2d(n_filters_spat, momentum=0.1, affine=True, eps=1e-5)
        self.square = Square()
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_time_length),
                                 stride=(1, pool_time_stride))
        self.safe_log = SafeLog()
        self.drop = nn.Dropout(p=drop_prob)

        # Flattened size from a dry run rather than by hand, so the model
        # adapts to every time window without edits.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            n_features = self._features(dummy).shape[1]

        self.classifier = nn.Linear(n_features, n_classes)
        self.n_features = n_features

    def _features(self, x):
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn(x)
        x = self.square(x)
        x = self.pool(x)
        x = self.safe_log(x)
        x = self.drop(x)
        return torch.flatten(x, start_dim=1)

    def forward(self, x_eeg):
        # Loader hands over (batch, n_channels, n_timepoints); ShallowConvNet
        # wants a channel dimension of 1 in front.
        if x_eeg.dim() == 3:
            x_eeg = x_eeg.unsqueeze(1)
        return self.classifier(self._features(x_eeg))


class EEGDataset(Dataset):
    """Dataset for EEG data."""

    def __init__(self, eeg_data, labels):
        self.eeg_data = torch.FloatTensor(eeg_data)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'eeg': self.eeg_data[idx], 'label': self.labels[idx]}


# ===========================================================================
# LOAD DATA ONCE — all 8 classes, all subjects
# ===========================================================================
def get_subject_number(path):
    basename = os.path.basename(path)
    match = re.search(r"subject(\d+)", basename, re.IGNORECASE)
    return int(match.group(1)) if match else 0


# Subjects are found from the .npz files themselves (same rule as
# run_personalized.py), so the two scripts always see the same subject list.
all_data, all_labels, subject_ids_list = [], [], []
subject_list = sorted({get_subject_number(p) for p in glob.glob(
    os.path.join(DATASET_ROOT, "subject*_eeg_epochs_*.npz"))})
print(f"\nFound {len(subject_list)} subjects: {subject_list}")
assert subject_list, f"no subject*_eeg_epochs_*.npz files in {DATASET_ROOT}"

for sid in subject_list:
    eeg_candidates = glob.glob(os.path.join(DATASET_ROOT, f"subject{sid}_eeg_epochs_*.npz"))
    assert len(eeg_candidates) == 1, f"subject{sid}: expected 1 .npz, found {eeg_candidates}"
    with np.load(eeg_candidates[0], allow_pickle=True) as z:
        # float32 / int64, the same types run_personalized.py loads
        data = z["data"].astype(np.float32).copy()
        labels = z["labels"].astype(np.int64).copy()
    all_data.append(data)
    all_labels.append(labels)
    subject_ids_list.extend([sid] * len(labels))

X_ALL = np.concatenate(all_data, axis=0)
Y_ALL = np.concatenate(all_labels, axis=0)
SUBJ_ALL = np.array(subject_ids_list)
del all_data, all_labels
gc.collect()

n_channels = X_ALL.shape[1]
full_len = X_ALL.shape[2]
TEST_TIME_WINDOWS_SAMPLES_FINAL = [min(s, full_len) for s in TEST_TIME_WINDOWS_SAMPLES]
TEST_TIME_WINDOWS_MS_FINAL = list(TEST_TIME_WINDOWS_MS)

print(f"Loaded X_ALL {X_ALL.shape} ({X_ALL.dtype}, "
      f"{X_ALL.nbytes / 1e9:.2f} GB) | subjects={len(np.unique(SUBJ_ALL))}")

# ShallowConvNet's temporal kernel is 25 samples and its pooling window is 75
# with stride 15. At 250 Hz the shortest window (800 ms = 200 samples) leaves
# 176 after the temporal conv, so 7 pooled steps. This guard catches it if a
# window is ever shortened further.
_min_samples = 25 + 75
for t_ms, t_samp in zip(TEST_TIME_WINDOWS_MS_FINAL, TEST_TIME_WINDOWS_SAMPLES_FINAL):
    assert t_samp >= _min_samples, (
        f"{t_ms}ms is {t_samp} samples; ShallowConvNet needs at least "
        f"{_min_samples} at this kernel/pool setting")


# ===========================================================================
# ONE PAIR — run_fine.py's training loop, on the shared folds
# ===========================================================================
def run_pair(class_a, class_b):
    pair_name = f"{JOINT_NAMES[class_a]}/{JOINT_NAMES[class_b]}"
    pair_slug = pair_name.replace("/", "_")
    pair_dir = os.path.join(RESULTS_ROOT, pair_slug)
    partial_dir = os.path.join(pair_dir, "_partial")
    os.makedirs(partial_dir, exist_ok=True)

    # already finished?
    if os.path.exists(os.path.join(pair_dir, "wide_subject_x_window.csv")):
        print(f"\n{pair_name}: already complete, skipping")
        return pd.read_csv(os.path.join(pair_dir, "per_subject.csv"))

    print(f"\n{'=' * 68}")
    print(f"PAIR: {pair_name} (classes {class_a} vs {class_b})")
    print(f"{'=' * 68}")

    unique_subjects = [int(s) for s in np.unique(SUBJ_ALL)]   # [P5] python ints
    all_subject_results = []

    for subj_id in unique_subjects:
        subj_id = int(subj_id)
        ckpt = os.path.join(partial_dir, f"S{subj_id}.pkl")
        if os.path.exists(ckpt):
            with open(ckpt, "rb") as f:
                all_subject_results.append(pickle.load(f))
            print(f"S{subj_id:>2}: loaded from checkpoint")
            continue

        # all 320 trials of this subject, all 8 classes
        subj_rows = np.where(SUBJ_ALL == subj_id)[0]
        X8, y8 = X_ALL[subj_rows], Y_ALL[subj_rows]
        in_pair = (y8 == class_a) | (y8 == class_b)
        y_subj = np.where(y8 == class_b, 1, 0)        # only read where in_pair

        if in_pair.sum() < 10 or len(np.unique(y8[in_pair])) < 2:
            print(f"Skipping Subject {subj_id}: insufficient data")
            continue

        # 8-class folds, shared with run_personalized.py; same folds for every
        # window and every pair
        cv_splits = subject_splits(y8, subj_id, n_folds)

        subject_time_window_results = {}
        t_subj_start = time.time()

        for tw_idx, t_ms in enumerate(TEST_TIME_WINDOWS_MS_FINAL):
            t_samples = TEST_TIME_WINDOWS_SAMPLES_FINAL[tw_idx]
            X_subj_tw = X8[:, :, :t_samples]
            fold_results_tw = []

            for fold_idx, (tr8, va8, te8) in enumerate(cv_splits):
                # restrict the 8-class fold to this pair's two classes
                train_indices = tr8[in_pair[tr8]]
                val_indices = va8[in_pair[va8]]
                test_indices = te8[in_pair[te8]]

                X_train, y_train = X_subj_tw[train_indices], y_subj[train_indices]
                X_val, y_val = X_subj_tw[val_indices], y_subj[val_indices]
                X_test, y_test = X_subj_tw[test_indices], y_subj[test_indices]

                X_train_norm, X_val_norm, X_test_norm = z_score_normalize(
                    X_train, X_val, X_test)

                aug_seed = int(seed + int(subj_id) + int(fold_idx) + int(tw_idx) * 100)
                X_train_augmented = add_gaussian_noise_augmentation(
                    X_train_norm, noise_level=NOISE_LEVEL, random_seed=aug_seed)
                X_train_final = np.concatenate([X_train_norm, X_train_augmented], axis=0)
                y_train_final = np.concatenate([y_train, y_train], axis=0)

                train_dataset = EEGDataset(X_train_final, y_train_final)
                val_dataset = EEGDataset(X_val_norm, y_val)
                test_dataset = EEGDataset(X_test_norm, y_test)

                loader_seed = int(seed + int(subj_id) * 1000
                                  + int(fold_idx) * 100 + int(tw_idx))
                train_generator = torch.Generator()
                train_generator.manual_seed(loader_seed)

                # [P4] worker_init_fn on all three loaders
                train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=0,
                                          generator=train_generator,
                                          worker_init_fn=seed_worker)
                val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                        shuffle=False, num_workers=0,
                                        worker_init_fn=seed_worker)
                test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0,
                                         worker_init_fn=seed_worker)

                model_init_seed = int(seed + int(subj_id) * 1000
                                      + int(fold_idx) * 100 + int(tw_idx))
                torch.manual_seed(model_init_seed)   # [P5] int cast
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(model_init_seed)

                model = ShallowConvNet(
                    n_channels=n_channels, n_timepoints=t_samples,
                    n_classes=n_classes).to(device)
                model.apply(init_weights)

                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=5)

                best_val_acc, best_val_loss = -1.0, float('inf')
                best_model_state = None

                for epoch in range(n_epochs):
                    train_loss, train_acc = train_epoch(model, train_loader,
                                                        criterion, optimizer, device)
                    val_loss, val_acc, _, _ = validate(model, val_loader,
                                                       criterion, device)
                    scheduler.step(val_loss)

                    # best val accuracy; ties broken by lower val loss
                    if val_acc > best_val_acc or (val_acc == best_val_acc
                                                  and val_loss < best_val_loss):
                        best_val_acc, best_val_loss = val_acc, val_loss
                        best_model_state = {k: v.detach().clone()
                                            for k, v in model.state_dict().items()}

                if best_model_state is not None:
                    model.load_state_dict(best_model_state)

                test_loss, test_acc, test_preds, test_labels = validate(
                    model, test_loader, criterion, device)
                test_preds = np.array(test_preds)
                test_labels = np.array(test_labels)

                fold_results_tw.append({
                    'fold': fold_idx + 1,
                    'test_accuracy': test_acc,
                    'test_loss': test_loss,
                    'test_preds': test_preds,
                    'test_labels': test_labels,
                    'n_test_samples': len(test_labels),
                    'val_accuracy': best_val_acc,
                })

                del model, best_model_state, optimizer

            all_test_preds_tw = np.concatenate([r['test_preds'] for r in fold_results_tw])
            all_test_labels_tw = np.concatenate([r['test_labels'] for r in fold_results_tw])

            subject_time_window_results[t_ms] = {
                'test_accuracy': accuracy_score(all_test_labels_tw, all_test_preds_tw) * 100,
                'test_preds': all_test_preds_tw,
                'test_labels': all_test_labels_tw,
                'confusion_matrix': confusion_matrix(all_test_labels_tw, all_test_preds_tw),
                'n_test_samples': len(all_test_labels_tw),
                'fold_results': fold_results_tw,
            }

        all_subject_results.append({'subject_id': subj_id, 'n_folds': n_folds,
                                    'time_window_results': subject_time_window_results})
        with open(ckpt + ".tmp", "wb") as f:
            pickle.dump(all_subject_results[-1], f)
        os.replace(ckpt + ".tmp", ckpt)

        print(f"S{subj_id:>2}: " + " ".join(
            f"{t}ms={subject_time_window_results[t]['test_accuracy']:6.2f}%"
            for t in TEST_TIME_WINDOWS_MS_FINAL)
            + f"   [{time.time() - t_subj_start:5.1f}s]")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # [P7] SAVE PER-SUBJECT + PER-FOLD RESULTS FOR THIS PAIR
    # -----------------------------------------------------------------------
    subj_rows, fold_rows = [], []
    for r in sorted(all_subject_results, key=lambda x: x['subject_id']):
        sid = int(r['subject_id'])
        for w in TEST_TIME_WINDOWS_MS_FINAL:
            twr = r['time_window_results'][w]
            subj_rows.append(dict(pair=pair_name, subject=sid, window_ms=w,
                                  accuracy=float(twr['test_accuracy']),
                                  n_test=int(twr['n_test_samples'])))
            for f in twr['fold_results']:
                fold_rows.append(dict(pair=pair_name, subject=sid, window_ms=w,
                                      fold=int(f['fold']),
                                      accuracy=float(f['test_accuracy']),
                                      val_accuracy=float(f['val_accuracy']),
                                      n_test=int(f['n_test_samples'])))

    df_subj = pd.DataFrame(subj_rows)
    df_fold = pd.DataFrame(fold_rows)

    # subjects x windows, fixed row/column order — what the paired tests consume
    wide = (df_subj.pivot(index='subject', columns='window_ms', values='accuracy')
            .reindex(columns=TEST_TIME_WINDOWS_MS_FINAL).sort_index())

    df_subj.to_csv(f"{pair_dir}/per_subject.csv", index=False)
    df_fold.to_csv(f"{pair_dir}/per_fold.csv", index=False)
    wide.to_csv(f"{pair_dir}/wide_subject_x_window.csv")

    payload = dict(pair=pair_name, class_a=int(class_a), class_b=int(class_b),
                   model="ShallowConvNet", dataset_root=DATASET_ROOT,
                   windows=TEST_TIME_WINDOWS_MS_FINAL, seed=seed,
                   gpu=GPU, torch=torch.__version__,
                   subjects=[int(s) for s in wide.index],
                   matrix=wide.to_numpy().tolist())
    with open(f"{pair_dir}/results.pkl", "wb") as f:
        pickle.dump(payload, f)
    with open(f"{pair_dir}/results.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSAVED {pair_name} -> {pair_dir}")
    print(wide.round(2).to_string())
    print("mean:", wide.mean().round(2).to_dict())
    if wide.isna().any().any():
        print("WARNING: missing subject/window cells — a subject was skipped.")

    gc.collect()

    return df_subj


# ===========================================================================
# ALL 28 PAIRS
# ===========================================================================
t0 = time.time()
per_pair_frames = []

for pair_idx, (a, b) in enumerate(ALL_PAIRS, start=1):
    elapsed = time.time() - t0
    print(f"\n### pair {pair_idx}/{len(ALL_PAIRS)}  "
          f"[elapsed {elapsed / 60:.1f} min]")
    per_pair_frames.append(run_pair(a, b))

# ===========================================================================
# COMBINED SUMMARY ACROSS ALL PAIRS
# ===========================================================================
combined = pd.concat(per_pair_frames, ignore_index=True)
combined.to_csv(os.path.join(RESULTS_ROOT, "all_pairs_per_subject.csv"), index=False)

# pairs x windows, mean over subjects — the table that compares to the paper
pair_x_window = (combined.groupby(['pair', 'window_ms'])['accuracy'].mean()
                 .unstack('window_ms')
                 .reindex(columns=TEST_TIME_WINDOWS_MS_FINAL))
pair_x_window.to_csv(os.path.join(RESULTS_ROOT, "pair_x_window_mean.csv"))

# subjects x pairs at the full window, for the per-subject spread
full_w = TEST_TIME_WINDOWS_MS_FINAL[-1]
subj_x_pair = (combined[combined.window_ms == full_w]
               .pivot(index='subject', columns='pair', values='accuracy'))
subj_x_pair.to_csv(os.path.join(RESULTS_ROOT, f"subject_x_pair_{full_w}ms.csv"))

print(f"\n{'=' * 68}")
print(f"ALL {len(ALL_PAIRS)} PAIRS COMPLETE in {(time.time() - t0) / 60:.1f} min")
print(f"model=ShallowConvNet | gpu={GPU} | torch={torch.__version__} | seed={seed}")
print(f"{'=' * 68}")
print("\nMean accuracy per pair (sorted, descending), at each window:")
print(pair_x_window.sort_values(full_w, ascending=False).round(2).to_string())
print(f"\nGrand mean over all pairs and subjects, by window:")
print(combined.groupby('window_ms')['accuracy'].mean().round(2).to_string())
print(f"\nPer-subject spread at {full_w} ms (mean over the 28 pairs):")
print(subj_x_pair.mean(axis=1).round(2).to_string())
print(f"\nWrote: all_pairs_per_subject.csv, pair_x_window_mean.csv, "
      f"subject_x_pair_{full_w}ms.csv")


# ===========================================================================
# REPRODUCIBILITY CHECK
# Run ONE pair twice, restarting the runtime in between, changing nothing.
# Copy the first output dir to <PAIR>_run2, then:
#
#   a = pd.read_csv(f"{RESULTS_ROOT}/HOC_SAA/wide_subject_x_window.csv", index_col=0)
#   b = pd.read_csv(f"{RESULTS_ROOT}/HOC_SAA_run2/wide_subject_x_window.csv", index_col=0)
#   print("max |diff| =", (a - b).abs().values.max(), "pp")
#
# 0.0      -> deterministic; proceed
# non-zero -> check the printed gpu= name matched across both runs
# ===========================================================================
