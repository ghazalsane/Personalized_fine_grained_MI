"""Validate explicit resumption without changing completed seed-run artifacts."""
import hashlib
import itertools
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = [800, 1500, 3000, 4000]
JOINTS = ["HOC", "WFE", "WAA", "EPS", "EFE", "SPS", "SAA", "SFE"]
PAIRS = {f"{JOINTS[a]}_{JOINTS[b]}": (a, b) for a, b in itertools.combinations(range(8), 2)}
SUMMARIES = ["all_pairs_per_subject.csv", "pair_x_window_mean.csv", "subject_x_pair_4000ms.csv"]
PAIR_FILES = ["per_subject.csv", "per_fold.csv", "wide_subject_x_window.csv", "results.json", "results.pkl"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    stat = path.stat()
    return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def validate_checkpoint(path):
    match = re.fullmatch(r"S([1-9]|1[0-8])\.pkl", path.name)
    require(match is not None, f"Unexpected checkpoint name: {path}")
    # These are trusted checkpoints created by this experiment, not external inputs.
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    require(saved["subject_id"] == int(match[1]) and saved["n_folds"] == 5,
            f"Checkpoint subject/fold mismatch: {path}")
    require(set(saved["time_window_results"]) == set(WINDOWS), f"Missing checkpoint windows: {path}")
    for window in WINDOWS:
        result = saved["time_window_results"][window]
        require(np.isfinite(result["test_accuracy"]) and 0 <= result["test_accuracy"] <= 100
                and result["n_test_samples"] == 80, f"Invalid subject metrics: {path}")
        folds = result["fold_results"]
        require(len(folds) == 5 and sorted(f["fold"] for f in folds) == list(range(1, 6)),
                f"Invalid checkpoint folds: {path}")
        require(all(f["n_test_samples"] == 16 and np.isfinite(f["test_accuracy"])
                    and np.isfinite(f["val_accuracy"])
                    and 0 <= f["test_accuracy"] <= 100 and 0 <= f["val_accuracy"] <= 100
                    for f in folds), f"Invalid fold metrics: {path}")


def validate_completed_pair(directory, identity):
    for name in PAIR_FILES:
        require((directory / name).is_file(), f"Incomplete pair completion marker: {directory / name}")
    require((directory / "_partial").is_dir(), f"Missing checkpoint directory: {directory}")
    a, b = PAIRS[directory.name]
    pair = f"{JOINTS[a]}/{JOINTS[b]}"
    subject = pd.read_csv(directory / "per_subject.csv")
    fold = pd.read_csv(directory / "per_fold.csv")
    for table, keys, expected, count in (
        (subject, ["subject", "window_ms"], set(itertools.product(range(1, 19), WINDOWS)), 80),
        (fold, ["subject", "window_ms", "fold"], set(itertools.product(range(1, 19), WINDOWS, range(1, 6))), 16),
    ):
        require(len(table) == len(expected) and not table.duplicated(keys).any(),
                f"Invalid completed-pair row coverage: {directory}")
        require(set(map(tuple, table[keys].to_numpy())) == expected and (table["pair"] == pair).all(),
                f"Completed-pair keys mismatch: {directory}")
        require(np.isfinite(table["accuracy"]).all() and table["accuracy"].between(0, 100).all()
                and (table["n_test"] == count).all(), f"Invalid completed-pair metrics: {directory}")
    wide = pd.read_csv(directory / "wide_subject_x_window.csv", index_col=0)
    wide.columns = wide.columns.astype(int)
    expected_wide = subject.pivot(index="subject", columns="window_ms", values="accuracy").reindex(columns=WINDOWS)
    pd.testing.assert_frame_equal(wide, expected_wide, check_names=False, atol=1e-12, rtol=0)
    metadata = json.loads((directory / "results.json").read_text())
    require(metadata["seed"] == identity["seed"] and metadata["pair"] == pair
            and metadata["windows"] == WINDOWS and metadata["subjects"] == list(range(1, 19)),
            f"Completed-pair identity mismatch: {directory}")
    np.testing.assert_allclose(metadata["matrix"], wide.to_numpy(), atol=1e-12, rtol=0)


def prepare_results(root, identity, resume=False):
    root = Path(root)
    seed = identity["seed"]
    require(seed in (100042, 200042, 300042) and root.name == f"mi_cmp_baseline_seed{seed}",
            "Only the three non-default seed directories are allowed")
    require(not root.is_symlink() and root.resolve() == root.absolute(), "Symlink result paths are forbidden")
    marker = root / "_seed_identity.json"
    if not resume:
        require(not root.exists(), f"Fresh run refuses existing results: {root}")
        root.mkdir()
        with marker.open("x") as stream:
            json.dump(identity, stream, indent=2)
            stream.write("\n")
        return dict(mode="fresh", protected={}, completed_pairs=0, checkpoints=0)
    require(root.is_dir() and marker.is_file(), "Resume requires an existing directory and seed identity")
    require(json.loads(marker.read_text()) == identity, "Resume source/seed/dataset/environment identity mismatch")
    protected = {str(marker.relative_to(root)): fingerprint(marker)}
    for path in root.rglob("*"):
        require(not path.is_symlink(), f"Symlink artifact is forbidden: {path}")
    for name in SUMMARIES:
        require(not (root / name).exists(), "Run-level summary already exists; review finalization before resuming")
    completed = 0
    checkpoints = 0
    for directory in root.iterdir():
        if not directory.is_dir():
            require(directory.name == marker.name, f"Unexpected top-level artifact: {directory}")
            continue
        require(directory.name in PAIRS, f"Unexpected pair directory: {directory}")
        for checkpoint in (directory / "_partial").glob("*.pkl"):
            validate_checkpoint(checkpoint)
            protected[str(checkpoint.relative_to(root))] = fingerprint(checkpoint)
            checkpoints += 1
        if (directory / "wide_subject_x_window.csv").exists():
            validate_completed_pair(directory, identity)
            completed += 1
            for path in directory.rglob("*"):
                if path.is_file():
                    protected[str(path.relative_to(root))] = fingerprint(path)
    require(completed < len(PAIRS), "All pairs are complete; no training remains to resume")
    return dict(mode="resume", protected=protected, completed_pairs=completed, checkpoints=checkpoints)


def verify_protected(root, protected):
    root = Path(root)
    for name, expected in protected.items():
        path = root / name
        require(path.is_file() and not path.is_symlink() and fingerprint(path) == expected,
                f"Completed artifact was changed: {path}")


if __name__ == "__main__":
    import sys
    require(len(sys.argv) == 3, "Usage: seed_resume.py RESULTS_ROOT READINESS_JSON")
    readiness = json.loads(Path(sys.argv[2]).read_text())
    verify_protected(Path(sys.argv[1]), readiness["resume_state"]["protected"])
    print("PASS: all previously completed results and checkpoints retain their hashes and modification times")
