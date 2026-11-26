#!/usr/bin/env python3
"""Publish/verify existing seed-run metrics without loading EEG or training.

Build: python scripts/prepare_seed_publication.py --source-root /path/to/data
Check: python scripts/prepare_seed_publication.py --verify-only
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/shallowconvnet/array-62018478"
SEEDS = [100042, 200042, 300042]
WINDOWS = [800, 1500, 3000, 4000]
NAMES = ["HOC", "WFE", "WAA", "EPS", "EFE", "SPS", "SAA", "SFE"]
PAIRS = list(itertools.combinations(NAMES, 2))
KEYS = ["pair", "subject", "window_ms"]
SOURCE_SHA = "731535d4179164cf46744d638ee052622b72b1e7a529b1aa2dda4bc62f3a2af8"
TOP = ["_seed_identity.json", "all_pairs_per_subject.csv", "pair_x_window_mean.csv", "subject_x_pair_4000ms.csv"]
PAIR_FILES = ["per_subject.csv", "per_fold.csv", "wide_subject_x_window.csv", "results.json"]
FILES = TOP + [f"{a}_{b}/{name}" for a, b in PAIRS for name in PAIR_FILES]
BASELINE = REPO / "results/shallowconvnet/job-61441374/tables/all_pairs_per_subject.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, **kwargs):
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def check(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


def ordered(frame, keys=KEYS):
    return frame.sort_values(keys).reset_index(drop=True)


def validate_run(root, seed):
    identity = json.loads((root / "_seed_identity.json").read_text())
    check(identity["seed"] == seed and identity["source_sha256"] == SOURCE_SHA, "Seed/source identity mismatch")
    data = read(root / "all_pairs_per_subject.csv")
    expected = set(itertools.product(["/".join(x) for x in PAIRS], range(1, 19), WINDOWS))
    check(len(data) == 2016 and not data.duplicated(KEYS).any()
          and set(data[KEYS].itertuples(index=False, name=None)) == expected, "Subject coverage mismatch")
    check(np.isfinite(data.accuracy).all() and data.accuracy.between(0, 100).all()
          and data.n_test.eq(80).all(), "Invalid subject metrics")
    parts = []
    for a, b in PAIRS:
        folder = root / f"{a}_{b}"
        subject = read(folder / "per_subject.csv")
        folds = read(folder / "per_fold.csv")
        check(len(folds) == 360 and not folds.duplicated(KEYS + ["fold"]).any(), "Fold coverage mismatch")
        check(set(folds[KEYS + ["fold"]].itertuples(index=False, name=None)) ==
              {("/".join((a, b)), s, w, f) for s, w, f in itertools.product(range(1, 19), WINDOWS, range(1, 6))},
              "Missing fold keys")
        check(folds.n_test.eq(16).all() and np.isfinite(folds[["accuracy", "val_accuracy"]]).all().all()
              and folds.accuracy.between(0, 100).all() and folds.val_accuracy.between(0, 100).all(),
              "Invalid fold metrics")
        close(folds.groupby(KEYS).accuracy.mean().sort_index(),
              subject.set_index(KEYS).accuracy.sort_index())
        wide = read(folder / "wide_subject_x_window.csv", index_col=0)
        wide.columns = wide.columns.astype(int)
        expected_wide = subject.pivot(index="subject", columns="window_ms", values="accuracy").reindex(columns=WINDOWS)
        pd.testing.assert_frame_equal(wide, expected_wide, check_names=False, atol=1e-12, rtol=0)
        metadata = json.loads((folder / "results.json").read_text())
        check(metadata["seed"] == seed and metadata["subjects"] == list(range(1, 19))
              and metadata["windows"] == WINDOWS and metadata["pair"] == "/".join((a, b)),
              "Pair JSON mismatch")
        close(metadata["matrix"], wide.to_numpy())
        parts.append(subject)
    pd.testing.assert_frame_equal(ordered(data), ordered(pd.concat(parts, ignore_index=True)),
                                  atol=1e-12, rtol=0)
    pair = read(root / "pair_x_window_mean.csv", index_col=0)
    pair.columns = pair.columns.astype(int)
    expected_pair = data.groupby(["pair", "window_ms"]).accuracy.mean().unstack().reindex(columns=WINDOWS)
    pd.testing.assert_frame_equal(pair, expected_pair, check_names=False, atol=1e-12, rtol=0)
    full = read(root / "subject_x_pair_4000ms.csv", index_col=0)
    expected_full = data[data.window_ms.eq(4000)].pivot(index="subject", columns="pair", values="accuracy")
    pd.testing.assert_frame_equal(full, expected_full, check_names=False, atol=1e-12, rtol=0)
    return data


def derived(frames):
    matrix = pd.concat({seed: df.set_index(KEYS).accuracy.sort_index()
                        for seed, df in frames.items()}, axis=1)
    check(len(matrix) == 2016 and not matrix.isna().any().any(), "Seed coverage differs")
    tables = {}
    for name, seeds in [("baseline_seed_mean_three_new.csv", SEEDS),
                        ("baseline_seed_mean_four.csv", [42] + SEEDS)]:
        result = matrix[seeds].mean(axis=1).rename("accuracy").reset_index()
        result["n_test"] = 80
        result["n_seeds"] = len(seeds)
        tables[name] = result
    tables["all_seeds_per_subject.csv"] = pd.concat(
        [df.assign(seed=seed) for seed, df in frames.items()], ignore_index=True)
    means = matrix.groupby(level="window_ms").mean()
    tables["per_seed_window_summary.csv"] = means.rename_axis("window_ms").reset_index().melt(
        id_vars="window_ms", var_name="seed", value_name="mean_accuracy_pct")
    tables["window_summary.csv"] = pd.DataFrame([
        dict(seed_set=label, window_ms=w, n_seeds=len(seeds),
             mean_accuracy_pct=means.loc[w, seeds].mean(),
             between_seed_grand_mean_sd_pp=means.loc[w, seeds].std(ddof=1))
        for label, seeds in [("three_new", SEEDS), ("all_four", [42] + SEEDS)] for w in WINDOWS])
    return tables, means


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        check(args.source_root is not None, "--source-root is required to build")
        OUT.mkdir(parents=True, exist_ok=False)
        (OUT / "aggregates").mkdir()
        (OUT / "source").mkdir()
        manifest = {}
        for index, seed in enumerate(SEEDS):
            source = args.source_root / f"mi_cmp_baseline_seed{seed}"
            target = OUT / f"seed-{seed}"
            manifest[str(seed)] = {}
            for name in FILES:
                src, dst = source / name, target / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                check(sha(src) == sha(dst), "Copy mismatch: " + name)
                manifest[str(seed)][name] = sha(src)
            readiness = args.source_root.parent / "logs" / f"shallow_seed_62018478_{index}" / "readiness.json"
            shutil.copy2(readiness, target / "readiness.json")
            for filename, current in [("run_shallow_all.py", REPO / "run_shallow_all.py"),
                                      ("submit_seeds.sh", REPO / "submit_seeds.sh"),
                                      ("seed_resume.py", REPO / "scripts/seed_resume.py")]:
                snapshot = readiness.parent / filename
                check(sha(snapshot) == sha(current), "Current code differs from executed snapshot: " + filename)
                if index == 0:
                    shutil.copy2(snapshot, OUT / "source" / filename)
        original_source = args.source_root.parent / "code/run_shallow_all.py"
        check(sha(original_source) == "e2ea42c2db7f0861caaaece7042bd81c69fd54b9555b3b6dbdb90db7463fec2e",
              "Original seed-42 source changed")
        shutil.copy2(original_source, OUT / "source/run_shallow_all_seed42.py")
        (OUT / "source_output_sha256.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((OUT / "source_output_sha256.json").read_text())
    frames = {42: read(BASELINE)}
    for seed in SEEDS:
        root = OUT / f"seed-{seed}"
        frames[seed] = validate_run(root, seed)
        for name in FILES:
            check(sha(root / name) == manifest[str(seed)][name], "Published source metric changed")
            if args.source_root:
                check(sha(args.source_root / f"mi_cmp_baseline_seed{seed}" / name) == manifest[str(seed)][name],
                      "Scratch source metric changed")
    check(sha(OUT / "source/run_shallow_all.py") == SOURCE_SHA, "Executed source snapshot changed")
    check(sha(OUT / "source/run_shallow_all_seed42.py") ==
          "e2ea42c2db7f0861caaaece7042bd81c69fd54b9555b3b6dbdb90db7463fec2e",
          "Archived original seed-42 source changed")
    tables, means = derived(frames)
    if args.verify_only:
        for name, expected in tables.items():
            actual = read(OUT / "aggregates" / name)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, atol=1e-12, rtol=0)
        for line in (OUT / "SHA256SUMS").read_text().splitlines():
            digest, name = line.split("  ", 1)
            check(sha(OUT / name) == digest, "Published artifact hash changed: " + name)
    else:
        for name, table in tables.items():
            table.to_csv(OUT / "aggregates" / name, index=False, float_format="%.17g")
        validation = json.loads((REPO / "docs/seed-results-validation.json").read_text())
        for w in WINDOWS:
            close(means.loc[w].mean(), validation["all_four_seeds_summary"][str(w)]["mean_percent"])
        provenance = dict(array_job_id=62018478, reference_job=61441374, seeds=SEEDS,
                          state="COMPLETED", exit_codes=["0:0"] * 3,
                          elapsed=["05:46:42", "05:47:06", "05:47:14"],
                          account="m4431_g", requested_qos="shared", resolved_qos="gpu_shared",
                          constraint="gpu&hbm40g", gpu="NVIDIA A100-SXM4-40GB", gpu_count_per_task=1,
                          cpus_per_task=32, memory_per_task_GiB=55, time_limit="08:00:00",
                          source_sha256=SOURCE_SHA, python="3.12.11", torch="2.8.0+cu129",
                          cuda="12.9", cudnn="9.10.2",
                          folds="42 + subject_id, unchanged across training seeds")
        (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        verification = dict(status="PASS", seeds=SEEDS, copied_source_files_per_seed=len(FILES),
                            rows_per_seed=2016, folds_per_seed=10080, all_seed_rows=8064,
                            source_csv_and_json_byte_identical=True, per_pair_and_root_summaries_consistent=True,
                            numeric_tolerance_absolute_percentage_points=1e-12,
                            original_baseline_preserved=True,
                            checkpoint_prediction_validation="../../../docs/seed-results-validation.json")
        (OUT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
        rows = ["| Seed | 800 ms | 1500 ms | 3000 ms | 4000 ms |",
                "| --- | ---: | ---: | ---: | ---: |"]
        for seed in [42] + SEEDS:
            rows.append("| " + str(seed) + " | " + " | ".join(f"{means.loc[w, seed]:.2f}" for w in WINDOWS) + " |")
        rows.append("| **Four-seed mean** | " + " | ".join(f"**{means.loc[w].mean():.2f}**" for w in WINDOWS) + " |")
        readme = """# ShallowConvNet: three additional training seeds

All three Perlmutter array tasks completed with exit code 0:0. The original
seed-42 baseline is retained unchanged in [job-61441374](../job-61441374/README.md).

Mean test accuracy (%) over all 18 subjects and 28 joint pairs:

""" + "\n".join(rows) + """

Each new seed has 2,016 subject/pair/window results and 10,080 fold results.
All four windows (800, 1500, 3000, 4000 ms), 28 pairs, 18 subjects, five folds,
50 epochs, and original preprocessing/model/optimizer settings are retained.
Training seeds vary initialization, training order, and noise augmentation;
train/validation/test folds remain fixed at 42 + subject_id.

## Results and verification

Each seed-<seed>/ directory contains all 116 original CSV/JSON metric and
identity files, byte-for-byte, plus the actual GPU startup readiness report.
Pair directories contain per_subject.csv, per_fold.csv,
wide_subject_x_window.csv, and results.json. Root summaries retain their
original filenames. The seed-42 comparison reads the existing published table.

aggregates/ contains per-cell means for the three new seeds and all four
seeds, a combined 8,064-row subject table, per-seed window means, and window
summaries. The SD in window_summary.csv is the sample SD across seed grand
means (ddof=1), in percentage points; it is not a confidence interval or
between-subject SD. Seed variability is descriptive; these correlated
measurements do not by themselves establish significance.

source/ retains the exact executed training script, array launcher, and
resume helper, plus the archived original seed-42 source. provenance.json records job resources and software;
source_output_sha256.json and SHA256SUMS protect the published metrics.
[Full checkpoint/prediction validation](../../../docs/seed-results-validation.json)
also checked every saved prediction and confusion matrix against the tables.

No EEG arrays, binary checkpoint/prediction pickles, caches, or private backup
archives are included. These remain in verified private Perlmutter storage.

## Runtime and resources

| Array task | Training seed | Elapsed |
| --- | ---: | --- |
| 62018478_0 | 100042 | 05:46:42 |
| 62018478_1 | 200042 | 05:47:06 |
| 62018478_2 | 300042 | 05:47:14 |

Each task used one A100 40 GB GPU, 32 logical CPUs, and 55 GiB host RAM under
account m4431_g with shared QOS (resolved gpu_shared). The eight-hour limit
provided sufficient headroom. PyTorch was 2.8.0+cu129 with CUDA 12.9.
Checkpoint resumption is explicit and identity-checked; see
[the resume procedure](../../../docs/seed-resume.md).

## Verify without training

From the repository root, with NumPy and pandas installed:

~~~bash
python scripts/prepare_seed_publication.py --verify-only
~~~

On Perlmutter, additionally compare the publication byte-for-byte against
the original seed-run outputs (read only):

~~~bash
python scripts/prepare_seed_publication.py --verify-only \\
  --source-root /pscratch/sd/a/anis/shallowconvnet/data
~~~

Building a new package requires --source-root and an absent target publication
directory. Verification never imports training code or loads EEG/checkpoints.
"""
        (OUT / "README.md").write_text(readme)
        paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
        (OUT / "SHA256SUMS").write_text("".join(f"{sha(p)}  {p.relative_to(OUT)}\n" for p in paths))
    print(f"PASS: seeds {SEEDS}, 348 original source CSV/JSON files, 30,240 folds; aggregates and artifact hashes verified")


if __name__ == "__main__":
    main()
