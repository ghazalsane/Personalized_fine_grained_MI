"""Combine ShallowConvNet baseline runs from several seeds.

Usage:
    python aggregate_seeds.py DIR_SEED_A DIR_SEED_B [DIR_SEED_C ...]

Each DIR holds all_pairs_per_subject.csv from one run. Writes
baseline_seed_mean.csv (accuracy averaged over seeds, per pair/subject/window)
and prints the seed-to-seed noise floor.
"""
import sys
from pathlib import Path

import pandas as pd

KEYS = ["pair", "subject", "window_ms"]
dirs = sys.argv[1:]
if len(dirs) < 2:
    sys.exit(__doc__)
if len({Path(d).resolve() for d in dirs}) != len(dirs):
    sys.exit("Each input must be a distinct run directory")

runs = []
for i, d in enumerate(dirs):
    df = pd.read_csv(f"{d.rstrip('/')}/all_pairs_per_subject.csv")
    indexed = df.set_index(KEYS)[["accuracy", "n_test"]].sort_index()
    if indexed.empty or not indexed.index.is_unique or indexed.isna().any().any():
        sys.exit(f"Empty, duplicate, or missing result cells in {d}")
    if i and not indexed.index.equals(runs[0].index):
        sys.exit("runs do not cover the same pair/subject/window cells")
    runs.append(indexed.add_suffix(f"_{i}"))
allr = pd.concat(runs, axis=1, join="inner")
assert len(allr) == len(runs[0]), "runs do not cover the same pair/subject/window cells"

acc = allr.filter(like="accuracy_")
ntest = allr.filter(like="n_test_")
assert (ntest.nunique(axis=1) == 1).all(), "test-trial counts differ between runs"
for i in range(1, acc.shape[1]):
    same = (acc.iloc[:, 0] == acc.iloc[:, i]).mean()
    if same > 0.5:
        print(f"WARNING: run {i} is {same:.0%} identical to run 0 - "
              f"the seed probably did not change ({dirs[i]})")

out = acc.mean(axis=1).rename("accuracy").reset_index()
out["n_seeds"] = acc.shape[1]
out.to_csv("baseline_seed_mean.csv", index=False)

# subject means over the 28 pairs, one column per seed
subj = acc.groupby(level=["window_ms", "subject"]).mean()
print(f"{acc.shape[1]} seeds, {len(acc)} cells\n")
print("window   grand mean per seed            seed SD of a subject's 28-pair mean")
for w, g in subj.groupby(level="window_ms"):
    means = "  ".join(f"{m:5.2f}" for m in g.mean())
    sd = g.std(axis=1, ddof=1)
    print(f"{w:>5} ms  {means:<28}  median {sd.median():.2f} pp, max {sd.max():.2f} pp")
print("\nWrote baseline_seed_mean.csv")
