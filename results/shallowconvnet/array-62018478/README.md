# ShallowConvNet: three additional training seeds

All three Perlmutter array tasks completed with exit code 0:0. The original
seed-42 baseline is retained unchanged in [job-61441374](../job-61441374/README.md).

Mean test accuracy (%) over all 18 subjects and 28 joint pairs:

| Seed | 800 ms | 1500 ms | 3000 ms | 4000 ms |
| --- | ---: | ---: | ---: | ---: |
| 42 | 57.92 | 58.69 | 59.50 | 58.71 |
| 100042 | 57.73 | 58.88 | 59.18 | 58.67 |
| 200042 | 57.87 | 58.58 | 59.40 | 59.05 |
| 300042 | 57.60 | 59.10 | 59.01 | 59.01 |
| **Four-seed mean** | **57.78** | **58.81** | **59.27** | **58.86** |

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
python scripts/prepare_seed_publication.py --verify-only \
  --source-root /pscratch/sd/a/anis/shallowconvnet/data
~~~

Building a new package requires --source-root and an absent target publication
directory. Verification never imports training code or loads EEG/checkpoints.
