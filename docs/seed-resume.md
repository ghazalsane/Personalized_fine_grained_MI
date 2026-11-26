# Checkpoint resumption for the three additional seeds

The approved fresh array is 62018478: task 0 uses seed 100042, task 1 uses
200042, and task 2 uses 300042. Submission evidence and current-at-check
scheduler state are recorded in seed-submission.json.

The scheduler resources and training source are unchanged from validation.
The launcher now has an explicit MI_RESUME flag. Fresh submissions use
MI_RESUME=0 and continue to reject existing result directories.

## Resume behavior

MI_RESUME=1 requires the exact existing non-default seed directory and its
_seed_identity.json. A fresh task writes this identity before training begins.
The identity includes seed, training source SHA-256, dataset fingerprint,
Python/package/CUDA/cuDNN versions, and GPU model. A mismatch blocks reuse.
The launcher also retains the per-seed lock, preventing concurrent writers.

A resume reads completed _partial/S<ID>.pkl checkpoints and skips completed
pairs. Saved subject checkpoints and all files in completed pairs are hashed,
and their sizes and modification times are recorded. A post-run check verifies
they remain unchanged. Completed-pair CSV coverage, values, metadata, and wide
matrix are validated before accepting the completion marker. Corrupt or
truncated artifacts block resumption for review.

The training implementation remains the exact downloaded run_shallow_all.py.
Resume granularity is one completed subject within a pair. An interrupted
subject restarts its four windows and five folds; optimizer/epoch state is
not saved. Valid completed subjects are loaded rather than fitted again.

A run with an existing top-level summary, or with all 28 pairs complete, is
refused for training resumption. If interruption occurred during final summary
writing, inspect/finalize those summaries separately. Existing summaries are
never overwritten by this resume path.

Only mi_cmp_baseline_seed100042, mi_cmp_baseline_seed200042, and
mi_cmp_baseline_seed300042 are accepted. The original mi_cmp_baseline directory
is excluded by both path and seed checks.

## Verification

Eleven CPU-only synthetic checks passed. They exercised the unchanged run_pair:
a completed pair was skipped; a saved subject loaded before a synthetic
interruption at the next subject; and an incomplete pair was rebuilt from saved
subject checkpoints. Previously completed files retained SHA-256 values,
sizes, and modification times. No model fit or CUDA operation was executed.

The tests also checked rejection of fresh-mode directory reuse, changed
identity, the original baseline path, missing identity, a truncated completion
marker, and an existing top-level summary. See seed-resume-validation.json.

Synthetic fixtures are kept under:
 /pscratch/sd/a/anis/shallowconvnet/validation/resume-SYNTHETIC-20261126T055310Z

This is a controlled checkpoint-path test, not a live cancelled GPU training
run. No active training job was interrupted or cancelled.

## Future approved replacement submission

After an interruption, inspect the batch stdout/stderr, sacct status, source and
environment, and checkpoint directory. Once the replacement submission is
explicitly approved, use only the affected array index. For seed 100042:

~~~bash
sbatch --array=0 --export=ALL,MI_RESUME=1 /global/homes/a/anis/shallowconvnet/submit_seeds.sh
~~~

Use index 1 for seed 200042 or index 2 for seed 300042. New array IDs create new
log files while reusing that seed's existing checkpoint directory. Do not
submit a replacement while the original task is still running. This example
has not been executed; only the approved three fresh tasks were submitted.

## Changed files and checks

Updated submit_seeds.sh; added scripts/seed_resume.py and
scripts/check_seed_resume.py; updated README and the seed-run documentation and
validation records. A pre-change backup is retained in:
 /global/homes/a/anis/backups/shallowconvnet/pre-submit-resume-20261126T055157Z

Python compilation, bash -n, git diff --check, the resume regression check,
and the complete lightweight seed preflight passed before submission.
The latter revalidated dependencies, dataset headers/labels, all splits,
output separation, and the original baseline's 648 file hashes and mtimes.
A post-submission integrity comparison also passed.

Reproduce the synthetic resume check with GPUs hidden and one CPU library
thread; fixtures and outputs remain separate from experiment results:

~~~bash
module load pytorch/2.8.0
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python /global/homes/a/anis/shallowconvnet/scripts/check_seed_resume.py
~~~
