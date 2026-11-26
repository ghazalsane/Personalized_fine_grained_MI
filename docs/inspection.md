# Preparation evidence — 2026-11-16

Original project: `/pscratch/sd/a/anis/shallowconvnet/code`, commit `b80ade9`.
The original repository is clean and its training file is byte-identical to
the permanent HOME clone. No dataset files were written or changed.

Training source SHA256: `e2ea42c2db7f0861caaaece7042bd81c69fd54b9555b3b6dbdb90db7463fec2e`.

Read-only discovery used `module spider`, `module show`, `sacctmgr show assoc`,
`sacctmgr show user`, `sacctmgr show qos`, `sinfo`, `scontrol show partition`,
`iris user anis`, `showquota`, Git status/history, AST inspection, NPZ headers
and label arrays. Account/QOS membership confirms `m4431_g` with `shared`.
The live shared-QOS limit is 48 hours and at most two GPUs per job.

`module load pytorch/2.8.0` and all required imports passed. Module loading and
`pip freeze --all` also passed with `set -euo pipefail`. Exact package and module
records are retained in the inspection scratch directory below. The module
`pytorch/2.6.0-1` was inspected but is deprecated; it is not used by these jobs.

`bash -n` passed for both job files and the common environment helper. Python
compilation passed for the original training source and both new Python tools.
Both batch files refused direct execution on the login node, and the GPU
validation mode refused execution outside an srun compute-node step. These
guard checks do not allocate GPUs or submit jobs.

The metadata-only validation ran with CUDA_VISIBLE_DEVICES empty and a single
CPU library thread. It checked all subjects, classes, counts and original folds
for all 28 pairs. Each pair/fold has 24 train, 8 validation and 8 test trials per
class, with disjoint splits and each trial tested once across five folds.
An isolated synthetic checkpoint exercised the original completed-subject load
and completed-pair skip branches with no fitting. Synthetic fixture accuracies
are not experiment results. The progress utility correctly reports zero
completed experimental pairs/checkpoints and no runtime estimate yet.

Inspection output:
`/pscratch/sd/a/anis/shallowconvnet/validation/inspection-20261116/metadata.json`.
Its neighboring `modules.txt` and `packages.txt` record the verified environment.

No GPU validation, full EEG payload scan, training, live interruption test,
performance measurement, scheduler submission test, or Slurm submission was
performed. No job ID exists for this preparation. GPU functionality and runtime
remain pending the user's approval and the prepared validation job.
