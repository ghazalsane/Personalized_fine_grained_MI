# Three additional ShallowConvNet seeds

All three tasks in array 62018478 completed successfully. See the
[published result tables and aggregates](../results/shallowconvnet/array-62018478/README.md)
and [completed-run validation](seed-results-validation.json).
The submission and preflight snapshots below retain their original timestamps.

Prepared on 2026-11-26. Pre-submission checks PASS. Following explicit approval,
array **62018478** was submitted with three independent tasks. See
[submission evidence](seed-submission.json) and the
[checkpoint resume procedure](seed-resume.md). The validation details below
record checks performed before submission.

The repository is /global/homes/a/anis/shallowconvnet. The three downloads came
from /global/homes/a/anis/baselines and have been copied into that repository.
The training source remains byte-identical to the downloaded update.
The historical source in /pscratch/sd/a/anis/shallowconvnet/code is unchanged.
The new launcher executes a checked snapshot of the updated HOME source.

## Backups and original results

The original run_shallow_all.py and README, all three unmodified downloads,
the initial Git status, a baseline integrity manifest, and the reproducible
preflight checker are saved in:

/global/homes/a/anis/backups/shallowconvnet/pre-seed-update-20261126T053707Z/

Original source: run_shallow_all.py.original
SHA-256: e2ea42c2db7f0861caaaece7042bd81c69fd54b9555b3b6dbdb90db7463fec2e

Updated source SHA-256:
731535d4179164cf46744d638ee052622b72b1e7a529b1aa2dda4bc62f3a2af8

All 648 files (7,267,803 bytes) in
/pscratch/sd/a/anis/shallowconvnet/data/mi_cmp_baseline retain their SHA-256,
sizes, and modification times. The directory was read only during validation.
No original output was deleted, rewritten, resumed, or rerun.

## Resources compared with successful job 61441374

| Setting | Previous job | Prepared array, per task |
| --- | --- | --- |
| Account | m4431_g | m4431_g |
| Requested QOS | shared | shared |
| Resolved QOS / partition | gpu_shared / shared_gpu_ss11 | Routing chosen by Slurm at submission |
| Requested constraint | gpu&hbm40g | gpu&hbm40g |
| Effective constraint | gpu&a100&hbm40g | Requires the same A100 40 GB model at runtime |
| GPU | 1 NVIDIA A100-SXM4-40GB | 1 GPU per task |
| Nodes / tasks | 1 / 1 | 1 / 1 |
| CPUs | 32 logical / 16 physical | 32 logical / 16 physical |
| CPU binding / library threads | cores / 16 | cores / 16 |
| Host memory | 55 GiB | 55 GiB, explicitly requested |
| Time limit | 48 hours | 8 hours |
| Measured elapsed | 05:46:27, exit 0:0 | Not yet run |

Read-only sacctmgr inspection confirms anis has access to m4431_g and shared.
Both shared and gpu_shared currently allow 48 hours; gpu_shared allows up to
two GPUs per job. Each array task requests one GPU. The three tasks may run
concurrently when resources are available. The live shared partition has
MaxMemPerCPU=1796 MiB, so 55 GiB is below the 56.125 GiB limit for 32 CPUs.

The previous log contains all 504 subject/pair training units, zero loaded
checkpoints, zero completed-pair skips, and completion of all 28 pairs.
Eight hours provides 02:13:33, or about 38.55%, above that full-run measurement.
Changed seeds and shared-node conditions can change runtime; eight hours is a
recommendation based on the measured run, rather than a guaranteed duration.

Official references:
[NERSC QOS limits and GPU constraints](https://docs.nersc.gov/jobs/policy/),
[Perlmutter jobs and binding](https://docs.nersc.gov/systems/perlmutter/running-jobs/),
[PyTorch modules](https://docs.nersc.gov/machinelearning/pytorch/).

## Environment and dataset checks

The module pytorch/2.8.0 remains available. All required imports passed with
user-site packages disabled, GPUs hidden, and CPU libraries limited to one thread.

| Component | Current and previous job |
| --- | --- |
| Python | 3.12.11 |
| PyTorch | 2.8.0+cu129 |
| CUDA build / cuDNN | 12.9 / 9.10.2 |
| NumPy / pandas | 2.1.2 / 2.3.3 |
| scikit-learn / SciPy | 1.7.2 / 1.16.2 |
| joblib / threadpoolctl | 1.5.2 / 3.6.0 |

No packages were installed. Module records and package versions are retained
in seed-preflight.json. At job startup, the launcher rechecks these versions,
the source hash, dataset file metadata, allocated CPU/memory settings, and one
visible NVIDIA A100-SXM4-40GB. It refuses CPU fallback.

Dataset:
 /pscratch/sd/a/anis/shallowconvnet/data/datasetpaper_4_40hz_MIonly

All 18 top-level subject NPZ files exist. Each has float32 data shaped
(320, 62, 1000), int64 labels shaped (320,), and classes 0 through 7 with
40 trials each. Headers, labels, file sizes, and modification times match
job 61441374's fingerprint:
f34eb2e3db901b69aff58f005b8a8b6bc0485b8bc4896e1f5b1e7cb204bff072

Only headers and labels were read for this check; EEG payloads were not scanned.
The earlier successful GPU validation remains historical evidence. Actual CUDA
availability on the future allocation will be checked after approved submission.

## Seeds, folds, and output isolation

| Array index | Training seed | Results directory under /pscratch/sd/a/anis/shallowconvnet/data/ |
| --- | --- | --- |
| 0 | 100042 | mi_cmp_baseline_seed100042 |
| 1 | 200042 | mi_cmp_baseline_seed200042 |
| 2 | 300042 | mi_cmp_baseline_seed300042 |

All three output paths are distinct, absent, and resolve to the expected scratch
locations. The launcher sets MI_SEED and PYTHONHASHSEED before Python starts.
It refuses invalid array indices, existing result directories or symlinks, and
simultaneous jobs using the same seed. Automatic requeue is disabled.

The explicit MI_RESUME=1 mode permits identity-checked checkpoint reuse; fresh
mode continues to reject existing paths. Completed results are protected and
validated as described in seed-resume.md.

Training functions, architecture, preprocessing, optimizer, 50 epochs, batch
size 32, four windows, and all 28 pairs are unchanged. AST comparison found only
the intended training-seed assignment, results-root assignment, and seed print.
The fold seed remains 42 + subject_id. All train, validation, and test indices
were compared against the original implementation for all subjects and all
three new seeds. Disjointness, per-class counts, and all pair restrictions passed.

The training script retains its seed-42 default. Use the prepared array launcher
for these additional runs; executing the training file directly without MI_SEED
would select the historical baseline path.

Each task writes separate stdout/stderr:
 /pscratch/sd/a/anis/shallowconvnet/logs/shallow_seed_ARRAYID_TASKID.out
 /pscratch/sd/a/anis/shallowconvnet/logs/shallow_seed_ARRAYID_TASKID.err

A matching per-task log directory retains source snapshots, hashes, Git status,
modules, packages, readiness evidence, temporary files, and caches.
The launcher does not source the historical job_common.sh or invoke its ready
mode, both of which select the original baseline path.

## Script checks and aggregation

Python compilation passed for both supplied Python scripts and the launcher's
embedded preflight. bash -n and git diff --check passed. Direct launcher execution
on the login node exited before module loading, writes, or srun. ShellCheck is
unavailable. No sbatch test-only request was made.

The downloaded aggregator's inner join could silently discard extra cells from
a later run. This was reproduced with synthetic data and corrected by requiring
exact key coverage across runs. The updated aggregator also rejects duplicate
directories, duplicate keys, empty tables, and missing values. Existing test-count
consistency checks remain. Nine synthetic checks passed, including expected
three-seed means with reordered rows and rejection of invalid inputs.
Synthetic outputs are isolated from experimental results at:

/pscratch/sd/a/anis/shallowconvnet/validation/seed-preflight-synthetic-20261126T054230Z/

Aggregation writes baseline_seed_mean.csv into the working directory. After the
jobs complete, run it from a new scratch aggregation directory, never from the
original mi_cmp_baseline directory. No experimental aggregation has been run.

## Approved submission

One array submission creates three independent one-GPU training tasks:

~~~bash
sbatch --parsable --export=ALL,MI_RESUME=0 /global/homes/a/anis/shallowconvnet/submit_seeds.sh
~~~

This command returned array ID 62018478. All three tasks were verified queued
on shared_gpu_ss11 with QOS gpu_shared. Per-task resource requests and exact
stdout/stderr locations are preserved in seed-submission.json.

Changed repository files: run_shallow_all.py, submit_seeds.sh,
aggregate_seeds.py, README.md, docs/seed-runs.md,
docs/seed-update-installation.json, and docs/seed-preflight.json.
Existing uncommitted work was preserved; no commit, pull, or push was performed.

## Verification commands and evidence

The machine-readable record is docs/seed-preflight.json. The checker is preserved
with the backup as the historical fresh-run check. It requires absent seed output
directories, so do not run it against active or existing results. The separate
resume regression check can run without touching experimental results.

~~~bash
module load pytorch/2.8.0
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python /global/homes/a/anis/backups/shallowconvnet/pre-seed-update-20261126T053707Z/preflight-check.py

bash -n /global/homes/a/anis/shallowconvnet/submit_seeds.sh
git -C /global/homes/a/anis/shallowconvnet diff --check
sacct -j 61441374 --format=JobID,Account,Partition,QOS,State,Elapsed,Timelimit,AllocCPUS,ReqMem,Constraints,ExitCode -P
sacctmgr -n -P show assoc where user=anis account=m4431_g format=Cluster,Account,User,Partition,QOS,DefaultQOS
sacctmgr -n -P show qos shared,gpu_shared format=Name,MaxWall,MaxTRESPerJob
module spider pytorch/2.8.0
~~~

Full EEG rescanning, new CUDA execution, training results, and actual runtime for
the changed seeds remain pending compute-node execution after explicit approval.
