# ShallowConvNet on NERSC Perlmutter

Current status: all three tasks in array 62018478 completed with exit code 0:0.
See [published results](../results/shallowconvnet/array-62018478/README.md).
The dated preparation notes below preserve the original setup history.

## Additional seeds prepared 2026-11-26

The updated training source, submit_seeds.sh, and aggregate_seeds.py are installed
in this HOME repository. See [the three-seed preflight report](seed-runs.md)
and [machine-readable validation](seed-preflight.json) for the backup,
resource comparison with job 61441374, separate output paths, and proposed
eight-hour array submission. Array **62018478** has now been submitted with
three tasks. See [submission evidence](seed-submission.json) and the
[explicit checkpoint resume procedure](seed-resume.md).

The sections below describe the original seed-42 setup. The additional seeds
use submit_seeds.sh, which executes the updated HOME source and selects separate
scratch results directories. The historical scratch source and all original
mi_cmp_baseline results remain unchanged.

Prepared for anis on 2026-11-16. No GPU job has been submitted and no training
has been run as part of preparation.

This is a permanent local clone of the research repository at commit `b80ade9`.
The launch scripts execute the user's original, unchanged file:
`/pscratch/sd/a/anis/shallowconvnet/code/run_shallow_all.py`.
A byte-identical source copy is retained in this HOME repository. No changes
have been pushed to GitHub. The original scratch repository remains clean.

## Environment and dataset

Use the NERSC-managed `pytorch/2.8.0` module. All required imports passed with
user-site packages disabled. No pip/conda installation is needed.

| Component | Verified version |
| --- | --- |
| Python | 3.12.11 |
| PyTorch | 2.8.0+cu129 |
| CUDA build | 12.9 |
| cuDNN | 9.10.2 |
| NumPy | 2.1.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| SciPy (scikit-learn dependency) | 1.16.2 |
| joblib (scikit-learn dependency) | 1.5.2 |
| threadpoolctl (scikit-learn dependency) | 3.6.0 |

Direct external dependencies are `numpy`, `pandas`, `torch`, and `scikit-learn`.
Standard-library imports are `os`, `gc`, `random`, `glob`, `re`, `json`, `time`,
`pickle`, `itertools`, and `warnings`. `google.colab.drive` is an optional import
inside an ImportError handler; it is not required on Perlmutter. There is no
need for braindecode, MNE, TensorFlow, torchvision, torchaudio, or a separate
CUDA toolkit installation.

The launcher exports:

```bash
export MI_DRIVE_ROOT=/pscratch/sd/a/anis/shallowconvnet/data
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
module load pytorch/2.8.0
```

`PYTHONHASHSEED` is set before Python starts; the original script's assignment
alone occurs too late to select the interpreter's hash seed. Other original
seeding/determinism settings run unchanged. CPU library threads are bounded to
16 physical cores; Slurm allocates 32 logical CPUs. No mixed precision, TF32,
architecture, data-loader, preprocessing, or experiment settings are changed.

Data remains read-only to the inspection code. The exact loader glob finds
subjects 1 through 18, with one top-level NPZ each, under:
`/pscratch/sd/a/anis/shallowconvnet/data/datasetpaper_4_40hz_MIonly`.
Each contains float32 EEG `(320, 62, 1000)` and int64 labels `(320,)`, with
classes 0..7 occurring 40 times each. The existing `subject1/` etc. subfolders
are ignored by the original loader. Together there are 5,760 trials and about
1.43 GB of float32 EEG before temporary copies; RAM is well within the request.

The original configuration is preserved: 28 pairs, 18 subjects, 4 windows
(800/1500/3000/4000 ms), 5 folds, 50 epochs, batch size 32, seed 42, noise 0.15,
Adam lr=0.001/weight_decay=1e-4, and the original scheduler/selection rule.
That is 10,080 model fits and 504,000 training epochs in total.
No reference software lockfile or reference-run results were supplied, so
matching a historical numerical result is not yet established.

## Slurm resources and submission

Read-only `sacctmgr` inspection confirmed `anis` is associated with GPU account
`m4431_g` and has access to `shared`. CPU default `m4431` is not used. `iris user
anis` also confirmed a positive GPU allocation balance. Recheck with that
command if the allocation changes before submission.

Both scripts request one A100 40 GB GPU using `gpu&hbm40g`, `shared`, one node,
one task, 32 logical CPUs (=16 physical cores), 55 GiB host RAM, and core binding.
The live shared partition reports MaxMemPerCPU=1796 MiB, so 55 GiB is below
32 x 1796 MiB = 56.125 GiB. NERSC documents a nominal 64 GB host-memory share
per GPU; requesting 64 GiB explicitly would exceed that live per-CPU limit.
The full request uses the documented and locally confirmed shared-QOS maximum
of 48 hours. That is a scheduling limit, not a measured runtime prediction.
There is no hard-coded partition name: NERSC routes `--qos=shared` with the GPU
constraint. Both scripts explicitly disable automatic requeueing.

The log directory already exists; Slurm needs it before opening stdout/stderr.
All commands below are for the user to run after approving job submission.

First submit the ten-minute validation job:

```bash
sbatch /global/homes/a/anis/shallowconvnet/jobs/validate_shared.sbatch
```

Wait for COMPLETED with exit code 0. Its report must say PASS (gpu):

```bash
sacct -j VALIDATION_JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode
cat /pscratch/sd/a/anis/shallowconvnet/validation/gpu_passed.json
```

Then submit the full experiment:

```bash
sbatch /global/homes/a/anis/shallowconvnet/jobs/shallow_shared.sbatch
```

The validation job reads every EEG payload to check finiteness, checks the
original splits for every subject/pair/fold, requires one visible A100,
and runs the original model's deterministic CUDA forward/backward kernels at
all four windows on two examples. It does not fit a model or change the
experiment. It also checks completed-subject/pair resume paths in an isolated,
explicitly synthetic fixture under its own log directory. Synthetic fixture
accuracy is test data and must not be reported as an experiment result.

The full job requires successful GPU validation and compares source SHA256,
dataset file metadata, package versions, CUDA/cuDNN, and GPU model with that
report. An additional identity file in the results directory prevents mixing
runs with different identities on resumption. The dataset identity uses names,
sizes, mtimes, array headers and label counts, rather than a cryptographic hash
of all EEG values. A full validation must be repeated after source/data or
software changes. Concurrent full jobs using these output paths are blocked.

## Outputs and checkpoint behavior

Results: `/pscratch/sd/a/anis/shallowconvnet/data/mi_cmp_baseline/`.
For each pair, the unchanged script writes:

- `per_subject.csv`: 18 subjects x 4 windows = 72 rows.
- `per_fold.csv`: 18 subjects x 4 windows x 5 folds = 360 rows.
- `wide_subject_x_window.csv`: the 18 x 4 accuracy matrix/completion marker.
- `results.json` and `results.pkl`: pair matrix and run metadata.
- `_partial/S1.pkl` through `_partial/S18.pkl`: per-subject metrics,
  fold predictions/labels, and confusion matrices, saved when that subject finishes.

After all 28 pairs finish, it writes:

- `all_pairs_per_subject.csv`: 2,016 rows.
- `pair_x_window_mean.csv`: 28 x 4 mean accuracies across subjects.
- `subject_x_pair_4000ms.csv`: 18 x 28 full-window accuracy matrix.

Logs: `/pscratch/sd/a/anis/shallowconvnet/logs/scn-baseline-JOBID.out` and `.err`.
Each job also has `logs/JOBNAME-JOBID/` with a source copy, source commit/hash,
package freeze, module list, validation/readiness report and temporary files.
Caches and temporary I/O are redirected to scratch. No data/results/logs are
added to Git; the HOME clone also includes defensive Git exclusions.

Resubmit the same full job script after an interruption or time limit, after
reviewing its failure log and approving another submission. Completed pairs
are skipped. Completed subjects within an unfinished pair load their original
`_partial/S<ID>.pkl` results. The subject checkpoint write uses a temporary file
and atomic `os.replace`, protecting the prior checkpoint from interruption.
An in-progress subject is retrained in full: there is no mid-epoch, mid-fold,
optimizer-state, or trained-weight checkpoint. At most the current subject's
20 model fits (4 windows x 5 folds) need repeating.

Resume branch behavior was tested without fitting: an isolated synthetic
subject checkpoint was loaded through the original `run_pair`, then the
completed-pair skip branch was exercised. A live interrupted-training test
has not been run. The original CSV completion marker is not written atomically;
if interruption happens during pair-output writing, inspect those CSVs before
resubmitting. A truncated marker must not be treated as complete. Do not delete
checkpoints or result directories for a normal resume.

## Monitoring and runtime

Replace JOB_ID with the ID printed by sbatch:

```bash
squeue -u anis
squeue -j JOB_ID -o '%.18i %.9P %.20j %.8T %.12M %.12l %R'
tail -f /pscratch/sd/a/anis/shallowconvnet/logs/scn-baseline-JOB_ID.out
```

The original program logs after an entire subject/pair unit; it does not emit
per-epoch progress. There can be a substantial interval before the first
subject line. Ctrl-C exits `tail`, leaving the batch job running.

```bash
python3 /global/homes/a/anis/shallowconvnet/scripts/progress.py
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode,AllocTRES
```

The progress utility reads result/checkpoint names and subject timing lines.
For a fresh run, total compute time is approximately 504 x mean seconds per
completed subject/pair unit. Remaining time is approximately
(504 - completed units) x representative seconds per unit. Wait for several
subjects; the utility uses the median of the latest ten timing lines. Queue
waits, startup, output writing and redoing the interrupted subject add time.
For example, 120 seconds per unit would imply about 16.8 hours from scratch;
this is arithmetic only, not a performance claim. If the measured estimate
exceeds 48 hours, use subsequent approved resubmissions with the same paths.

After sbatch reports a job ID, Slurm owns the job. You can close SSH/VS Code,
shut your laptop, or disconnect your computer; the job and scratch files stay
on Perlmutter. Reconnect later to inspect the queue and logs. No tmux/nohup is
needed for a submitted batch job. A direct `bash jobs/shallow_shared.sbatch`
is not a submission and is intentionally blocked on login nodes.

Scratch persists across job exits and laptop disconnects, but is not backed up
and is subject to purge. Keep this source/launch setup in HOME; arrange an
approved CFS/HPSS or off-site copy for irreplaceable results. No shared-project
storage was changed by this preparation.

## Checks and sources

Shell syntax, Python syntax, NPZ headers/labels, original fold indices and
isolated checkpoint reuse passed on the login node with GPUs hidden. Full EEG
payload validation, CUDA execution, training accuracy and runtime remain pending
the approved compute-node validation/training jobs. No sbatch/salloc/srun job
has been submitted or allocated during preparation.

- [NERSC PyTorch modules](https://docs.nersc.gov/machinelearning/pytorch/)
- [NERSC shared QOS and limits](https://docs.nersc.gov/jobs/policy/)
- [Perlmutter GPU jobs and binding](https://docs.nersc.gov/systems/perlmutter/running-jobs/)
- [Iris GPU accounts](https://docs.nersc.gov/iris/iris-for-users/)

The metadata-only check is safe on login nodes and reads only headers/labels:

```bash
module load pytorch/2.8.0
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python /global/homes/a/anis/shallowconvnet/scripts/validate_perlmutter.py \
  --mode metadata \
  --report /pscratch/sd/a/anis/shallowconvnet/validation/inspection-NEW-NAME/metadata.json
```

Use a new report directory for each check because its resume fixture is created
exclusively, protecting any earlier inspection artifacts.
