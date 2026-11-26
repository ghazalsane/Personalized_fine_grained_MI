# Personalized fine-grained motor imagery

Research implementations in run_shallow_all.py and run_personalized.py.

## Verified ShallowConvNet baseline results

The original [seed-42 baseline, job 61441374](results/shallowconvnet/job-61441374/README.md)
and its publication figures remain available unchanged.

[Array 62018478](results/shallowconvnet/array-62018478/README.md) adds training
seeds 100042, 200042, and 300042. All three tasks completed successfully.
The package includes every original CSV/JSON result, exact executed source,
provenance, and three-seed/four-seed aggregates.

Mean test accuracy (%) across all 18 subjects and 28 joint pairs:

| Seed | 800 ms | 1500 ms | 3000 ms | 4000 ms |
| --- | ---: | ---: | ---: | ---: |
| 42 | 57.92 | 58.69 | 59.50 | 58.71 |
| 100042 | 57.73 | 58.88 | 59.18 | 58.67 |
| 200042 | 57.87 | 58.58 | 59.40 | 59.05 |
| 300042 | 57.60 | 59.10 | 59.01 | 59.01 |
| **Four-seed mean** | **57.78** | **58.81** | **59.27** | **58.86** |

Every new run contains 2,016 subject/pair/window accuracy values and 10,080
fold results. All windows and all subjects are reported. Training seeds vary
initialization, training order, and augmentation; evaluation folds stay fixed
at 42 + subject_id. Seed variation is descriptive, not a significance test.

## Perlmutter setup and resumption

The validated environment is pytorch/2.8.0 (Python 3.12.11,
PyTorch 2.8.0+cu129, CUDA 12.9). Each array task used one A100 40 GB GPU,
32 logical CPUs, 55 GiB host memory, and an eight-hour limit in shared QOS
under m4431_g. All three completed in approximately 5 hours 47 minutes.

- [Perlmutter environment and original setup record](docs/perlmutter.md)
- [Seed configuration, paths, and validation](docs/seed-runs.md)
- [Explicit checkpoint resumption](docs/seed-resume.md)
- [Completed-run validation and prediction consistency](docs/seed-results-validation.json)

submit_seeds.sh maps its three task indices to the intended training seeds,
uses separate result directories, and refuses directory reuse in fresh mode.
Explicit MI_RESUME=1 requires a matching source/seed/data/environment identity,
reloads completed subject checkpoints, and skips completed pairs. An incomplete
subject repeats its four windows and five folds. Scheduler actions require
explicit approval; no training is started by the verification commands below.

The root training file now accepts MI_SEED; its default remains 42.
run_personalized.py is unchanged. The original seed-42 training file is
retained in the publication source archive and Git history.

## Verify the published results without training

With NumPy and pandas installed, run from the repository root:

~~~bash
python scripts/prepare_shallowconvnet_publication.py --verify-only
python scripts/prepare_seed_publication.py --verify-only
~~~

On Perlmutter, add source comparisons to check the copied metrics against
the unchanged scratch result directories:

~~~bash
python scripts/prepare_shallowconvnet_publication.py --verify-only \
  --source-results /pscratch/sd/a/anis/shallowconvnet/data/mi_cmp_baseline
python scripts/prepare_seed_publication.py --verify-only \
  --source-root /pscratch/sd/a/anis/shallowconvnet/data
~~~

The baseline verifier checks the archived historical source, so the seed
extension does not invalidate provenance for the original run. Synthetic
checkpoint-resumption tests are in scripts/check_seed_resume.py.

## Stored artifacts

The repository contains metric tables, result JSON, aggregate summaries,
source snapshots, and validation records. EEG arrays, binary checkpoint and
prediction pickles, caches, full environment freezes, and private backup
archives are excluded. Complete seed-run artifacts, including checkpoints,
also have a verified private HOME archive on Perlmutter. Original results
were neither modified nor rerun while publishing this package.

For plotting and regeneration of the original baseline figures, see
[the original publication commands](results/shallowconvnet/job-61441374/README.md#regeneration-and-verification).
