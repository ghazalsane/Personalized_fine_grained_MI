#!/bin/bash
# Three seed runs; MI_RESUME=1 explicitly allows verified checkpoint reuse.
# Submit with: sbatch /global/homes/a/anis/shallowconvnet/submit_seeds.sh
#SBATCH --job-name=shallow_seeds
#SBATCH --account=m4431_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm40g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=55G
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --chdir=/global/homes/a/anis/shallowconvnet
#SBATCH --output=/pscratch/sd/a/anis/shallowconvnet/logs/shallow_seed_%A_%a.out
#SBATCH --error=/pscratch/sd/a/anis/shallowconvnet/logs/shallow_seed_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} || -z ${SLURM_ARRAY_JOB_ID:-} || $(hostname -s) != nid* ]]; then
    echo 'This launcher requires a Slurm array on a Perlmutter compute node.' >&2
    exit 1
fi
case ${SLURM_ARRAY_TASK_ID:-} in
    0|1|2) ;;
    *) echo 'Expected array task 0, 1, or 2.' >&2; exit 1 ;;
esac

SEEDS=(100042 200042 300042)
export MI_SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
export PYTHONHASHSEED="$MI_SEED"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONOPTIMIZE
export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

export PROJECT_ROOT=/global/homes/a/anis/shallowconvnet
export WORK_ROOT=/pscratch/sd/a/anis/shallowconvnet
export MI_DRIVE_ROOT="$WORK_ROOT/data"
export RESULTS_ROOT="$MI_DRIVE_ROOT/mi_cmp_baseline_seed${MI_SEED}"
export RUN_DIR="$WORK_ROOT/logs/shallow_seed_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# Lock only this seed. The historical baseline launcher/helper is not invoked.
exec 9>"$WORK_ROOT/validation/seed-${MI_SEED}.lock"
flock -n 9 || { echo "Another job is using seed $MI_SEED." >&2; exit 1; }
export MI_RESUME="${MI_RESUME:-0}"
case $MI_RESUME in
    0)
        if [[ -e $RESULTS_ROOT || -L $RESULTS_ROOT ]]; then
            echo "Fresh run refuses existing results: $RESULTS_ROOT" >&2
            exit 1
        fi ;;
    1)
        if [[ ! -d $RESULTS_ROOT || -L $RESULTS_ROOT ]]; then
            echo "Resume requires an existing seed directory: $RESULTS_ROOT" >&2
            exit 1
        fi ;;
    *) echo 'MI_RESUME must be 0 or 1.' >&2; exit 1 ;;
esac
mkdir -p "$RUN_DIR/tmp" "$RUN_DIR/cache"
export TMPDIR="$RUN_DIR/tmp"
export XDG_CACHE_HOME="$RUN_DIR/cache"
export TORCH_HOME="$RUN_DIR/cache/torch"
export CUDA_CACHE_PATH="$RUN_DIR/cache/cuda"

module load pytorch/2.8.0
module -t list 2>"$RUN_DIR/modules.txt"
python -m pip freeze --all >"$RUN_DIR/packages.txt"
git -C "$PROJECT_ROOT" rev-parse HEAD >"$RUN_DIR/source_commit.txt"
git -C "$PROJECT_ROOT" status --porcelain=v1 >"$RUN_DIR/source_status.txt"
cp "$PROJECT_ROOT/run_shallow_all.py" "$RUN_DIR/run_shallow_all.py"
cp "$PROJECT_ROOT/submit_seeds.sh" "$RUN_DIR/submit_seeds.sh"
cp "$PROJECT_ROOT/scripts/seed_resume.py" "$RUN_DIR/seed_resume.py"
sha256sum "$RUN_DIR/run_shallow_all.py" >"$RUN_DIR/source_sha256.txt"

# Read-only checks against job 61441374; never call its baseline-writing ready mode.
srun --unbuffered --cpu-bind=cores --gpus-per-task=1 python -u - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import torch

work = Path(os.environ['WORK_ROOT'])
run = Path(os.environ['RUN_DIR'])
old = json.loads((work / 'logs/scn-baseline-61441374/readiness.json').read_text())
assert old['passed'] and old['mode'] == 'ready'
identity = old['identity']
assert platform.python_version() == identity['python'], 'Python version changed'
for package, version in identity['packages'].items():
    assert importlib.metadata.version(package) == version, (package, version)
assert torch.version.cuda == identity['torch_cuda']
assert torch.backends.cudnn.version() == identity['cudnn']
assert torch.cuda.is_available(), 'CUDA is required; CPU fallback is forbidden'
assert torch.cuda.device_count() == 1, 'Exactly one visible GPU is required'
assert torch.cuda.get_device_name(0) == identity['gpu'], 'GPU model changed'
assert int(os.environ['SLURM_CPUS_PER_TASK']) == 32
assert int(os.environ['SLURM_MEM_PER_NODE']) == 55 * 1024
seed = int(os.environ['MI_SEED'])
assert seed in (100042, 200042, 300042)
assert os.environ['PYTHONHASHSEED'] == str(seed)
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
data = Path(os.environ['MI_DRIVE_ROOT']) / 'datasetpaper_4_40hz_MIonly'
for record in old['subjects']:
    candidates = list(data.glob(f"subject{record['subject']}_eeg_epochs_*.npz"))
    assert len(candidates) == 1 and candidates[0].name == record['name']
    stat = candidates[0].stat()
    assert (stat.st_size, stat.st_mtime_ns) == (record['size'], record['mtime_ns'])
result = Path(os.environ['RESULTS_ROOT'])
assert result.resolve() == work / f'data/mi_cmp_baseline_seed{seed}'
assert not result.is_symlink()
source = run / 'run_shallow_all.py'
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == '731535d4179164cf46744d638ee052622b72b1e7a529b1aa2dda4bc62f3a2af8', 'Training source changed since preflight'
import sys
sys.path.insert(0, str(run))
from seed_resume import prepare_results
seed_identity = dict(schema_version=1, seed=seed, source_sha256=digest,
    dataset_fingerprint=identity['dataset_fingerprint'],
    environment={key: identity[key] for key in ('python', 'packages', 'torch_cuda', 'cudnn', 'gpu')})
resume_state = prepare_results(result, seed_identity, resume=os.environ['MI_RESUME'] == '1')
(run / 'readiness.json').write_text(json.dumps(dict(
    passed=True, seed=seed, results_root=str(result), source_sha256=digest,
    reference_environment=identity, reference_job=61441374, resume_state=resume_state), indent=2) + '\n')
print(f'PASS: seed={seed}, gpu={torch.cuda.get_device_name(0)}, results={result}')
PY

srun --unbuffered --cpu-bind=cores --gpus-per-task=1 \
    python -u "$RUN_DIR/run_shallow_all.py"
python "$RUN_DIR/seed_resume.py" "$RESULTS_ROOT" "$RUN_DIR/readiness.json"
