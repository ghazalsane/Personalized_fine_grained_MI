#!/bin/bash
#SBATCH --account=m4431_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm40g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=55G
#SBATCH --time=48:00:00
#SBATCH --job-name=scn-baseline
#SBATCH --chdir=/pscratch/sd/a/anis/shallowconvnet/code
#SBATCH --output=/pscratch/sd/a/anis/shallowconvnet/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/a/anis/shallowconvnet/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
source /global/homes/a/anis/shallowconvnet/scripts/job_common.sh

# Keep independent jobs from writing the same original checkpoint paths.
exec 9>"$WORK_ROOT/validation/experiment.lock"
flock -n 9 || { echo 'Another baseline job is using these results.' >&2; exit 1; }
test -f "$WORK_ROOT/validation/gpu_passed.json" || {
    echo 'Run validate_shared.sbatch successfully before the full experiment.' >&2
    exit 1
}
srun --unbuffered --cpu-bind=cores --gpus-per-task=1 \
    python -u "$PROJECT_ROOT/scripts/validate_perlmutter.py" \
    --mode ready --validated "$WORK_ROOT/validation/gpu_passed.json" \
    --report "$RUN_DIR/readiness.json"

# Execute the original file directly with its unchanged experiment settings.
srun --unbuffered --cpu-bind=cores --gpus-per-task=1 python -u "$SOURCE_SCRIPT"
