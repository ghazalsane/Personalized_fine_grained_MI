#!/bin/bash
#SBATCH --account=m4431_g
#SBATCH --qos=shared
#SBATCH --constraint=gpu&hbm40g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=55G
#SBATCH --time=00:10:00
#SBATCH --job-name=scn-validate
#SBATCH --chdir=/pscratch/sd/a/anis/shallowconvnet/code
#SBATCH --output=/pscratch/sd/a/anis/shallowconvnet/logs/%x-%j.out
#SBATCH --error=/pscratch/sd/a/anis/shallowconvnet/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail
source /global/homes/a/anis/shallowconvnet/scripts/job_common.sh

exec 8>"$WORK_ROOT/validation/validation.lock"
flock -n 8 || { echo 'Another validation job is running.' >&2; exit 1; }
srun --unbuffered --cpu-bind=cores --gpus-per-task=1 \
    python -u "$PROJECT_ROOT/scripts/validate_perlmutter.py" \
    --mode gpu --report "$RUN_DIR/validation.json"

# Only a completely successful GPU validation creates this marker.
cp "$RUN_DIR/validation.json" "$WORK_ROOT/validation/gpu_passed.json.tmp"
mv "$WORK_ROOT/validation/gpu_passed.json.tmp" "$WORK_ROOT/validation/gpu_passed.json"
echo "Validation passed: $WORK_ROOT/validation/gpu_passed.json"
