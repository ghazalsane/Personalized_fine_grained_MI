#!/bin/bash
# Sourced by batch scripts; guard before any GPU operation.
if [[ -z ${SLURM_JOB_ID:-} || $(hostname -s) != nid* ]]; then
    echo 'This launcher must run on a Perlmutter compute node inside Slurm.' >&2
    exit 1
fi
export PROJECT_ROOT=/global/homes/a/anis/shallowconvnet
export WORK_ROOT=/pscratch/sd/a/anis/shallowconvnet
export SOURCE_SCRIPT="$WORK_ROOT/code/run_shallow_all.py"
export MI_DRIVE_ROOT="$WORK_ROOT/data"
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export SLURM_CPU_BIND=cores
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16
module load pytorch/2.8.0
export RUN_DIR="$WORK_ROOT/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR" "$WORK_ROOT/validation" "$WORK_ROOT/cache"
export XDG_CACHE_HOME="$WORK_ROOT/cache"
export TORCH_HOME="$WORK_ROOT/cache/torch"
export CUDA_CACHE_PATH="$WORK_ROOT/cache/cuda"
export TMPDIR="$RUN_DIR/tmp"
mkdir -p "$TMPDIR"
date -u '+UTC start: %Y-%m-%dT%H:%M:%SZ'
printf 'job=%s host=%s source=%s\n' "$SLURM_JOB_ID" "$(hostname -s)" "$SOURCE_SCRIPT"
printf 'MI_DRIVE_ROOT=%s\nresults=%s/mi_cmp_baseline\n' "$MI_DRIVE_ROOT" "$MI_DRIVE_ROOT"
module -t list 2>"$RUN_DIR/modules.txt"
python -m pip freeze --all >"$RUN_DIR/packages.txt"
git -C "$WORK_ROOT/code" rev-parse HEAD >"$RUN_DIR/source_commit.txt"
git -C "$WORK_ROOT/code" status --porcelain=v1 >"$RUN_DIR/source_status.txt"
sha256sum "$SOURCE_SCRIPT" >"$RUN_DIR/source_sha256.txt"
cp "$SOURCE_SCRIPT" "$RUN_DIR/run_shallow_all.py"
