#!/usr/bin/env bash
# Invoke manually ON the remote host. No SSH, Git operations, or installation.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
exec "${PYTHON:-python}" "$script_dir/run_analysis.py" "$@"
