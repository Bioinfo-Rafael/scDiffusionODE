#!/usr/bin/env bash
# Activate the existing conda environment, then launch all stages in background.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" ]]; then
  echo "conda not found. Initialize conda, or activate the environment and run scripts/run_all.py directly." >&2
  exit 1
fi
exec "$CONDA_BIN" run --no-capture-output -n "${TWOSTEP_CONDA_ENV:-scdiffusion}" \
  python -B "$SCRIPT_DIR/run_all.py" "$@"
