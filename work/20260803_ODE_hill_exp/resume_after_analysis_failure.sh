#!/usr/bin/env bash
set -euo pipefail

SUITE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SUITE_DIR}/../.." && pwd)"
BATCH_ID="${1:-20260803_full_30000}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${SUITE_DIR}/batches/${BATCH_ID}"
LOG_PATH="${LOG_DIR}/resume_after_analysis_failure.log"

COMMAND=(
  "${PYTHON_BIN}"
  "${SUITE_DIR}/scripts/resume_after_analysis_failure.py"
  --batch-id "${BATCH_ID}"
  --failed-experiment standard_hybrid_lincomb__exp
  --remaining-experiment
  ts_soft_tau80_hybrid_lincomb__hill_after_linear
  ts_soft_tau80_hybrid_lincomb__racipe
  ts_soft_tau80_hybrid_lincomb__exp
  standard_hybrid_single__hill_after_linear
  standard_hybrid_single__racipe
  standard_hybrid_single__exp
)

if [[ "${RUN_MODE:-background}" == "foreground" ]]; then
  cd "${REPO_DIR}"
  exec "${COMMAND[@]}"
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_DIR}"
nohup "${COMMAND[@]}" >"${LOG_PATH}" 2>&1 < /dev/null &
PID=$!
printf 'RECOVERY_PID=%s\n' "${PID}"
printf "RECOVERY_LOG='%s'\n" "${LOG_PATH}"
