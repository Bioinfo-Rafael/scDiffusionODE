#!/usr/bin/env bash
# Run on the remote host in the existing model environment.
# Preflight and analysis are both detached; analysis starts only if preflight succeeds.
set -euo pipefail
analysis_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
analysis_python="$(command -v "${PYTHON:-python}")"
for argument in "$@"; do
  case "$argument" in
    --output|--output=*|--check-only|--smoke)
      echo "run_background.sh manages --output/--check-only; use run.py for smoke tests." >&2
      exit 2
      ;;
  esac
done
stamp="$(TZ=Asia/Tokyo date +%Y%m%d_%H%M%S)"
job_dir="$analysis_dir/outputs/job_${stamp}_$$"
mkdir -p "$job_dir"
nohup bash -c '
  set -euo pipefail
  analysis_python=$1
  entry=$2
  job_dir=$3
  shift 3
  finish() {
    exit_status=$?
    printf "%s\n" "$exit_status" > "$job_dir/exit_code"
  }
  trap finish EXIT
  echo "=== Checkpoint/data preflight ==="
  "$analysis_python" -u "$entry" "$@" --check-only --output "$job_dir/check"
  echo "=== Preflight passed; starting full analysis ==="
  "$analysis_python" -u "$entry" "$@" --output "$job_dir/analysis"
' _ "$analysis_python" "$analysis_dir/run.py" "$job_dir" "$@" \
  > "$job_dir/console.log" 2>&1 < /dev/null &
analysis_pid=$!
echo "$analysis_pid" > "$job_dir/launcher.pid"
echo "Started preflight -> full analysis in background. Launcher PID: $analysis_pid"
echo "Job directory: $job_dir"
echo "Monitor: tail -f \"$job_dir/console.log\""
