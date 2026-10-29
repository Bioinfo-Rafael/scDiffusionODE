#!/usr/bin/env bash
# Run from any directory. Environment setup is deliberately separate from execution.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
cd "$repo_root"
export PYTHON="${PYTHON:-python}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

if [[ "${1:-}" == --worker ]]; then
    shift
    job_dir=$1
    shift
    set +e
    "$PYTHON" -u "$script_dir/run_divergence_analysis.py" --output "$job_dir/analysis" "$@"
    result=$?
    printf '%s\n' "$result" > "$job_dir/exit_code"
    exit "$result"
fi

for arg in "$@"; do
    case "$arg" in
        --output|--output=*|--resume|--replot|--replot=*)
            echo 'For resume/replot/custom output, call run_divergence_analysis.py directly.' >&2
            exit 2 ;;
        -h|--help)
            exec "$PYTHON" "$script_dir/run_divergence_analysis.py" --help ;;
    esac
done
preflight=$("$PYTHON" "$script_dir/check_environment.py" "$@")
printf '%s\n' "$preflight"
if ! command -v "${JULIA:-julia}" >/dev/null; then
    echo 'Julia missing: set JULIA=/absolute/path/to/julia (tested 1.10.10).' >&2
    exit 2
fi
job_root=${DIVERGENCE_JOB_ROOT:-"$repo_root/work/20261028_Projection/outputs/divergence_analysis/jobs"}
mkdir -p "$job_root"
job_root=$(cd -- "$job_root" && pwd)
job_dir=$(mktemp -d "$job_root/job_$(date +%Y%m%d_%H%M%S)_XXXXXX")
git rev-parse HEAD > "$job_dir/source_commit"
git status --short -- work/20261028_Projection > "$job_dir/source_status"
printf '%q ' "$PYTHON" "$script_dir/run_divergence_analysis.py" --output "$job_dir/analysis" "$@" > "$job_dir/command.txt"
printf '\n' >> "$job_dir/command.txt"
printf '%s\n' "$preflight" > "$job_dir/environment.json"
job_pid=$("$PYTHON" - "$script_dir/run_divergence_background.sh" "$job_dir" "$@" <<'PY'
import subprocess
import sys
from pathlib import Path
script, directory, *args = sys.argv[1:]
with open(Path(directory)/'console.log', 'w') as log:
    child = subprocess.Popen(['bash', script, '--worker', directory, *args],
                             stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
print(child.pid)
PY
)
printf '%s\n' "$job_pid" > "$job_dir/launcher.pid"
printf 'PID=%s\nJOB_DIR=%s\nLOG=%s/console.log\nOUTPUT=%s/analysis\n' "$job_pid" "$job_dir" "$job_dir" "$job_dir"
