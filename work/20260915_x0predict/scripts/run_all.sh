#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$ROOT/work/20260915_x0predict/.mplconfig"
cd "$ROOT"
exec "${PYTHON:-python}" -m work.20260915_x0predict.scripts.run_all "$@"
