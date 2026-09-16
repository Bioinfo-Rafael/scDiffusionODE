#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$ROOT/work/20260916_x0predict_hybrid_additive/.mplconfig"
exec "${PYTHON:-python}" -m work.20260916_x0predict_hybrid_additive.scripts.run_all "$@"
