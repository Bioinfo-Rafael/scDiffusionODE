#!/bin/bash
# Thin wrapper to launch the 17-experiment matrix via the Python launcher (20260609).
# 引数はそのまま launcher に渡す（--dry-run / --only / --skip-existing / --gpu N）。
#
#   bash run_all_5x3.sh --dry-run
#   bash run_all_5x3.sh --gpu 0
#   bash run_all_5x3.sh --only lowrank__ratio_reg,baseline_cellunet

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="/home/suzuki/.conda/envs/scdiffusion/bin:$PATH"

echo "=========================================="
echo "Hybrid 5x3 + baseline launcher"
echo "  dir: ${DIR}"
echo "  args: $*"
echo "=========================================="

python "${DIR}/run_experiments_5x3.py" "$@"
