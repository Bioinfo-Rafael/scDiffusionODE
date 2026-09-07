#!/bin/bash
# Launcher for hybrid_norm_mode=ratio_reg (20260609). 数値パラメタは train_hybridnorm_20260609.sh に集約。
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/train_hybridnorm_20260609.sh" --hybrid_norm_mode ratio_reg "$@"
