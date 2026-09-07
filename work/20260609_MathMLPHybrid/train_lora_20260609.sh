#!/bin/bash
# Launcher for model_type=lora (20260609). 数値パラメタは train_mathmlp_20260609.sh に集約。
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DIR}/train_mathmlp_20260609.sh" --model_type lora "$@"
