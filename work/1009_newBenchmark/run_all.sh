#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export STAGE1_DEVICE="${STAGE1_DEVICE:-auto}"
# Arguments shared by every stage: --config /absolute/path/to/config.json
"$PYTHON_BIN" "$HERE/prepare_data.py" "$@"
# Resume only incomplete training; reuse a completed checkpoint.
"$PYTHON_BIN" - "$HERE" "$@" <<'PY'
import os
import sys
sys.path.insert(0, sys.argv[1])
from common import config, parser, paths, resolve_checkpoint, load_checkpoint
from train import train
p=parser('pipeline'); args=p.parse_args(sys.argv[2:]); c=config(args.config)
if (paths(c)/'checkpoints/latest.json').exists():
    _, ck, _, checkpoint = load_checkpoint(c)
    if ck['step'] < c['training']['total_steps']:
        train(c, target_device=os.environ['STAGE1_DEVICE'], resume=checkpoint)
else:
    train(c, target_device=os.environ['STAGE1_DEVICE'])
PY
"$PYTHON_BIN" "$HERE/sample.py" "$@" --device "$STAGE1_DEVICE"
"$PYTHON_BIN" "$HERE/visualize.py" "$@" --scope samples
"$PYTHON_BIN" "$HERE/export_velocity.py" "$@" --device "$STAGE1_DEVICE"
"$PYTHON_BIN" "$HERE/visualize.py" "$@" --scope all
"$PYTHON_BIN" "$HERE/visualize.py" "$@" --scope erythroid
# Exit 2 is intentional scientific inapplicability, never successful benchmarking.
"$PYTHON_BIN" "$HERE/evaluate.py" "$@"
