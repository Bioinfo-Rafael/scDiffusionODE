#!/usr/bin/env python3
"""Run all requested 0707 experiment configs sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(WORK_ROOT / "configs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    args = parser.parse_args()
    names = (
        "hybrid_reverse_lincomb.json", "hybrid_ts_soft_lincomb.json",
        "lincomb_only_raw.json", "lincomb_softmax_gate.json",
        "lincomb_sparse_reg.json", "lincomb_entropy_reg.json",
    )
    for name in names:
        command = [sys.executable, HERE / "run_pipeline_0707.py", "--config", Path(args.config_dir) / name]
        if args.dry_run:
            command.append("--dry-run")
        if args.skip_sample:
            command.append("--skip-sample")
        if args.skip_viz:
            command.append("--skip-viz")
        print(" ".join(str(v) for v in command), flush=True)
        completed = subprocess.run(command)
        if completed.returncode:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

