#!/usr/bin/env python3
"""Run all requested 0707 experiment configs sequentially."""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
CONFIG_NAMES = (
    "hybrid_reverse_lincomb.json",
    "lincomb_only_raw.json",
    "lincomb_softmax_gate.json",
    "lincomb_sparse_reg.json",
    "lincomb_entropy_reg.json",
    "hybrid_ts_soft_lincomb.json",
)


def _normalize_only_name(value):
    name = str(value).strip()
    if Path(name).name != name:
        raise ValueError(f"--only accepts canonical config names, not paths: {value}")
    if not name.endswith(".json"):
        name = f"{name}.json"
    return name


def _select_names(only_values):
    if not only_values:
        return CONFIG_NAMES
    valid = set(CONFIG_NAMES)
    wanted = []
    unknown = []
    for value in only_values:
        name = _normalize_only_name(value)
        if name not in valid:
            unknown.append(name)
        elif name not in wanted:
            wanted.append(name)
    if unknown:
        raise ValueError(f"unknown config(s) for --only: {', '.join(unknown)}")
    wanted = set(wanted)
    return tuple(name for name in CONFIG_NAMES if name in wanted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default=str(WORK_ROOT / "configs"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--only", nargs="+", default=(), metavar="CONFIG",
                        help="run only selected canonical configs (with or without .json)")
    args = parser.parse_args()
    try:
        names = _select_names(args.only)
    except ValueError as exc:
        parser.error(str(exc))
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
