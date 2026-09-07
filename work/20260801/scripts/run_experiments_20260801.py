#!/usr/bin/env python3
"""Train and analyze all three 20260801 Hybrid + Softmax conditions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
CONFIG_NAMES = (
    "hybrid_softmax_gate.json",
    "hybrid_ts_soft_softmax_gate.json",
    "hybrid_ts_soft_gentle_tau80_softmax_gate.json",
)


def run_stage(label: str, command: list[object]) -> None:
    print(f"\n### {label}\n{' '.join(str(value) for value in command)}", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-integrated-umap", action="store_true")
    args, extra = parser.parse_known_args()
    for config_name in CONFIG_NAMES:
        command = [sys.executable, HERE / "run_pipeline_20260801.py", "--config", SUITE_ROOT / "configs" / config_name]
        if args.dry_run:
            command.append("--dry-run")
        if args.skip_existing:
            command.append("--skip-existing")
        command.extend(extra)
        run_stage(config_name, command)
    if args.dry_run:
        return
    run_stage("comparison summary", [sys.executable, HERE / "summarize_results_20260801.py"])
    if not args.skip_integrated_umap:
        run_stage("integrated UMAP", [sys.executable, SUITE_ROOT / "integrated_analysis" / "run_integrated_umap_20260801.py"])


if __name__ == "__main__":
    main()
