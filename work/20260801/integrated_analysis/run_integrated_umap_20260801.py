#!/usr/bin/env python3
"""Run the shared integrated-UMAP tool for the three 20260801 conditions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
SHARED_SCRIPT = REPO_ROOT / "work" / "20260707_lincomb" / "integrated_analysis" / "run_all_models_facets_0707.py"
EXPERIMENTS = (
    "hybrid_softmax_gate",
    "hybrid_ts_soft_softmax_gate",
    "hybrid_ts_soft_gentle_tau80_softmax_gate",
)


def main() -> None:
    command = [
        sys.executable, SHARED_SCRIPT,
        "--work_root", SUITE_ROOT,
        "--output_root", SUITE_ROOT / "integrated_analysis" / "outputs",
        "--models", *EXPERIMENTS,
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command, cwd=REPO_ROOT).returncode)


if __name__ == "__main__":
    main()
