#!/usr/bin/env python3
"""Facet the 20260801 baselines and 20260802 norm-control runs together."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
BASE_ROOT = REPO_ROOT / "work" / "20260801"
SHARED_SCRIPT = REPO_ROOT / "work" / "20260707_lincomb" / "integrated_analysis" / "run_all_models_facets_0707.py"
BASE_EXPERIMENTS = (
    "hybrid_softmax_gate",
    "hybrid_ts_soft_softmax_gate",
    "hybrid_ts_soft_gentle_tau80_softmax_gate",
)
NORM_EXPERIMENTS = (
    "hybrid_softmax_gate_ratio_reg",
    "hybrid_softmax_gate_normed_learned_scale",
    "hybrid_softmax_gate_scale_model_x",
    "hybrid_softmax_gate_scale_model_ml_emb",
    "hybrid_ts_soft_softmax_gate_ratio_reg",
    "hybrid_ts_soft_softmax_gate_normed_learned_scale",
    "hybrid_ts_soft_softmax_gate_scale_model_x",
    "hybrid_ts_soft_softmax_gate_scale_model_ml_emb",
    "hybrid_ts_soft_gentle_tau80_softmax_gate_ratio_reg",
    "hybrid_ts_soft_gentle_tau80_softmax_gate_normed_learned_scale",
    "hybrid_ts_soft_gentle_tau80_softmax_gate_scale_model_x",
    "hybrid_ts_soft_gentle_tau80_softmax_gate_scale_model_ml_emb",
)
EXPERIMENTS = BASE_EXPERIMENTS + NORM_EXPERIMENTS


def _link_directory(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"refusing to replace non-matching combined-root entry: {link}")
    link.symlink_to(target, target_is_directory=True)


def main() -> None:
    dry_run = "--dry_run" in sys.argv[1:]
    combined_root = SUITE_ROOT / "integrated_analysis" / "combined_work_root"
    if not dry_run:
        for name in BASE_EXPERIMENTS:
            _link_directory(combined_root / "runs" / name, BASE_ROOT / "runs" / name)
            _link_directory(combined_root / "samples" / name, BASE_ROOT / "samples" / name)
        for name in NORM_EXPERIMENTS:
            _link_directory(combined_root / "runs" / name, SUITE_ROOT / "runs" / name)
            _link_directory(combined_root / "samples" / name, SUITE_ROOT / "samples" / name)
    command = [
        sys.executable, SHARED_SCRIPT,
        "--work_root", combined_root,
        "--output_root", SUITE_ROOT / "integrated_analysis" / "outputs",
        "--models", *EXPERIMENTS,
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.run(command, cwd=REPO_ROOT).returncode)


if __name__ == "__main__":
    main()
