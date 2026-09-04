#!/usr/bin/env python3
"""Run one or all read-only post-hoc analyses for a completed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from analysis.diffusion_diagnostics import analyze_diffusion_diagnostics  # noqa: E402
from analysis.hematopoietic import (  # noqa: E402
    analyze_drift_velocity,
    analyze_generated_umap,
)
from analysis.loss_analysis import analyze_loss  # noqa: E402
from analysis.parameter_evolution import analyze_parameter_evolution  # noqa: E402
from common import requested_config, validate_run_directory  # noqa: E402


STAGES = ("loss", "parameters", "diagnostics", "velocity", "umap")


def analyze_run(args: argparse.Namespace) -> dict:
    run = validate_run_directory(args.run_dir)
    config = requested_config(run)
    requested = STAGES if args.stage == "all" else (args.stage,)
    results = {}
    for stage in requested:
        if stage == "loss":
            result = analyze_loss(
                run,
                rolling_window=int(
                    args.rolling_window
                    if args.rolling_window is not None
                    else config.get("loss_rolling_window", 100)
                ),
                force=args.force,
            )
        elif stage == "parameters":
            result = analyze_parameter_evolution(
                run, max_points=args.parameter_points, force=args.force
            )
        elif stage == "diagnostics":
            result = analyze_diffusion_diagnostics(
                run,
                step=int(
                    args.timestep_step
                    if args.timestep_step is not None
                    else config.get("analysis_timestep_step", 20)
                ),
                max_cells=int(
                    args.max_cells
                    if args.max_cells is not None
                    else config.get("analysis_max_cells", 2000)
                ),
                batch_size=int(
                    args.batch_size
                    if args.batch_size is not None
                    else config.get("analysis_batch_size", 128)
                ),
                seed=int(args.seed if args.seed is not None else config.get("seed", 1234)),
                device=args.device,
                ema_rate=args.ema_rate,
                force=args.force,
            )
        elif stage == "velocity":
            result = analyze_drift_velocity(
                run,
                device=args.device,
                ema_rate=args.ema_rate,
                max_cells=int(
                    args.hematopoietic_max_cells
                    if args.hematopoietic_max_cells is not None
                    else config.get("hematopoietic_max_cells", 0)
                ),
                seed=int(args.seed if args.seed is not None else config.get("seed", 1234)),
                n_jobs=int(
                    args.n_jobs
                    if args.n_jobs is not None
                    else config.get("velocity_n_jobs", 32)
                ),
                force=args.force,
            )
        elif stage == "umap":
            result = analyze_generated_umap(
                run,
                device=args.device,
                ema_rate=args.ema_rate,
                max_cells=int(
                    args.hematopoietic_max_cells
                    if args.hematopoietic_max_cells is not None
                    else config.get("hematopoietic_max_cells", 0)
                ),
                seed=int(args.seed if args.seed is not None else config.get("seed", 1234)),
                force=args.force,
            )
        else:
            raise AssertionError(stage)
        results[stage] = result
    return {"status": "completed", "run_dir": str(run), "stages": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--ema-rate", default="")
    parser.add_argument("--rolling-window", type=int, default=None)
    parser.add_argument("--parameter-points", type=int, default=8)
    parser.add_argument("--timestep-step", type=int, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--hematopoietic-max-cells", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = analyze_run(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
