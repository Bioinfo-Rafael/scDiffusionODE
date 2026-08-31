#!/usr/bin/env python3
"""Detailed post-hoc analysis CLI for one run or all twelve conditions."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.runner import (  # noqa: E402
    AnalysisOptions,
    analyze_run,
    create_diffusion,
    discover_run_directories,
    parse_timestep_spec,
    summarize_runs,
)
from scripts.common import read_json  # noqa: E402


def _parse_for_runs(spec, runs):
    if not spec:
        return None
    config = read_json(Path(runs[0]) / "exp_config.json")
    return parse_timestep_spec(spec, create_diffusion(config).num_timesteps)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", action="append", default=[])
    parser.add_argument("--all-runs", action="store_true")
    parser.add_argument("--batch-id", default="")
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument("--quick", action="store_true")
    preset.add_argument("--full", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary-only", action="store_true")
    mode.add_argument("--gradient-only", action="store_true")
    parser.add_argument("--cells", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--timesteps", default="")
    parser.add_argument("--gradient-timesteps", default="")
    parser.add_argument("--gradient-cells", type=int)
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summary-output", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    require_all = bool(args.all_runs or args.summary_only)
    runs = discover_run_directories(
        run_dirs=args.run_dir,
        batch_id=args.batch_id,
        require_all=require_all,
    )
    if not runs:
        raise FileNotFoundError("no 20260830 run directories discovered")
    preset = "full" if args.full else "quick" if args.quick else "default"
    options = AnalysisOptions(
        preset=preset,
        cells=args.cells,
        batch_size=args.batch_size,
        timesteps=_parse_for_runs(args.timesteps, runs),
        gradient_timesteps=_parse_for_runs(args.gradient_timesteps, runs),
        gradient_cells=args.gradient_cells,
        rolling_window=args.rolling_window,
        seed=args.seed,
        device=args.device,
        force=args.force,
        gradient_only=args.gradient_only,
    )
    results = []
    if not args.summary_only:
        for index, run in enumerate(runs, 1):
            print(f"[{index}/{len(runs)}] analyze {run}", flush=True)
            results.append(analyze_run(run, options))
    if args.all_runs or args.summary_only:
        output = (
            Path(args.summary_output).expanduser().resolve()
            if args.summary_output
            else SUITE_ROOT / "analysis_results" / (
                args.batch_id or datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        )
        summary = summarize_runs(runs, output, require_all=True)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(results[0], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
