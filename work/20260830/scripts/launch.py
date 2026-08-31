#!/usr/bin/env python3
"""Canonical ordered launcher for all twelve 20260830 conditions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import EXPERIMENT_ORDER, new_batch_id, run_dir, safe_component  # noqa: E402


def commands(experiment, batch_id, args):
    target = run_dir(experiment, batch_id)
    config = SUITE_ROOT / "configs" / "experiments" / f"{experiment}.json"
    python = sys.executable
    if args.smoke:
        return [[python, str(SUITE_ROOT / "tests" / "smoke.py")]]
    if args.sample_only:
        return [[python, str(HERE / "sample.py"), "--run-dir", str(target)]]
    if args.analysis_only:
        analysis = [python, str(HERE / "analyze.py"), "--run-dir", str(target)]
        if args.analysis_full:
            analysis.append("--full")
        return [analysis]
    train = [python, str(HERE / "train.py"), "--config", str(config), "--run-dir", str(target)]
    if args.resume:
        train.extend(["--resume", args.resume])
    analysis = [python, str(HERE / "analyze.py"), "--run-dir", str(target)]
    if args.analysis_full:
        analysis.append("--full")
    return [
        train,
        [python, str(HERE / "sample.py"), "--run-dir", str(target)],
        analysis,
    ]


def summary_command(batch_id):
    return [
        sys.executable,
        str(HERE / "analyze.py"),
        "--all-runs",
        "--batch-id",
        batch_id,
        "--summary-only",
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", choices=EXPERIMENT_ORDER)
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--resume", nargs="?", const="auto", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample-only", action="store_true")
    mode.add_argument("--analysis-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--analysis-full",
        action="store_true",
        help="run per-condition analysis over all diffusion timesteps",
    )
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.analysis_full and (args.sample_only or args.smoke):
        parser.error("--analysis-full requires the normal pipeline or --analysis-only")
    selected = tuple(args.experiment or EXPERIMENT_ORDER)
    if args.resume not in ("", "auto") and len(selected) != 1:
        raise ValueError("an explicit resume checkpoint requires exactly one experiment")
    batch_id = safe_component(args.batch_id or new_batch_id(), "batch id")
    plan = [
        {"experiment": experiment, "run_dir": str(run_dir(experiment, batch_id)), "commands": command_list}
        for experiment in selected
        for command_list in [commands(experiment, batch_id, args)]
    ]
    run_summary = (
        selected == EXPERIMENT_ORDER and not args.sample_only and not args.smoke
    )
    summary = summary_command(batch_id) if run_summary else None
    if args.dry_run:
        print(json.dumps({"batch_id": batch_id, "runs": plan, "summary": summary}, indent=2))
        return 0
    if args.background:
        log_dir = SUITE_ROOT / "runs" / "_launcher_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{batch_id}.log"
        child = [value for value in sys.argv if value != "--background"]
        if "--batch-id" not in child:
            child.extend(["--batch-id", batch_id])
        with log.open("ab", buffering=0) as handle:
            process = subprocess.Popen(
                child,
                cwd=SUITE_ROOT.parent.parent,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=os.environ.copy(),
            )
        print(f"BACKGROUND_PID={process.pid}\nBACKGROUND_LOG='{log}'")
        return 0
    if args.smoke:
        subprocess.run(commands(selected[0], batch_id, args)[0], cwd=SUITE_ROOT.parent.parent, check=True)
        return 0
    for index, item in enumerate(plan, 1):
        print(f"[{index}/{len(plan)}] {item['experiment']}", flush=True)
        for command in item["commands"]:
            subprocess.run(command, cwd=SUITE_ROOT.parent.parent, check=True)
    if summary is not None:
        print("[summary] all 12 conditions", flush=True)
        subprocess.run(summary, cwd=SUITE_ROOT.parent.parent, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
