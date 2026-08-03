#!/usr/bin/env python3
"""Ordered launcher for the twelve isolated 20260803 experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (SUITE_ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from common import (  # noqa: E402
    BATCHES_ROOT,
    CONFIG_ROOT,
    apply_set_overrides,
    command_string,
    generate_batch_id,
    load_experiment_config,
    now_iso,
    run_dir_for,
    select_experiments,
    validate_path_component,
    write_json,
)


def _flatten(values: Sequence[Sequence[str]] | None) -> list[str]:
    return [item for group in (values or ()) for item in group]


def _stage_commands(
    experiment: str,
    batch_id: str,
    args: argparse.Namespace,
) -> list[tuple[str, list[str]]]:
    run_dir = run_dir_for(experiment, batch_id)
    config = CONFIG_ROOT / f"{experiment}.json"
    python = sys.executable
    if args.smoke:
        return [("smoke", [
            python, str(SUITE_ROOT / "tests" / "smoke.py"),
            "--experiment", experiment,
            "--batch-id", batch_id,
        ])]
    if args.sample_only:
        return [("sample", [python, str(HERE / "sample.py"), "--run-dir", str(run_dir)])]
    if args.analysis_only:
        return [("analysis", [
            python, str(HERE / "analyze.py"), "--run-dir", str(run_dir),
        ])]

    train = [
        python, str(HERE / "train.py"),
        "--config", str(config),
        "--run-dir", str(run_dir),
    ]
    if args.resume:
        train.extend(["--resume", args.resume])
    for value in args.set_values:
        train.extend(["--set", value])
    sample = [python, str(HERE / "sample.py"), "--run-dir", str(run_dir)]
    resolved = apply_set_overrides(load_experiment_config(config), args.set_values)
    analysis = [
        python, str(HERE / "analyze.py"), "--run-dir", str(run_dir),
        "--max-cells", str(resolved.get("analysis_max_cells", 2000)),
        "--batch-size", str(resolved.get("analysis_batch_size", 128)),
        "--t-values", str(resolved.get("analysis_t_values", "")),
        "--device", str(resolved.get("device", "auto")),
    ]
    return [("train", train), ("sample", sample), ("analysis", analysis)]


def _print_plan(experiments: Sequence[str], batch_id: str, args: argparse.Namespace) -> None:
    payload = {
        "dry_run": True,
        "batch_id": batch_id,
        "experiments": list(experiments),
        "runs": [
            {
                "experiment": experiment,
                "run_dir": str(run_dir_for(experiment, batch_id)),
                "commands": [
                    {"stage": stage, "argv": command_string(command)}
                    for stage, command in _stage_commands(experiment, batch_id, args)
                ],
            }
            for experiment in experiments
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _background(args: argparse.Namespace, batch_id: str) -> int:
    batch_dir = BATCHES_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    log_path = batch_dir / "launcher.log"
    child_args = [value for value in sys.argv[1:] if value != "--background"]
    if "--batch-id" not in child_args:
        child_args.extend(["--batch-id", batch_id])
    command = [sys.executable, str(Path(__file__).resolve()), *child_args]
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )
    write_json(batch_dir / "background.json", {
        "status": "started",
        "started_at": now_iso(),
        "pid": process.pid,
        "command": command_string(command),
        "log_path": str(log_path),
    })
    print(f"BACKGROUND_PID={process.pid}")
    print(f"BACKGROUND_LOG='{log_path}'")
    return 0


def _run(args: argparse.Namespace, experiments: Sequence[str], batch_id: str) -> int:
    batch_dir = BATCHES_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    status_path = batch_dir / ("smoke_batch.json" if args.smoke else "batch.json")
    status: dict[str, Any] = {
        "status": "running",
        "started_at": now_iso(),
        "batch_id": batch_id,
        "experiments": list(experiments),
        "command": command_string(),
        "results": [],
    }
    write_json(status_path, status)
    try:
        if args.smoke:
            unit_command = [
                sys.executable, "-m", "unittest", "-v",
                str(SUITE_ROOT / "tests" / "test_models.py"),
            ]
            print(f"[unit] {command_string(unit_command)}", flush=True)
            subprocess.run(unit_command, cwd=REPO_ROOT, check=True)

        for index, experiment in enumerate(experiments, start=1):
            result: dict[str, Any] = {
                "experiment": experiment,
                "run_dir": str(run_dir_for(experiment, batch_id)),
                "status": "running",
                "started_at": now_iso(),
                "stages": [],
            }
            status["results"].append(result)
            write_json(status_path, status)
            print(f"[{index}/{len(experiments)}] {experiment}", flush=True)
            for stage, command in _stage_commands(experiment, batch_id, args):
                print(f"[{stage}] {command_string(command)}", flush=True)
                started = now_iso()
                subprocess.run(command, cwd=REPO_ROOT, check=True)
                result["stages"].append({
                    "stage": stage,
                    "status": "completed",
                    "started_at": started,
                    "finished_at": now_iso(),
                    "command": command_string(command),
                })
                write_json(status_path, status)
            result["status"] = "completed"
            result["finished_at"] = now_iso()
            write_json(status_path, status)
        status["status"] = "completed"
        status["finished_at"] = now_iso()
        write_json(status_path, status)
        print(f"BATCH_STATUS='{status_path}'")
        return 0
    except BaseException as exc:
        if status.get("results"):
            status["results"][-1]["status"] = "failed"
            status["results"][-1]["failed_at"] = now_iso()
            status["results"][-1]["error"] = f"{type(exc).__name__}: {exc}"
        status.update({
            "status": "failed",
            "failed_at": now_iso(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        write_json(status_path, status)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--family", action="append", nargs="+", default=[])
    parser.add_argument("--experiment", action="append", nargs="+", default=[])
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--resume", nargs="?", const="auto", default="")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--sample-only", action="store_true")
    mode.add_argument("--analysis-only", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--set", dest="set_values", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    families = _flatten(args.family)
    requested = _flatten(args.experiment)
    experiments = select_experiments(families, requested)
    batch_id = validate_path_component(args.batch_id or generate_batch_id(), "batch id")
    if args.resume and (args.sample_only or args.analysis_only or args.smoke):
        raise ValueError("--resume is valid only for the training pipeline")
    if args.resume not in ("", "auto") and len(experiments) != 1:
        raise ValueError("an explicit checkpoint path requires exactly one experiment")
    if args.set_values and (args.sample_only or args.analysis_only or args.smoke):
        raise ValueError("--set is valid only for the training pipeline")
    if args.dry_run:
        _print_plan(experiments, batch_id, args)
        return 0
    if args.background:
        return _background(args, batch_id)
    return _run(args, experiments, batch_id)


if __name__ == "__main__":
    raise SystemExit(main())
