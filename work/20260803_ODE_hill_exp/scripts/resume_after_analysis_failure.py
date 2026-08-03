#!/usr/bin/env python3
"""Recover one failed analysis, then run only the explicitly remaining conditions.

This orchestrator is intentionally state-aware: completed training, sampling,
and analysis stages are skipped only when their recorded artifacts still exist.
An interrupted training stage resumes from a complete raw/optimizer/EMA bundle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (SUITE_ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from common import (  # noqa: E402
    BATCHES_ROOT,
    CONFIG_ROOT,
    command_string,
    latest_checkpoint_bundle,
    load_experiment_config,
    now_iso,
    read_json,
    run_dir_for,
    select_experiments,
    validate_path_component,
    write_json,
)


FAILED_DEFAULT = "standard_hybrid_lincomb__exp"
REMAINING_DEFAULT = (
    "ts_soft_tau80_hybrid_lincomb__hill_after_linear",
    "ts_soft_tau80_hybrid_lincomb__racipe",
    "ts_soft_tau80_hybrid_lincomb__exp",
    "standard_hybrid_single__hill_after_linear",
    "standard_hybrid_single__racipe",
    "standard_hybrid_single__exp",
)


def _manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    return read_json(path) if path.is_file() else {}


def _config(run_dir: Path, experiment: str) -> dict[str, Any]:
    actual = run_dir / "exp_config.json"
    return read_json(actual) if actual.is_file() else load_experiment_config(
        CONFIG_ROOT / f"{experiment}.json"
    )


def _artifact_completed(
    manifest: Mapping[str, Any], stage: str, manifest_key: str
) -> bool:
    stage_info = manifest.get("stages", {}).get(stage, {})
    value = manifest.get(manifest_key) or stage_info.get(manifest_key)
    return stage_info.get("status") == "completed" and bool(value) and Path(str(value)).is_file()


def _analysis_completed(manifest: Mapping[str, Any]) -> bool:
    stage_info = manifest.get("stages", {}).get("analysis", {})
    value = manifest.get("analysis_manifest_path")
    if value:
        path = Path(str(value))
    else:
        directory = manifest.get("analysis_path") or stage_info.get("analysis_path")
        path = Path(str(directory)) / "analysis_manifest.json" if directory else Path()
    return stage_info.get("status") == "completed" and bool(value or manifest.get("analysis_path")) and path.is_file()


def _train_completed(run_dir: Path, config: Mapping[str, Any]) -> bool:
    manifest = _manifest(run_dir)
    if manifest.get("stages", {}).get("train", {}).get("status") != "completed":
        return False
    bundle = latest_checkpoint_bundle(run_dir, config["ema_rate"], require_complete=True)
    if bundle is None:
        raise RuntimeError(f"training is marked completed but has no complete bundle: {run_dir}")
    target = int(config.get("total_steps", config["lr_anneal_steps"]))
    if int(bundle["step"]) < target:
        raise RuntimeError(
            f"training is marked completed at step {bundle['step']}, below target {target}: {run_dir}"
        )
    return True


def _analysis_command(run_dir: Path, config: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        str(HERE / "analyze.py"),
        "--run-dir",
        str(run_dir),
        "--max-cells",
        str(config.get("analysis_max_cells", 2000)),
        "--batch-size",
        str(config.get("analysis_batch_size", 128)),
        "--t-values",
        str(config.get("analysis_t_values", "")),
        "--device",
        str(config.get("device", "auto")),
    ]


def _commands_for_remaining(
    experiment: str, batch_id: str
) -> list[tuple[str, list[str]]]:
    run_dir = run_dir_for(experiment, batch_id)
    config = _config(run_dir, experiment)
    commands: list[tuple[str, list[str]]] = []
    if not (run_dir.exists() and _train_completed(run_dir, config)):
        train = [
            sys.executable,
            str(HERE / "train.py"),
            "--config",
            str(CONFIG_ROOT / f"{experiment}.json"),
            "--run-dir",
            str(run_dir),
        ]
        if run_dir.exists():
            train.extend(["--resume", "auto"])
        commands.append(("train", train))
        # Later stage decisions must be made again after training completes.
        return commands
    manifest = _manifest(run_dir)
    if not _artifact_completed(manifest, "sample", "sample_path"):
        commands.append(("sample", [
            sys.executable, str(HERE / "sample.py"), "--run-dir", str(run_dir)
        ]))
        return commands
    if not _analysis_completed(manifest):
        commands.append(("analysis", _analysis_command(run_dir, config)))
    return commands


def _execute(
    stage: str,
    experiment: str,
    command: Sequence[str],
    status: dict[str, Any],
    status_path: Path,
) -> None:
    entry = {
        "experiment": experiment,
        "stage": stage,
        "status": "running",
        "started_at": now_iso(),
        "command": command_string(command),
    }
    status["events"].append(entry)
    write_json(status_path, status)
    print(f"[{experiment}:{stage}] {entry['command']}", flush=True)
    subprocess.run(list(command), cwd=REPO_ROOT, check=True)
    entry.update({"status": "completed", "finished_at": now_iso()})
    write_json(status_path, status)


def _dry_plan(
    failed: str, remaining: Sequence[str], batch_id: str
) -> dict[str, Any]:
    plan: list[dict[str, Any]] = []
    failed_run = run_dir_for(failed, batch_id)
    failed_config = _config(failed_run, failed)
    failed_manifest = _manifest(failed_run) if failed_run.exists() else {}
    if not _analysis_completed(failed_manifest):
        command = _analysis_command(failed_run, failed_config)
        plan.append({"experiment": failed, "stage": "analysis", "command": command_string(command)})
    for experiment in remaining:
        run_dir = run_dir_for(experiment, batch_id)
        if not run_dir.exists():
            train = [
                sys.executable, str(HERE / "train.py"), "--config",
                str(CONFIG_ROOT / f"{experiment}.json"), "--run-dir", str(run_dir),
            ]
            plan.extend([
                {"experiment": experiment, "stage": "train", "command": command_string(train)},
                {"experiment": experiment, "stage": "sample", "command": "after successful train"},
                {"experiment": experiment, "stage": "analysis", "command": "after successful sample"},
            ])
        else:
            for stage, command in _commands_for_remaining(experiment, batch_id):
                plan.append({"experiment": experiment, "stage": stage, "command": command_string(command)})
    return {"dry_run": True, "batch_id": batch_id, "plan": plan}


def run_recovery(args: argparse.Namespace) -> int:
    batch_id = validate_path_component(args.batch_id, "batch id")
    selected = select_experiments(experiments=args.remaining_experiments)
    if args.failed_experiment in selected:
        raise ValueError("failed experiment must not also appear in --remaining-experiment")
    if args.dry_run:
        print(json.dumps(
            _dry_plan(args.failed_experiment, selected, batch_id),
            indent=2, ensure_ascii=False,
        ))
        return 0

    batch_dir = BATCHES_ROOT / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    status_path = batch_dir / "resume_after_analysis_failure.json"
    status: dict[str, Any] = {
        "status": "running",
        "batch_id": batch_id,
        "failed_experiment": args.failed_experiment,
        "remaining_experiments": list(selected),
        "started_at": now_iso(),
        "events": [],
    }
    write_json(status_path, status)
    try:
        failed_run = run_dir_for(args.failed_experiment, batch_id)
        if not failed_run.is_dir():
            raise FileNotFoundError(f"failed run directory does not exist: {failed_run}")
        failed_config = _config(failed_run, args.failed_experiment)
        if not _train_completed(failed_run, failed_config):
            raise RuntimeError("failed experiment training is not completed")
        failed_manifest = _manifest(failed_run)
        if not _artifact_completed(failed_manifest, "sample", "sample_path"):
            raise RuntimeError("failed experiment sampling is not completed or sample is missing")
        if not _analysis_completed(failed_manifest):
            _execute(
                "analysis", args.failed_experiment,
                _analysis_command(failed_run, failed_config), status, status_path,
            )

        for experiment in selected:
            # Re-evaluate state after each subprocess so train -> sample ->
            # analysis proceeds without rerunning a completed artifact.
            while True:
                commands = _commands_for_remaining(experiment, batch_id)
                if not commands:
                    break
                stage, command = commands[0]
                _execute(stage, experiment, command, status, status_path)

        status.update({"status": "completed", "finished_at": now_iso()})
        write_json(status_path, status)
        print(f"RECOVERY_STATUS='{status_path}'")
        return 0
    except BaseException as exc:
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
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--failed-experiment", default=FAILED_DEFAULT)
    parser.add_argument(
        "--remaining-experiment", dest="remaining_experiments", nargs="+",
        default=list(REMAINING_DEFAULT),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_recovery(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
