#!/usr/bin/env python3
"""Resumable training -> sampling -> analysis pipeline for Model A and B."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from common import (  # noqa: E402
    CONFIG_ROOT,
    MODEL_CONFIGS,
    MODEL_ORDER,
    all_raw_checkpoints,
    latest_raw_checkpoint,
    load_experiment_config,
    read_json,
    resolve_sampling_checkpoint,
    run_directory,
    safe_component,
    validate_run_directory,
    write_json,
)


ANALYSIS_STAGES = ("loss", "parameters", "diagnostics", "velocity", "umap")
ANALYSIS_DIRECTORIES = {
    "loss": "loss",
    "parameters": "parameter_evolution",
    "diagnostics": "diffusion_diagnostics",
    "velocity": "drift_velocity",
    "umap": "hematopoietic_umap",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _latest_segment_status(run: Path) -> dict:
    statuses = sorted(run.glob("segments/segment_*/status.json"))
    return read_json(statuses[-1]) if statuses else {}


def decide_training_action(run_dir, target_steps: int) -> str:
    """Return ``run``, ``resume``, ``skip``, or ``blocked_running``."""

    run = Path(run_dir)
    if not run.exists():
        return "run"
    status = _latest_segment_status(run)
    if status.get("status") == "running":
        return "blocked_running"
    raw = all_raw_checkpoints(run)
    final_step = max((int(path.stem.replace("model", "")) for path in raw), default=-1)
    # The inherited TrainLoop writes a checkpoint after the update but before
    # incrementing ``self.step``.  On the rare boundary where the last loop
    # index is itself a save interval, the completed bundle is labelled
    # ``target_steps - 1``; otherwise its final explicit save is labelled
    # ``target_steps``.  Both represent a completed run and must be skipped.
    completed_floor = max(0, int(target_steps) - 1)
    if status.get("status") == "completed" and final_step >= completed_floor:
        return "skip"
    return "resume" if latest_raw_checkpoint(run) is not None else "run"


def analysis_stage_complete(run_dir, stage: str) -> bool:
    directory = ANALYSIS_DIRECTORIES[stage]
    run = Path(run_dir)
    path = run / "analysis" / directory / "metadata.json"
    if not path.is_file():
        return False
    metadata = read_json(path)
    if metadata.get("status") != "completed":
        return False
    if stage == "loss":
        current = [
            str(item.resolve())
            for item in sorted(run.glob("segments/segment_*/loss_components.csv"))
        ]
        return bool(current) and metadata.get("source_files") == current
    if stage == "parameters":
        sources = metadata.get("sources", [])
        recorded = {
            Path(str(item.get("path", ""))).expanduser().resolve()
            for item in sources
        }
        raw = all_raw_checkpoints(run)
        initial = (run / "initial_forward_state.pt").resolve()
        return bool(raw) and initial in recorded and raw[-1].resolve() in recorded
    try:
        checkpoint, step, rate = resolve_sampling_checkpoint(run)
        recorded_checkpoint = Path(
            str(metadata.get("checkpoint", ""))
        ).expanduser().resolve()
    except (FileNotFoundError, ValueError, TypeError):
        return False
    return (
        recorded_checkpoint == checkpoint.resolve()
        and int(metadata.get("checkpoint_step", -1)) == int(step)
        and str(metadata.get("ema_rate", "")) == str(rate)
    )


def sample_stage_complete(run_dir, ema_rate=None) -> bool:
    run = Path(run_dir)
    try:
        checkpoint, _step, _rate = resolve_sampling_checkpoint(run, ema_rate)
    except (FileNotFoundError, ValueError):
        return False
    for sidecar in (run / "samples").glob("*.json"):
        metadata = read_json(sidecar)
        if (
            metadata.get("status") == "completed"
            and Path(metadata.get("checkpoint_path", "")).expanduser().resolve()
            == checkpoint.resolve()
            and sidecar.with_suffix(".npz").is_file()
        ):
            return True
    return False


def _selected_models(spec: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(value.strip() for value in spec.split(",") if value.strip()))
    if not values:
        raise ValueError("--models must select at least one model")
    invalid = [value for value in values if value not in MODEL_ORDER]
    if invalid:
        raise ValueError(f"unknown model(s): {invalid}; expected {MODEL_ORDER}")
    return values


def build_pipeline_plan(args: argparse.Namespace) -> list[dict]:
    batch = safe_component(args.batch_id, "batch id")
    config_by_model = dict(zip(MODEL_ORDER, MODEL_CONFIGS))
    plan = []
    for model_name in _selected_models(args.models):
        config_path = CONFIG_ROOT / config_by_model[model_name]
        overrides = list(args.set_values)
        if args.device != "auto":
            overrides.append(f"device={json.dumps(args.device)}")
        config, provenance = load_experiment_config(
            config_path,
            base_path=args.base_config,
            set_values=overrides,
        )
        run = validate_run_directory(run_directory(config["experiment"], batch))
        action = decide_training_action(run, int(config.get("lr_anneal_steps", 0)))
        train_command = [
            args.python,
            str(HERE / "train.py"),
            "--config",
            provenance["experiment_config_path"],
            "--base-config",
            provenance["base_config_path"],
            "--run-dir",
            str(run),
        ]
        if action == "resume":
            train_command.extend(["--resume", "auto"])
        for value in overrides:
            train_command.extend(["--set", value])
        sample_command = [
            args.python,
            str(HERE / "sample.py"),
            "--run-dir",
            str(run),
            "--device",
            args.device,
        ]
        analysis = []
        for stage in args.analysis_stages:
            command = [
                args.python,
                str(HERE / "analyze.py"),
                "--run-dir",
                str(run),
                "--stage",
                stage,
                "--device",
                args.device,
            ]
            if args.force_analysis:
                command.append("--force")
            analysis.append(
                {
                    "stage": stage,
                    "action": (
                        "run"
                        if args.force_analysis or not analysis_stage_complete(run, stage)
                        else "skip"
                    ),
                    "command": command,
                }
            )
        plan.append(
            {
                "model": model_name,
                "experiment": config["experiment"],
                "run_dir": str(run),
                "training": {"action": action, "command": train_command},
                "sampling": {
                    "action": "skip" if sample_stage_complete(run) else "run",
                    "command": sample_command,
                },
                "analysis": analysis,
            }
        )
    return plan


def _update_pipeline_status(run: Path, stage: str, payload: Mapping) -> None:
    path = run / "pipeline_status.json"
    existing = read_json(path) if path.is_file() else {"stages": {}}
    existing.setdefault("stages", {})[stage] = dict(payload)
    existing["updated_at"] = _now()
    write_json(path, existing)


def _execute_stage(run: Path, stage: str, command: Sequence[str]) -> None:
    _update_pipeline_status(
        run,
        stage,
        {"status": "running", "started_at": _now(), "command": list(command)},
    )
    try:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
    except BaseException as error:
        _update_pipeline_status(
            run,
            stage,
            {
                "status": "failed",
                "failed_at": _now(),
                "command": list(command),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    _update_pipeline_status(
        run,
        stage,
        {"status": "completed", "finished_at": _now(), "command": list(command)},
    )


def run_pipeline(args: argparse.Namespace) -> dict:
    plan = build_pipeline_plan(args)
    if args.dry_run:
        return {"status": "dry_run", "batch_id": args.batch_id, "plan": plan}
    results = []
    for model_plan in plan:
        run = Path(model_plan["run_dir"])
        run.mkdir(parents=True, exist_ok=True)
        training = model_plan["training"]
        if training["action"] == "blocked_running":
            raise RuntimeError(f"training is already marked running: {run}")
        if training["action"] != "skip":
            _execute_stage(run, "training", training["command"])
        else:
            _update_pipeline_status(run, "training", {"status": "skipped_completed"})
        if model_plan["sampling"]["action"] != "skip":
            _execute_stage(run, "sampling", model_plan["sampling"]["command"])
        else:
            _update_pipeline_status(run, "sampling", {"status": "skipped_completed"})
        for analysis in model_plan["analysis"]:
            stage_name = f"analysis_{analysis['stage']}"
            if analysis["action"] == "skip":
                _update_pipeline_status(run, stage_name, {"status": "skipped_completed"})
            else:
                _execute_stage(run, stage_name, analysis["command"])
        results.append({"model": model_plan["model"], "run_dir": str(run), "status": "completed"})
    return {"status": "completed", "batch_id": args.batch_id, "results": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--models", default=",".join(MODEL_ORDER))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--base-config", default=None)
    parser.add_argument(
        "--set", dest="set_values", action="append", default=[], metavar="KEY=JSON_VALUE"
    )
    parser.add_argument(
        "--analysis-stages",
        default=",".join(ANALYSIS_STAGES),
        help="comma-separated loss,parameters,diagnostics,velocity,umap",
    )
    parser.add_argument("--force-analysis", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    stages = tuple(
        dict.fromkeys(value.strip() for value in args.analysis_stages.split(",") if value.strip())
    )
    invalid = [stage for stage in stages if stage not in ANALYSIS_STAGES]
    if not stages or invalid:
        raise ValueError(f"invalid analysis stages {invalid}; expected {ANALYSIS_STAGES}")
    args.analysis_stages = stages
    result = run_pipeline(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
