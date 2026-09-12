"""Discover completed EMA runs and restore using unchanged suite helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .source_imports import ROOT, activate_suite, analysis_helpers


@dataclass
class RunSelection:
    suite: str
    experiment: str
    run_dir: Path
    config: dict
    checkpoint: Path
    step: int
    ema_rate: str


def select_run(
    suite: str, run_dir: Path, checkpoint: str = "", expected_step: int = 30000
) -> RunSelection:
    common, sample = activate_suite(suite)
    run = common.validate_run_dir(run_dir)
    config = common.read_json(run / "exp_config.json")
    common.validate_config(config)
    if config["experiment"] != run.parent.name:
        raise ValueError(f"config experiment does not match run directory: {run}")
    manifest = common.read_json(run / "manifest.json")
    if manifest.get("stages", {}).get("train", {}).get("status") != "completed":
        raise ValueError(f"training is not marked completed: {run}")
    if int(config["total_steps"]) != expected_step:
        raise ValueError(f"run target is {config['total_steps']}, expected {expected_step}: {run}")
    path = sample._choose_checkpoint(run, config, checkpoint)
    step = common.checkpoint_step(path)
    if step != expected_step or not path.name.startswith("ema_"):
        raise ValueError(f"expected final EMA step {expected_step}, got {path}")
    rate = path.name[len("ema_") :].rsplit("_", 1)[0]
    if rate not in common.ema_rates(config["ema_rate"]):
        raise ValueError(f"checkpoint EMA rate {rate} not in config: {config['ema_rate']}")
    return RunSelection(suite, config["experiment"], run, config, path, step, rate)


def discover_suite(suite: str) -> dict:
    common, _ = activate_suite(suite)
    report: dict[str, Any] = {"suite": suite, "experiments": {}}
    for experiment in common.EXPERIMENT_ORDER:
        candidates, rejected = [], []
        for run in sorted((common.RUNS_ROOT / experiment).glob("*")):
            if not run.is_dir():
                continue
            try:
                selected = select_run(suite, run)
                candidates.append(
                    {
                        "run_dir": str(run.resolve()),
                        "checkpoint": str(selected.checkpoint),
                        "step": selected.step,
                        "ema_rate": selected.ema_rate,
                    }
                )
            except (ValueError, OSError, KeyError, TypeError) as exc:
                rejected.append({"run_dir": str(run.resolve()), "reason": str(exc)})
        report["experiments"][experiment] = {"candidates": candidates, "rejected": rejected}
    return report


def resolve_entry(entry: dict, report: dict) -> dict:
    experiment = entry["experiment"]
    if entry.get("checkpoint") and not entry.get("run_dir"):
        raise ValueError("an explicit checkpoint also requires an explicit run_dir in models.json")
    if entry["suite"] != report["suite"] or experiment not in report["experiments"]:
        raise ValueError(f"unknown suite/experiment mapping: {entry}")
    if entry.get("run_dir"):
        run = Path(entry["run_dir"]).expanduser()
        run = run if run.is_absolute() else ROOT / run
        if run.parent.name != experiment:
            raise ValueError(f"mapping experiment {experiment} differs from run {run}")
        return {**entry, "run_dir": str(run.resolve())}
    candidates = report["experiments"][experiment]["candidates"]
    if len(candidates) != 1:
        raise ValueError(
            f"{entry['suite']}/{experiment}: expected one completed final EMA run, found {len(candidates)}. Set run_dir/checkpoint in models.json. Candidates: {candidates}"
        )
    return {**entry, **candidates[0]}


def restore(selection: RunSelection, device_name: str = "cpu") -> tuple[Any, Any, list[str], Any]:
    import torch

    _, sample = activate_suite(selection.suite)
    helpers = analysis_helpers(selection.suite)
    device = sample._select_device(torch, device_name)
    genes = sample._genes(selection.config)
    diffusion = helpers.build_diffusion(selection.config)
    if (
        diffusion.num_timesteps != 1000
        or diffusion.original_num_steps != 1000
        or diffusion.timestep_map != list(range(1000))
    ):
        raise ValueError(
            "post-hoc sampling requires an unchanged full 1000-step diffusion schedule"
        )
    if selection.config.get("use_ddim", False) or selection.config.get("class_cond", False):
        raise ValueError(
            "this experiment requires unconditional ancestral p_sample; config requests DDIM or class conditioning"
        )
    model = helpers.load_model(selection.config, genes, diffusion, selection.checkpoint, device)
    return model, diffusion, genes, device
