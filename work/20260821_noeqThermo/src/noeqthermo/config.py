"""Configuration loading and validation for the landscape/flux pipeline."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[2]


def load_config(mode: str, path: str | Path | None = None) -> dict[str, Any]:
    """Load one complete mode configuration; modes never silently mix settings."""

    mode = str(mode).lower()
    if mode not in {"smoke", "full"}:
        raise ValueError("mode must be 'smoke' or 'full'")
    config_path = Path(path) if path else HERE / "configs" / f"{mode}.json"
    with config_path.expanduser().resolve().open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise TypeError("configuration root must be a JSON object")
    if str(config.get("mode")) != mode:
        raise ValueError(f"config mode {config.get('mode')!r} does not match --mode {mode!r}")
    validate_config(config)
    config["_config_path"] = str(config_path.expanduser().resolve())
    return config


def _positive(mapping: Mapping[str, Any], name: str, *, integer: bool = False) -> None:
    value = mapping.get(name)
    if integer and (not isinstance(value, int) or isinstance(value, bool)):
        raise TypeError(f"{name} must be an integer")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail early on settings that would make models incomparable or arrays invalid."""

    required = {"selection", "umap", "dynamo", "sde", "lap", "plot"}
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"missing configuration sections: {missing}")
    selection = config["selection"]
    if selection.get("obs_key") != "Superclass" or selection.get("value") != "Erythropoietic":
        raise ValueError("this analysis is restricted to Superclass == Erythropoietic")
    for name in ("seed", "max_cells"):
        value = config.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    for name in ("n_neighbors", "n_pcs", "n_components"):
        _positive(config["umap"], name, integer=True)
    for name in ("grid_num", "topography_samples", "restart_num", "M"):
        _positive(config["dynamo"], name, integer=True)
    if str(config["dynamo"].get("version")) != "1.4.1":
        raise ValueError("this implementation is audited and pinned to Dynamo 1.4.1")
    sde = config["sde"]
    for name in ("trajectories", "steps", "burn_in", "grid_size", "checkpoint_every"):
        _positive(sde, name, integer=True)
    if sde["burn_in"] >= sde["steps"]:
        raise ValueError("sde.burn_in must be smaller than sde.steps")
    if sde.get("parameterization") not in {"umap_calibrated", "paper_literal"}:
        raise ValueError("sde.parameterization must be umap_calibrated or paper_literal")
    if sde["parameterization"] == "paper_literal":
        _positive(sde, "dt")
        _positive(sde, "D")
    else:
        for name in ("drift_step_fraction", "noise_step_fraction", "dt_min", "dt_max"):
            _positive(sde, name)
        if sde["dt_min"] > sde["dt_max"]:
            raise ValueError("sde.dt_min must not exceed sde.dt_max")
    if not 0 <= float(sde.get("bounds_quantile", -1)) < 0.5:
        raise ValueError("sde.bounds_quantile must be in [0, 0.5)")
    if float(sde.get("bounds_margin", -1)) < 0:
        raise ValueError("sde.bounds_margin must be non-negative")
    _positive(config["lap"], "n_points", integer=True)
    if int(config["lap"].get("max_pairs", -1)) < 0:
        raise ValueError("lap.max_pairs must be non-negative")


__all__ = ["load_config", "validate_config"]
