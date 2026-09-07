"""Small, suite-local config and run helpers."""

from __future__ import annotations

import copy
import json
import re
from datetime import datetime
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUITE_ROOT.parent.parent
CONFIG_ROOT = SUITE_ROOT / "configs"
RUNS_ROOT = SUITE_ROOT / "runs"
RUNS2_ROOT = SUITE_ROOT / "runs2"
RUN_ROOTS = {"runs": RUNS_ROOT, "runs2": RUNS2_ROOT}
EXPERIMENT_ORDER = (
    "01_centered_signed_hill_lambda0p1",
    "02_centered_signed_hill_lambda1",
    "03_centered_signed_hill_lambda10",
    "04_shifted_hill_rho_lambda0p1",
    "05_shifted_hill_rho_lambda1",
    "06_shifted_hill_rho_lambda10",
    "07_hill_after_linear_lambda0p1",
    "08_hill_after_linear_lambda1",
    "09_hill_after_linear_lambda10",
    "10_simple_softplus_lambda0p1",
    "11_simple_softplus_lambda1",
    "12_simple_softplus_lambda10",
)
EXPLORATORY_EXPERIMENT_ORDER = (
    "13_centered_signed_hill_lambda0p01",
    "14_centered_signed_hill_lambda0p001",
    "15_centered_signed_hill_lambda10_to_0p001",
    "16_shifted_hill_rho_lambda0p01",
    "17_shifted_hill_rho_lambda0p001",
    "18_shifted_hill_rho_lambda10_to_0p001",
    "19_hill_after_linear_lambda0p01",
    "20_hill_after_linear_lambda0p001",
    "21_hill_after_linear_lambda10_to_0p001",
    "22_simple_softplus_lambda0p01",
    "23_simple_softplus_lambda0p001",
    "24_simple_softplus_lambda10_to_0p001",
)
ALL_EXPERIMENTS = EXPERIMENT_ORDER + EXPLORATORY_EXPERIMENT_ORDER
_EXPERIMENT_SPECS = {
    name: {
        "ode_type": ode_type,
        "start": start,
        "schedule": schedule,
        "end": end,
    }
    for name, ode_type, start, schedule, end in (
        *(
            (name, ode_type, value, "constant", None)
            for name, (ode_type, value) in zip(
                EXPERIMENT_ORDER,
                (
                    (ode_type, value)
                    for ode_type in (
                        "centered_signed_hill", "shifted_hill_rho",
                        "hill_after_linear", "simple_softplus",
                    )
                    for value in (0.1, 1.0, 10.0)
                ),
            )
        ),
        *(
            (name, ode_type, start, schedule, end)
            for name, (ode_type, start, schedule, end) in zip(
                EXPLORATORY_EXPERIMENT_ORDER,
                (
                    (ode_type, start, schedule, end)
                    for ode_type in (
                        "centered_signed_hill", "shifted_hill_rho",
                        "hill_after_linear", "simple_softplus",
                    )
                    for start, schedule, end in (
                        (0.01, "constant", None),
                        (0.001, "constant", None),
                        (10.0, "log_linear", 0.001),
                    )
                ),
            )
        ),
    )
}
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    temporary.replace(destination)


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config):
    name = str(config.get("experiment", ""))
    if name not in _EXPERIMENT_SPECS:
        raise ValueError(f"experiment is not registered, got {name!r}")
    spec = _EXPERIMENT_SPECS[name]
    if config.get("ode_type") != spec["ode_type"]:
        raise ValueError(f"{name} must use {spec['ode_type']}")
    if float(config.get("cell_ode_reg_lambda_20260830")) != spec["start"]:
        raise ValueError(f"{name} must start with lambda={spec['start']}")
    schedule = str(config.get("cell_ode_reg_schedule_20260830", "constant"))
    if schedule != spec["schedule"]:
        raise ValueError(f"{name} must use schedule={spec['schedule']}")
    configured_end = config.get("cell_ode_reg_lambda_end_20260830")
    if spec["end"] is None:
        if configured_end is not None:
            raise ValueError(f"{name} must not set a schedule end weight")
    elif float(configured_end) != spec["end"]:
        raise ValueError(f"{name} must end with lambda={spec['end']}")
    if int(config.get("K", 1)) != 1 or config.get("gate_mode") != "none":
        raise ValueError("all conditions must be single ODE without a gate")
    if float(config.get("off_mask_lambda", 5.0)) != 5.0:
        raise ValueError("off_mask_lambda is inherited and fixed at 5.0")
    if float(config.get("ode_reg_lambda", 1.0)) != 1.0:
        raise ValueError("ode_reg_lambda is inherited and fixed at 1.0")
    if int(config.get("total_steps", 0)) != 100000:
        raise ValueError("all 20260830 conditions must use total_steps=100000")
    if int(config.get("lr_anneal_steps", 0)) != 100000:
        raise ValueError("all 20260830 conditions must use lr_anneal_steps=100000")
    if int(config.get("total_steps")) != int(config.get("lr_anneal_steps")):
        raise ValueError("total_steps and lr_anneal_steps must be identical")
    if int(config.get("cell_ode_reg_schedule_steps_20260830", 0)) != int(config["total_steps"]):
        raise ValueError("consistency schedule steps must equal total_steps")
    if int(config.get("detailed_loss_flush_interval", 0)) <= 0:
        raise ValueError("detailed_loss_flush_interval must be positive")


def load_experiment_config(experiment_or_path):
    value = Path(str(experiment_or_path))
    if value.suffix == ".json" and value.exists():
        override_path = value
    else:
        name = value.stem
        override_path = CONFIG_ROOT / "experiments" / f"{name}.json"
    config = deep_merge(read_json(CONFIG_ROOT / "base.json"), read_json(override_path))
    validate_config(config)
    return config


def safe_component(value, label):
    text = str(value)
    if not _SAFE.fullmatch(text) or Path(text).name != text:
        raise ValueError(f"unsafe {label}: {value!r}")
    return text


def new_batch_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def resolve_runs_root(name="runs"):
    try:
        return RUN_ROOTS[str(name)]
    except KeyError as error:
        raise ValueError(f"runs root must be one of {tuple(RUN_ROOTS)}") from error


def assert_allowed_run_dir(path):
    target = Path(path).resolve()
    for root in RUN_ROOTS.values():
        resolved_root = root.resolve()
        try:
            target.relative_to(resolved_root)
        except ValueError:
            continue
        if target != resolved_root:
            return target
    raise ValueError("run-dir must be an experiment/batch directory below runs/ or runs2/")


def run_dir(experiment, batch_id, runs_root="runs"):
    return (
        resolve_runs_root(runs_root)
        / safe_component(experiment, "experiment")
        / safe_component(batch_id, "batch id")
    )


def checkpoint_files(run_path):
    root = Path(run_path) / "checkpoints"
    raw = sorted(root.glob("segment_*/model/model*.pt"))
    return raw


def latest_raw_checkpoint(run_path):
    paths = sorted(
        checkpoint_files(run_path),
        key=lambda path: int(path.stem.replace("model", "")),
        reverse=True,
    )
    for raw in paths:
        step = int(raw.stem.replace("model", ""))
        if step == 0:
            continue
        optimizer = raw.with_name(f"opt{step:06d}.pt")
        ema = list(raw.parent.glob(f"ema_*_{step:06d}.pt"))
        if optimizer.is_file() and ema:
            return raw
    return None


def choose_sampling_checkpoint(run_path, ema_rate):
    raw = latest_raw_checkpoint(run_path)
    if raw is None:
        raise FileNotFoundError("no checkpoint found")
    step = int(raw.stem.replace("model", ""))
    rate = str(ema_rate).split(",")[0]
    ema = raw.with_name(f"ema_{rate}_{step:06d}.pt")
    return ema if ema.exists() else raw


__all__ = [name for name in globals() if not name.startswith("_")]
