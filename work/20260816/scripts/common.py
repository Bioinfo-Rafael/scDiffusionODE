"""Shared, suite-local configuration, run, and checkpoint helpers.

Nothing in this module writes outside :data:`SUITE_ROOT`.  Input data and the
edge-prior TSV may live elsewhere, but every cache, run, sample, log, and
analysis artifact is kept below this experiment suite.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
WORK_ROOT = SUITE_ROOT  # Compatibility with the 20260707 helper naming.
REPO_ROOT = SUITE_ROOT.parent.parent
CONFIG_ROOT = SUITE_ROOT / "configs"
RUNS_ROOT = SUITE_ROOT / "runs"
BATCHES_ROOT = SUITE_ROOT / "batches"

ODE_TYPES = ("centered_signed_hill", "shifted_hill_rho")
MODEL_FAMILIES = (
    "linear_hybrid_lincomb",
    "ts_soft_tau80_hybrid_lincomb",
)
EXPERIMENT_ORDER = (
    "linear_centered_signed_hill",
    "linear_shifted_hill_rho",
    "ts_soft_tau80_centered_signed_hill",
    "ts_soft_tau80_shifted_hill_rho",
)
# Friendly aliases for callers/tests.
EXPERIMENTS = EXPERIMENT_ORDER
EXPERIMENT_NAMES = EXPERIMENT_ORDER
EXPERIMENT_TO_PARTS = {
    "linear_centered_signed_hill": (
        "linear_hybrid_lincomb", "centered_signed_hill"
    ),
    "linear_shifted_hill_rho": (
        "linear_hybrid_lincomb", "shifted_hill_rho"
    ),
    "ts_soft_tau80_centered_signed_hill": (
        "ts_soft_tau80_hybrid_lincomb", "centered_signed_hill"
    ),
    "ts_soft_tau80_shifted_hill_rho": (
        "ts_soft_tau80_hybrid_lincomb", "shifted_hill_rho"
    ),
}

_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RAW_CHECKPOINT_RE = re.compile(r"^model(?P<step>[0-9]+)\.pt$")
_EMA_CHECKPOINT_RE = re.compile(r"^ema_(?P<rate>.+)_(?P<step>[0-9]+)\.pt$")


class ConfigurationError(ValueError):
    """Raised when an experiment config violates the fixed comparison matrix."""


class RunSafetyError(ValueError):
    """Raised when a path would escape or overwrite the isolated suite."""


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in ("true", "1", "yes", "y", "on"):
        return True
    if lowered in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"invalid boolean: {value!r}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def read_json(path: os.PathLike[str] | str) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_write_text(
    path: os.PathLike[str] | str,
    text: str,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically write UTF-8 text in the destination directory."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and destination.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if not overwrite and destination.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {destination}")
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return destination


def write_json(
    path: os.PathLike[str] | str,
    payload: Any,
    *,
    overwrite: bool = True,
) -> Path:
    text = json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        default=_json_default,
    ) + "\n"
    return atomic_write_text(path, text, overwrite=overwrite)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries; lists and scalars are replaced."""

    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def mutate_json(
    path: os.PathLike[str] | str,
    mutator: Callable[[dict[str, Any]], MutableMapping[str, Any] | None],
) -> dict[str, Any]:
    """Read-modify-replace a JSON object while holding an advisory file lock."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.with_name(destination.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        payload = read_json(destination) if destination.exists() else {}
        if not isinstance(payload, dict):
            raise TypeError(f"expected JSON object in {destination}")
        changed = mutator(payload)
        if changed is not None:
            payload = dict(changed)
        write_json(destination, payload)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return payload


def update_json(
    path: os.PathLike[str] | str,
    updates: Mapping[str, Any],
    *,
    recursive: bool = True,
) -> dict[str, Any]:
    def apply(payload: dict[str, Any]) -> dict[str, Any]:
        return deep_merge(payload, updates) if recursive else {**payload, **updates}

    return mutate_json(path, apply)


def validate_path_component(value: str, label: str = "path component") -> str:
    component = str(value).strip()
    if (
        not component
        or component in (".", "..")
        or not _SAFE_COMPONENT_RE.fullmatch(component)
        or Path(component).name != component
    ):
        raise RunSafetyError(f"invalid {label}: {value!r}")
    return component


def ensure_within(path: os.PathLike[str] | str, root: os.PathLike[str] | str) -> Path:
    candidate = Path(path).expanduser().resolve()
    parent = Path(root).expanduser().resolve()
    try:
        candidate.relative_to(parent)
    except ValueError as exc:
        raise RunSafetyError(f"path must remain below {parent}: {candidate}") from exc
    return candidate


def _canonical_experiment(value: str) -> str:
    name = str(value).strip()
    if name.endswith(".json"):
        name = name[:-5]
    if Path(name).name != name:
        raise ConfigurationError(f"experiment must be a canonical name, not a path: {value}")
    if name not in EXPERIMENT_TO_PARTS:
        raise ConfigurationError(
            f"unknown experiment {value!r}; expected one of: {', '.join(EXPERIMENT_ORDER)}"
        )
    return name


def select_experiments(
    families: Sequence[str] | None = None,
    experiments: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return a subset in the required family-outer/ODE-inner order."""

    family_values = tuple(str(value).strip() for value in (families or ()))
    unknown_families = sorted(set(family_values).difference(MODEL_FAMILIES))
    if unknown_families:
        raise ConfigurationError(f"unknown model family/families: {', '.join(unknown_families)}")
    experiment_values = tuple(_canonical_experiment(value) for value in (experiments or ()))
    family_set = set(family_values)
    experiment_set = set(experiment_values)
    selected = tuple(
        name
        for name in EXPERIMENT_ORDER
        if (not family_set or EXPERIMENT_TO_PARTS[name][0] in family_set)
        and (not experiment_set or name in experiment_set)
    )
    if not selected:
        raise ConfigurationError("model-family and experiment filters select no conditions")
    return selected


def experiment_config_path(experiment: str) -> Path:
    name = _canonical_experiment(experiment)
    return CONFIG_ROOT / f"{name}.json"


def _require_number(config: Mapping[str, Any], key: str, minimum: float = 0.0) -> float:
    if key not in config:
        raise ConfigurationError(f"missing required config key: {key}")
    try:
        value = float(config[key])
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{key} must be numeric, got {config[key]!r}") from exc
    if value < minimum:
        raise ConfigurationError(f"{key} must be >= {minimum}, got {value}")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate matrix identity and comparison-critical training settings."""

    required = ("experiment", "model_family", "ode_type", "data_dir", "edge_tsv_path")
    missing = [key for key in required if key not in config or config[key] in (None, "")]
    if missing:
        raise ConfigurationError(f"missing required config keys: {', '.join(missing)}")

    experiment = _canonical_experiment(str(config["experiment"]))
    family = str(config["model_family"])
    ode_type = str(config["ode_type"])
    if family not in MODEL_FAMILIES:
        raise ConfigurationError(f"unknown model_family: {family!r}")
    if ode_type not in ODE_TYPES:
        raise ConfigurationError(f"unknown ode_type: {ode_type!r}")
    expected_parts = EXPERIMENT_TO_PARTS[experiment]
    if (family, ode_type) != expected_parts:
        raise ConfigurationError(
            "experiment identity mismatch: "
            f"{experiment!r} requires family/type={expected_parts!r}"
        )

    try:
        components = int(config.get("K", config.get("num_components", -1)))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("LinComb K must be the integer 8") from exc
    if components != 8:
        raise ConfigurationError(f"all LinComb conditions require K=8, got {components}")
    if str(config.get("gate_mode", "")).lower() != "softmax":
        raise ConfigurationError("all 20260816 conditions require gate_mode=softmax")

    regime_mode = str(config.get("regime_gate_mode", "none"))
    if family == "ts_soft_tau80_hybrid_lincomb":
        if regime_mode.lower() != "ts_i_vs_ii_iii":
            raise ConfigurationError("TS_soft_tau80 requires regime_gate_mode=Ts_I_vs_II_III")
        if str(config.get("regime_gate_type", "")).lower() != "sigmoid":
            raise ConfigurationError("TS_soft_tau80 requires regime_gate_type=sigmoid")
        if float(config.get("gate_tau", -1)) != 80.0:
            raise ConfigurationError("TS_soft_tau80 requires gate_tau=80")
        if bool_value(config.get("rescale_timesteps", False)):
            raise ConfigurationError("TS_soft_tau80 requires rescale_timesteps=false")
    elif regime_mode.lower() != "none":
        raise ConfigurationError("linear Hybrid requires regime_gate_mode=none")

    if "use_mask_reg" in config and not bool_value(config["use_mask_reg"]):
        raise ConfigurationError("the TSV-derived edge prior must remain enabled")
    if "SoftReg" in config and not bool_value(config["SoftReg"]):
        raise ConfigurationError("the TSV-derived prior must remain a soft penalty")
    _require_number(config, "off_mask_lambda", 0.0)

    for key in ("batch_size", "diffusion_steps", "save_interval", "log_interval"):
        if key in config and int(config[key]) <= 0:
            raise ConfigurationError(f"{key} must be > 0")
    if "microbatch" in config and int(config["microbatch"]) == 0:
        raise ConfigurationError("microbatch must be -1 or > 0")
    total_steps = config.get("total_steps")
    anneal_steps = config.get("lr_anneal_steps")
    if total_steps is None and anneal_steps is None:
        raise ConfigurationError("total_steps or lr_anneal_steps must be configured")
    if total_steps is not None and int(total_steps) <= 0:
        raise ConfigurationError("total_steps must be > 0")
    if anneal_steps is not None and int(anneal_steps) <= 0:
        raise ConfigurationError("lr_anneal_steps must be > 0")
    if total_steps is not None and anneal_steps is not None and int(total_steps) != int(anneal_steps):
        raise ConfigurationError("total_steps and lr_anneal_steps must match")

    device = str(config.get("device", "auto")).lower()
    if device not in ("auto", "cpu", "cuda", "mps"):
        raise ConfigurationError("device must be one of auto, cpu, cuda, mps")


def load_experiment_config(
    config_path: os.PathLike[str] | str,
    base_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    """Deep-merge ``configs/base.json`` with one condition config."""

    selected_path = Path(config_path).expanduser()
    if not selected_path.exists() and Path(str(config_path)).name == str(config_path):
        candidate = CONFIG_ROOT / str(config_path)
        if not candidate.suffix:
            candidate = candidate.with_suffix(".json")
        selected_path = candidate
    selected_path = selected_path.resolve()
    actual_base_path = Path(base_path).expanduser().resolve() if base_path else CONFIG_ROOT / "base.json"
    base = read_json(actual_base_path)
    selected = read_json(selected_path)
    if not isinstance(base, dict) or not isinstance(selected, dict):
        raise ConfigurationError("base and experiment configs must contain JSON objects")
    merged = deep_merge(base, selected)
    validate_config(merged)
    return merged


def config_provenance(
    config_path: os.PathLike[str] | str,
    base_path: os.PathLike[str] | str | None = None,
) -> dict[str, Any]:
    selected = Path(config_path).expanduser().resolve()
    base = Path(base_path).expanduser().resolve() if base_path else CONFIG_ROOT / "base.json"
    return {
        "base_config_path": str(base),
        "base_config_sha256": file_sha256(base),
        "experiment_config_path": str(selected),
        "experiment_config_sha256": file_sha256(selected),
    }


def parse_scalar(text: str) -> Any:
    stripped = str(text).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return stripped


def apply_set_overrides(config: Mapping[str, Any], values: Iterable[str]) -> dict[str, Any]:
    """Apply repeatable ``KEY=JSON_VALUE`` overrides, including dotted keys."""

    result = copy.deepcopy(dict(config))
    for item in values:
        if "=" not in item:
            raise ConfigurationError(f"--set expects KEY=VALUE, got {item!r}")
        dotted_key, raw_value = item.split("=", 1)
        keys = [key.strip() for key in dotted_key.split(".")]
        if not keys or any(not key for key in keys):
            raise ConfigurationError(f"invalid --set key: {dotted_key!r}")
        target: dict[str, Any] = result
        for key in keys[:-1]:
            current = target.get(key)
            if current is None:
                current = {}
                target[key] = current
            if not isinstance(current, dict):
                raise ConfigurationError(f"cannot set nested key below non-object: {dotted_key!r}")
            target = current
        target[keys[-1]] = parse_scalar(raw_value)
    validate_config(result)
    return result


def _resolve_input_path(value: str) -> str:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from local_paths import resolve_path as legacy_resolve_path

    resolved = legacy_resolve_path(str(value))
    path = Path(resolved).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return str(path.resolve()) if path.exists() else str(path.absolute())


def resolve_config_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve read-only inputs and force writable caches below this suite."""

    resolved = copy.deepcopy(dict(config))
    for key in ("data_dir", "edge_tsv_path"):
        if resolved.get(key):
            resolved[key] = _resolve_input_path(str(resolved[key]))

    cache_value = str(resolved.get("ts_cache_path", "") or "")
    if cache_value:
        cache_path = Path(cache_value).expanduser()
        if not cache_path.is_absolute():
            if cache_path.parts and cache_path.parts[0] == "work":
                cache_path = REPO_ROOT / cache_path
            else:
                cache_path = SUITE_ROOT / cache_path
        cache_path = ensure_within(cache_path, SUITE_ROOT)
        resolved["ts_cache_path"] = str(cache_path)
    return resolved


def generate_batch_id(suffix: str = "") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        stamp += "_" + validate_path_component(suffix, "batch suffix")
    return stamp


def run_dir_for(experiment: str, batch_id: str) -> Path:
    name = _canonical_experiment(experiment)
    run_id = validate_path_component(batch_id, "batch id")
    return RUNS_ROOT / name / run_id


def validate_run_dir(
    run_dir: os.PathLike[str] | str,
    experiment: str | None = None,
) -> Path:
    candidate = ensure_within(run_dir, RUNS_ROOT)
    relative = candidate.relative_to(RUNS_ROOT.resolve())
    if len(relative.parts) != 2:
        raise RunSafetyError(
            f"run directory must be {RUNS_ROOT}/<experiment>/<batch-id>: {candidate}"
        )
    canonical = _canonical_experiment(relative.parts[0])
    validate_path_component(relative.parts[1], "batch id")
    if experiment is not None and canonical != _canonical_experiment(experiment):
        raise RunSafetyError(
            f"run directory experiment {canonical!r} does not match {experiment!r}"
        )
    return candidate


def create_run_dir(experiment: str, batch_id: str = "") -> Path:
    actual_batch_id = batch_id or generate_batch_id()
    run_dir = run_dir_for(experiment, actual_batch_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir.resolve()


def latest_run_dir(experiment: str, *, require_manifest: bool = True) -> Path:
    name = _canonical_experiment(experiment)
    parent = RUNS_ROOT / name
    candidates = []
    if parent.is_dir():
        for candidate in parent.iterdir():
            if not candidate.is_dir():
                continue
            if require_manifest and not (candidate / "manifest.json").is_file():
                continue
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(f"no run directories found for {name}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name)).resolve()


def artifact_dirs(run_dir: os.PathLike[str] | str) -> dict[str, Path]:
    run = validate_run_dir(run_dir)
    return {
        "run": run,
        "train": run / "train",
        "checkpoints": run / "train" / "checkpoints",
        "samples": run / "samples",
        "analysis": run / "analysis",
        "viz": run / "viz",
        "logs": run / "logs",
        "commands": run / "commands",
        "results": run / "results",
    }


def prepare_run_dirs(run_dir: os.PathLike[str] | str) -> dict[str, Path]:
    paths = artifact_dirs(run_dir)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def update_manifest(run_dir: os.PathLike[str] | str, **updates: Any) -> dict[str, Any]:
    run = validate_run_dir(run_dir)
    manifest_path = run / "manifest.json"

    def apply(payload: dict[str, Any]) -> dict[str, Any]:
        if "created_at" not in payload:
            payload["created_at"] = now_iso()
        merged = deep_merge(payload, updates)
        merged["updated_at"] = now_iso()
        return merged

    return mutate_json(manifest_path, apply)


def update_stage(
    run_dir: os.PathLike[str] | str,
    stage: str,
    status: str,
    **details: Any,
) -> dict[str, Any]:
    if status not in ("pending", "running", "completed", "failed", "skipped"):
        raise ValueError(f"invalid stage status: {status}")
    stage_payload = {"status": status, **details}
    return update_manifest(run_dir, stages={stage: stage_payload})


def command_string(argv: Sequence[object] | None = None) -> str:
    values = sys.argv if argv is None else argv
    return " ".join(shlex.quote(str(value)) for value in values)


def record_command(run_dir: os.PathLike[str] | str, stage: str, argv: Sequence[object] | None = None) -> Path:
    paths = prepare_run_dirs(run_dir)
    attempt = 1
    while True:
        destination = paths["commands"] / f"{stage}_attempt_{attempt:03d}.txt"
        if not destination.exists():
            return atomic_write_text(destination, command_string(argv) + "\n", overwrite=False)
        attempt += 1


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
        "captured_at": now_iso(),
    }


def file_sha256(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def config_fingerprint(config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ema_rates(value: Any) -> tuple[str, ...]:
    if isinstance(value, (float, int)):
        raw_rates = (value,)
    else:
        raw_rates = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not raw_rates:
        raise ConfigurationError("ema_rate must contain at least one rate")
    rates = []
    for raw in raw_rates:
        rate = float(raw)
        if not 0.0 <= rate <= 1.0:
            raise ConfigurationError(f"invalid EMA rate: {raw}")
        normalized = str(rate)
        if normalized not in rates:
            rates.append(normalized)
    return tuple(rates)


def checkpoint_step(path: os.PathLike[str] | str) -> int:
    name = Path(path).name
    raw_match = _RAW_CHECKPOINT_RE.fullmatch(name)
    if raw_match:
        return int(raw_match.group("step"))
    ema_match = _EMA_CHECKPOINT_RE.fullmatch(name)
    if ema_match:
        return int(ema_match.group("step"))
    raise ValueError(f"unrecognized checkpoint filename: {name}")


def checkpoint_bundle_from_raw(
    raw_checkpoint: os.PathLike[str] | str,
    configured_ema_rates: Any,
) -> dict[str, Any]:
    raw_path = Path(raw_checkpoint).expanduser().resolve()
    match = _RAW_CHECKPOINT_RE.fullmatch(raw_path.name)
    if not match:
        raise RunSafetyError(
            "resume requires a raw modelNNNNNN.pt checkpoint; EMA paths are sampling-only"
        )
    step = int(match.group("step"))
    width = len(match.group("step"))
    suffix = f"{step:0{width}d}"
    rates = ema_rates(configured_ema_rates)
    ema_paths = {rate: raw_path.parent / f"ema_{rate}_{suffix}.pt" for rate in rates}
    optimizer_path = raw_path.parent / f"opt{suffix}.pt"
    missing = [str(path) for path in (raw_path, optimizer_path, *ema_paths.values()) if not path.is_file()]
    return {
        "step": step,
        "raw_checkpoint_path": str(raw_path),
        "ema_checkpoint_path": str(ema_paths[rates[0]]),
        "ema_checkpoint_paths": {rate: str(path) for rate, path in ema_paths.items()},
        "optimizer_checkpoint_path": str(optimizer_path),
        "checkpoint_path": str(ema_paths[rates[0]]),
        "checkpoint_directory": str(raw_path.parent),
        "complete": not missing,
        "missing": missing,
    }


def discover_checkpoint_bundles(
    run_dir: os.PathLike[str] | str,
    configured_ema_rates: Any,
) -> list[dict[str, Any]]:
    run = validate_run_dir(run_dir)
    checkpoint_root = run / "train" / "checkpoints"
    bundles: list[dict[str, Any]] = []
    if checkpoint_root.is_dir():
        for raw_path in checkpoint_root.glob("segment_*/*/model*.pt"):
            if _RAW_CHECKPOINT_RE.fullmatch(raw_path.name):
                bundles.append(checkpoint_bundle_from_raw(raw_path, configured_ema_rates))
    bundles.sort(
        key=lambda bundle: (
            int(bundle["step"]),
            Path(bundle["raw_checkpoint_path"]).stat().st_mtime_ns,
        )
    )
    return bundles


def latest_checkpoint_bundle(
    run_dir: os.PathLike[str] | str,
    configured_ema_rates: Any,
    *,
    require_complete: bool = True,
) -> dict[str, Any] | None:
    bundles = discover_checkpoint_bundles(run_dir, configured_ema_rates)
    if require_complete:
        bundles = [bundle for bundle in bundles if bundle["complete"]]
    return bundles[-1] if bundles else None


def validate_resume_bundle(
    run_dir: os.PathLike[str] | str,
    raw_checkpoint: os.PathLike[str] | str,
    configured_ema_rates: Any,
) -> dict[str, Any]:
    run = validate_run_dir(run_dir)
    raw = ensure_within(raw_checkpoint, run / "train" / "checkpoints")
    bundle = checkpoint_bundle_from_raw(raw, configured_ema_rates)
    if bundle["step"] == 0:
        raise RunSafetyError(
            "model000000.pt cannot safely restore optimizer/EMA with the legacy TrainLoop; "
            "resume from step >= 1"
        )
    if not bundle["complete"]:
        raise FileNotFoundError(
            "incomplete resume checkpoint bundle; missing: " + ", ".join(bundle["missing"])
        )
    return bundle


def next_checkpoint_segment(run_dir: os.PathLike[str] | str) -> tuple[int, Path]:
    root = artifact_dirs(run_dir)["checkpoints"]
    root.mkdir(parents=True, exist_ok=True)
    existing = []
    for candidate in root.glob("segment_*"):
        if candidate.is_dir():
            try:
                existing.append(int(candidate.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    number = max(existing, default=-1) + 1
    segment = root / f"segment_{number:03d}"
    segment.mkdir(parents=False, exist_ok=False)
    return number, segment


def checkpoint_metadata(bundle: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(bundle)
    files = {}
    for key in ("raw_checkpoint_path", "ema_checkpoint_path", "optimizer_checkpoint_path"):
        path = Path(str(bundle[key]))
        files[key] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    result["files"] = files
    return result


def build_model_from_config(config: Mapping[str, Any], gene_list: Sequence[str], timesteps: int, device: Any) -> Any:
    """Import the suite-local model factory lazily to keep CLI dry-runs light."""

    if str(SUITE_ROOT) not in sys.path:
        sys.path.insert(0, str(SUITE_ROOT))
    from models import build_model_from_config as model_factory

    return model_factory(dict(config), list(gene_list), timesteps, device)
