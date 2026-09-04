"""Suite-local configuration, path, run, and checkpoint helpers.

All generated artifacts are constrained to this experiment's ``runs/``
directory.  Input paths may be absolute or repository-relative; legacy remote
paths are passed through :mod:`local_paths` so the repository's established
local fallback continues to work.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
CONFIG_ROOT = SUITE_ROOT / "configs"
RUNS_ROOT = SUITE_ROOT / "runs"
PROTECTED_MANIFEST = SUITE_ROOT / "protected_core_sha256.json"

MODEL_CONFIGS = (
    "model_a_stationary_qd_dense.json",
    "model_b_free_affine_dense.json",
)
MODEL_ORDER = ("stationary_qd", "free_affine")

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RAW_CHECKPOINT = re.compile(r"^model(?P<step>[0-9]+)\.pt$")
_EMA_CHECKPOINT = re.compile(
    r"^ema_(?P<rate>[^_]+)_(?P<step>[0-9]+)\.pt$"
)


class ConfigurationError(ValueError):
    """Raised when a CLI/config value is inconsistent or unsafe."""


def read_json(path: os.PathLike[str] | str) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: os.PathLike[str] | str, payload: Any) -> Path:
    """Atomically write one UTF-8 JSON artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)
    return destination


def file_sha256(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gene_order_sha256(genes: Sequence[str]) -> str:
    """Hash ordered names with explicit indices and separators."""

    digest = hashlib.sha256()
    for index, gene in enumerate(genes):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(gene).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings; lists and scalar values are replaced."""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def safe_component(value: Any, label: str) -> str:
    component = str(value).strip()
    if (
        not component
        or component in {".", ".."}
        or Path(component).name != component
        or _SAFE_COMPONENT.fullmatch(component) is None
    ):
        raise ConfigurationError(f"unsafe {label}: {value!r}")
    return component


def ensure_within(path: os.PathLike[str] | str, root: os.PathLike[str] | str) -> Path:
    candidate = Path(path).expanduser().resolve()
    allowed = Path(root).expanduser().resolve()
    try:
        candidate.relative_to(allowed)
    except ValueError as exc:
        raise ConfigurationError(
            f"path must remain below {allowed}: {candidate}"
        ) from exc
    return candidate


def resolve_repository_path(value: os.PathLike[str] | str) -> Path:
    """Resolve an input path using the established repository fallback."""

    text = str(value)
    if not text:
        raise ConfigurationError("input path must not be empty")
    # Imported lazily so common.py remains usable for config dry-runs in a
    # minimal Python environment.
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from local_paths import resolve_path

    resolved = Path(resolve_path(text)).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def resolve_config_paths(config: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve read-only data/edge paths relative to the repository."""

    resolved = copy.deepcopy(dict(config))
    for key in ("data_dir", "edge_tsv_path"):
        if resolved.get(key):
            resolved[key] = str(resolve_repository_path(resolved[key]))
    return resolved


def _config_candidate(value: os.PathLike[str] | str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.is_file():
            return repo_candidate.resolve()
    if len(candidate.parts) == 1:
        name = candidate.name
        if not name.endswith(".json"):
            name += ".json"
        suite_candidate = CONFIG_ROOT / name
        if suite_candidate.is_file():
            return suite_candidate.resolve()
    raise FileNotFoundError(f"experiment config not found: {value}")


def load_experiment_config(
    experiment_or_path: os.PathLike[str] | str,
    *,
    base_path: Optional[os.PathLike[str] | str] = None,
    set_values: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Merge ``base.json`` with a model config and resolve input paths."""

    override_path = _config_candidate(experiment_or_path)
    actual_base = (
        _config_candidate(base_path)
        if base_path is not None
        else (CONFIG_ROOT / "base.json").resolve()
    )
    if not actual_base.is_file():
        raise FileNotFoundError(f"base config not found: {actual_base}")
    base = read_json(actual_base)
    override = read_json(override_path)
    if not isinstance(base, Mapping) or not isinstance(override, Mapping):
        raise ConfigurationError("base and experiment configs must be JSON objects")
    merged = deep_merge(base, override)
    merged = apply_set_overrides(merged, set_values)
    merged = resolve_config_paths(merged)
    validate_common_config(merged)
    provenance = {
        "base_config_path": str(actual_base),
        "base_config_sha256": file_sha256(actual_base),
        "experiment_config_path": str(override_path),
        "experiment_config_sha256": file_sha256(override_path),
    }
    return merged, provenance


def apply_set_overrides(
    config: Mapping[str, Any], values: Iterable[str]
) -> dict[str, Any]:
    """Apply repeatable ``KEY=JSON_VALUE`` command-line overrides."""

    result = copy.deepcopy(dict(config))
    for item in values:
        if "=" not in item:
            raise ConfigurationError(f"--set expects KEY=VALUE, got {item!r}")
        dotted_key, raw = item.split("=", 1)
        keys = [part.strip() for part in dotted_key.split(".")]
        if not keys or any(not key for key in keys):
            raise ConfigurationError(f"invalid --set key: {dotted_key!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = result
        for key in keys[:-1]:
            child = target.get(key)
            if child is None:
                child = {}
                target[key] = child
            if not isinstance(child, dict):
                raise ConfigurationError(
                    f"cannot set a child below non-object key: {dotted_key!r}"
                )
            target = child
        target[keys[-1]] = value
    return result


def validate_common_config(config: Mapping[str, Any]) -> None:
    """Validate only entrypoint-level requirements; model policy stays local."""

    missing = [
        key
        for key in ("experiment", "forward_model", "data_dir", "batch_size")
        if config.get(key) in (None, "")
    ]
    if missing:
        raise ConfigurationError(
            "missing required config keys: " + ", ".join(missing)
        )
    safe_component(config["experiment"], "experiment")
    if str(config["forward_model"]).lower() not in MODEL_ORDER:
        raise ConfigurationError(
            f"forward_model must be one of {MODEL_ORDER}, got "
            f"{config['forward_model']!r}"
        )
    for key in ("batch_size", "diffusion_steps", "log_interval", "save_interval"):
        if key in config and int(config[key]) <= 0:
            raise ConfigurationError(f"{key} must be positive")
    total_steps = int(config.get("lr_anneal_steps", config.get("total_steps", 0)))
    if total_steps <= 0:
        raise ConfigurationError("lr_anneal_steps or total_steps must be positive")


def new_batch_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_directory(experiment: str, batch_id: str) -> Path:
    return (
        RUNS_ROOT
        / safe_component(experiment, "experiment")
        / safe_component(batch_id, "batch id")
    )


def validate_run_directory(path: os.PathLike[str] | str) -> Path:
    candidate = ensure_within(path, RUNS_ROOT)
    relative = candidate.relative_to(RUNS_ROOT.resolve())
    if len(relative.parts) != 2:
        raise ConfigurationError(
            "run directory must be runs/<experiment>/<batch-id>: "
            f"{candidate}"
        )
    safe_component(relative.parts[0], "experiment")
    safe_component(relative.parts[1], "batch id")
    return candidate


def next_segment_directory(run_path: os.PathLike[str] | str) -> Path:
    root = validate_run_directory(run_path) / "segments"
    existing = []
    if root.is_dir():
        for candidate in root.glob("segment_[0-9][0-9][0-9]"):
            if candidate.is_dir():
                existing.append(int(candidate.name.rsplit("_", 1)[-1]))
    return root / f"segment_{max(existing, default=-1) + 1:03d}"


def checkpoint_step(path: os.PathLike[str] | str) -> Optional[int]:
    match = _RAW_CHECKPOINT.fullmatch(Path(path).name)
    return int(match.group("step")) if match is not None else None


def ema_checkpoint_metadata(
    path: os.PathLike[str] | str,
) -> Optional[tuple[float, str, int]]:
    match = _EMA_CHECKPOINT.fullmatch(Path(path).name)
    if match is None:
        return None
    label = match.group("rate")
    try:
        rate = float(label)
    except ValueError:
        return None
    return rate, label, int(match.group("step"))


def requested_config(run_path: os.PathLike[str] | str) -> dict[str, Any]:
    run = validate_run_directory(run_path)
    path = run / "requested_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"requested config is missing: {path}")
    payload = read_json(path)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping):
        raise ConfigurationError(f"invalid requested config payload: {path}")
    return dict(payload["config"])


def all_raw_checkpoints(run_path: os.PathLike[str] | str) -> list[Path]:
    """Return raw checkpoints ordered by numeric step and modification time."""

    run = validate_run_directory(run_path)
    candidates = []
    for raw in run.glob("segments/segment_*/model/model*.pt"):
        step = checkpoint_step(raw)
        if raw.is_file() and step is not None:
            candidates.append(raw.resolve())
    return sorted(
        candidates,
        key=lambda path: (checkpoint_step(path), path.stat().st_mtime_ns, str(path)),
    )


def resolve_sampling_checkpoint(
    run_path: os.PathLike[str] | str,
    ema_rate: str | float | None = None,
) -> tuple[Path, int, str]:
    """Choose the highest-step EMA checkpoint, never a raw checkpoint."""

    run = validate_run_directory(run_path)
    config = requested_config(run)
    configured = tuple(
        float(token.strip())
        for token in str(config.get("ema_rate", "")).split(",")
        if token.strip()
    )
    if ema_rate in (None, ""):
        if not configured:
            raise ConfigurationError("ema_rate is absent from requested config")
        selected_rate = max(configured)
    else:
        selected_rate = float(ema_rate)
        if configured and selected_rate not in configured:
            raise ConfigurationError(
                f"EMA rate {selected_rate} was not configured; available={configured}"
            )
    matches = []
    for path in run.glob("segments/segment_*/model/ema_*.pt"):
        metadata = ema_checkpoint_metadata(path)
        if not path.is_file() or metadata is None:
            continue
        rate, label, step = metadata
        if rate == selected_rate and path.with_name(f"model{step:06d}.pt").is_file():
            matches.append((step, path.stat().st_mtime_ns, path.resolve(), label))
    if not matches:
        raise FileNotFoundError(
            f"no EMA checkpoint for rate {selected_rate} below {run}"
        )
    step, _mtime, path, label = max(matches, key=lambda item: (item[0], item[1]))
    return path, int(step), str(label)


def expected_ema_checkpoint_names(
    run_path: os.PathLike[str] | str, step: int
) -> tuple[str, ...]:
    """Return every EMA peer required by the run's recorded configuration."""

    run = validate_run_directory(run_path)
    requested = run / "requested_config.json"
    if not requested.is_file():
        return ()
    payload = read_json(requested)
    if not isinstance(payload, Mapping):
        raise ConfigurationError(f"invalid requested config payload: {requested}")
    config = payload.get("config", payload)
    if not isinstance(config, Mapping):
        raise ConfigurationError(f"invalid config object in {requested}")
    raw_rates = config.get("ema_rate", "")
    if isinstance(raw_rates, (int, float)):
        rate_tokens = (raw_rates,)
    else:
        rate_tokens = tuple(
            token.strip() for token in str(raw_rates).split(",") if token.strip()
        )
    if not rate_tokens:
        raise ConfigurationError(f"ema_rate is empty in {requested}")
    try:
        labels = tuple(str(float(token)) for token in rate_tokens)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"invalid ema_rate in {requested}: {raw_rates!r}") from exc
    return tuple(f"ema_{label}_{int(step):06d}.pt" for label in labels)


def _has_complete_ema_peers(raw: Path, run: Path, step: int) -> bool:
    expected = expected_ema_checkpoint_names(run, step)
    if expected:
        return all(raw.with_name(name).is_file() for name in expected)
    # Legacy/incomplete run metadata cannot prove which rates were requested;
    # retain the prior conservative minimum of requiring at least one EMA peer.
    return bool(tuple(raw.parent.glob(f"ema_*_{step:06d}.pt")))


def complete_raw_checkpoints(run_path: os.PathLike[str] | str) -> list[Path]:
    """Return raw checkpoints having optimizer and every configured EMA peer."""

    run = validate_run_directory(run_path)
    candidates = []
    for raw in run.glob("segments/segment_*/model/model*.pt"):
        step = checkpoint_step(raw)
        # Core TrainLoop treats parsed step zero as a fresh run and therefore
        # would not restore optimizer/EMA state from a model000000.pt bundle.
        if step is None or step == 0:
            continue
        optimizer = raw.with_name(f"opt{step:06d}.pt")
        if optimizer.is_file() and _has_complete_ema_peers(raw, run, step):
            candidates.append(raw.resolve())
    return sorted(
        candidates,
        key=lambda path: (checkpoint_step(path) or -1, path.stat().st_mtime_ns),
    )


def latest_raw_checkpoint(run_path: os.PathLike[str] | str) -> Optional[Path]:
    checkpoints = complete_raw_checkpoints(run_path)
    return checkpoints[-1] if checkpoints else None


def resolve_resume_checkpoint(
    run_path: os.PathLike[str] | str, request: str
) -> str:
    """Resolve ``''``, ``auto``, or an in-run raw checkpoint path."""

    run = validate_run_directory(run_path)
    latest = latest_raw_checkpoint(run)
    if not request:
        if latest is not None:
            raise RuntimeError(
                "run already contains a complete checkpoint; use --resume auto "
                "or choose a new run directory"
            )
        return ""
    if request == "auto":
        return "" if latest is None else str(latest)
    candidate = Path(request).expanduser()
    if not candidate.is_absolute():
        repo_candidate = REPO_ROOT / candidate
        candidate = repo_candidate if repo_candidate.exists() else candidate
    candidate = candidate.resolve()
    ensure_within(candidate, run)
    if (
        not candidate.is_file()
        or checkpoint_step(candidate) is None
        or checkpoint_step(candidate) == 0
    ):
        raise FileNotFoundError(f"raw model checkpoint not found: {candidate}")
    step = checkpoint_step(candidate)
    if not candidate.with_name(f"opt{step:06d}.pt").is_file():
        raise FileNotFoundError(f"optimizer peer is missing for {candidate}")
    if not _has_complete_ema_peers(candidate, run, step):
        expected = expected_ema_checkpoint_names(run, step)
        raise FileNotFoundError(
            f"one or more EMA peers are missing for {candidate}; expected={expected}"
        )
    return str(candidate)


def model_metadata(model: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    process = model.forward_process
    metadata = {
        "experiment": str(config["experiment"]),
        "wrapper_class": type(model).__name__,
        "denoiser_class": type(model.denoiser).__name__,
        "forward_process_class": type(process).__name__,
        "forward_model": str(config["forward_model"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "forward_parameter_count": sum(
            parameter.numel() for parameter in process.parameters()
        ),
        "forward_dtype": str(next(process.parameters()).dtype),
        "loss_mode": str(config.get("loss_mode", "paper_elbo")),
        "training_only": False,
        "generation_sampler": "custom_reverse_sde_euler_maruyama",
        "standard_ddpm_sampler_supported": False,
    }
    if str(config["forward_model"]) == "stationary_qd":
        metadata["stationary_covariance_evaluation"] = {
            "default": "I - Phi Phi^T",
            "cancellation_regime": "adaptive integral Taylor series",
            "selection_rule": "physical_time <= 8 * dimension * dtype_epsilon",
            "changes_declared_sde": False,
            "adds_jitter_or_clipping": False,
        }
    return metadata


__all__ = [name for name in globals() if not name.startswith("_")]
