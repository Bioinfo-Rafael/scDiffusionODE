"""Run, data, checkpoint, and metric helpers shared by all analyses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from models.factory import build_experiment_components
from scripts.common import (
    SUITE_ROOT,
    gene_order_sha256,
    requested_config,
    resolve_sampling_checkpoint,
    validate_run_directory,
)


def configure_runtime_caches() -> None:
    root = SUITE_ROOT / "runs" / ".runtime_cache"
    numba = root / "numba"
    matplotlib = root / "matplotlib"
    numba.mkdir(parents=True, exist_ok=True)
    matplotlib.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib))


def select_device(name: str) -> torch.device:
    selected = str(name).lower()
    if selected == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if selected == "mps":
        raise ValueError("MPS is not supported by the exact dense matrix path")
    if selected not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(selected)


def gene_names_from_adata(adata, gene_column: str = "gene_name") -> list[str]:
    if gene_column not in adata.var.columns:
        raise KeyError(f"AnnData .var lacks gene column {gene_column!r}")
    genes = [str(value) for value in adata.var[gene_column].tolist()]
    if len(genes) != int(adata.n_vars) or len(set(genes)) != len(genes):
        raise ValueError("ordered gene names must be unique and one-to-one")
    return genes


def dense_float32(matrix) -> np.ndarray:
    import scipy.sparse as sp

    result = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    result = np.asarray(result, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("matrix must be finite and two-dimensional")
    return result


def training_matrix(adata, config: Mapping[str, Any]) -> np.ndarray:
    layer = config.get("data_layer", config.get("layer"))
    source = adata.layers[layer] if layer else adata.X
    return dense_float32(source)


def load_run_data(run_dir):
    configure_runtime_caches()
    import scanpy as sc

    run = validate_run_directory(run_dir)
    config = requested_config(run)
    data_path = Path(str(config["data_dir"])).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"configured AnnData does not exist: {data_path}")
    adata = sc.read_h5ad(data_path)
    genes = gene_names_from_adata(
        adata, str(config.get("gene_column", "gene_name"))
    )
    return run, config, data_path, adata, genes


def load_state_dict(path: str | Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = Path(path).expanduser().resolve()
    value = torch.load(checkpoint, map_location=device)
    if not isinstance(value, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    state = value.get("model_state_dict", value)
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint state is invalid: {checkpoint}")
    return state


def load_model(
    config: Mapping[str, Any],
    genes: Sequence[str],
    checkpoint: str | Path,
    device: torch.device,
):
    components = build_experiment_components(config, genes, device)
    state = load_state_dict(checkpoint, device)
    components.model.load_state_dict(state, strict=True)
    components.model.eval()
    return components


def load_final_ema(run_dir, *, device: str = "auto", ema_rate=None):
    run, config, data_path, adata, genes = load_run_data(run_dir)
    selected_device = select_device(device)
    checkpoint, step, rate = resolve_sampling_checkpoint(run, ema_rate)
    components = load_model(config, genes, checkpoint, selected_device)
    return {
        "run_dir": run,
        "config": config,
        "data_path": data_path,
        "adata": adata,
        "genes": genes,
        "gene_order_sha256": gene_order_sha256(genes),
        "device": selected_device,
        "checkpoint": checkpoint,
        "checkpoint_step": step,
        "ema_rate": rate,
        "components": components,
    }


def metadata_matches_final_ema(
    metadata: Mapping[str, Any], run_dir, *, ema_rate=None
) -> bool:
    """Return true only when an analysis names the current selected EMA."""

    try:
        checkpoint, step, rate = resolve_sampling_checkpoint(run_dir, ema_rate)
        recorded = Path(str(metadata.get("checkpoint", ""))).expanduser().resolve()
    except (FileNotFoundError, ValueError, TypeError):
        return False
    return (
        metadata.get("status") == "completed"
        and recorded == checkpoint.resolve()
        and int(metadata.get("checkpoint_step", -1)) == int(step)
        and str(metadata.get("ema_rate", "")) == str(rate)
    )


def timestep_grid(num_timesteps: int, step: int = 20) -> list[int]:
    num_timesteps, step = int(num_timesteps), int(step)
    if num_timesteps <= 0 or step <= 0:
        raise ValueError("num_timesteps and step must be positive")
    values = list(range(0, num_timesteps, step))
    if values[-1] != num_timesteps - 1:
        values.append(num_timesteps - 1)
    return values


def sample_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Historical per-cell Pearson correlation across genes."""

    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(dim=1) / (
        a.norm(dim=1) * b.norm(dim=1)
    ).clamp_min(eps)


def sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b).reshape(a.shape[0], -1).square().mean(dim=1)


def sample_norm(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a.reshape(a.shape[0], -1), dim=1)


def summarize(values: np.ndarray | torch.Tensor) -> dict[str, float]:
    array = (
        values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    )
    array = np.asarray(array, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise FloatingPointError("metric values must be non-empty and finite")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "median": float(np.median(array)),
    }


__all__ = [
    "configure_runtime_caches",
    "dense_float32",
    "gene_names_from_adata",
    "load_final_ema",
    "load_model",
    "load_run_data",
    "metadata_matches_final_ema",
    "sample_corr",
    "sample_mse",
    "sample_norm",
    "select_device",
    "summarize",
    "timestep_grid",
    "training_matrix",
]
