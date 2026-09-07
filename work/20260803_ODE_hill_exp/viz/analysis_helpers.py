"""Run restoration, numerical preflight, metrics, and plots for this suite.

This module deliberately lives below ``work/20260803_ODE_hill_exp``.  It may
import stable helpers from the repository, but it never modifies an existing
training, sampling, model, or visualization implementation.

The central policy is intentionally strict: an input, checkpoint parameter,
model output, metric, or embedding containing NaN/Inf aborts analysis.  Values
are never repaired with ``nan_to_num``.  This makes a failed preflight visible
instead of turning a numerical failure into a plausible-looking figure.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch


SAMPLE_KEYS = ("cell_gen", "samples", "generated", "x")
REGULARIZATION_MARKERS = (
    "reg", "penalty", "off_mask", "offmask", "sparse", "entropy", "ratio",
)
LABEL_PRIORITY = ("Superclass", "Subclass", "ClassAnn", "celltype", "final_annotation")


@dataclass(frozen=True)
class AnalysisMetadata:
    """Metadata repeated in every plot title."""

    ode_type: str
    model_family: str
    checkpoint: str
    analyzed_at: str

    def title(self, subject: str) -> str:
        return (
            f"{subject}\nODE={self.ode_type} | family={self.model_family} | "
            f"checkpoint={self.checkpoint} | datetime={self.analyzed_at}"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _resolve_path(value: str | os.PathLike[str], run_dir: Path) -> Path:
    """Resolve absolute, run-relative, repository-relative, and legacy paths."""

    raw = Path(value).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend((run_dir / raw, run_dir.parents[3] / raw))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    try:
        from local_paths import resolve_path as resolve_legacy_path

        resolved = Path(resolve_legacy_path(str(value))).expanduser()
        if resolved.exists():
            return resolved.resolve()
    except Exception:
        pass
    return raw.resolve() if raw.is_absolute() else (run_dir / raw).resolve()


def _first_existing(values: Iterable[Any], run_dir: Path) -> Path | None:
    for value in values:
        if value in (None, ""):
            continue
        candidate = _resolve_path(str(value), run_dir)
        if candidate.is_file():
            return candidate
    return None


def _newest(paths: Iterable[Path]) -> Path | None:
    existing = [p for p in paths if p.is_file()]
    return max(existing, key=lambda p: (p.stat().st_mtime_ns, str(p))) if existing else None


def discover_run_inputs(
    run_dir: Path,
    *,
    checkpoint_override: str = "",
    sample_override: str = "",
) -> dict[str, Any]:
    """Restore every analysis input from one independent run directory."""

    run_dir = run_dir.resolve()
    config_path = run_dir / "exp_config.json"
    manifest_path = run_dir / "manifest.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing run config: {config_path}")
    config = _read_json(config_path)
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    for key in ("experiment", "model_family", "ode_type"):
        if not config.get(key):
            raise KeyError(f"exp_config.json is missing required key '{key}'")

    checkpoint = _first_existing(
        (
            checkpoint_override,
            manifest.get("checkpoint_path"),
            manifest.get("ema_checkpoint_path"),
            manifest.get("raw_checkpoint_path"),
            config.get("checkpoint_path"),
        ),
        run_dir,
    )
    if checkpoint is None:
        checkpoint = _newest(
            list(run_dir.glob("train/**/ema_*.pt"))
            + list(run_dir.glob("train/**/model*.pt"))
            + list(run_dir.glob("**/ema_*.pt"))
        )
    if checkpoint is None:
        raise FileNotFoundError(
            "no checkpoint found in manifest or run directory; pass --checkpoint"
        )

    sample_path = _first_existing(
        (sample_override, manifest.get("sample_path"), config.get("sample_path")), run_dir
    )
    if sample_path is None:
        sample_path = _newest(run_dir.glob("samples/*.npz"))
    if sample_path is None:
        raise FileNotFoundError(
            "no generated sample found in manifest or run_dir/samples; pass --sample-path"
        )

    data_value = config.get("data_dir") or manifest.get("data_dir")
    if not data_value:
        raise KeyError("exp_config.json is missing data_dir")
    data_path = _resolve_path(str(data_value), run_dir)
    if not data_path.is_file():
        raise FileNotFoundError(f"data file does not exist: {data_path}")

    loss_path = _first_existing(
        (manifest.get("loss_path"), config.get("loss_path")), run_dir
    )
    if loss_path is None:
        preferred = [
            run_dir / "train" / "loss_details.csv",
            run_dir / "train" / "checkpoints" / "loss_details.csv",
        ]
        loss_path = _first_existing(preferred, run_dir)
    if loss_path is None:
        loss_path = _newest(run_dir.glob("train/**/loss_details.csv"))
    if loss_path is None:
        raise FileNotFoundError("training loss_details.csv was not found below run_dir/train")
    loss_paths = sorted(
        run_dir.glob("train/**/loss_details.csv"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
    )
    if loss_path not in loss_paths:
        loss_paths.append(loss_path)

    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "manifest_path": manifest_path,
        "config": config,
        "manifest": manifest,
        "checkpoint": checkpoint,
        "sample_path": sample_path,
        "data_path": data_path,
        "loss_path": loss_path,
        "loss_paths": loss_paths,
    }


def assert_finite(name: str, value: Any) -> None:
    """Fail loudly on NaN/Inf; never sanitize an array for plotting."""

    if torch.is_tensor(value):
        finite = torch.isfinite(value)
        if not bool(finite.all().item()):
            bad = int((~finite).sum().item())
            total = int(value.numel())
            raise FloatingPointError(f"preflight failed: {name} has {bad}/{total} non-finite values")
        return
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        return
    finite = np.isfinite(array)
    if not bool(finite.all()):
        bad = int((~finite).sum())
        raise FloatingPointError(
            f"preflight failed: {name} has {bad}/{array.size} non-finite values"
        )


def _dense(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    elif hasattr(value, "todense"):
        value = value.todense()
    return np.asarray(value)


def load_real_subset(
    data_path: Path,
    *,
    layer: str | None,
    max_cells: int,
    seed: int,
) -> tuple[np.ndarray, list[str], np.ndarray, str]:
    """Read only the selected cells from the (large) h5ad."""

    adata = sc.read_h5ad(data_path, backed="r")
    try:
        n_obs = int(adata.n_obs)
        take = n_obs if max_cells <= 0 else min(n_obs, int(max_cells))
        rng = np.random.default_rng(seed)
        indices = (
            np.arange(n_obs, dtype=np.int64)
            if take == n_obs
            else np.sort(rng.choice(n_obs, take, replace=False))
        )
        if layer:
            if layer not in adata.layers:
                raise KeyError(f"configured input layer '{layer}' is absent from {data_path}")
            source = adata.layers[layer][indices]
        else:
            source = adata.X[indices]
        x = _dense(source).astype(np.float32, copy=False)
        if "gene_name" in adata.var:
            genes = [str(v) for v in adata.var["gene_name"].tolist()]
        else:
            genes = [str(v) for v in adata.var_names.tolist()]
        if len(genes) != int(adata.n_vars) or len(set(genes)) != len(genes):
            raise ValueError("gene_name must be a one-to-one list matching AnnData variables")
        label_col = next((c for c in LABEL_PRIORITY if c in adata.obs), "")
        labels = (
            np.asarray(adata.obs.iloc[indices][label_col].astype(str).tolist(), dtype=object)
            if label_col
            else np.repeat("real", len(indices)).astype(object)
        )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    assert_finite("real data subset", x)
    return x, genes, labels, label_col


def load_generated_subset(sample_path: Path, *, max_cells: int, seed: int) -> np.ndarray:
    with np.load(sample_path, allow_pickle=False) as archive:
        key = next((candidate for candidate in SAMPLE_KEYS if candidate in archive.files), None)
        if key is None and len(archive.files) == 1:
            key = archive.files[0]
        if key is None:
            raise KeyError(
                f"no generated array key {SAMPLE_KEYS} in {sample_path}; keys={archive.files}"
            )
        generated = np.asarray(archive[key], dtype=np.float32)
    if generated.ndim != 2:
        raise ValueError(f"generated samples must be a 2-D matrix, got {generated.shape}")
    if max_cells > 0 and len(generated) > max_cells:
        rng = np.random.default_rng(seed + 1)
        generated = generated[np.sort(rng.choice(len(generated), max_cells, replace=False))]
    assert_finite("generated samples", generated)
    return generated


def load_loss_table(loss_paths: Sequence[Path] | Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    paths = [loss_paths] if isinstance(loss_paths, Path) else list(loss_paths)
    if not paths:
        raise ValueError("no loss tables were provided")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True, sort=False)
    if "step" in frame.columns:
        frame = frame.sort_values("step").drop_duplicates(subset=["step"], keep="last")
    if frame.empty:
        raise ValueError(f"loss table is empty: {paths}")
    numeric = list(frame.select_dtypes(include=[np.number]).columns)
    if not numeric:
        raise ValueError(f"loss table has no numeric columns: {paths}")
    for column in numeric:
        assert_finite(f"loss column '{column}'", frame[column].to_numpy())
    x_candidates = [c for c in numeric if c.lower() in ("step", "samples", "iteration", "iter")]
    x_column = x_candidates[0] if x_candidates else numeric[0]
    values = [c for c in numeric if c != x_column]
    regularizers = [
        c for c in values if any(marker in c.lower() for marker in REGULARIZATION_MARKERS)
    ]
    return frame, [x_column, *values], regularizers


def build_diffusion(config: Mapping[str, Any]):
    from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

    args = model_and_diffusion_defaults()
    for key in args:
        if key in config:
            args[key] = config[key]
    _, diffusion = create_model_and_diffusion(**args)
    return diffusion


def load_model(
    config: Mapping[str, Any],
    genes: Sequence[str],
    diffusion: Any,
    checkpoint: Path,
    device: torch.device,
):
    from guided_diffusion import dist_util
    from ODE.ode_20260609_mathmlp import clean_state_dict
    from models.factory import build_model_from_config

    model = build_model_from_config(config, list(genes), diffusion.num_timesteps, device)
    state = dist_util.load_state_dict(str(checkpoint), map_location="cpu")
    state = clean_state_dict(state)
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint did not contain a state_dict mapping: {checkpoint}")
    for name, tensor in state.items():
        if torch.is_tensor(tensor):
            assert_finite(f"checkpoint tensor '{name}'", tensor)
    # Strict loading is intentional: analysis must never run with silently initialized parameters.
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if torch.is_tensor(tensor):
            assert_finite(f"restored model tensor '{name}'", tensor)
    return model


def parse_t_values(value: str | Sequence[int] | None, num_timesteps: int) -> list[int]:
    if value is None or value == "":
        values = [0, (num_timesteps - 1) // 2, num_timesteps - 1]
    elif isinstance(value, str):
        values = [int(v.strip()) for v in value.split(",") if v.strip()]
    else:
        values = [int(v) for v in value]
    values = sorted(set(values))
    if not values:
        raise ValueError("--t-values must contain at least one timestep")
    bad = [v for v in values if not 0 <= v < num_timesteps]
    if bad:
        raise ValueError(f"timesteps outside [0, {num_timesteps - 1}]: {bad}")
    return values


def _checkpoint_label(path: Path) -> str:
    return path.stem


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "checkpoint"


def _save_figure(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_loss(
    frame: pd.DataFrame,
    numeric: Sequence[str],
    regularizers: Sequence[str],
    output_dir: Path,
    meta: AnalysisMetadata,
) -> list[str]:
    x_column = numeric[0]
    values = list(numeric[1:])
    if not values:
        raise ValueError("loss table has no value columns after the step column")
    ncols = min(3, max(1, len(values)))
    nrows = int(math.ceil(len(values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.7 * nrows), squeeze=False)
    for ax, column in zip(axes.ravel(), values):
        ax.plot(frame[x_column], frame[column], lw=1.25)
        ax.set(xlabel=x_column, ylabel=column, title=column)
        ax.grid(True, alpha=0.25)
    for ax in axes.ravel()[len(values):]:
        ax.set_visible(False)
    fig.suptitle(meta.title("Training loss and logged terms"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    created = [_save_figure(fig, output_dir / "training_loss_all_terms.png")]

    if regularizers:
        fig, ax = plt.subplots(figsize=(9, 5))
        for column in regularizers:
            ax.plot(frame[x_column], frame[column], label=column, lw=1.3)
        ax.set(xlabel=x_column, ylabel="regularization value")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(meta.title("Every logged regularization loss"), fontsize=10)
        fig.tight_layout()
        created.append(_save_figure(fig, output_dir / "regularization_losses.png"))
    frame.to_csv(output_dir / "training_loss_table.csv", index=False)
    return created


def _plot_real_generated_umap(
    real: np.ndarray,
    generated: np.ndarray,
    real_labels: np.ndarray,
    label_col: str,
    output_dir: Path,
    meta: AnalysisMetadata,
    seed: int,
) -> tuple[list[str], str]:
    if real.shape[1] != generated.shape[1]:
        raise ValueError(
            f"real/generated feature mismatch: {real.shape[1]} vs {generated.shape[1]}"
        )
    combined = np.concatenate((real, generated), axis=0).astype(np.float32, copy=False)
    assert_finite("combined real/generated data", combined)
    n = len(combined)
    if n < 3:
        raise ValueError("at least three combined cells are required for an embedding")
    adata = sc.AnnData(combined)
    if n >= 5:
        n_comps = max(2, min(50, combined.shape[1] - 1, n - 2))
        sc.pp.pca(adata, n_comps=n_comps, random_state=seed)
        neighbors = max(2, min(15, n - 1))
        sc.pp.neighbors(adata, n_neighbors=neighbors, n_pcs=n_comps, random_state=seed)
        sc.tl.umap(adata, min_dist=0.5, random_state=seed)
        embedding = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
        method = "UMAP"
    else:
        # A tiny smoke run cannot fit a neighbor graph reliably.  Keep the fallback explicit.
        from sklearn.decomposition import PCA

        embedding = PCA(n_components=2, random_state=seed).fit_transform(combined)
        method = "PCA fallback (fewer than 5 cells)"
    assert_finite(f"{method} embedding", embedding)
    origin = np.asarray(["real"] * len(real) + ["generated"] * len(generated))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    real_mask = origin == "real"
    axes[0].scatter(embedding[real_mask, 0], embedding[real_mask, 1], s=8, alpha=0.35,
                    c="#7f7f7f", label="real")
    axes[0].scatter(embedding[~real_mask, 0], embedding[~real_mask, 1], s=12, alpha=0.8,
                    c="#d62728", label="generated")
    axes[0].legend(frameon=False)
    axes[0].set_title("Origin")
    if label_col:
        categories = pd.unique(real_labels)
        cmap = plt.get_cmap("tab20")
        for index, category in enumerate(categories[:20]):
            mask = real_labels == category
            axes[1].scatter(embedding[: len(real)][mask, 0], embedding[: len(real)][mask, 1],
                            s=7, alpha=0.45, color=cmap(index % 20), label=str(category))
        axes[1].scatter(embedding[len(real):, 0], embedding[len(real):, 1], s=12,
                        alpha=0.8, color="#d62728", label="generated")
        axes[1].legend(frameon=False, fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
        axes[1].set_title(f"Real {label_col} + generated")
    else:
        axes[1].scatter(embedding[:, 0], embedding[:, 1], s=9, alpha=0.55,
                        c=np.where(real_mask, "#7f7f7f", "#d62728"))
        axes[1].set_title("No real annotation column available")
    for ax in axes:
        ax.set(xlabel=f"{method} 1", ylabel=f"{method} 2")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(meta.title(f"Real versus generated comparison ({method})"), fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = _save_figure(fig, output_dir / "real_vs_generated_umap.png")
    table = pd.DataFrame({
        "embedding_1": embedding[:, 0], "embedding_2": embedding[:, 1], "origin": origin,
    })
    table.to_csv(output_dir / "real_vs_generated_embedding.csv", index=False)
    return [path], method


def _sample_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    return (a * b).sum(dim=1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)


def _sample_norm(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(a.reshape(a.shape[0], -1), dim=1)


def _call_ml(model: Any, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    try:
        return model.ml_model(x, t)
    except TypeError:
        return model.ml_model(x, t, None)


def _ode_weight(model: Any, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Return the *actual* ODE weight used by the hybrid forward policy."""

    if not hasattr(model, "ml_model"):
        return torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype), "ode_only"
    regime_mode = str(getattr(model, "regime_gate_mode", "none")).lower()
    if regime_mode != "none":
        if not hasattr(model, "_regime_ode_weight"):
            raise AttributeError("enabled regime gate has no _regime_ode_weight implementation")
        weight = model._regime_ode_weight(t, x.device, x.dtype)
        return weight.reshape(x.shape[0], -1)[:, :1], "regime_gate"
    if not hasattr(model, "_scheduler"):
        raise AttributeError("hybrid model has neither regime gate nor _scheduler")
    r = model._scheduler(t, device=x.device, dtype=x.dtype)
    r = r.reshape(x.shape[0], -1)[:, :1]
    if bool(getattr(model, "reverse_coef", False)):
        return 1.0 - r, "reverse_scheduler"
    return r, "standard_scheduler"


def _moments(values: Sequence[np.ndarray]) -> tuple[float, float]:
    # Metrics are produced as float32 tensors.  A finite value can still be as
    # large as ~1e38; NumPy's float32 std squares centered values and may then
    # overflow even though every input is finite.  Aggregate in float64 so the
    # finite preflight tests the metric itself rather than an avoidable
    # reduction overflow.  This is a dtype-only change, not clipping/sanitizing.
    array = np.concatenate([
        np.asarray(value, dtype=np.float64).reshape(-1) for value in values
    ])
    assert_finite("metric accumulator", array)
    return float(array.mean()), float(array.std())


@torch.no_grad()
def evaluate_model_io(
    model: Any,
    diffusion: Any,
    real: np.ndarray,
    t_values: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    """Compare denoiser output/branches with known q_sample noise."""

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rows: list[dict[str, Any]] = []
    policy = "ode_only"
    for raw_t in t_values:
        accum: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(real), batch_size):
            x0 = torch.from_numpy(real[start:start + batch_size]).to(device=device, dtype=torch.float32)
            t_raw = torch.full((len(x0),), int(raw_t), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t_raw, noise=noise)
            model_t = diffusion._scale_timesteps(t_raw)
            total = model(x_t, model_t)
            assert_finite(f"model output at t={raw_t}", total)
            fields: dict[str, torch.Tensor] = {
                "model_corr": _sample_corr(total, noise),
                "model_dot_per_dim": (total * noise).reshape(len(total), -1).mean(dim=1),
                "model_cosine": torch.nn.functional.cosine_similarity(
                    total.reshape(len(total), -1), noise.reshape(len(total), -1), dim=1
                ),
                "model_mse": (total - noise).reshape(len(total), -1).pow(2).mean(dim=1),
                "model_norm": _sample_norm(total),
                "noise_norm": _sample_norm(noise),
            }
            if hasattr(model, "ode_model"):
                ode = model.ode_model(x_t, model_t)
                assert_finite(f"ODE branch output at t={raw_t}", ode)
                fields.update({
                    "ode_corr": _sample_corr(ode, noise),
                    "ode_dot_per_dim": (ode * noise).reshape(len(ode), -1).mean(dim=1),
                    "ode_mse": (ode - noise).reshape(len(ode), -1).pow(2).mean(dim=1),
                    "ode_raw_norm": _sample_norm(ode),
                })
                if hasattr(model, "ml_model"):
                    ml = _call_ml(model, x_t, model_t)
                    assert_finite(f"CellUnet branch output at t={raw_t}", ml)
                    weight, policy = _ode_weight(model, model_t, x_t)
                    while weight.dim() < ode.dim():
                        weight = weight.unsqueeze(-1)
                    ode_contrib = weight * ode
                    ml_contrib = (1.0 - weight) * ml
                    ode_norm = _sample_norm(ode)
                    ml_norm = _sample_norm(ml)
                    ode_weighted_norm = _sample_norm(ode_contrib)
                    ml_weighted_norm = _sample_norm(ml_contrib)
                    eps = 1e-12
                    fields.update({
                        "ml_corr": _sample_corr(ml, noise),
                        "ml_dot_per_dim": (ml * noise).reshape(len(ml), -1).mean(dim=1),
                        "ml_mse": (ml - noise).reshape(len(ml), -1).pow(2).mean(dim=1),
                        "ml_raw_norm": ml_norm,
                        "ode_weighted_norm": ode_weighted_norm,
                        "ml_weighted_norm": ml_weighted_norm,
                        "raw_ode_to_ml_norm_ratio": ode_norm / ml_norm.clamp_min(eps),
                        "weighted_ode_to_ml_norm_ratio": (
                            ode_weighted_norm / ml_weighted_norm.clamp_min(eps)
                        ),
                        "ode_weighted_norm_fraction": (
                            ode_weighted_norm /
                            (ode_weighted_norm + ml_weighted_norm).clamp_min(eps)
                        ),
                        "ode_weight": weight.reshape(len(weight), -1)[:, 0],
                    })
            for name, tensor in fields.items():
                assert_finite(f"model I/O metric '{name}' at t={raw_t}", tensor)
                accum.setdefault(name, []).append(tensor.detach().cpu().numpy())
        row: dict[str, Any] = {"t": int(raw_t)}
        for name, chunks in accum.items():
            mean, std = _moments(chunks)
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    for column in frame.select_dtypes(include=[np.number]).columns:
        assert_finite(f"model I/O table '{column}'", frame[column].to_numpy())
    return frame, policy


def _metric_plot(
    table: pd.DataFrame,
    bases: Sequence[str],
    ylabel: str,
    subject: str,
    path: Path,
    meta: AnalysisMetadata,
) -> str | None:
    available = [base for base in bases if f"{base}_mean" in table]
    if not available:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = table["t"].to_numpy()
    for base in available:
        y = table[f"{base}_mean"].to_numpy()
        std = table[f"{base}_std"].to_numpy()
        ax.plot(x, y, marker="o", label=base)
        ax.fill_between(x, y - std, y + std, alpha=0.14)
    ax.set(xlabel="forward diffusion timestep t", ylabel=ylabel)
    ax.set_title(meta.title(subject), fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save_figure(fig, path)


def plot_model_io(
    table: pd.DataFrame,
    output_dir: Path,
    meta: AnalysisMetadata,
    *,
    hybrid: bool,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "model_io_metrics.csv", index=False)
    created: list[str] = []
    specs = [
        (("model_corr", "ode_corr", "ml_corr"), "Pearson correlation", "Model I/O correlation",
         "correlation.png"),
        (("model_dot_per_dim", "ode_dot_per_dim", "ml_dot_per_dim"), "mean dot per dimension",
         "Model I/O dot product", "dot_product.png"),
        (("model_mse", "ode_mse", "ml_mse"), "MSE", "Model I/O mean squared error", "mse.png"),
    ]
    for bases, ylabel, subject, filename in specs:
        path = _metric_plot(table, bases, ylabel, subject, output_dir / filename, meta)
        if path:
            created.append(path)
    if hybrid:
        for bases, ylabel, subject, filename in (
            (("ode_raw_norm", "ml_raw_norm"), "branch L2 norm", "Raw ODE and CellUnet branch norms",
             "branch_raw_norm.png"),
            (("ode_weighted_norm", "ml_weighted_norm"), "weighted branch L2 norm",
             "Weighted ODE and CellUnet branch norms", "branch_weighted_norm.png"),
            (("raw_ode_to_ml_norm_ratio", "weighted_ode_to_ml_norm_ratio"), "ODE / CellUnet norm",
             "ODE-to-CellUnet branch norm ratio (denominator floor=1e-12)", "branch_norm_ratio.png"),
            (("ode_weight",), "ODE coefficient", "Actual hybrid timestep coefficient",
             "hybrid_timestep_coefficient.png"),
        ):
            path = _metric_plot(table, bases, ylabel, subject, output_dir / filename, meta)
            if path:
                created.append(path)
    return created


def _get_tensor_attr(module: Any, names: Sequence[str]) -> torch.Tensor | None:
    for name in names:
        if not hasattr(module, name):
            continue
        value = getattr(module, name)
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if torch.is_tensor(value):
            return value.detach()
    return None


def _field(model: Any) -> Any:
    return getattr(model, "ode_model", model)


def is_lincomb_model(model: Any, config: Mapping[str, Any]) -> bool:
    ode = _field(model)
    return bool(
        hasattr(ode, "coeff_net")
        or bool(getattr(ode, "is_lincomb", False))
        or "lincomb" in str(config.get("model_family", ""))
    )


def _coefficients(ode: Any, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if hasattr(ode, "get_gate_values"):
        value = ode.get_gate_values(x, t)
        if isinstance(value, Mapping):
            for key in ("coefficients", "probabilities", "a"):
                if key in value:
                    return value[key]
        if torch.is_tensor(value):
            return value
    if hasattr(ode, "_coeffs"):
        return ode._coeffs(x, t)
    if hasattr(ode, "effective_coefficients"):
        return ode.effective_coefficients(x, t)
    raise AttributeError("LinComb field exposes no coefficient helper")


def _component_outputs(ode: Any, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    for name in ("component_outputs", "get_component_outputs"):
        if hasattr(ode, name):
            value = getattr(ode, name)(x, t)
            if torch.is_tensor(value):
                return value
    raise AttributeError("LinComb field exposes no component_outputs/get_component_outputs helper")


@torch.no_grad()
def evaluate_lincomb(
    model: Any,
    real: np.ndarray,
    t_values: Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    ode = _field(model)
    rows: list[dict[str, Any]] = []
    if hasattr(ode, "num_components"):
        expected_k = int(ode.num_components)
    elif isinstance(getattr(ode, "K", None), int):
        expected_k = int(ode.K)
    else:
        expected_k = 8
    if expected_k != 8:
        raise ValueError(f"LinComb analysis requires 8 components, restored model reports {expected_k}")
    for t_value in t_values:
        per_component: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(real), batch_size):
            x = torch.from_numpy(real[start:start + batch_size]).to(device=device, dtype=torch.float32)
            t = torch.full((len(x),), float(t_value), device=device, dtype=x.dtype)
            a = _coefficients(ode, x, t)
            outputs = _component_outputs(ode, x, t)
            if a.ndim != 2 or outputs.ndim != 3:
                raise ValueError(
                    f"unexpected LinComb shapes: coefficients={tuple(a.shape)}, "
                    f"components={tuple(outputs.shape)}"
                )
            if a.shape[1] != 8 or outputs.shape[1] != 8:
                raise ValueError("LinComb coefficient/component dimension must be exactly 8")
            assert_finite(f"LinComb coefficients at t={t_value}", a)
            assert_finite(f"LinComb component outputs at t={t_value}", outputs)
            weighted = a.unsqueeze(-1) * outputs
            assert_finite(f"LinComb weighted contributions at t={t_value}", weighted)
            component_norm = torch.linalg.vector_norm(outputs, dim=-1)
            contribution_norm = torch.linalg.vector_norm(weighted, dim=-1)
            share = contribution_norm / contribution_norm.sum(dim=1, keepdim=True).clamp_min(1e-12)
            hybrid_weight, _ = _ode_weight(model, t, x)
            effective = hybrid_weight * a
            for key, value in {
                "a": a,
                "abs_a": a.abs(),
                "component_norm": component_norm,
                "contribution_norm": contribution_norm,
                "contribution_share": share,
                "effective_coefficient": effective,
            }.items():
                assert_finite(f"LinComb {key} at t={t_value}", value)
                per_component.setdefault(key, []).append(value.detach().cpu().numpy())
        stacked = {key: np.concatenate(value, axis=0) for key, value in per_component.items()}
        for component in range(8):
            row: dict[str, Any] = {"t": int(t_value), "component": component}
            for key, array in stacked.items():
                values = array[:, component]
                row[f"{key}_mean"] = float(values.mean())
                row[f"{key}_std"] = float(values.std())
            rows.append(row)
    table = pd.DataFrame(rows)
    for column in table.select_dtypes(include=[np.number]).columns:
        assert_finite(f"LinComb table '{column}'", table[column].to_numpy())
    return table


def plot_lincomb(table: pd.DataFrame, output_dir: Path, meta: AnalysisMetadata) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "lincomb_component_metrics.csv", index=False)
    created: list[str] = []
    specs = (
        ("a", "LinComb coefficient a_k", "All 8 LinComb coefficients", "lincomb_coefficients.png"),
        ("contribution_norm", "||a_k f_k(x)||_2", "Component-specific weighted contribution",
         "component_contribution.png"),
        ("contribution_share", "||a_k f_k|| / sum_l ||a_l f_l||",
         "Component contribution share", "component_contribution_share.png"),
        ("effective_coefficient", "w_ODE(t) a_k",
         "Effective coefficient (hybrid ODE weight times LinComb coefficient)",
         "effective_coefficient.png"),
    )
    for base, ylabel, subject, filename in specs:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for component in range(8):
            subset = table[table["component"] == component].sort_values("t")
            x = subset["t"].to_numpy()
            y = subset[f"{base}_mean"].to_numpy()
            std = subset[f"{base}_std"].to_numpy()
            ax.plot(x, y, marker="o", label=f"k={component}")
            ax.fill_between(x, y - std, y + std, alpha=0.1)
        ax.set(xlabel="timestep t", ylabel=ylabel)
        ax.set_title(meta.title(subject), fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        created.append(_save_figure(fig, output_dir / filename))
    return created


def _as_numpy(name: str, value: torch.Tensor | None) -> np.ndarray:
    if value is None:
        raise AttributeError(f"restored ODE does not expose required semantic parameter '{name}'")
    assert_finite(f"ODE semantic parameter '{name}'", value)
    return value.detach().cpu().numpy().astype(np.float64, copy=False)


def semantic_parameters(model: Any, ode_type: str) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    ode = _field(model)
    ode_type = str(ode_type).lower()
    values: dict[str, np.ndarray] = {}
    if ode_type == "hill_after_linear":
        values["W"] = _as_numpy("W", _get_tensor_attr(ode, ("W", "expert_W")))
        values["K"] = _as_numpy("K", _get_tensor_attr(ode, ("K",)))
        values["V"] = _as_numpy("V", _get_tensor_attr(ode, ("V",)))
        values["delta"] = _as_numpy("delta", _get_tensor_attr(ode, ("delta",)))
    elif ode_type == "racipe":
        values["r"] = _as_numpy("r", _get_tensor_attr(ode, ("r",)))
        values["lambda"] = _as_numpy(
            "lambda", _get_tensor_attr(ode, ("lambdas", "lambda_values", "lambda_"))
        )
        values["K"] = _as_numpy("K", _get_tensor_attr(ode, ("K",)))
        values["G"] = _as_numpy("G", _get_tensor_attr(ode, ("G",)))
        values["delta"] = _as_numpy("delta", _get_tensor_attr(ode, ("delta",)))
    elif ode_type == "exp":
        values["W"] = _as_numpy("W", _get_tensor_attr(ode, ("W", "expert_W")))
        values["bias"] = _as_numpy("bias", _get_tensor_attr(ode, ("b", "bias", "expert_b")))
        values["delta"] = _as_numpy("delta", _get_tensor_attr(ode, ("delta",)))
    else:
        raise ValueError(f"unsupported ode_type for parameter analysis: {ode_type}")
    mask_tensor = _get_tensor_attr(ode, ("mask", "edge_mask"))
    mask = None if mask_tensor is None else mask_tensor.detach().cpu().numpy()
    if mask is not None:
        assert_finite("ODE edge mask", mask)
        mask = np.asarray(mask).astype(bool)
    return values, mask


def _sample_values(array: np.ndarray, limit: int, seed: int) -> np.ndarray:
    flat = np.asarray(array).reshape(-1)
    if len(flat) <= limit:
        return flat
    rng = np.random.default_rng(seed)
    return flat[rng.choice(len(flat), limit, replace=False)]


def _parameter_distribution(
    name: str,
    value: np.ndarray,
    output_dir: Path,
    meta: AnalysisMetadata,
    *,
    note: str = "",
    seed: int = 0,
) -> str:
    sampled = _sample_values(value, 200_000, seed)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sampled, bins=80, alpha=0.85, color="#4c78a8")
    ax.axvline(float(np.mean(value)), color="#d62728", ls="--", label=f"mean={np.mean(value):.4g}")
    ax.set(xlabel=name, ylabel="count (sampled for large tensors)")
    ax.legend(frameon=False)
    subject = f"{name} parameter distribution"
    if note:
        subject += f" — {note}"
    ax.set_title(meta.title(subject), fontsize=9)
    fig.tight_layout()
    return _save_figure(fig, output_dir / f"{name}_distribution.png")


def _matrix_components(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.ndim == 2:
        return matrix[None, ...]
    if matrix.ndim == 3:
        return matrix
    raise ValueError(f"edge matrix must have shape (d,d) or (K,d,d), got {matrix.shape}")


def _edge_component_summary(
    matrix_name: str,
    matrix: np.ndarray,
    mask: np.ndarray | None,
    output_dir: Path,
    meta: AnalysisMetadata,
    *,
    penalty_note: str,
) -> tuple[list[str], pd.DataFrame]:
    components = _matrix_components(matrix)
    rows: list[dict[str, Any]] = []
    for k, component in enumerate(components):
        row: dict[str, Any] = {
            "component": k,
            "mean": float(component.mean()),
            "std": float(component.std()),
            "mean_abs": float(np.abs(component).mean()),
            "l2_norm": float(np.linalg.norm(component)),
            "min": float(component.min()),
            "max": float(component.max()),
        }
        if mask is not None:
            if mask.shape != component.shape:
                raise ValueError(f"mask/matrix shape mismatch: {mask.shape} vs {component.shape}")
            on = np.abs(component[mask])
            off = np.abs(component[~mask])
            row.update({
                "mean_abs_on": float(on.mean()) if len(on) else 0.0,
                "mean_abs_off": float(off.mean()) if len(off) else 0.0,
                "off_on_ratio": (
                    float(off.mean() / max(float(on.mean()), 1e-12)) if len(off) and len(on) else 0.0
                ),
            })
        rows.append(row)
    table = pd.DataFrame(rows)
    for column in table.select_dtypes(include=[np.number]).columns:
        assert_finite(f"{matrix_name} component summary '{column}'", table[column].to_numpy())
    table.to_csv(output_dir / f"{matrix_name}_component_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(table["component"], table["mean_abs"])
    axes[0].set(xlabel="component", ylabel=f"mean |{matrix_name}|")
    axes[1].bar(table["component"], table["l2_norm"])
    axes[1].set(xlabel="component", ylabel=f"||{matrix_name}||_2")
    fig.suptitle(meta.title(f"{matrix_name} component summary — {penalty_note}"), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    created = [_save_figure(fig, output_dir / f"{matrix_name}_component_summary.png")]

    if mask is not None:
        on_all = np.concatenate([c[mask].reshape(-1) for c in components])
        off_all = np.concatenate([c[~mask].reshape(-1) for c in components])
        on_sample = _sample_values(on_all, 100_000, 100)
        off_sample = _sample_values(off_all, 100_000, 101)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(on_sample, bins=80, density=True, alpha=0.55, label="on-mask")
        ax.hist(off_sample, bins=80, density=True, alpha=0.55, label="off-mask")
        ax.set(xlabel=matrix_name, ylabel="density (sampled)")
        ax.legend(frameon=False)
        ax.set_title(meta.title(f"{matrix_name}: on-mask versus off-mask — {penalty_note}"), fontsize=9)
        fig.tight_layout()
        created.append(_save_figure(fig, output_dir / f"{matrix_name}_on_off_distribution.png"))
    return created, table


def _top_edges(
    matrix_name: str,
    matrix: np.ndarray,
    mask: np.ndarray | None,
    genes: Sequence[str],
    output_dir: Path,
    meta: AnalysisMetadata,
    *,
    top_n: int = 40,
    penalty_note: str,
) -> list[str]:
    components = _matrix_components(matrix)
    d = len(genes)
    if components.shape[1:] != (d, d):
        raise ValueError(
            f"{matrix_name} edge dimensions {components.shape[1:]} do not match {d} genes"
        )
    flat_abs = np.abs(components).reshape(-1)
    take = min(int(top_n), len(flat_abs))
    indices = np.argpartition(flat_abs, -take)[-take:]
    indices = indices[np.argsort(flat_abs[indices])[::-1]]
    rows: list[dict[str, Any]] = []
    for flat_index in indices:
        k, target, source = np.unravel_index(flat_index, components.shape)
        rows.append({
            "rank": len(rows) + 1,
            "component": int(k),
            "source_gene": str(genes[source]),
            "target_gene": str(genes[target]),
            "value": float(components[k, target, source]),
            "abs_value": float(abs(components[k, target, source])),
            "mask": ("on" if bool(mask[target, source]) else "off") if mask is not None else "unknown",
        })
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / f"{matrix_name}_top_edges.csv", index=False)
    shown = table.iloc[: min(20, len(table))].iloc[::-1]
    labels = [
        f"k={r.component} {r.source_gene}->{r.target_gene} [{r.mask}]"
        for r in shown.itertuples()
    ]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(shown))))
    colors = np.where(shown["value"].to_numpy() >= 0, "#2ca02c", "#d62728")
    ax.barh(np.arange(len(shown)), shown["value"], color=colors)
    ax.set_yticks(np.arange(len(shown)), labels=labels, fontsize=7)
    ax.set_xlabel(matrix_name)
    ax.set_title(meta.title(f"Top {len(shown)} |{matrix_name}| edges — {penalty_note}"), fontsize=9)
    fig.tight_layout()
    return [_save_figure(fig, output_dir / f"{matrix_name}_top_edges.png")]


def plot_ode_parameters(
    model: Any,
    ode_type: str,
    genes: Sequence[str],
    output_dir: Path,
    meta: AnalysisMetadata,
    *,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    values, mask = semantic_parameters(model, ode_type)
    created: list[str] = []
    summary: dict[str, Any] = {
        "ode_type": ode_type,
        "parameters": {},
        "mask_available": mask is not None,
    }
    for index, (name, value) in enumerate(values.items()):
        assert_finite(f"semantic parameter {name}", value)
        note = ""
        if ode_type == "racipe" and name == "r":
            note = "off-mask penalty target is r=log(lambda), not lambda"
        created.append(
            _parameter_distribution(name, value, output_dir, meta, note=note, seed=seed + index)
        )
        summary["parameters"][name] = {
            "shape": list(value.shape),
            "min": float(value.min()), "max": float(value.max()),
            "mean": float(value.mean()), "std": float(value.std()),
        }

    matrix_name = "r" if ode_type == "racipe" else "W"
    penalty_note = (
        "off-mask penalty target: r=log(lambda)"
        if ode_type == "racipe"
        else f"off-mask penalty target: {matrix_name}"
    )
    matrix = values[matrix_name]
    matrix_figs, component_table = _edge_component_summary(
        matrix_name, matrix, mask, output_dir, meta, penalty_note=penalty_note
    )
    created.extend(matrix_figs)
    created.extend(
        _top_edges(
            matrix_name, matrix, mask, genes, output_dir, meta,
            penalty_note=penalty_note,
        )
    )
    summary["edge_parameter"] = matrix_name
    summary["off_mask_penalty_target"] = matrix_name
    summary["component_summary"] = component_table.to_dict(orient="records")

    if ode_type == "racipe":
        r = values["r"]
        tolerance = 1e-8
        counts = np.asarray([
            np.count_nonzero(r > tolerance),
            np.count_nonzero(r < -tolerance),
            np.count_nonzero(np.abs(r) <= tolerance),
        ], dtype=np.int64)
        proportions = counts / max(int(counts.sum()), 1)
        state_table = pd.DataFrame({
            "state": ["activation (r>0)", "repression (r<0)", "neutral (r=0)"],
            "count": counts, "proportion": proportions,
            "neutral_tolerance": tolerance,
        })
        state_table.to_csv(output_dir / "racipe_edge_state_proportions.csv", index=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.bar(state_table["state"], proportions, color=["#2ca02c", "#d62728", "#7f7f7f"])
        ax.set(ylabel="fraction of edges", ylim=(0, 1))
        ax.tick_params(axis="x", rotation=15)
        ax.set_title(meta.title(
            "RACIPE activation/repression/neutral edges; penalty is on r=log(lambda)"
        ), fontsize=9)
        fig.tight_layout()
        created.append(_save_figure(fig, output_dir / "racipe_edge_state_proportions.png"))
        summary["off_mask_penalty_target"] = "r = log(lambda)"
        summary["edge_states"] = state_table.to_dict(orient="records")

    _write_json(output_dir / "parameter_summary.json", summary)
    return created, summary


def _planned_outputs(config: Mapping[str, Any]) -> dict[str, Any]:
    family = str(config["model_family"])
    lincomb = "lincomb" in family
    hybrid = family != "ode_only_lincomb"
    return {
        "loss": True,
        "real_generated_umap": True,
        "model_io": True,
        "hybrid_branch_and_timestep": hybrid,
        "lincomb_coefficients_components_effective": lincomb,
        "ode_parameters": str(config["ode_type"]),
        "single_omits_lincomb": not lincomb,
        "ode_only_omits_cell_and_hybrid": not hybrid,
    }


def update_run_manifest(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "manifest.json"
    payload = _read_json(path) if path.is_file() else {}
    payload.update(updates)
    _write_json(path, payload)


def analyze_run(
    run_dir: str | os.PathLike[str],
    *,
    checkpoint: str = "",
    sample_path: str = "",
    max_cells: int = 2000,
    t_values: str | Sequence[int] | None = None,
    batch_size: int = 128,
    device_name: str = "auto",
    seed: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Restore and analyze one run.  This is the CLI-independent entry point."""

    inputs = discover_run_inputs(
        Path(run_dir), checkpoint_override=checkpoint, sample_override=sample_path
    )
    config = inputs["config"]
    selected_seed = int(config.get("seed", 1234) if seed is None else seed)
    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"
    device = torch.device(device_name)
    analyzed_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_label = _checkpoint_label(inputs["checkpoint"])
    output_dir = inputs["run_dir"] / "analysis" / f"{_slug(checkpoint_label)}_{stamp}"
    planned = _planned_outputs(config)
    if dry_run:
        return {
            "dry_run": True,
            "run_dir": str(inputs["run_dir"]),
            "config_path": str(inputs["config_path"]),
            "checkpoint": str(inputs["checkpoint"]),
            "sample_path": str(inputs["sample_path"]),
            "data_path": str(inputs["data_path"]),
            "loss_path": str(inputs["loss_path"]),
            "loss_paths": [str(path) for path in inputs["loss_paths"]],
            "output_dir": str(output_dir),
            "planned_outputs": planned,
        }

    # Load and validate every external input before plotting.
    real, genes, real_labels, label_col = load_real_subset(
        inputs["data_path"], layer=config.get("ts_layer"), max_cells=max_cells,
        seed=selected_seed,
    )
    generated = load_generated_subset(
        inputs["sample_path"], max_cells=max_cells, seed=selected_seed
    )
    if generated.shape[1] != len(genes):
        raise ValueError(
            f"sample dimension {generated.shape[1]} does not match model/data dimension {len(genes)}"
        )
    loss_frame, numeric_loss_columns, regularizer_columns = load_loss_table(inputs["loss_paths"])
    diffusion = build_diffusion(config)
    selected_t = parse_t_values(t_values, diffusion.num_timesteps)
    model = load_model(config, genes, diffusion, inputs["checkpoint"], device)
    meta = AnalysisMetadata(
        ode_type=str(config["ode_type"]),
        model_family=str(config["model_family"]),
        checkpoint=checkpoint_label,
        analyzed_at=analyzed_at,
    )
    hybrid = hasattr(model, "ml_model")
    lincomb = is_lincomb_model(model, config)
    expected_hybrid = str(config["model_family"]) != "ode_only_lincomb"
    expected_lincomb = "lincomb" in str(config["model_family"])
    if hybrid != expected_hybrid:
        raise RuntimeError(
            f"restored structure/config mismatch: ml_model={hybrid}, family={config['model_family']}"
        )
    if lincomb != expected_lincomb:
        raise RuntimeError(
            f"restored structure/config mismatch: lincomb={lincomb}, family={config['model_family']}"
        )

    # Precompute finite model diagnostics before any figure is written.
    io_table, coefficient_policy = evaluate_model_io(
        model, diffusion, real, selected_t, batch_size=batch_size, device=device,
        seed=selected_seed,
    )
    lincomb_table = (
        evaluate_lincomb(model, real, selected_t, batch_size=batch_size, device=device)
        if lincomb
        else None
    )
    # semantic_parameters performs the parameter-level finite preflight here.
    semantic_parameters(model, str(config["ode_type"]))

    output_dir.mkdir(parents=True, exist_ok=False)
    created: list[str] = []
    created.extend(_plot_loss(
        loss_frame, numeric_loss_columns, regularizer_columns, output_dir / "loss", meta
    ))
    umap_paths, embedding_method = _plot_real_generated_umap(
        real, generated, real_labels, label_col, output_dir / "umap", meta, selected_seed
    )
    created.extend(umap_paths)
    created.extend(plot_model_io(io_table, output_dir / "model_io", meta, hybrid=hybrid))
    if lincomb_table is not None:
        created.extend(plot_lincomb(lincomb_table, output_dir / "lincomb", meta))
    parameter_paths, parameter_summary = plot_ode_parameters(
        model, str(config["ode_type"]), genes, output_dir / "parameters", meta,
        seed=selected_seed,
    )
    created.extend(parameter_paths)

    skips = {
        "lincomb_plots": "single ODE model has no LinComb coefficients"
        if not lincomb else "",
        "hybrid_plots": "ODE-only model has no CellUnet branch or hybrid coefficient"
        if not hybrid else "",
    }
    result: dict[str, Any] = {
        "status": "completed",
        "run_dir": str(inputs["run_dir"]),
        "output_dir": str(output_dir),
        "config_path": str(inputs["config_path"]),
        "checkpoint": str(inputs["checkpoint"]),
        "sample_path": str(inputs["sample_path"]),
        "data_path": str(inputs["data_path"]),
        "loss_path": str(inputs["loss_path"]),
        "loss_paths": [str(path) for path in inputs["loss_paths"]],
        "ode_type": str(config["ode_type"]),
        "model_family": str(config["model_family"]),
        "checkpoint_label": checkpoint_label,
        "datetime": analyzed_at,
        "device": str(device),
        "max_cells": int(max_cells),
        "n_real_cells": int(len(real)),
        "n_generated_cells": int(len(generated)),
        "n_genes": int(len(genes)),
        "t_values": list(selected_t),
        "hybrid": hybrid,
        "lincomb": lincomb,
        "hybrid_coefficient_policy": coefficient_policy,
        "embedding_method": embedding_method,
        "regularization_columns": regularizer_columns,
        "nonfinite_policy": "fail; no nan_to_num or silent replacement",
        "ratio_denominator_floor": 1e-12,
        "metric_aggregation_dtype": "float64",
        "conditional_skips": skips,
        "parameter_summary": parameter_summary,
        "created_files": [str(Path(path).relative_to(output_dir)) for path in created],
    }
    _write_json(output_dir / "analysis_manifest.json", result)
    manifest = inputs["manifest"]
    stages = dict(manifest.get("stages", {}))
    stages["analysis"] = {
        "status": "completed",
        "finished_at": analyzed_at,
        "analysis_path": str(output_dir),
        "checkpoint_path": str(inputs["checkpoint"]),
    }
    update_run_manifest(
        inputs["run_dir"], stages=stages, analysis_path=str(output_dir),
        analysis_manifest_path=str(output_dir / "analysis_manifest.json"),
    )
    return result
