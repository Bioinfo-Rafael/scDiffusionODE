"""Diffusion-time epsilon, drift, and score diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from diffusion.objectives import score_from_noise, weighted_noise_quadratic
from scripts.common import validate_run_directory, write_json
from .common import (
    dense_float32,
    load_final_ema,
    metadata_matches_final_ema,
    sample_corr,
    sample_mse,
    sample_norm,
    summarize,
    timestep_grid,
)


DRIFT_NOISE_WARNING = (
    "diagnostic comparison; forward drift and Gaussian noise have different semantics"
)


def metric_bundle(
    epsilon_prediction: torch.Tensor,
    epsilon: torch.Tensor,
    drift: torch.Tensor,
    score_prediction: torch.Tensor,
    score_true: torch.Tensor,
    weighted_quadratic: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-cell quantities before mean/std/median aggregation."""

    epsilon_norm = sample_norm(epsilon)
    prediction_norm = sample_norm(epsilon_prediction)
    drift_norm = sample_norm(drift)
    true_score_norm = sample_norm(score_true)
    predicted_score_norm = sample_norm(score_prediction)
    eps = torch.finfo(epsilon_norm.dtype).eps
    return {
        "model_vs_epsilon_mse": sample_mse(epsilon_prediction, epsilon),
        "model_vs_epsilon_corr": sample_corr(epsilon_prediction, epsilon),
        "epsilon_norm": epsilon_norm,
        "model_norm": prediction_norm,
        "model_to_epsilon_norm_ratio": prediction_norm
        / epsilon_norm.clamp_min(eps),
        "drift_vs_epsilon_mse": sample_mse(drift, epsilon),
        "drift_vs_epsilon_corr": sample_corr(drift, epsilon),
        "drift_norm": drift_norm,
        "drift_to_epsilon_norm_ratio": drift_norm
        / epsilon_norm.clamp_min(eps),
        "drift_vs_model_mse": sample_mse(drift, epsilon_prediction),
        "drift_vs_model_corr": sample_corr(drift, epsilon_prediction),
        "drift_to_model_norm_ratio": drift_norm
        / prediction_norm.clamp_min(eps),
        "score_mse": sample_mse(score_prediction, score_true),
        "score_corr": sample_corr(score_prediction, score_true),
        "true_score_norm": true_score_norm,
        "predicted_score_norm": predicted_score_norm,
        "score_norm_ratio": predicted_score_norm / true_score_norm.clamp_min(eps),
        "weighted_score_quadratic": weighted_quadratic,
    }


def _append_summary(row: dict[str, float], name: str, values: torch.Tensor) -> None:
    for statistic, value in summarize(values).items():
        row[f"{name}_{statistic}"] = value


def _select_indices(size: int, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or maximum >= size:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(size, int(maximum), replace=False)).astype(np.int64)


def compute_timestep_metrics(
    model,
    time_map,
    real: np.ndarray,
    *,
    timesteps: Sequence[int],
    batch_size: int,
    noise_seed: int,
) -> list[dict[str, float | int | str]]:
    """Use one cell subset and the same fixed epsilon matrix at every timestep."""

    matrix = dense_float32(real)
    process = model.forward_process
    parameter = next(process.parameters())
    device, dtype = parameter.device, parameter.dtype
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(noise_seed))
    fixed_noise = torch.randn(matrix.shape, generator=generator, dtype=torch.float64)
    rows: list[dict[str, float | int | str]] = []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for timestep in timesteps:
                index_tensor = torch.tensor(float(timestep), dtype=torch.float64)
                physical = time_map.fractional_index_to_time(index_tensor)
                dummy = torch.zeros((1, process.dim), device=device, dtype=dtype)
                stats = process.transition_stats(dummy, physical)
                accumulated: dict[str, list[torch.Tensor]] = {}
                for start in range(0, len(matrix), int(batch_size)):
                    stop = min(start + int(batch_size), len(matrix))
                    clean = torch.from_numpy(matrix[start:stop]).to(
                        device=device, dtype=dtype
                    )
                    epsilon = fixed_noise[start:stop].to(device=device, dtype=dtype)
                    if hasattr(stats, "z"):
                        mean = stats.transition_mean(clean)
                        noisy = mean + stats.noise_transform(epsilon)
                    else:
                        mean = clean @ stats.transition_matrix.transpose(-1, -2)
                        mean = mean + stats.affine_shift
                        noisy = mean + epsilon @ stats.cholesky.transpose(-1, -2)
                    t = torch.full(
                        (len(clean),),
                        float(timestep),
                        device=device,
                        dtype=torch.float32,
                    )
                    epsilon_prediction = model.predict_noise(noisy, t)
                    drift = process.drift(noisy)
                    if hasattr(stats, "z"):
                        score_prediction = process.conditional_score(stats, epsilon_prediction)
                        score_true = process.conditional_score(stats, epsilon)
                        weighted = process.noise_metric_quadratic(stats, epsilon_prediction - epsilon)
                    else:
                        score_prediction = score_from_noise(
                            stats.cholesky, epsilon_prediction
                        )
                        score_true = score_from_noise(stats.cholesky, epsilon)
                        weighted = weighted_noise_quadratic(
                            stats.cholesky,
                            stats.diffusion_covariance,
                            epsilon_prediction - epsilon,
                        )
                    bundle = metric_bundle(
                        epsilon_prediction,
                        epsilon,
                        drift,
                        score_prediction,
                        score_true,
                        weighted,
                    )
                    for name, values in bundle.items():
                        if not bool(torch.isfinite(values).all().detach().item()):
                            raise FloatingPointError(
                                f"{name} is non-finite at timestep {timestep}"
                            )
                        accumulated.setdefault(name, []).append(values.cpu())
                row: dict[str, float | int | str] = {
                    "timestep": int(timestep),
                    "physical_time": float(physical),
                    "covariance_evaluation": getattr(
                        stats, "covariance_evaluation", "augmented_van_loan"
                    ),
                    "covariance_series_terms": int(
                        getattr(stats, "covariance_series_terms", 0)
                    ),
                }
                for name, chunks in accumulated.items():
                    _append_summary(row, name, torch.cat(chunks))
                rows.append(row)
    finally:
        model.train(was_training)
    return rows


def _plot_lines(
    frame: pd.DataFrame,
    metrics: Sequence[str],
    *,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = frame["timestep"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for metric in metrics:
        mean = frame[f"{metric}_mean"].to_numpy(float)
        std = frame[f"{metric}_std"].to_numpy(float)
        ax.plot(x, mean, marker="o", markersize=3, linewidth=1.4, label=metric)
        ax.fill_between(x, mean - std, mean + std, alpha=0.12)
    ax.set(xlabel="forward diffusion timestep", ylabel=ylabel, title=title)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_panels(
    frame: pd.DataFrame,
    groups: Sequence[Sequence[str]],
    *,
    group_ylabels: Sequence[str],
    title: str,
    output: Path,
) -> None:
    """One subplot per group of commensurate metrics; groups of a
    different unit (MSE vs. correlation vs. a raw norm) never share an
    axis (dataviz anti-pattern: never combine differently scaled
    measures on one y-axis). A group may hold >1 metric only when they
    share units, e.g. the two epsilon norms."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = frame["timestep"].to_numpy()
    fig, axes = plt.subplots(
        len(groups), 1, figsize=(10.5, 3.2 * len(groups)), sharex=True
    )
    axes = np.atleast_1d(axes)
    for ax, group, group_ylabel in zip(axes, groups, group_ylabels):
        for metric in group:
            mean = frame[f"{metric}_mean"].to_numpy(float)
            std = frame[f"{metric}_std"].to_numpy(float)
            ax.plot(x, mean, marker="o", markersize=3, linewidth=1.4, label=metric)
            ax.fill_between(x, mean - std, mean + std, alpha=0.12)
        ax.set_ylabel(group_ylabel, fontsize=9)
        ax.grid(alpha=0.22)
        if len(group) > 1:
            ax.legend(frameon=False, fontsize=8)
    axes[-1].set_xlabel("forward diffusion timestep")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_diffusion_diagnostics(
    run_dir,
    *,
    step: int = 20,
    max_cells: int = 2000,
    batch_size: int = 128,
    seed: int = 1234,
    device: str = "auto",
    ema_rate=None,
    force: bool = False,
) -> dict:
    run = validate_run_directory(run_dir)
    output = run / "analysis" / "diffusion_diagnostics"
    metadata_path = output / "metadata.json"
    if metadata_path.is_file() and not force:
        with metadata_path.open(encoding="utf-8") as handle:
            if metadata_matches_final_ema(
                json.load(handle), run, ema_rate=ema_rate
            ):
                return {"status": "skipped_completed", "metadata": str(metadata_path)}
    loaded = load_final_ema(run, device=device, ema_rate=ema_rate)
    indices = _select_indices(int(loaded["adata"].n_obs), int(max_cells), int(seed))
    layer = loaded["config"].get("data_layer", loaded["config"].get("layer"))
    subset = loaded["adata"][indices]
    source = subset.layers[layer] if layer else subset.X
    real = dense_float32(source)
    timesteps = timestep_grid(loaded["components"].time_map.num_timesteps, step)
    rows = compute_timestep_metrics(
        loaded["components"].model,
        loaded["components"].time_map,
        real,
        timesteps=timesteps,
        batch_size=batch_size,
        noise_seed=seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "timestep_metrics.csv", index=False)
    _plot_lines(
        frame,
        (
            "model_vs_epsilon_mse",
            "drift_vs_epsilon_mse",
            "drift_vs_model_mse",
        ),
        ylabel="per-cell MSE across genes",
        title=f"MSE comparison\n{DRIFT_NOISE_WARNING}",
        output=output / "mse_comparisons.png",
    )
    _plot_lines(
        frame,
        (
            "model_vs_epsilon_corr",
            "drift_vs_epsilon_corr",
            "drift_vs_model_corr",
        ),
        ylabel="per-cell Pearson correlation across genes",
        title=f"Correlation comparison\n{DRIFT_NOISE_WARNING}",
        output=output / "correlation_comparisons.png",
    )
    _plot_lines(
        frame,
        ("epsilon_norm", "model_norm"),
        ylabel="per-cell L2 norm",
        title="Epsilon vs. model-predicted noise norms",
        output=output / "epsilon_vs_model_norms.png",
    )
    # drift_vs_epsilon_{mse,corr} and drift_vs_model_{mse,corr} are already
    # plotted correctly above (each metric alongside its own unit family, in
    # mse_comparisons.png / correlation_comparisons.png); a combined
    # MSE+correlation figure would just repeat those two lines on one
    # mismatched axis, so it is not regenerated here.
    _plot_panels(
        frame,
        (
            ("score_mse",),
            ("score_corr",),
            ("true_score_norm", "predicted_score_norm"),
            ("score_norm_ratio",),
            ("weighted_score_quadratic",),
        ),
        group_ylabels=(
            "score MSE",
            "score Pearson correlation",
            "per-cell L2 norm",
            "predicted / true score norm ratio",
            "weighted quadratic",
        ),
        title="Conditional score-space diagnostics and paper quadratic",
        output=output / "score_metrics.png",
    )
    adata_file = getattr(loaded["adata"], "file", None)
    if adata_file is not None:
        adata_file.close()
    metadata = {
        "status": "completed",
        "checkpoint": str(loaded["checkpoint"]),
        "checkpoint_step": loaded["checkpoint_step"],
        "ema_rate": loaded["ema_rate"],
        "cell_indices": indices.tolist(),
        "noise_seed": int(seed),
        "noise_policy": "same fixed Gaussian epsilon rows reused at every timestep",
        "timestep_step": int(step),
        "timesteps": timesteps,
        "correlation": "per-cell Pearson correlation across genes",
        "drift_noise_warning": DRIFT_NOISE_WARNING,
        "weighted_quadratic": (
            "0.5 * (s_theta-s_phi)^T a_s (s_theta-s_phi), sum-form"
        ),
    }
    write_json(metadata_path, metadata)
    return {"status": "completed", "metadata": str(metadata_path)}


__all__ = [
    "DRIFT_NOISE_WARNING",
    "analyze_diffusion_diagnostics",
    "compute_timestep_metrics",
    "metric_bundle",
]
