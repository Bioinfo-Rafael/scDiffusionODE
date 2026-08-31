"""Publication-oriented plots with explicit timestep and loss semantics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _finish(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _condition_title(metadata: dict) -> str:
    return (
        f"{metadata.get('ode_type', 'unknown ODE')} | "
        f"lambda={metadata.get('cell_ode_reg_lambda_20260830', 'unknown')} | "
        f"checkpoint training_step={metadata.get('checkpoint_training_step', 'unknown')}"
    )


def _line(ax, frame, column, label=None):
    ax.plot(frame["diffusion_timestep"], frame[column], label=label or column, linewidth=1.8)
    ax.set_xlabel("diffusion_timestep")
    ax.grid(alpha=0.25)


def _rolling_plot(ax, history, column, window, color):
    median = f"{column}_rolling_median_w{window}"
    q25 = f"{column}_rolling_q25_w{window}"
    q75 = f"{column}_rolling_q75_w{window}"
    x = history["training_step"].to_numpy(dtype=float)
    med = history[median].to_numpy(dtype=float)
    lower = history[q25].to_numpy(dtype=float)
    upper = history[q75].to_numpy(dtype=float)
    raw = history[column].to_numpy(dtype=float)
    ax.plot(x, med, color=color, label="rolling median")
    ax.fill_between(
        x, lower, upper, color=color,
        alpha=0.2, label="rolling Q25-Q75",
    )
    ax.scatter(x, raw, s=8, alpha=0.25, color=color, label="raw")
    ax.set_xlabel("training_step")
    ax.set_ylabel(column)
    ax.grid(alpha=0.25)


def plot_gradient_figures(
    gradient_metrics: pd.DataFrame,
    output_dir,
    metadata: dict,
) -> list[Path]:
    if gradient_metrics.empty:
        return []
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    title = _condition_title(metadata)
    created = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for diffusion_timestep, group in gradient_metrics.groupby("diffusion_timestep"):
        axes[0].plot(group["checkpoint_training_step"], group["cell_diffusion_gradient_norm"], "o-", label=f"diffusion_timestep={diffusion_timestep} g_diff")
        axes[0].plot(group["checkpoint_training_step"], group["cell_consistency_gradient_norm"], "x--", label=f"diffusion_timestep={diffusion_timestep} g_cons")
        axes[1].plot(group["checkpoint_training_step"], group["ode_prior_gradient_norm"], "o-", label=f"diffusion_timestep={diffusion_timestep} g_prior")
        axes[1].plot(group["checkpoint_training_step"], group["ode_consistency_gradient_norm"], "x--", label=f"diffusion_timestep={diffusion_timestep} g_cons")
    for ax, branch in zip(axes, ("CellUnet", "ODE")):
        ax.set_xlabel("checkpoint_training_step"); ax.set_ylabel("global gradient L2 norm")
        ax.set_title(branch); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    fig.suptitle(f"Post-hoc gradient norms (no optimizer step)\n{title}")
    path = output / "11_gradient_norms.png"; _finish(fig, path); created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for diffusion_timestep, group in gradient_metrics.groupby("diffusion_timestep"):
        axes[0].plot(group["checkpoint_training_step"], group["cell_gradient_cosine"], "o-", label=f"diffusion_timestep={diffusion_timestep}")
        axes[1].plot(group["checkpoint_training_step"], group["ode_gradient_cosine"], "o-", label=f"diffusion_timestep={diffusion_timestep}")
    for ax, branch in zip(axes, ("Cell: diffusion vs consistency", "ODE: prior vs consistency")):
        ax.axhline(0, color="black", linewidth=1); ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("checkpoint_training_step"); ax.set_ylabel("gradient cosine")
        ax.set_title(branch); ax.legend(fontsize=7); ax.grid(alpha=0.25)
    fig.suptitle(f"Gradient cooperation (>0) / conflict (<0)\n{title}")
    path = output / "12_gradient_cosine.png"; _finish(fig, path); created.append(path)
    return created


def plot_run_figures(
    diffusion_metrics: pd.DataFrame,
    cell_ode_metrics: pd.DataFrame,
    loss_history: pd.DataFrame,
    loss_fraction: pd.DataFrame,
    gradient_metrics: pd.DataFrame,
    output_dir,
    *,
    rolling_window: int,
    zoom_timestep_min: int = 1,
) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = cell_ode_metrics.iloc[0].to_dict()
    title = _condition_title(metadata)
    created = []

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, column, ylabel in zip(
        axes,
        ("cell_target_pearson_global", "cell_target_mse_mean", "cell_target_cosine_mean"),
        ("global Pearson", "per-cell MSE mean", "per-cell cosine mean"),
    ):
        _line(ax, diffusion_metrics, column)
        ax.set_ylabel(ylabel)
    fig.suptitle(f"CellUnet vs exact diffusion target ({diffusion_metrics.iloc[0]['diffusion_target']})\n{title}")
    path = output / "01_cell_target_metrics_vs_t.png"; _finish(fig, path); created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _line(axes[0], cell_ode_metrics, "cell_ode_pearson_global")
    _line(axes[1], cell_ode_metrics, "cell_ode_cosine_mean")
    axes[0].set_ylabel("Cell-ODE global Pearson")
    axes[1].set_ylabel("Cell-ODE per-cell cosine mean")
    fig.suptitle(f"CellUnet vs ODE correlation and cosine | full diffusion timestep range\n{title}")
    path = output / "02_cell_ode_corr_cos_vs_t.png"; _finish(fig, path); created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _line(axes[0], cell_ode_metrics, "cell_ode_mse_mean")
    _line(axes[1], cell_ode_metrics, "cell_ode_nmse_mean")
    axes[0].set_ylabel("Cell-ODE raw MSE mean")
    axes[1].set_ylabel("Cell-ODE symmetric NMSE mean")
    fig.suptitle(f"CellUnet vs ODE error | full range, linear y scale\n{title}")
    path = output / "03_cell_ode_mse_nmse_vs_t.png"; _finish(fig, path); created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _line(axes[0], cell_ode_metrics, "cell_norm_mean")
    _line(axes[1], cell_ode_metrics, "ode_norm_mean")
    axes[0].set_ylabel("mean per-cell ||Cell||2")
    axes[1].set_ylabel("mean per-cell ||ODE||2")
    fig.suptitle(f"Branch norms | full diffusion timestep range\n{title}")
    path = output / "04_cell_ode_norm_vs_t.png"; _finish(fig, path); created.append(path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    _line(ax, cell_ode_metrics, "ode_cell_norm_ratio_mean")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="ratio=1")
    ax.set_ylabel("mean ||ODE||2 / (||Cell||2 + eps)")
    ax.legend()
    ax.set_title(f"ODE/Cell norm ratio | full diffusion timestep range\n{title}")
    path = output / "05_cell_ode_norm_ratio_vs_t.png"; _finish(fig, path); created.append(path)

    zoom = cell_ode_metrics[cell_ode_metrics["diffusion_timestep"] >= zoom_timestep_min]
    if zoom.empty:
        zoom = cell_ode_metrics.copy()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, column, label in zip(
        axes.flat,
        ("cell_ode_pearson_global", "cell_ode_cosine_mean", "cell_ode_mse_mean", "cell_ode_nmse_mean"),
        ("global Pearson", "cosine mean", "raw MSE mean", "NMSE mean"),
    ):
        _line(ax, zoom, column)
        ax.set_ylabel(label)
    fig.suptitle(
        f"Cell-ODE zoom: diffusion_timestep >= {zoom_timestep_min}; excluded t < {zoom_timestep_min}\n{title}"
    )
    explicit = output / f"06_cell_ode_metrics_zoom_tge{zoom_timestep_min}.png"
    _finish(fig, explicit); created.append(explicit)
    # Required stable name, with the exclusion still explicit in the title.
    data = explicit.read_bytes()
    stable = output / "06_cell_ode_metrics_zoom.png"
    stable.write_bytes(data); created.append(stable)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, column, label in zip(
        axes.flat,
        ("cell_ode_mse_mean", "cell_ode_nmse_mean", "cell_norm_mean", "ode_norm_mean"),
        ("raw MSE mean", "NMSE mean", "Cell norm mean", "ODE norm mean"),
    ):
        positive = cell_ode_metrics[["diffusion_timestep", column]].copy()
        positive.loc[positive[column] <= 0, column] = np.nan
        _line(ax, positive, column)
        ax.set_yscale("log")
        ax.set_ylabel(f"{label} (log y; non-positive omitted)")
    fig.suptitle(f"Positive Cell-ODE metrics | full range, log y scale\n{title}")
    path = output / "07_cell_ode_metrics_log.png"; _finish(fig, path); created.append(path)

    if not loss_history.empty:
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        for ax, column, color in zip(
            axes,
            ("diffusion_loss_raw", "ode_regularization_raw", "cell_ode_consistency_raw_20260830"),
            ("tab:blue", "tab:orange", "tab:green"),
        ):
            _rolling_plot(ax, loss_history, column, rolling_window, color)
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"Raw loss components on independent y axes | rolling median w={rolling_window}\n{title}")
        path = output / "08_loss_components_raw.png"; _finish(fig, path); created.append(path)

        fig, ax = plt.subplots(figsize=(10, 5))
        for column, label in (
            ("diffusion_loss_raw", "diffusion raw/final contribution"),
            ("ode_regularization_weighted", "ODE final weighted contribution"),
            ("cell_ode_consistency_weighted_20260830", "Cell-ODE final weighted contribution"),
        ):
            median_column = f"{column}_rolling_median_w{rolling_window}"
            value = loss_history[median_column].where(
                loss_history[median_column] > 0, np.nan
            )
            ax.plot(
                loss_history["training_step"], value,
                label=f"{label} (rolling median w={rolling_window})",
            )
            raw = loss_history[column].where(loss_history[column] > 0, np.nan)
            ax.scatter(
                loss_history["training_step"], raw, s=7, alpha=0.18,
            )
        ax.set_yscale("log")
        ax.set_xlabel("training_step")
        ax.set_ylabel("loss contribution (log y; non-positive omitted)")
        ax.legend(); ax.grid(alpha=0.25)
        ax.set_title(f"Weighted contributions entering total loss\n{title}")
        path = output / "09_loss_components_weighted_log.png"; _finish(fig, path); created.append(path)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.stackplot(
            loss_fraction["training_step"].to_numpy(dtype=float),
            loss_fraction["diffusion_contribution_fraction"].to_numpy(dtype=float),
            loss_fraction["ode_regularization_contribution_fraction"].to_numpy(dtype=float),
            loss_fraction["cell_ode_contribution_fraction_20260830"].to_numpy(dtype=float),
            labels=("diffusion", "ODE regularization", "Cell-ODE consistency"),
            alpha=0.85,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("training_step"); ax.set_ylabel("fraction of three-term loss sum")
        ax.legend(loc="upper right"); ax.grid(alpha=0.2)
        ax.set_title(f"Loss contribution fraction (0-1)\n{title}")
        path = output / "10_loss_contribution_fraction.png"; _finish(fig, path); created.append(path)

    if not gradient_metrics.empty:
        created.extend(plot_gradient_figures(gradient_metrics, output, metadata))
    return created


HEATMAP_COLUMNS = (
    "cell_target_pearson_global_timestep_mean",
    "cell_target_mse_mean_timestep_mean",
    "cell_ode_pearson_global_timestep_mean",
    "cell_ode_cosine_mean_timestep_mean",
    "cell_ode_nmse_mean_timestep_mean",
    "norm_ratio_deviation_from_1_timestep_mean",
    "late_cell_ode_fraction_20260830",
    "cell_gradient_cosine_mean",
    "cell_gradient_norm_ratio_mean",
)


def plot_summary_figures(summary: pd.DataFrame, output_dir) -> list[Path]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    created = []
    ordered = summary.sort_values(["ode_type", "cell_ode_reg_lambda_20260830"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    columns = (
        "cell_target_pearson_global_timestep_mean",
        "cell_target_mse_mean_timestep_mean",
        "cell_ode_pearson_global_timestep_mean",
        "cell_ode_nmse_mean_timestep_mean",
    )
    for ax, column in zip(axes.flat, columns):
        for ode_type, group in ordered.groupby("ode_type"):
            ax.plot(group["cell_ode_reg_lambda_20260830"], group[column], "o-", label=ode_type)
        ax.set_xscale("log"); ax.set_xlabel("cell_ode_reg_lambda_20260830")
        ax.set_ylabel(column); ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Lambda comparison: diffusion task and Cell-ODE agreement")
    path = output / "13_lambda_comparison.png"; _finish(fig, path); created.append(path)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    lambdas = sorted(ordered["cell_ode_reg_lambda_20260830"].unique())
    for row, value in enumerate(lambdas[:3]):
        subset = ordered[ordered["cell_ode_reg_lambda_20260830"] == value]
        axes[row, 0].bar(subset["ode_type"], subset["cell_target_mse_mean_timestep_mean"])
        axes[row, 1].bar(subset["ode_type"], subset["cell_ode_nmse_mean_timestep_mean"])
        axes[row, 0].set_ylabel(f"lambda={value}\nCell-target MSE")
        axes[row, 1].set_ylabel(f"lambda={value}\nCell-ODE NMSE")
        for ax in axes[row]:
            ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.25)
    fig.suptitle("ODE family comparison at fixed consistency lambda")
    path = output / "14_ode_family_comparison.png"; _finish(fig, path); created.append(path)

    matrix = ordered.set_index("experiment").reindex(columns=HEATMAP_COLUMNS)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.45 * len(matrix))))
    sns.heatmap(matrix, cmap="viridis", annot=False, ax=ax)
    ax.set_title("Condition summary heatmap | raw metric values")
    path = output / "15_condition_summary_heatmap_raw.png"; _finish(fig, path); created.append(path)

    std = matrix.std(axis=0, ddof=0).replace(0, np.nan)
    standardized = ((matrix - matrix.mean(axis=0)) / std).fillna(0.0)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.45 * len(matrix))))
    sns.heatmap(standardized, cmap="vlag", center=0, annot=False, ax=ax)
    ax.set_title("Condition summary heatmap | column-wise z-score (constant columns=0)")
    path = output / "16_condition_summary_heatmap_standardized.png"; _finish(fig, path); created.append(path)
    return created


__all__ = [
    "HEATMAP_COLUMNS",
    "plot_gradient_figures",
    "plot_run_figures",
    "plot_summary_figures",
]
