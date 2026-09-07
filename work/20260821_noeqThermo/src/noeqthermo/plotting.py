"""Publication-style figures following the visual grammar of the released work."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .landscape import LandscapeResult

FIXED_POINT_STYLES = {
    -1: ("Attractor", "#a6d854", "o"),
    0: ("Saddle", "white", "o"),
    1: ("Repellor", "#542788", "o"),
}


def _finish(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _cell_scatter(ax: plt.Axes, coordinates: np.ndarray, labels: Sequence[str], cfg: Mapping[str, Any]) -> None:
    labels = np.asarray(labels, dtype=str)
    categories = list(dict.fromkeys(labels.tolist()))
    cmap = plt.get_cmap("tab20", max(len(categories), 1))
    for index, category in enumerate(categories):
        selected = labels == category
        ax.scatter(
            coordinates[selected, 0],
            coordinates[selected, 1],
            s=float(cfg["cell_size"]),
            alpha=float(cfg["alpha"]),
            color=cmap(index),
            linewidths=0,
            label=category,
            rasterized=True,
        )


def _stream(ax: plt.Axes, result: LandscapeResult, *, color: str = "black", density: float = 1.4) -> None:
    speed = np.hypot(result.field_x, result.field_y)
    linewidth = 0.25 + 1.5 * speed / (np.nanpercentile(speed, 95) + np.finfo(float).eps)
    ax.streamplot(
        result.Xgrid[0],
        result.Ygrid[:, 0],
        result.field_x,
        result.field_y,
        color=color,
        density=density,
        linewidth=linewidth,
        arrowsize=0.8,
        minlength=0.08,
    )


def _fixed_points(ax: plt.Axes, points: np.ndarray, types: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    types = np.asarray(types, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        return
    for index, (point, raw_type) in enumerate(zip(points, types)):
        if not np.isfinite(point).all() or not np.isfinite(raw_type):
            continue
        ftype = int(raw_type)
        name, color, marker = FIXED_POINT_STYLES.get(ftype, ("Unknown", "grey", "o"))
        ax.scatter(
            point[0],
            point[1],
            s=180,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=1.4,
            zorder=8,
            label=name if name not in ax.get_legend_handles_labels()[1] else None,
        )
        ax.text(point[0], point[1], str(index), ha="center", va="center", fontsize=8, zorder=9)


def _heatmap(ax: plt.Axes, result: LandscapeResult, values: np.ndarray, *, cmap: str, label: str, norm=None):
    mesh = ax.pcolormesh(
        result.Xgrid,
        result.Ygrid,
        values,
        shading="auto",
        cmap=cmap,
        norm=norm,
        rasterized=True,
    )
    plt.colorbar(mesh, ax=ax, label=label)
    return mesh


def _normalized_quiver(
    ax: plt.Axes,
    result: LandscapeResult,
    x_component: np.ndarray,
    y_component: np.ndarray,
    stride: int,
    *,
    color: str = "black",
) -> None:
    magnitude = np.hypot(x_component, y_component)
    normalized_x = np.divide(x_component, magnitude, out=np.zeros_like(x_component), where=magnitude > 0)
    normalized_y = np.divide(y_component, magnitude, out=np.zeros_like(y_component), where=magnitude > 0)
    sl = np.s_[::stride, ::stride]
    ax.quiver(
        result.Xgrid[sl],
        result.Ygrid[sl],
        normalized_x[sl],
        normalized_y[sl],
        color=color,
        angles="xy",
        scale_units="xy",
        scale=1.8,
        width=0.003,
        headwidth=4,
        zorder=5,
    )


def plot_core_figures(
    result: LandscapeResult,
    coordinates: np.ndarray,
    labels: Sequence[str],
    fixed_points: np.ndarray,
    fixed_types: np.ndarray,
    output_dir: str | Path,
    cfg: Mapping[str, Any],
) -> list[str]:
    """Create figures 01--09 with stable names and consistent axes."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(cfg["dpi"])
    stride = int(cfg["quiver_stride"])
    created: list[str] = []

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    _cell_scatter(ax, coordinates, labels, cfg)
    _stream(ax, result)
    ax.set(title="Erythropoietic continuous UMAP vector field", xlabel="UMAP1", ylabel="UMAP2")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    name = "01_umap_vector_field.png"
    _finish(fig, output_dir / name, dpi)
    created.append(name)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    _cell_scatter(ax, coordinates, labels, cfg)
    _stream(ax, result)
    _fixed_points(ax, fixed_points, fixed_types)
    ax.set(title="Vector-field topography", xlabel="UMAP1", ylabel="UMAP2")
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="best")
    name = "02_topography_fixed_points.png"
    _finish(fig, output_dir / name, dpi)
    created.append(name)

    finite_curl = result.curl[np.isfinite(result.curl)]
    curl_scale = float(np.quantile(np.abs(finite_curl), 0.98)) if len(finite_curl) else 1.0
    curl_scale = max(curl_scale, np.finfo(float).eps)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    _heatmap(
        ax,
        result,
        result.curl,
        cmap="coolwarm",
        label=r"UMAP curl $\partial F_y/\partial x-\partial F_x/\partial y$",
        norm=TwoSlopeNorm(vmin=-curl_scale, vcenter=0.0, vmax=curl_scale),
    )
    _stream(ax, result, color="#222222", density=1.0)
    ax.set(title="Curl of reconstructed 2D UMAP field", xlabel="UMAP1", ylabel="UMAP2")
    name = "03_curl_umap.png"
    _finish(fig, output_dir / name, dpi)
    created.append(name)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    _heatmap(ax, result, result.probability, cmap="turbo", label=r"$P_{ss}$")
    ax.set(title="Steady-state probability", xlabel="UMAP1", ylabel="UMAP2")
    name = "04_steady_state_probability.png"
    _finish(fig, output_dir / name, dpi)
    created.append(name)

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        result.Xgrid,
        result.Ygrid,
        result.potential_for_gradient,
        cmap="jet",
        linewidth=0,
        antialiased=True,
        alpha=0.92,
    )
    mesh_stride = max(1, result.Xgrid.shape[0] // 25)
    ax.plot_wireframe(
        result.Xgrid,
        result.Ygrid,
        result.potential_for_gradient + 0.02,
        rstride=mesh_stride,
        cstride=mesh_stride,
        color="black",
        linewidth=0.25,
        alpha=0.65,
    )
    fig.colorbar(surface, ax=ax, shrink=0.65, pad=0.1, label=r"Potential $U=-\log P_{ss}$")
    ax.set(xlabel="UMAP1", ylabel="UMAP2", zlabel="Potential", title="Nonequilibrium potential landscape")
    ax.view_init(elev=45, azim=30)
    name = "05_potential_landscape.png"
    _finish(fig, output_dir / name, dpi)
    created.append(name)

    vector_panels = [
        ("06_mean_force.png", "Mean force", result.mean_force_x, result.mean_force_y),
        ("07_gradient_force.png", r"Gradient force $-D\nabla U$", result.gradient_force_x, result.gradient_force_y),
        ("08_probability_flux.png", r"Probability flux $J=FP-D\nabla P$", result.probability_flux_x, result.probability_flux_y),
    ]
    for name, title, x_component, y_component in vector_panels:
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        _heatmap(ax, result, result.potential_for_gradient, cmap="turbo", label="Potential")
        _normalized_quiver(ax, result, x_component, y_component, stride)
        ax.set(title=title, xlabel="UMAP1", ylabel="UMAP2")
        _finish(fig, output_dir / name, dpi)
        created.append(name)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    _heatmap(ax, result, result.potential_for_gradient, cmap="turbo", label="Potential")
    _normalized_quiver(
        ax,
        result,
        result.rotational_force_x,
        result.rotational_force_y,
        stride,
        color="white",
    )
    _stream(ax, result, color="black", density=1.0)
    _fixed_points(ax, fixed_points, fixed_types)
    ax.set(
        title=r"Landscape, deterministic field, and rotational force $J/P_{ss}$",
        xlabel="UMAP1",
        ylabel="UMAP2",
    )
    name = "09_landscape_flux_overlay.png"
    _finish(fig, output_dir / name, dpi)
    created.append(name)
    return created


def plot_lap(
    result: LandscapeResult,
    coordinates: np.ndarray,
    labels: Sequence[str],
    paths: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    cfg: Mapping[str, Any],
) -> None:
    """Overlay Dynamo LAPs with action-colored points as in the PNAS notebook."""

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    _cell_scatter(ax, coordinates, labels, {**cfg, "alpha": 0.25})
    _stream(ax, result, color="#777777", density=0.8)
    color_values = []
    for path in paths:
        prediction = np.asarray(path["prediction"], dtype=np.float64)
        action = np.asarray(path["action"], dtype=np.float64)
        if len(action) != len(prediction):
            action = np.linspace(0.0, 1.0, len(prediction))
        color_values.extend(action.tolist())
        ax.plot(prediction[:, 0], prediction[:, 1], color="black", linewidth=2.0, zorder=7)
        scatter = ax.scatter(
            prediction[:, 0],
            prediction[:, 1],
            c=action,
            cmap="viridis",
            s=32,
            edgecolor="none",
            zorder=8,
        )
        ax.scatter(*prediction[0], s=100, marker="o", facecolor="white", edgecolor="black", zorder=9)
        ax.scatter(*prediction[-1], s=110, marker="*", facecolor="#fdae61", edgecolor="black", zorder=9)
        ax.text(*prediction[0], str(path["start_label"]), fontsize=8, ha="right")
        ax.text(*prediction[-1], str(path["end_label"]), fontsize=8, ha="left")
    if color_values:
        fig.colorbar(scatter, ax=ax, label="Cumulative action")
    ax.set(title="Dynamo least-action paths", xlabel="UMAP1", ylabel="UMAP2")
    _finish(fig, Path(output_path), int(cfg["dpi"]))


__all__ = ["plot_core_figures", "plot_lap"]
