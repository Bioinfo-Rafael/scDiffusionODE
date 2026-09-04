"""Raw-checkpoint evolution of dense forward-process parameters."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from models.factory import build_forward_process
from scripts.common import (
    all_raw_checkpoints,
    checkpoint_step,
    requested_config,
    validate_run_directory,
    write_json,
)
from .common import gene_names_from_adata, load_run_data, load_state_dict


@dataclass(frozen=True)
class SnapshotSource:
    label: str
    training_step: int
    path: Path
    pretraining: bool


def select_snapshot_sources(run_dir, *, max_points: int = 8) -> list[SnapshotSource]:
    """Choose 5--10 chronological stages when enough raw checkpoints exist."""

    run = validate_run_directory(run_dir)
    max_points = int(max_points)
    if not 5 <= max_points <= 10:
        raise ValueError("max_points must be between 5 and 10")
    initial = run / "initial_forward_state.pt"
    if not initial.is_file():
        raise FileNotFoundError(f"pre-training forward state is missing: {initial}")
    raw = all_raw_checkpoints(run)
    if not raw:
        raise FileNotFoundError("no raw checkpoints found for parameter evolution")
    slots = max_points - 1
    if len(raw) > slots:
        indices = np.linspace(0, len(raw) - 1, slots)
        selected_indices = sorted(set(int(round(value)) for value in indices))
        raw = [raw[index] for index in selected_indices]
    sources = [SnapshotSource("initial", -1, initial.resolve(), True)]
    for path in raw:
        step = checkpoint_step(path)
        sources.append(
            SnapshotSource(
                f"step {step} (post-update raw)", int(step), path, False
            )
        )
    return sources


def load_forward_snapshot(
    config: Mapping,
    genes: Sequence[str],
    source: SnapshotSource,
    *,
    device: torch.device = torch.device("cpu"),
):
    process = build_forward_process(config, genes, device)
    if source.pretraining:
        payload = torch.load(source.path, map_location=device)
        if payload.get("semantics") != (
            "pre-training forward-process state before any optimizer update"
        ):
            raise ValueError("initial state lacks explicit pre-training semantics")
        state = payload["state_dict"]
    else:
        wrapper_state = load_state_dict(source.path, device)
        prefix = "forward_process."
        state = {
            name[len(prefix) :]: value
            for name, value in wrapper_state.items()
            if name.startswith(prefix)
        }
        if not state:
            raise KeyError(f"raw checkpoint lacks forward_process parameters: {source.path}")
    process.load_state_dict(state, strict=True)
    process.eval()
    return process


def model_a_arrays(process) -> dict[str, np.ndarray]:
    with torch.no_grad():
        q = process.q_matrix().detach().cpu().double().numpy()
        d = process.d_matrix().detach().cpu().double().numpy()
    a = q + d
    return {"Q": q, "D": d, "A": a, "F": -a}


def model_b_arrays(process) -> dict[str, np.ndarray]:
    with torch.no_grad():
        w = process.drift_matrix().detach().cpu().double().numpy()
        b = process.drift_bias().detach().cpu().double().numpy()
    return {"W": w, "b": b}


def _edge_masks(process) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dim = int(process.dim)
    diagonal = np.eye(dim, dtype=bool)
    if process.grn_mask_target_source.numel():
        mask = process.grn_mask_target_source.detach().cpu().numpy() > 0.5
    else:
        mask = np.zeros((dim, dim), dtype=bool)
    return mask & ~diagonal, ~mask & ~diagonal, diagonal


def _summary(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q05": np.nan,
            "q95": np.nan,
            "frobenius_norm": 0.0,
            "max_absolute_value": 0.0,
        }
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q95": float(np.quantile(flat, 0.95)),
        "frobenius_norm": float(np.linalg.norm(flat)),
        "max_absolute_value": float(np.max(np.abs(flat))),
    }


def parameter_categories(process) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    known, unknown, diagonal = _edge_masks(process)
    if hasattr(process, "q_matrix"):
        arrays = model_a_arrays(process)
        q, d, a = arrays["Q"], arrays["D"], arrays["A"]
        categories = {
            "Q off-diagonal": q[~diagonal],
            "D diagonal": d[diagonal],
            "D off-diagonal": d[~diagonal],
            "A Known GRN edges": a[known],
            "A Unknown GRN edges": a[unknown],
            "Q all": q,
            "D all": d,
            "A all": a,
            "F all": arrays["F"],
        }
        diagnostics = {
            "q_skew_symmetry_residual": float(np.max(np.abs(q + q.T))),
            "d_minimum_eigenvalue": float(np.linalg.eigvalsh(d).min()),
            "stationarity_residual": float(
                np.linalg.norm(-a + (-a).T + 2.0 * d)
            ),
        }
    else:
        arrays = model_b_arrays(process)
        w, b = arrays["W"], arrays["b"]
        categories = {
            "W all": w,
            "W diagonal": w[diagonal],
            "W off-diagonal": w[~diagonal],
            "W Known GRN edges": w[known],
            "W Unknown GRN edges": w[unknown],
            "b": b,
        }
        diagnostics = {}
    return categories, diagnostics


def _plot_histograms(
    snapshots: Sequence[tuple[SnapshotSource, dict[str, np.ndarray]]],
    rows: Sequence[str],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows, n_cols = len(rows), len(snapshots)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False
    )
    for column, (source, categories) in enumerate(snapshots):
        for row, name in enumerate(rows):
            ax = axes[row, column]
            values = np.asarray(categories[name]).reshape(-1)
            if values.size:
                ax.hist(values, bins=60, alpha=0.72, color=f"C{row % 10}")
                annotation = f"mean:{values.mean():.2e}\nstd:{values.std():.2e}"
            else:
                annotation = "no entries"
            ax.text(
                0.97,
                0.95,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.6},
            )
            if row == 0:
                ax.set_title(source.label, fontsize=9)
            if column == 0:
                ax.set_ylabel(name, fontweight="bold")
    fig.suptitle("Forward-parameter distributions across raw checkpoints")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_parameter_evolution(
    run_dir,
    *,
    max_points: int = 8,
    force: bool = False,
) -> dict:
    run = validate_run_directory(run_dir)
    output = run / "analysis" / "parameter_evolution"
    metadata_path = output / "metadata.json"
    if metadata_path.is_file() and not force:
        with metadata_path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        expected_sources = select_snapshot_sources(run, max_points=max_points)
        if (
            existing.get("status") == "completed"
            and [item.get("path") for item in existing.get("sources", [])]
            == [str(source.path) for source in expected_sources]
        ):
            return {"status": "skipped_completed", "metadata": str(metadata_path)}
    _run, config, _data_path, adata, genes = load_run_data(run)
    sources = select_snapshot_sources(run, max_points=max_points)
    snapshots = []
    summary_rows = []
    diagnostics_rows = []
    for source in sources:
        process = load_forward_snapshot(config, genes, source)
        categories, diagnostics = parameter_categories(process)
        snapshots.append((source, categories))
        for name, values in categories.items():
            summary_rows.append(
                {
                    "training_step": source.training_step,
                    "checkpoint_label": source.label,
                    "checkpoint_path": str(source.path),
                    "pretraining": source.pretraining,
                    "parameter_group": name,
                    **_summary(values),
                }
            )
        diagnostics_rows.append(
            {
                "training_step": source.training_step,
                "checkpoint_label": source.label,
                "checkpoint_path": str(source.path),
                **diagnostics,
            }
        )
    family = str(config["forward_model"])
    plot_rows = (
        (
            "Q off-diagonal",
            "D diagonal",
            "D off-diagonal",
            "A Known GRN edges",
            "A Unknown GRN edges",
        )
        if family == "stationary_qd"
        else (
            "W all",
            "W diagonal",
            "W off-diagonal",
            "W Known GRN edges",
            "W Unknown GRN edges",
            "b",
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(output / "parameter_summary.csv", index=False)
    pd.DataFrame(diagnostics_rows).to_csv(
        output / "parameter_diagnostics.csv", index=False
    )
    _plot_histograms(snapshots, plot_rows, output / "parameter_histograms.png")
    adata_file = getattr(adata, "file", None)
    if adata_file is not None:
        adata_file.close()
    metadata = {
        "status": "completed",
        "forward_model": family,
        "checkpoint_policy": "pre-training state plus chronological raw checkpoints; never EMA",
        "sources": [
            {
                "label": source.label,
                "training_step": source.training_step,
                "path": str(source.path),
                "pretraining": source.pretraining,
            }
            for source in sources
        ],
        "histogram_bins": 60,
        "mask_orientation": "target_source",
    }
    write_json(metadata_path, metadata)
    return {"status": "completed", "metadata": str(metadata_path)}


__all__ = [
    "SnapshotSource",
    "analyze_parameter_evolution",
    "load_forward_snapshot",
    "model_a_arrays",
    "model_b_arrays",
    "parameter_categories",
    "select_snapshot_sources",
]
