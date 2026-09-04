"""Raw, weighted, and contribution-fraction loss history analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from scripts.common import validate_run_directory, write_json


RAW_COLUMNS = (
    "path_loss_raw",
    "terminal_kl_raw",
    "boundary_nll_raw",
    "grn_penalty_raw",
)
WEIGHTED_COLUMNS = (
    "path_final_per_dim",
    "terminal_final_per_dim",
    "boundary_final_per_dim",
    "grn_penalty_final_weighted",
)


def loss_component_files(run_dir) -> list[Path]:
    run = validate_run_directory(run_dir)
    return sorted(run.glob("segments/segment_*/loss_components.csv"))


def load_loss_components(run_dir) -> pd.DataFrame:
    files = loss_component_files(run_dir)
    if not files:
        raise FileNotFoundError("no work-local loss_components.csv files found")
    frames = []
    for order, path in enumerate(files):
        frame = pd.read_csv(path)
        frame["loss_source_path"] = str(path.resolve())
        frame["loss_source_order"] = order
        frames.append(frame)
    history = pd.concat(frames, ignore_index=True)
    required = {
        "training_step",
        "total_loss",
        *RAW_COLUMNS,
        *WEIGHTED_COLUMNS,
    }
    missing = sorted(required.difference(history.columns))
    if missing:
        raise KeyError("loss component CSV is missing: " + ", ".join(missing))
    history = (
        history.sort_values(["training_step", "loss_source_order"])
        .drop_duplicates("training_step", keep="last")
        .reset_index(drop=True)
    )
    reconstructed = history[list(WEIGHTED_COLUMNS)].sum(axis=1)
    if not np.allclose(
        reconstructed.to_numpy(),
        history["total_loss"].to_numpy(),
        rtol=1e-6,
        atol=1e-7,
    ):
        raise ValueError("saved total_loss does not match weighted contributions")
    return history


def add_rolling_statistics(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    window: int,
) -> pd.DataFrame:
    result = frame.copy()
    window = int(window)
    if window <= 0:
        raise ValueError("rolling window must be positive")
    for column in columns:
        rolling = result[column].rolling(window=window, min_periods=1)
        result[f"{column}_rolling_median_w{window}"] = rolling.median()
        result[f"{column}_rolling_q25_w{window}"] = rolling.quantile(0.25)
        result[f"{column}_rolling_q75_w{window}"] = rolling.quantile(0.75)
    return result


def contribution_fractions(history: pd.DataFrame, *, window: int) -> pd.DataFrame:
    denominator = history[list(WEIGHTED_COLUMNS)].sum(axis=1)
    safe = denominator.where(denominator.abs() > 1e-30, np.nan)
    frame = pd.DataFrame(
        {
            "training_step": history["training_step"].astype(np.int64),
            "weighted_contribution_sum": denominator,
        }
    )
    for column in WEIGHTED_COLUMNS:
        frame[f"{column}_fraction"] = history[column] / safe
    frame = frame.fillna(0.0)
    return add_rolling_statistics(
        frame,
        tuple(f"{column}_fraction" for column in WEIGHTED_COLUMNS),
        window=window,
    )


def _plot_rolling(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    window: int,
    ylabel: str,
    title: str,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = frame["training_step"].to_numpy(dtype=np.int64)
    fig, ax = plt.subplots(figsize=(11, 6))
    for column in columns:
        median = frame[f"{column}_rolling_median_w{window}"].to_numpy(float)
        q25 = frame[f"{column}_rolling_q25_w{window}"].to_numpy(float)
        q75 = frame[f"{column}_rolling_q75_w{window}"].to_numpy(float)
        ax.plot(x, median, linewidth=1.6, label=column)
        ax.fill_between(x, q25, q75, alpha=0.14)
    ax.set(xlabel="training step", ylabel=ylabel, title=title)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_loss(run_dir, *, rolling_window: int = 100, force: bool = False) -> dict:
    run = validate_run_directory(run_dir)
    output = run / "analysis" / "loss"
    metadata_path = output / "metadata.json"
    if metadata_path.is_file() and not force:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        current_sources = [
            str(path.resolve()) for path in loss_component_files(run)
        ]
        if (
            metadata.get("status") == "completed"
            and metadata.get("source_files") == current_sources
            and int(metadata.get("rolling_window", -1)) == int(rolling_window)
        ):
            return {"status": "skipped_completed", "metadata": str(metadata_path)}
    output.mkdir(parents=True, exist_ok=True)
    history = load_loss_components(run)
    columns = (*RAW_COLUMNS, *WEIGHTED_COLUMNS, "total_loss")
    history = add_rolling_statistics(history, columns, window=rolling_window)
    fractions = contribution_fractions(history, window=rolling_window)
    history_path = output / "loss_components.csv"
    fraction_path = output / "loss_fractions.csv"
    history.to_csv(history_path, index=False)
    fractions.to_csv(fraction_path, index=False)
    _plot_rolling(
        history,
        RAW_COLUMNS,
        window=rolling_window,
        ylabel="raw sum-form quantity",
        title="Raw learnable-forward loss components",
        output=output / "loss_raw.png",
    )
    _plot_rolling(
        history,
        WEIGHTED_COLUMNS,
        window=rolling_window,
        ylabel="actual contribution to total loss",
        title="Per-dimension paper ELBO + external GRN contribution",
        output=output / "loss_weighted.png",
    )
    fraction_columns = tuple(f"{column}_fraction" for column in WEIGHTED_COLUMNS)
    _plot_rolling(
        fractions,
        fraction_columns,
        window=rolling_window,
        ylabel="signed contribution / signed contribution sum",
        title="Actual loss contribution fractions",
        output=output / "loss_fraction.png",
    )
    metadata = {
        "status": "completed",
        "source_files": [str(path.resolve()) for path in loss_component_files(run)],
        "rolling_window": int(rolling_window),
        "raw_columns": list(RAW_COLUMNS),
        "weighted_columns": list(WEIGHTED_COLUMNS),
        "fraction_semantics": (
            "signed weighted contribution divided by their signed sum; boundary "
            "NLL may be negative"
        ),
        "row_count": int(len(history)),
    }
    write_json(metadata_path, metadata)
    return {"status": "completed", "metadata": str(metadata_path)}


__all__ = [
    "RAW_COLUMNS",
    "WEIGHTED_COLUMNS",
    "add_rolling_statistics",
    "analyze_loss",
    "contribution_fractions",
    "load_loss_components",
    "loss_component_files",
]
