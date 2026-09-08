#!/usr/bin/env python3
"""Plot std/mean trajectories produced by parameter-distribution analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


WEIGHT_ORDER = ("10", "1", "0.1", "0.01", "0.001", "10 -> 0.001 (exp)")
COLORS = ("#7F0000", "#D95F02", "#E6AB02", "#1B9E77", "#377EB8", "#6A3D9A")
MARKERS = ("o", "s", "^", "D", "v", "P")
EXPECTED_STEPS = (0, 5000, 10000, 15000, 20000, 25000, 30000)
FIGURE_FILES = (
    "01_W_all_std_over_mean.csv",
    "02_W_mask_present_std_over_mean.csv",
    "03_W_mask_absent_std_over_mean.csv",
    "11_CellUnet_all_parameters_std_over_mean.csv",
    "12_CellUnet_all_weights_std_over_mean.csv",
    "13_CellUnet_all_biases_std_over_mean.csv",
)


def _validate(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    required = {"figure", "weight_label", "training_step", "std_over_mean"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns in {source}: {missing}")
    result = frame.copy()
    result["weight_label"] = result["weight_label"].astype(str)
    result["training_step"] = pd.to_numeric(result["training_step"], errors="raise").astype(int)
    result["std_over_mean"] = pd.to_numeric(result["std_over_mean"], errors="coerce")
    if result.duplicated(["weight_label", "training_step"]).any():
        raise ValueError(f"duplicate weight/step rows in {source}")
    observed_weights = set(result["weight_label"])
    if observed_weights != set(WEIGHT_ORDER):
        raise ValueError(
            f"weight labels differ in {source}: expected {WEIGHT_ORDER}, got {sorted(observed_weights)}"
        )
    for weight in WEIGHT_ORDER:
        steps = tuple(sorted(result.loc[result["weight_label"] == weight, "training_step"]))
        if steps != EXPECTED_STEPS:
            raise ValueError(f"{source}: {weight} has steps {steps}, expected {EXPECTED_STEPS}")
    return result


def plot_one(source: Path, output: Path, *, dpi: int = 220) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    frame = _validate(pd.read_csv(source), source)
    figure_name = str(frame["figure"].iloc[0])
    if frame["figure"].astype(str).nunique() != 1:
        raise ValueError(f"multiple figure labels in {source}")

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    finite_points = 0
    for weight, color, marker in zip(WEIGHT_ORDER, COLORS, MARKERS):
        subset = frame[frame["weight_label"] == weight].sort_values("training_step")
        x = subset["training_step"].to_numpy(dtype=float)
        y = subset["std_over_mean"].to_numpy(dtype=float)
        finite_points += int(np.isfinite(y).sum())
        ax.plot(x, y, color=color, marker=marker, linewidth=1.8,
                markersize=5.5, label=weight)

    ax.axhline(0.0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlim(EXPECTED_STEPS[0], EXPECTED_STEPS[-1])
    ax.set_xticks(EXPECTED_STEPS)
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: "0" if value == 0 else f"{int(value / 1000)}k")
    )
    ax.set_xlabel("Training step")
    ax.set_ylabel("std / mean")
    ax.set_title(f"{figure_name}: std / mean across training")
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.legend(title="Consistency weight", ncol=2, frameon=True)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return {"figure": figure_name, "source": str(source.resolve()),
            "output": str(output.resolve()), "finite_points": finite_points,
            "total_points": int(len(frame))}


def run(input_dir: Path, output_dir: Path, *, dpi: int = 220) -> dict:
    source_root = Path(input_dir).expanduser().resolve()
    destination_root = Path(output_dir).expanduser().resolve()
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")
    if not source_root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {source_root}")
    destination_root.mkdir(parents=True, exist_ok=True)

    results = []
    for filename in FIGURE_FILES:
        source = source_root / filename
        if not source.is_file():
            raise FileNotFoundError(f"missing input CSV: {source}")
        output = destination_root / filename.replace("_std_over_mean.csv", "_std_over_mean.png")
        print(f"PLOT {source.name} -> {output.name}", flush=True)
        results.append(plot_one(source, output, dpi=dpi))

    metadata = {
        "status": "completed", "input_directory": str(source_root),
        "output_directory": str(destination_root), "x_axis": "training_step",
        "y_axis": "std_over_mean", "smoothing": False,
        "weight_order": list(WEIGHT_ORDER), "expected_steps": list(EXPECTED_STEPS),
        "figures": results,
    }
    (destination_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("STD_OVER_MEAN_PLOTS_COMPLETE", flush=True)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot six weight trajectories from each std_over_mean parameter CSV."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="", help="default: <input-dir>/figures")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "figures"
    result = run(input_dir, output_dir, dpi=args.dpi)
    print(json.dumps({"status": result["status"],
                      "output_directory": result["output_directory"],
                      "figure_count": len(result["figures"])},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
