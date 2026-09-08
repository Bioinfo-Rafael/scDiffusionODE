#!/usr/bin/env python3
"""Plot hill_after_linear ODE and CellUnet parameters across checkpoints.

The comparison is intentionally fixed to the six consistency-weight conditions used
by the 20260830 suite.  Each output PNG is a 6 (weight) x 7 (checkpoint) grid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import read_json, write_json  # noqa: E402


@dataclass(frozen=True)
class Condition:
    label: str
    experiment: str
    runs_root_name: str


@dataclass(frozen=True)
class Checkpoint:
    label: str
    training_step: int
    use_ema: bool


CONDITIONS = (
    Condition("10", "09_hill_after_linear_lambda10", "runs"),
    Condition("1", "08_hill_after_linear_lambda1", "runs"),
    Condition("0.1", "07_hill_after_linear_lambda0p1", "runs"),
    Condition("0.01", "19_hill_after_linear_lambda0p01", "runs2"),
    Condition("0.001", "20_hill_after_linear_lambda0p001", "runs2"),
    Condition("10 -> 0.001 (exp)", "21_hill_after_linear_lambda10_to_0p001", "runs2"),
)
CHECKPOINTS = (
    Checkpoint("model\n0", 0, False),
    *(Checkpoint(f"EMA\n{step // 1000}k", step, True) for step in range(5000, 30001, 5000)),
)
TRAINABLE_PARAMETER_KEYS = ("W", "b", "raw_K", "raw_V", "raw_delta")
CATEGORY_ORDER = (
    "W_all",
    "W_mask_present",
    "W_mask_absent",
    "b",
    "raw_K",
    "K_effective",
    "raw_V",
    "V_effective",
    "raw_delta",
    "delta_effective",
)
CATEGORY_TITLES = {
    "W_all": "W: all entries",
    "W_mask_present": "W: mask present (mask = 1)",
    "W_mask_absent": "W: mask absent (mask = 0, diagonal included)",
    "b": "b (trainable)",
    "raw_K": "raw_K (trainable)",
    "K_effective": "K = softplus(raw_K) + epsilon",
    "raw_V": "raw_V (trainable)",
    "V_effective": "V = softplus(raw_V) + epsilon",
    "raw_delta": "raw_delta (trainable)",
    "delta_effective": "delta = softplus(raw_delta) + epsilon",
}
CELLUNET_AGGREGATE_ORDER = (
    "CellUnet_all_parameters",
    "CellUnet_all_weights",
    "CellUnet_all_biases",
)


def _checkpoint_filename(checkpoint: Checkpoint, ema_rate: str) -> str:
    if not checkpoint.use_ema:
        return f"model{checkpoint.training_step:06d}.pt"
    return f"ema_{ema_rate}_{checkpoint.training_step:06d}.pt"


def _checkpoint_path(run: Path, checkpoint: Checkpoint, ema_rate: str) -> Path:
    filename = _checkpoint_filename(checkpoint, ema_rate)
    candidates = list((run / "checkpoints").glob(f"segment_*/model/{filename}"))
    if not candidates:
        raise FileNotFoundError(f"missing {filename} below {run / 'checkpoints'}")

    def sort_key(path: Path) -> tuple[int, int, str]:
        try:
            segment = int(path.parents[1].name.removeprefix("segment_"))
        except ValueError:
            segment = -1
        return segment, path.stat().st_mtime_ns, str(path)

    return max(candidates, key=sort_key).resolve()


def _validate_run(run: Path, condition: Condition) -> tuple[dict, list[Path]]:
    config_path = run / "exp_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing exp_config.json: {run}")
    config = read_json(config_path)
    if config.get("experiment") != condition.experiment:
        raise ValueError(
            f"experiment mismatch in {config_path}: expected {condition.experiment!r}, "
            f"got {config.get('experiment')!r}"
        )
    if config.get("ode_type") != "hill_after_linear":
        raise ValueError(f"{run} is not a hill_after_linear run")
    ema_rate = str(config.get("ema_rate", "")).split(",")[0].strip()
    if not ema_rate:
        raise ValueError(f"ema_rate is missing from {config_path}")
    return config, [_checkpoint_path(run, item, ema_rate) for item in CHECKPOINTS]


def discover_common_batch(
    root: Path,
    conditions: Sequence[Condition],
    *,
    requested_batch_id: str = "",
) -> tuple[str, dict[str, Path]]:
    """Select one complete batch shared by every requested experiment."""

    root = Path(root).expanduser().resolve()
    by_experiment: dict[str, dict[str, Path]] = {}
    failures: list[str] = []
    for condition in conditions:
        experiment_root = root / condition.experiment
        candidates = (
            [experiment_root / requested_batch_id]
            if requested_batch_id
            else [path for path in experiment_root.iterdir() if path.is_dir()]
            if experiment_root.is_dir()
            else []
        )
        complete: dict[str, Path] = {}
        for candidate in candidates:
            try:
                _validate_run(candidate, condition)
            except (FileNotFoundError, KeyError, ValueError) as error:
                failures.append(f"{candidate}: {error}")
                continue
            complete[candidate.name] = candidate.resolve()
        by_experiment[condition.experiment] = complete

    common = set.intersection(
        *(set(paths) for paths in by_experiment.values())
    ) if by_experiment else set()
    if not common:
        requested = f" {requested_batch_id!r}" if requested_batch_id else ""
        detail = "\n".join(failures[-12:])
        message = f"no complete common batch{requested} under {root}"
        if detail:
            message += f"\nRejected candidates:\n{detail}"
        raise FileNotFoundError(message)

    def batch_mtime(batch_id: str) -> int:
        return max(
            by_experiment[condition.experiment][batch_id].stat().st_mtime_ns
            for condition in conditions
        )

    selected = max(common, key=lambda item: (batch_mtime(item), item))
    return selected, {
        condition.experiment: by_experiment[condition.experiment][selected]
        for condition in conditions
    }


def resolve_condition_runs(
    runs_root: Path,
    runs2_root: Path,
    *,
    runs_batch_id: str = "",
    runs2_batch_id: str = "",
) -> tuple[dict[str, Path], dict[str, str]]:
    canonical = tuple(item for item in CONDITIONS if item.runs_root_name == "runs")
    exploratory = tuple(item for item in CONDITIONS if item.runs_root_name == "runs2")
    canonical_batch, canonical_runs = discover_common_batch(
        runs_root, canonical, requested_batch_id=runs_batch_id
    )
    exploratory_batch, exploratory_runs = discover_common_batch(
        runs2_root, exploratory, requested_batch_id=runs2_batch_id
    )
    return (
        {**canonical_runs, **exploratory_runs},
        {"runs": canonical_batch, "runs2": exploratory_batch},
    )


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location="cpu")


def _clean_state_dict(payload) -> Mapping[str, torch.Tensor]:
    state = payload
    for wrapper in ("ema", "model", "state_dict"):
        if isinstance(state, Mapping) and wrapper in state and isinstance(state[wrapper], Mapping):
            state = state[wrapper]
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint does not contain a state_dict mapping")
    return {
        (name[len("module.") :] if str(name).startswith("module.") else str(name)): value
        for name, value in state.items()
    }


def _state_tensor(state: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    suffix = f"ode_model.{name}"
    matches = [
        value for key, value in state.items()
        if key == suffix or key.endswith(f".{suffix}")
    ]
    if len(matches) != 1:
        raise KeyError(f"expected exactly one checkpoint tensor ending in {suffix!r}, got {len(matches)}")
    value = matches[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"checkpoint value for {suffix} is not a tensor")
    return value.detach().cpu().float()


def _natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", value)
    )


def _cellunet_parameters(state: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    """Return all CellUnet trainable tensors stored below ``ml_model``.

    Cell_Unet has no running-statistic buffers, so every tensor in this namespace
    is a trainable parameter (Linear or LayerNorm weight/bias).
    """

    tensors: dict[str, np.ndarray] = {}
    for key, value in state.items():
        marker = "ml_model."
        if key.startswith(marker):
            name = key[len(marker) :]
        elif f".{marker}" in key:
            name = key.split(f".{marker}", 1)[1]
        else:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"CellUnet checkpoint value {key!r} is not a tensor")
        array = value.detach().cpu().float().numpy().reshape(-1)
        if not array.size:
            raise ValueError(f"CellUnet parameter {name!r} is empty")
        if not np.isfinite(array).all():
            raise ValueError(f"CellUnet parameter {name!r} contains non-finite values")
        tensors[name] = array
    if not tensors:
        raise KeyError("checkpoint contains no ml_model parameters")
    return dict(sorted(tensors.items(), key=lambda item: _natural_key(item[0])))


def _cellunet_categories(parameters: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    all_values = list(parameters.values())
    weights = [values for name, values in parameters.items() if name.endswith(".weight")]
    biases = [values for name, values in parameters.items() if name.endswith(".bias")]
    if not weights or not biases:
        raise ValueError("CellUnet checkpoint must contain both weight and bias parameters")
    categories = {
        "CellUnet_all_parameters": np.concatenate(all_values),
        "CellUnet_all_weights": np.concatenate(weights),
        "CellUnet_all_biases": np.concatenate(biases),
    }
    categories.update({f"CellUnet::{name}": values for name, values in parameters.items()})
    return categories


def load_parameter_categories(
    checkpoint_path: Path,
    *,
    positive_epsilon: float,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    categories, mask, _ = load_snapshot_categories(
        checkpoint_path, positive_epsilon=positive_epsilon
    )
    return {name: categories[name] for name in CATEGORY_ORDER}, mask


def load_snapshot_categories(
    checkpoint_path: Path,
    *,
    positive_epsilon: float,
    include_individual_cellunet: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray, tuple[str, ...]]:
    """Load ODE and CellUnet categories from one checkpoint in a single pass."""

    state = _clean_state_dict(_torch_load(Path(checkpoint_path)))
    tensors = {name: _state_tensor(state, name) for name in TRAINABLE_PARAMETER_KEYS}
    mask_tensor = _state_tensor(state, "mask")
    w = tensors["W"]
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"W must be square, got shape {tuple(w.shape)} in {checkpoint_path}")
    if tuple(mask_tensor.shape) != tuple(w.shape):
        raise ValueError(
            f"mask shape {tuple(mask_tensor.shape)} != W shape {tuple(w.shape)} in {checkpoint_path}"
        )
    mask = mask_tensor.numpy() > 0.5
    w_values = w.numpy()

    def array(name: str) -> np.ndarray:
        values = tensors[name].numpy().astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite {name} values in {checkpoint_path}")
        return values.reshape(-1)

    def positive(raw: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, raw) + float(positive_epsilon)

    raw_k = array("raw_K")
    raw_v = array("raw_V")
    raw_delta = array("raw_delta")
    categories = {
        "W_all": w_values.reshape(-1).astype(np.float64, copy=False),
        "W_mask_present": w_values[mask].reshape(-1).astype(np.float64, copy=False),
        "W_mask_absent": w_values[~mask].reshape(-1).astype(np.float64, copy=False),
        "b": array("b"),
        "raw_K": raw_k,
        "K_effective": positive(raw_k),
        "raw_V": raw_v,
        "V_effective": positive(raw_v),
        "raw_delta": raw_delta,
        "delta_effective": positive(raw_delta),
    }
    cell_parameters = _cellunet_parameters(state)
    cell_categories = _cellunet_categories(cell_parameters)
    if not include_individual_cellunet:
        cell_categories = {
            name: cell_categories[name] for name in CELLUNET_AGGREGATE_ORDER
        }
    categories.update(cell_categories)
    for name, values in categories.items():
        if values.size == 0:
            raise ValueError(f"parameter group {name} is empty in {checkpoint_path}")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values in {name}: {checkpoint_path}")
    return categories, mask, tuple(cell_parameters)


def summarize(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "count": int(flat.size),
        "mean": float(flat.mean()),
        "std": float(flat.std(ddof=0)),
        "median": float(np.median(flat)),
        "q05": float(np.quantile(flat, 0.05)),
        "q95": float(np.quantile(flat, 0.95)),
        "minimum": float(flat.min()),
        "maximum": float(flat.max()),
        "l2_norm": float(np.linalg.norm(flat)),
    }


def _histogram_edges(all_values: Sequence[np.ndarray], bins: int) -> np.ndarray:
    minimum = min(float(values.min()) for values in all_values)
    maximum = max(float(values.max()) for values in all_values)
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=1e-15):
        padding = max(abs(minimum) * 0.05, 1e-6)
        minimum -= padding
        maximum += padding
    return np.linspace(minimum, maximum, int(bins) + 1)


def plot_category_grid(
    snapshots: Mapping[tuple[str, int], Mapping[str, np.ndarray]],
    category: str,
    output_path: Path,
    *,
    bins: int = 200,
    dpi: int = 220,
    shared_edges: np.ndarray | None = None,
    y_max: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    values_list = [
        snapshots[(condition.experiment, checkpoint.training_step)][category]
        for condition in CONDITIONS
        for checkpoint in CHECKPOINTS
    ]
    edges = (
        np.asarray(shared_edges, dtype=np.float64)
        if shared_edges is not None
        else _histogram_edges(values_list, bins)
    )
    if edges.ndim != 1 or edges.size < 3 or not np.all(np.diff(edges) > 0):
        raise ValueError("histogram edges must be a strictly increasing 1D array")
    figure, axes = plt.subplots(
        len(CONDITIONS),
        len(CHECKPOINTS),
        figsize=(24, 18),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    is_cellunet = category.startswith("CellUnet")
    color = "#4D7C4A" if is_cellunet else "#3264A8" if category.startswith("W") else "#B5502D"
    for row, condition in enumerate(CONDITIONS):
        for column, checkpoint in enumerate(CHECKPOINTS):
            axis = axes[row, column]
            values = snapshots[(condition.experiment, checkpoint.training_step)][category]
            counts, _ = np.histogram(values, bins=edges)
            percentages = counts.astype(np.float64) * (100.0 / values.size)
            axis.bar(
                edges[:-1],
                percentages,
                width=np.diff(edges),
                align="edge",
                color=color,
                alpha=0.82,
                linewidth=0,
            )
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            axis.axvline(mean, color="#B22222", linewidth=1.0, linestyle="--")
            axis.text(
                0.97,
                0.94,
                f"mean {mean:.3e}\nstd  {std:.3e}\nn={values.size:,}",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.82, "pad": 2.5},
            )
            axis.grid(axis="y", alpha=0.2, linewidth=0.6)
            if y_max is not None:
                axis.set_ylim(0.0, float(y_max))
            axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-3, 3))
            axis.xaxis.set_major_formatter(formatter)
            axis.tick_params(labelsize=8)
            if row == 0:
                axis.set_title(checkpoint.label, fontsize=11, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"lambda: {condition.label}\npercent / bin", fontsize=10)
            if row == len(CONDITIONS) - 1:
                axis.set_xlabel("parameter value", fontsize=9)
    figure.suptitle(
        (
            f"CellUnet parameter distributions — {category.removeprefix('CellUnet::').replace('_', ' ')}"
            if is_cellunet
            else f"hill_after_linear parameter distributions — {CATEGORY_TITLES[category]}"
        ),
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)


def analyze(
    *,
    runs_root: Path,
    runs2_root: Path,
    output_dir: Path,
    runs_batch_id: str = "",
    runs2_batch_id: str = "",
    bins: int = 200,
    dpi: int = 220,
    force: bool = False,
    all_parameters: bool = False,
    independent_x: bool = False,
    x_min: float = -0.5,
    x_max: float = 0.5,
    y_max: float = 30.0,
) -> dict:
    if int(bins) < 2:
        raise ValueError("bins must be at least 2")
    if int(dpi) <= 0:
        raise ValueError("dpi must be positive")
    if not math.isfinite(float(x_min)) or not math.isfinite(float(x_max)) or x_min >= x_max:
        raise ValueError("x-min and x-max must be finite with x-min < x-max")
    if not math.isfinite(float(y_max)) or y_max <= 0:
        raise ValueError("y-max must be finite and positive")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not force:
        raise FileExistsError(f"output directory is not empty (use --force): {output}")
    output.mkdir(parents=True, exist_ok=True)

    runs, batches = resolve_condition_runs(
        runs_root,
        runs2_root,
        runs_batch_id=runs_batch_id,
        runs2_batch_id=runs2_batch_id,
    )
    print(f"runs batch:  {batches['runs']}", flush=True)
    print(f"runs2 batch: {batches['runs2']}", flush=True)

    snapshots: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    sources = []
    reference_mask: np.ndarray | None = None
    cellunet_parameter_names: tuple[str, ...] | None = None
    for condition in CONDITIONS:
        run = runs[condition.experiment]
        config, checkpoint_paths = _validate_run(run, condition)
        epsilon = float(config.get("positive_epsilon", 1e-6))
        for checkpoint, checkpoint_path in zip(CHECKPOINTS, checkpoint_paths):
            print(
                f"LOAD lambda={condition.label:<17} step={checkpoint.training_step:06d} "
                f"{checkpoint_path}",
                flush=True,
            )
            categories, mask, current_cellunet_names = load_snapshot_categories(
                checkpoint_path,
                positive_epsilon=epsilon,
                include_individual_cellunet=all_parameters,
            )
            if reference_mask is None:
                reference_mask = mask.copy()
            elif not np.array_equal(reference_mask, mask):
                raise ValueError(f"ODE mask changed across snapshots: {checkpoint_path}")
            if cellunet_parameter_names is None:
                cellunet_parameter_names = current_cellunet_names
            elif cellunet_parameter_names != current_cellunet_names:
                raise ValueError(f"CellUnet parameter names changed across snapshots: {checkpoint_path}")
            snapshots[(condition.experiment, checkpoint.training_step)] = categories
            sources.append({
                "weight_label": condition.label,
                "experiment": condition.experiment,
                "runs_root": condition.runs_root_name,
                "run_directory": str(run),
                "checkpoint_label": checkpoint.label.replace("\n", " "),
                "training_step": checkpoint.training_step,
                "checkpoint_kind": "EMA" if checkpoint.use_ema else "raw model",
                "checkpoint_path": str(checkpoint_path),
            })
    assert cellunet_parameter_names is not None
    complete_category_order = (
        *CATEGORY_ORDER,
        *CELLUNET_AGGREGATE_ORDER,
        *(f"CellUnet::{name}" for name in cellunet_parameter_names),
    )
    selected_categories = (
        complete_category_order
        if all_parameters
        else (
            "W_all",
            "W_mask_present",
            "W_mask_absent",
            *CELLUNET_AGGREGATE_ORDER,
        )
    )
    shared_edges = None
    if not independent_x:
        shared_edges = np.linspace(float(x_min), float(x_max), int(bins) + 1)

    summary_rows = []
    source_by_snapshot = {
        (source["experiment"], source["training_step"]): source for source in sources
    }
    for condition in CONDITIONS:
        for checkpoint in CHECKPOINTS:
            source = source_by_snapshot[(condition.experiment, checkpoint.training_step)]
            categories = snapshots[(condition.experiment, checkpoint.training_step)]
            for category in selected_categories:
                summary_rows.append({
                    "weight_label": condition.label,
                    "experiment": condition.experiment,
                    "runs_root": condition.runs_root_name,
                    "batch_id": batches[condition.runs_root_name],
                    "training_step": checkpoint.training_step,
                    "checkpoint_kind": "ema" if checkpoint.use_ema else "raw_model",
                    "checkpoint_path": source["checkpoint_path"],
                    "parameter_group": category,
                    **summarize(categories[category]),
                })

    figure_paths = []
    for category in selected_categories:
        index = complete_category_order.index(category) + 1
        safe_name = re.sub(r"[^A-Za-z0-9]+", "_", category).strip("_")
        destination = output / f"{index:02d}_{safe_name}.png"
        print(f"PLOT {destination.name}", flush=True)
        plot_category_grid(
            snapshots,
            category,
            destination,
            bins=bins,
            dpi=dpi,
            shared_edges=shared_edges,
            y_max=y_max,
        )
        figure_paths.append(str(destination))

    summary_path = output / "parameter_distribution_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    assert reference_mask is not None
    metadata = {
        "status": "completed",
        "model_family": "hill_after_linear",
        "batches": batches,
        "runs_root": str(Path(runs_root).expanduser().resolve()),
        "runs2_root": str(Path(runs2_root).expanduser().resolve()),
        "output_directory": str(output),
        "grid": {
            "rows": [condition.label for condition in CONDITIONS],
            "columns": [checkpoint.label.replace("\n", " ") for checkpoint in CHECKPOINTS],
        },
        "checkpoint_policy": (
            "raw model000000.pt (legacy loop filename after its first optimizer update); "
            "EMA checkpoints at exact 5,000-step intervals through 30,000"
        ),
        "ema_rate_policy": "first ema_rate value stored in each run's exp_config.json",
        "mask": {
            "orientation": "target_source",
            "shape": list(reference_mask.shape),
            "present_count": int(reference_mask.sum()),
            "absent_count": int((~reference_mask).sum()),
            "absent_definition": "mask == 0, including diagonal entries when absent",
        },
        "parameter_groups": list(selected_categories),
        "available_parameter_groups": list(complete_category_order),
        "ode_trainable_parameter_groups": ["W", "b", "raw_K", "raw_V", "raw_delta"],
        "derived_effective_groups": ["K_effective", "V_effective", "delta_effective"],
        "cellunet": {
            "aggregate_groups": list(CELLUNET_AGGREGATE_ORDER),
            "individual_parameter_names": list(cellunet_parameter_names),
            "individual_png_per_parameter": bool(all_parameters),
            "buffer_policy": "Cell_Unet has no running-statistic buffers; ml_model tensors are trainable parameters",
        },
        "histogram": {
            "bins": int(bins),
            "y_axis": "percent of entries per bin",
            "common_edges_within_each_png": True,
            "common_edges_across_all_pngs": not independent_x,
            "fixed_x_range": [float(x_min), float(x_max)] if not independent_x else None,
            "fixed_y_range": [0.0, float(y_max)],
            "values_outside_x_range": "excluded from displayed bars but retained in summary statistics",
            "full_min_max_range_without_clipping": True,
            "mean_line": "red dashed",
            "std_definition": "population standard deviation (ddof=0)",
        },
        "figures": figure_paths,
        "summary_csv": str(summary_path),
        "sources": sources,
    }
    write_json(output / "metadata.json", metadata)
    print("HILL_AFTER_LINEAR_PARAMETER_DISTRIBUTIONS_COMPLETE", flush=True)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create one 6x7 distribution grid per hill_after_linear ODE parameter "
            "from canonical runs and exploratory runs2 checkpoints."
        )
    )
    parser.add_argument("--runs-root", default=str(SUITE_ROOT / "runs"))
    parser.add_argument("--runs2-root", default=str(SUITE_ROOT / "runs2"))
    parser.add_argument("--runs-batch-id", default="", help="default: newest complete common batch")
    parser.add_argument("--runs2-batch-id", default="", help="default: newest complete common batch")
    parser.add_argument(
        "--output-dir",
        default=str(
            SUITE_ROOT
            / "analysis_results"
            / "hill_after_linear_parameter_distributions"
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
    )
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--x-min", type=float, default=-0.5)
    parser.add_argument("--x-max", type=float, default=0.5)
    parser.add_argument("--y-max", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--all-parameters",
        action="store_true",
        help="also plot ODE K/V/b/delta and every individual CellUnet tensor",
    )
    parser.add_argument(
        "--independent-x",
        action="store_true",
        help="choose a separate horizontal range for each PNG instead of one shared range",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metadata = analyze(
        runs_root=Path(args.runs_root),
        runs2_root=Path(args.runs2_root),
        output_dir=Path(args.output_dir),
        runs_batch_id=args.runs_batch_id,
        runs2_batch_id=args.runs2_batch_id,
        bins=args.bins,
        dpi=args.dpi,
        force=args.force,
        all_parameters=args.all_parameters,
        independent_x=args.independent_x,
        x_min=args.x_min,
        x_max=args.x_max,
        y_max=args.y_max,
    )
    print(json.dumps({
        "status": metadata["status"],
        "output_directory": metadata["output_directory"],
        "figure_count": len(metadata["figures"]),
        "summary_csv": metadata["summary_csv"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
