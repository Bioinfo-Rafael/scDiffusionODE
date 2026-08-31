"""Run discovery, checkpoint restoration, per-run analysis, and 12-run summary."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from guided_diffusion import dist_util
from guided_diffusion.script_util import create_gaussian_diffusion
from ODE.ode_20260609_mathmlp import clean_state_dict

from models import build_model_from_config
from scripts.common import (
    EXPERIMENT_ORDER,
    RUNS_ROOT,
    checkpoint_files,
    choose_sampling_checkpoint,
    read_json,
    validate_config,
    write_json,
)

from .gradients import (
    analyze_gradients,
    checkpoint_training_step,
    parameter_fingerprint,
    select_analysis_checkpoints,
)
from .loss_history import late_loss_summary, load_loss_history
from .metrics import evaluate_diffusion_timesteps
from .plotting import plot_gradient_figures, plot_run_figures, plot_summary_figures


DEFAULT_TIMESTEPS = (0, 249, 499, 616, 749, 999)
QUICK_TIMESTEPS = (0, 1, 10, 100, 499, 999)


@dataclass(frozen=True)
class AnalysisOptions:
    preset: str = "default"
    cells: int | None = None
    batch_size: int = 128
    timesteps: tuple[int, ...] | None = None
    gradient_timesteps: tuple[int, ...] | None = None
    gradient_cells: int | None = None
    rolling_window: int = 100
    seed: int = 1234
    device: str = "auto"
    force: bool = False
    gradient_only: bool = False


def parse_timestep_spec(spec: str, num_timesteps: int) -> tuple[int, ...]:
    """Parse comma values plus inclusive ``start-stop[:stride]`` ranges."""

    if not spec:
        return ()
    values: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            range_part, _, stride_part = token.partition(":")
            start_text, stop_text = range_part.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            stride = int(stride_part) if stride_part else 1
            if stride <= 0 or stop < start:
                raise ValueError(f"invalid timestep range: {token}")
            values.update(range(start, stop + 1, stride))
        else:
            values.add(int(token))
    result = tuple(sorted(values))
    if not result or result[0] < 0 or result[-1] >= num_timesteps:
        raise ValueError(f"timesteps must be within [0,{num_timesteps - 1}]")
    return result


def preset_timesteps(preset: str, num_timesteps: int) -> tuple[int, ...]:
    if preset == "full":
        return tuple(range(num_timesteps))
    source = QUICK_TIMESTEPS if preset == "quick" else DEFAULT_TIMESTEPS
    return tuple(value for value in source if value < num_timesteps)


def preset_cells(preset: str) -> int:
    return {"quick": 128, "default": 2048, "full": 4096}[preset]


def discover_run_directories(
    *,
    run_dirs: Sequence[str] | None = None,
    batch_id: str = "",
    require_all: bool = False,
) -> list[Path]:
    if run_dirs:
        found = [Path(value).expanduser().resolve() for value in run_dirs]
    else:
        found = []
        for experiment in EXPERIMENT_ORDER:
            root = RUNS_ROOT / experiment
            candidates = [
                path for path in root.iterdir()
                if path.is_dir() and (path / "exp_config.json").is_file()
            ] if root.is_dir() else []
            if batch_id:
                candidates = [path for path in candidates if path.name == batch_id]
            complete = []
            for candidate in candidates:
                try:
                    candidate_config = read_json(candidate / "exp_config.json")
                    choose_sampling_checkpoint(candidate, candidate_config["ema_rate"])
                    complete.append(candidate)
                except (FileNotFoundError, KeyError):
                    continue
            candidates = complete
            if candidates:
                found.append(max(candidates, key=lambda path: path.stat().st_mtime_ns).resolve())
    for run in found:
        if not (run / "exp_config.json").is_file():
            raise FileNotFoundError(f"missing exp_config.json: {run}")
        validate_config(read_json(run / "exp_config.json"))
    if require_all:
        names = {read_json(run / "exp_config.json")["experiment"] for run in found}
        missing = [name for name in EXPERIMENT_ORDER if name not in names]
        if missing:
            raise FileNotFoundError("missing canonical runs: " + ", ".join(missing))
    return sorted(found, key=lambda run: EXPERIMENT_ORDER.index(read_json(run / "exp_config.json")["experiment"]))


def select_device(name: str) -> torch.device:
    choice = str(name).lower()
    if choice == "auto":
        if torch.cuda.is_available():
            choice = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            choice = "mps"
        else:
            choice = "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(choice)


def create_diffusion(config: dict):
    return create_gaussian_diffusion(
        steps=int(config["diffusion_steps"]),
        learn_sigma=bool(config["learn_sigma"]),
        noise_schedule=str(config["noise_schedule"]),
        use_kl=bool(config["use_kl"]),
        predict_xstart=bool(config["predict_xstart"]),
        rescale_timesteps=bool(config["rescale_timesteps"]),
        rescale_learned_sigmas=bool(config["rescale_learned_sigmas"]),
        timestep_respacing=config["timestep_respacing"],
    )


def load_analysis_cells(config: dict, cells: int, seed: int) -> tuple[torch.Tensor, list[str], np.ndarray]:
    import scanpy as sc

    adata = sc.read_h5ad(config["data_dir"])
    if "gene_name" not in adata.var.columns:
        raise KeyError("AnnData .var must contain gene_name")
    genes = [str(value) for value in adata.var["gene_name"].tolist()]
    if len(genes) != adata.n_vars or len(set(genes)) != len(genes):
        raise ValueError("gene_name must be a unique one-to-one mapping")
    count = min(int(cells), int(adata.n_obs))
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(int(adata.n_obs), size=count, replace=False))
    source = adata.X[indices]
    if hasattr(source, "toarray"):
        source = source.toarray()
    array = np.asarray(source, dtype=np.float32)
    return torch.from_numpy(array), genes, indices


def load_checkpoint(model, checkpoint, device) -> None:
    state = clean_state_dict(dist_util.load_state_dict(str(checkpoint), map_location="cpu"))
    model.load_state_dict(state, strict=True)
    model.to(device)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[3],
        ).strip()
    except Exception:
        return "unknown"


def _analysis_dirs(run: Path) -> dict[str, Path]:
    root = run / "analysis" / "detailed_20260830"
    paths = {"root": root, "csv": root / "csv", "figures": root / "figures"}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def analyze_run(run_dir, options: AnalysisOptions) -> dict:
    run = Path(run_dir).resolve()
    config = read_json(run / "exp_config.json")
    validate_config(config)
    paths = _analysis_dirs(run)
    completion = paths["root"] / "analysis_complete.json"

    device = select_device(options.device)
    diffusion = create_diffusion(config)
    timesteps = options.timesteps or preset_timesteps(options.preset, diffusion.num_timesteps)
    cells = options.cells or preset_cells(options.preset)
    gradient_cells = options.gradient_cells or ({"quick": 8, "default": 32, "full": 64}[options.preset])
    gradient_timesteps = options.gradient_timesteps or (
        0, diffusion.num_timesteps // 2, diffusion.num_timesteps - 1
    )
    x_start, genes, indices = load_analysis_cells(config, cells, options.seed)
    np.save(paths["root"] / "cell_indices.npy", indices)
    final_checkpoint = choose_sampling_checkpoint(run, config["ema_rate"])
    final_training_step = checkpoint_training_step(final_checkpoint)
    base_metadata = {
        "experiment": config["experiment"],
        "ode_type": config["ode_type"],
        "cell_ode_reg_lambda_20260830": float(config["cell_ode_reg_lambda_20260830"]),
        "run_directory": str(run),
        "checkpoint_path": str(final_checkpoint.resolve()),
        "checkpoint_training_step": final_training_step,
    }
    selections = select_analysis_checkpoints(checkpoint_files(run))
    existing_config_path = paths["root"] / "analysis_config.json"
    reuse_partial = False
    stored = read_json(existing_config_path) if existing_config_path.exists() else {}
    if stored and not options.force:
        reuse_partial = (
            stored.get("checkpoint_path") == str(final_checkpoint.resolve())
            and stored.get("analysis_seed") == int(options.seed)
            and stored.get("analyzed_cells") == int(len(indices))
            and stored.get("gradient_cells") == int(min(gradient_cells, len(indices)))
            and stored.get("analysis_batch_size") == int(options.batch_size)
        )
    if (
        completion.exists()
        and stored
        and not options.force
        and not options.gradient_only
    ):
        matches = (
            reuse_partial
            and stored.get("diffusion_timesteps") == list(timesteps)
            and stored.get("gradient_diffusion_timesteps") == list(gradient_timesteps)
            and stored.get("rolling_window") == int(options.rolling_window)
        )
        if matches:
            return read_json(completion)
    analysis_config = {
        **base_metadata,
        "analysis_seed": int(options.seed),
        "analyzed_cells": int(len(indices)),
        "cell_indices_path": str((paths["root"] / "cell_indices.npy").resolve()),
        "diffusion_timesteps": list(timesteps),
        "gradient_diffusion_timesteps": list(gradient_timesteps),
        "gradient_cells": int(min(gradient_cells, len(indices))),
        "analysis_batch_size": int(options.batch_size),
        "dataset_path": config["data_dir"],
        "device": str(device),
        "preset": options.preset,
        "rolling_window": int(options.rolling_window),
        "git_commit": git_commit(),
        "training_config": config,
        "normal_metrics_grad_mode": False,
        "gradient_analysis_uses_optimizer": False,
        "gradient_checkpoints": selections,
    }
    write_json(paths["root"] / "analysis_config.json", analysis_config)

    model = build_model_from_config(config, genes, diffusion.num_timesteps, device)
    diffusion_metrics = pd.DataFrame()
    cell_ode_metrics = pd.DataFrame()
    history = pd.DataFrame()
    fractions = pd.DataFrame()
    if not options.gradient_only:
        load_checkpoint(model, final_checkpoint, device)
        before = parameter_fingerprint(model)
        diffusion_path = paths["csv"] / "diffusion_metrics_by_timestep.csv"
        cell_ode_path = paths["csv"] / "cell_ode_metrics_by_timestep.csv"
        if reuse_partial and diffusion_path.exists() and cell_ode_path.exists():
            diffusion_metrics = pd.read_csv(diffusion_path)
            cell_ode_metrics = pd.read_csv(cell_ode_path)
            diffusion_metrics = diffusion_metrics[
                diffusion_metrics["diffusion_timestep"].isin(timesteps)
                & (diffusion_metrics["checkpoint_path"] == str(final_checkpoint.resolve()))
            ]
            cell_ode_metrics = cell_ode_metrics[
                cell_ode_metrics["diffusion_timestep"].isin(timesteps)
                & (cell_ode_metrics["checkpoint_path"] == str(final_checkpoint.resolve()))
            ]
        completed_timesteps = set(diffusion_metrics.get("diffusion_timestep", [])) & set(
            cell_ode_metrics.get("diffusion_timestep", [])
        )
        for timestep in timesteps:
            if timestep in completed_timesteps:
                continue
            current_diffusion, current_ode = evaluate_diffusion_timesteps(
                model, diffusion, x_start, [timestep],
                batch_size=options.batch_size, seed=options.seed, device=device,
                metadata=base_metadata,
            )
            diffusion_metrics = pd.concat(
                [diffusion_metrics, current_diffusion], ignore_index=True
            )
            cell_ode_metrics = pd.concat(
                [cell_ode_metrics, current_ode], ignore_index=True
            )
            diffusion_metrics = diffusion_metrics.sort_values("diffusion_timestep")
            cell_ode_metrics = cell_ode_metrics.sort_values("diffusion_timestep")
            # Save every completed diffusion timestep so interrupted full runs resume.
            diffusion_metrics.to_csv(diffusion_path, index=False)
            cell_ode_metrics.to_csv(cell_ode_path, index=False)
        after = parameter_fingerprint(model)
        if before != after:
            raise RuntimeError("normal metric analysis mutated checkpoint state")
        history, fractions = load_loss_history(
            run, config, rolling_window=options.rolling_window
        )
        for frame in (history, fractions):
            for column, value in reversed((
                ("experiment", config["experiment"]),
                ("ode_type", config["ode_type"]),
                (
                    "cell_ode_reg_lambda_20260830",
                    float(config["cell_ode_reg_lambda_20260830"]),
                ),
            )):
                if column not in frame.columns:
                    frame.insert(0, column, value)
        history.to_csv(paths["csv"] / "loss_history.csv", index=False)
        fractions.to_csv(paths["csv"] / "loss_fraction.csv", index=False)

    gradient_path = paths["csv"] / "gradient_metrics.csv"
    if reuse_partial and gradient_path.exists() and gradient_path.stat().st_size > 1:
        try:
            gradients = pd.read_csv(gradient_path)
        except pd.errors.EmptyDataError:
            gradients = pd.DataFrame()
    else:
        gradients = pd.DataFrame()
    for selection in selections:
        completed_pairs = set()
        if not gradients.empty:
            completed_pairs = set(zip(
                gradients["checkpoint_path"], gradients["diffusion_timestep"]
            ))
        missing_timesteps = [
            timestep for timestep in gradient_timesteps
            if (selection["checkpoint_path"], timestep) not in completed_pairs
        ]
        if not missing_timesteps:
            continue
        load_checkpoint(model, selection["checkpoint_path"], device)
        metadata = {**base_metadata, **selection}
        current = analyze_gradients(
            model,
            diffusion,
            x_start[: min(gradient_cells, len(x_start))],
            missing_timesteps,
            cell_ode_lambda=float(config["cell_ode_reg_lambda_20260830"]),
            ode_reg_lambda=float(config["ode_reg_lambda"]),
            seed=options.seed,
            device=device,
            metadata=metadata,
        )
        gradients = pd.concat([gradients, current], ignore_index=True)
        gradients = gradients.sort_values(
            ["checkpoint_training_step", "diffusion_timestep"]
        )
        # Save after every checkpoint/timestep group for resumability.
        gradients.to_csv(gradient_path, index=False)
    if gradients.empty:
        gradients.to_csv(gradient_path, index=False)

    if options.gradient_only:
        existing_diff = paths["csv"] / "diffusion_metrics_by_timestep.csv"
        existing_ode = paths["csv"] / "cell_ode_metrics_by_timestep.csv"
        existing_loss = paths["csv"] / "loss_history.csv"
        existing_fraction = paths["csv"] / "loss_fraction.csv"
        if all(path.exists() for path in (existing_diff, existing_ode, existing_loss, existing_fraction)):
            diffusion_metrics = pd.read_csv(existing_diff)
            cell_ode_metrics = pd.read_csv(existing_ode)
            history = pd.read_csv(existing_loss)
            fractions = pd.read_csv(existing_fraction)
    if not diffusion_metrics.empty:
        plot_run_figures(
            diffusion_metrics, cell_ode_metrics, history, fractions, gradients,
            paths["figures"], rolling_window=options.rolling_window,
            zoom_timestep_min=1,
        )
    elif not gradients.empty:
        plot_gradient_figures(gradients, paths["figures"], base_metadata)

    result = {
        **base_metadata,
        "status": "completed",
        "analysis_root": str(paths["root"]),
        "analysis_config_path": str(paths["root"] / "analysis_config.json"),
        "csv_directory": str(paths["csv"]),
        "figure_directory": str(paths["figures"]),
        "normal_metrics_rows": int(len(diffusion_metrics)),
        "gradient_rows": int(len(gradients)),
    }
    if not options.gradient_only:
        write_json(completion, result)
    return result


def _region(timestep: int, num_timesteps: int) -> str:
    if timestep < num_timesteps / 3:
        return "low_diffusion_timestep"
    if timestep < 2 * num_timesteps / 3:
        return "mid_diffusion_timestep"
    return "high_diffusion_timestep"


def summarize_run(run_dir) -> tuple[dict, list[dict]]:
    run = Path(run_dir).resolve()
    config = read_json(run / "exp_config.json")
    root = run / "analysis" / "detailed_20260830" / "csv"
    diffusion = pd.read_csv(root / "diffusion_metrics_by_timestep.csv")
    cell_ode = pd.read_csv(root / "cell_ode_metrics_by_timestep.csv")
    history = pd.read_csv(root / "loss_history.csv")
    fraction = pd.read_csv(root / "loss_fraction.csv")
    gradient_path = root / "gradient_metrics.csv"
    gradients = pd.read_csv(gradient_path) if gradient_path.exists() and gradient_path.stat().st_size else pd.DataFrame()
    joined = diffusion.merge(
        cell_ode,
        on=[
            "experiment", "ode_type", "cell_ode_reg_lambda_20260830",
            "run_directory", "checkpoint_path", "checkpoint_training_step",
            "diffusion_timestep", "analyzed_cells", "diffusion_target",
        ],
        how="inner",
    )
    joined["norm_ratio_deviation_from_1"] = (
        joined["ode_cell_norm_ratio_mean"] - 1.0
    ).abs()
    metric_columns = (
        "cell_target_pearson_global",
        "cell_target_mse_mean",
        "cell_target_cosine_mean",
        "cell_ode_pearson_global",
        "cell_ode_cosine_mean",
        "cell_ode_mse_mean",
        "cell_ode_nmse_mean",
        "ode_cell_norm_ratio_mean",
        "norm_ratio_deviation_from_1",
    )
    base = {
        "experiment": config["experiment"],
        "ode_type": config["ode_type"],
        "cell_ode_reg_lambda_20260830": float(config["cell_ode_reg_lambda_20260830"]),
        "run_directory": str(run),
    }
    row = dict(base)
    for column in metric_columns:
        row[f"{column}_timestep_mean"] = float(joined[column].mean())
        row[f"{column}_timestep_median"] = float(joined[column].median())
    row.update(late_loss_summary(history, fraction))
    if not gradients.empty:
        for column in (
            "cell_gradient_cosine", "cell_gradient_norm_ratio",
            "ode_gradient_cosine", "ode_gradient_norm_ratio",
        ):
            row[f"{column}_mean"] = float(gradients[column].mean())
            row[f"{column}_median"] = float(gradients[column].median())
    else:
        for column in (
            "cell_gradient_cosine", "cell_gradient_norm_ratio",
            "ode_gradient_cosine", "ode_gradient_norm_ratio",
        ):
            row[f"{column}_mean"] = np.nan
            row[f"{column}_median"] = np.nan

    num_timesteps = int(config["diffusion_steps"])
    regional = []
    joined["diffusion_timestep_region"] = joined["diffusion_timestep"].map(
        lambda value: _region(int(value), num_timesteps)
    )
    for region, group in joined.groupby("diffusion_timestep_region"):
        region_row = {**base, "diffusion_timestep_region": region}
        for column in metric_columns:
            region_row[f"{column}_mean"] = float(group[column].mean())
            region_row[f"{column}_median"] = float(group[column].median())
        regional.append(region_row)
    return row, regional


def summarize_runs(run_dirs: Iterable[Path], output_root, *, require_all: bool = True) -> dict:
    runs = list(run_dirs)
    rows, regional = [], []
    all_diffusion, all_ode, all_loss, all_fraction, all_gradient = [], [], [], [], []
    reference_reproducibility = None
    for run in runs:
        analysis_root = Path(run) / "analysis" / "detailed_20260830"
        analysis_config_path = analysis_root / "analysis_config.json"
        indices_path = analysis_root / "cell_indices.npy"
        if analysis_config_path.exists() and indices_path.exists():
            current_config = read_json(analysis_config_path)
            current = {
                "analysis_seed": current_config.get("analysis_seed"),
                "dataset_path": current_config.get("dataset_path"),
                "diffusion_timesteps": current_config.get("diffusion_timesteps"),
                "cell_indices": np.load(indices_path).tolist(),
            }
            if reference_reproducibility is None:
                reference_reproducibility = current
            elif current != reference_reproducibility:
                raise ValueError(
                    "12-condition summary requires identical dataset, cell subset, "
                    "seed, and diffusion timesteps"
                )
        row, region_rows = summarize_run(run)
        rows.append(row); regional.extend(region_rows)
        csv_root = Path(run) / "analysis" / "detailed_20260830" / "csv"
        all_diffusion.append(pd.read_csv(csv_root / "diffusion_metrics_by_timestep.csv"))
        all_ode.append(pd.read_csv(csv_root / "cell_ode_metrics_by_timestep.csv"))
        all_loss.append(pd.read_csv(csv_root / "loss_history.csv"))
        all_fraction.append(pd.read_csv(csv_root / "loss_fraction.csv"))
        gradient_path = csv_root / "gradient_metrics.csv"
        if gradient_path.exists() and gradient_path.stat().st_size:
            all_gradient.append(pd.read_csv(gradient_path))
    summary = pd.DataFrame(rows)
    if require_all:
        names = set(summary["experiment"])
        missing = [name for name in EXPERIMENT_ORDER if name not in names]
        if missing or len(summary) != 12:
            raise ValueError("12-condition summary requires one run per canonical condition")
    order = {name: index for index, name in enumerate(EXPERIMENT_ORDER)}
    summary["canonical_order"] = summary["experiment"].map(order)
    summary = summary.sort_values("canonical_order")
    output = Path(output_root)
    csv_dir, figure_dir = output / "csv", output / "figures"
    csv_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_dir / "condition_summary.csv", index=False)
    pd.DataFrame(regional).to_csv(csv_dir / "condition_summary_by_timestep_region.csv", index=False)
    pd.concat(all_diffusion, ignore_index=True).to_csv(csv_dir / "diffusion_metrics_by_timestep.csv", index=False)
    pd.concat(all_ode, ignore_index=True).to_csv(csv_dir / "cell_ode_metrics_by_timestep.csv", index=False)
    pd.concat(all_loss, ignore_index=True).to_csv(csv_dir / "loss_history.csv", index=False)
    pd.concat(all_fraction, ignore_index=True).to_csv(csv_dir / "loss_fraction.csv", index=False)
    if all_gradient:
        pd.concat(all_gradient, ignore_index=True).to_csv(csv_dir / "gradient_metrics.csv", index=False)
    plot_summary_figures(summary, figure_dir)
    result = {
        "status": "completed",
        "conditions": int(len(summary)),
        "summary_root": str(output.resolve()),
        "csv_directory": str(csv_dir.resolve()),
        "figure_directory": str(figure_dir.resolve()),
        "git_commit": git_commit(),
    }
    write_json(output / "analysis_config.json", {
        **result,
        "run_directories": [str(Path(run).resolve()) for run in runs],
        "canonical_order": list(EXPERIMENT_ORDER),
        "timestep_regions": {
            "low_diffusion_timestep": "t < diffusion_steps/3",
            "mid_diffusion_timestep": "diffusion_steps/3 <= t < 2*diffusion_steps/3",
            "high_diffusion_timestep": "t >= 2*diffusion_steps/3",
        },
        "standardized_heatmap": "column-wise population z-score; constant columns set to 0",
    })
    return result


__all__ = [
    "AnalysisOptions",
    "analyze_run",
    "discover_run_directories",
    "parse_timestep_spec",
    "preset_cells",
    "preset_timesteps",
    "summarize_runs",
]
