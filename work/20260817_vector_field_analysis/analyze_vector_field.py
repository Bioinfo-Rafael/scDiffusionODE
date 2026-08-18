#!/usr/bin/env python3
"""Post-hoc vector-field analysis of a trained 20260803 single Hill ODE."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SOURCE_SUITE = REPO_ROOT / "work" / "20260803_ODE_hill_exp"
# The 20260803 helper imports ``models.factory`` by its suite-local package name.
# Put that suite before the repository root to prevent import ambiguity.
for search_path in (str(REPO_ROOT), str(SOURCE_SUITE), str(HERE)):
    if search_path in sys.path:
        sys.path.remove(search_path)
    sys.path.insert(0, search_path)

from viz.analysis_helpers import (  # noqa: E402
    _first_existing,
    _newest,
    _read_json,
    _resolve_path,
    build_diffusion,
    discover_run_inputs,
    load_generated_subset,
    load_model,
    load_real_subset,
)

from dynamo_analysis import (  # noqa: E402
    acceleration_gene_summary,
    aggregate_jacobians,
    evaluate_dataset,
    find_fixed_points,
    jacobian_gene_summary,
    jacobian_top_interactions,
    plot_metric_pca,
    plot_real_generated_metrics,
    plot_velocity_pca,
    representative_real_cells,
    save_jacobian_aggregates,
    sensitivity_aggregates,
    summarize_metrics,
)
from vector_field_adapter import TrainedODEVectorField, assert_finite  # noqa: E402


DYNAMO_COMMIT = "69ba903000a1dcd5aa7d8b36a9318c1b22c5d650"
EXPECTED_FAMILY = "standard_hybrid_single"
EXPECTED_ODE = "hill_after_linear"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True, default=str)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is false")
    if name == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("--device mps was requested but MPS is unavailable")
    return torch.device(name)


def _validate_target(config: Mapping[str, Any]) -> None:
    family = str(config.get("model_family", ""))
    ode_type = str(config.get("ode_type", ""))
    if family != EXPECTED_FAMILY or ode_type != EXPECTED_ODE:
        raise ValueError(
            "this analysis intentionally accepts only "
            f"model_family={EXPECTED_FAMILY}, ode_type={EXPECTED_ODE}; "
            f"got model_family={family!r}, ode_type={ode_type!r}"
        )


def _discover_vector_field_inputs(
    run_dir: Path, *, checkpoint: str, sample_path: str
) -> dict[str, Any]:
    """Reuse 20260803 discovery, without requiring its unrelated loss table."""

    try:
        return discover_run_inputs(
            run_dir,
            checkpoint_override=checkpoint,
            sample_override=sample_path,
        )
    except FileNotFoundError as exc:
        if "loss_details.csv" not in str(exc):
            raise

    run_dir = run_dir.resolve()
    config_path = run_dir / "exp_config.json"
    manifest_path = run_dir / "manifest.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing run config: {config_path}")
    config = _read_json(config_path)
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    restored_checkpoint = _first_existing(
        (
            checkpoint,
            manifest.get("checkpoint_path"),
            manifest.get("ema_checkpoint_path"),
            manifest.get("raw_checkpoint_path"),
            config.get("checkpoint_path"),
        ),
        run_dir,
    )
    if restored_checkpoint is None:
        restored_checkpoint = _newest(
            list(run_dir.glob("train/**/ema_*.pt"))
            + list(run_dir.glob("train/**/model*.pt"))
            + list(run_dir.glob("**/ema_*.pt"))
        )
    restored_sample = _first_existing(
        (sample_path, manifest.get("sample_path"), config.get("sample_path")), run_dir
    )
    if restored_sample is None:
        restored_sample = _newest(run_dir.glob("samples/*.npz"))
    data_value = config.get("data_dir") or manifest.get("data_dir")
    if restored_checkpoint is None:
        raise FileNotFoundError("no checkpoint found; pass --checkpoint")
    if restored_sample is None:
        raise FileNotFoundError("no generated sample found; pass --sample-path")
    if not data_value:
        raise KeyError("exp_config.json is missing data_dir")
    data_path = _resolve_path(str(data_value), run_dir)
    if not data_path.is_file():
        raise FileNotFoundError(f"data file does not exist: {data_path}")
    return {
        "run_dir": run_dir,
        "config_path": config_path,
        "manifest_path": manifest_path,
        "config": config,
        "manifest": manifest,
        "checkpoint": restored_checkpoint,
        "sample_path": restored_sample,
        "data_path": data_path,
        "loss_path": None,
        "loss_paths": [],
    }


def _choose_indices(size: int, limit: int, seed: int) -> np.ndarray:
    take = min(size, int(limit))
    if take <= 0:
        raise ValueError("the requested cell limit selected no cells")
    if take == size:
        return np.arange(size, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, take, replace=False)).astype(np.int64)


def _fit_pca(real: np.ndarray, generated: np.ndarray, n_components: int, seed: int):
    maximum = min(real.shape[0] - 1, real.shape[1])
    if maximum < 2:
        raise ValueError("at least 3 real cells and 2 genes are required for PCA plots")
    components = min(int(n_components), maximum)
    pca = PCA(n_components=components, svd_solver="randomized", random_state=seed)
    real_coordinates = pca.fit_transform(real)
    generated_coordinates = pca.transform(generated)
    assert_finite("real PCA coordinates", real_coordinates)
    assert_finite("generated PCA coordinates", generated_coordinates)
    assert_finite("PCA components", pca.components_)
    return pca, real_coordinates, generated_coordinates


def _output_dir(args: argparse.Namespace, inputs: Mapping[str, Any]) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment = _slug(str(inputs["config"].get("experiment", EXPECTED_FAMILY)))
    checkpoint = _slug(Path(inputs["checkpoint"]).stem)
    return (HERE / "outputs" / f"{experiment}__{checkpoint}__{stamp}").resolve()


def _created_files(output_dir: Path) -> list[str]:
    return [
        str(path.relative_to(output_dir))
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    ]


def _save_pca_table(
    path: Path,
    cell_metrics: pd.DataFrame,
    coordinates: Mapping[str, np.ndarray],
    projected_velocity: Mapping[str, np.ndarray],
) -> None:
    rows: list[pd.DataFrame] = []
    for dataset in ("real", "generated"):
        source = cell_metrics[cell_metrics["dataset"] == dataset].reset_index(drop=True)
        points = coordinates[dataset]
        velocity = projected_velocity[dataset]
        if len(source) != len(points):
            raise ValueError("PCA rows do not align with cell metrics")
        rows.append(
            pd.DataFrame(
                {
                    "cell_id": source["cell_id"],
                    "dataset": dataset,
                    "annotation": source["annotation"],
                    "PC1": points[:, 0],
                    "PC2": points[:, 1],
                    "velocity_PC1": velocity[:, 0],
                    "velocity_PC2": velocity[:, 1],
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)
    assert_finite("PCA output table", table.select_dtypes(include=[np.number]))
    table.to_csv(path, index=False)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _discover_vector_field_inputs(
        Path(args.run_dir).expanduser(),
        checkpoint=args.checkpoint,
        sample_path=args.sample_path,
    )
    config = inputs["config"]
    _validate_target(config)
    seed = int(config.get("seed", 1234) if args.seed is None else args.seed)
    device = _select_device(args.device)
    output_dir = _output_dir(args, inputs)
    try:
        output_dir.relative_to(inputs["run_dir"])
    except ValueError:
        pass
    else:
        raise ValueError(
            "--output-dir must be outside the source run directory so that the "
            "training/sampling run remains read-only"
        )

    preview = {
        "run_dir": str(inputs["run_dir"]),
        "config": str(inputs["config_path"]),
        "checkpoint": str(inputs["checkpoint"]),
        "real_data": str(inputs["data_path"]),
        "generated_samples": str(inputs["sample_path"]),
        "output_dir": str(output_dir),
        "device": str(device),
        "target": {"model_family": EXPECTED_FAMILY, "ode_type": EXPECTED_ODE},
    }
    if args.dry_run:
        return {"status": "dry_run", **preview}

    real, genes, real_labels, label_column = load_real_subset(
        inputs["data_path"],
        layer=config.get("ts_layer"),
        max_cells=args.max_real_cells,
        seed=seed,
    )
    generated = load_generated_subset(
        inputs["sample_path"],
        max_cells=args.max_generated_cells,
        seed=seed,
    )
    if real.shape[1] != generated.shape[1] or real.shape[1] != len(genes):
        raise ValueError(
            f"dimension mismatch: real={real.shape}, generated={generated.shape}, "
            f"genes={len(genes)}"
        )

    diffusion = build_diffusion(config)
    model = load_model(config, genes, diffusion, inputs["checkpoint"], device)
    if not hasattr(model, "ode_model") or not hasattr(model, "ml_model"):
        raise RuntimeError("restored model is not the expected standard Hybrid wrapper")
    if str(getattr(model, "model_family", "")) != EXPECTED_FAMILY:
        raise RuntimeError("restored model family does not match exp_config.json")
    adapter = TrainedODEVectorField(
        model.ode_model,
        device=device,
        batch_size=args.batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "analysis_manifest.json"
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": started,
        "read_only_source_run": True,
        "vector_field": "V(x) = restored model.ode_model(x); hybrid output is unused",
        "dynamo_refit_performed": False,
        "dynamo_source_commit": DYNAMO_COMMIT,
        "inputs": preview,
        "settings": vars(args),
        "seed": seed,
        "label_column": label_column,
        "n_genes": len(genes),
        "n_real_cells": len(real),
        "n_generated_cells": len(generated),
    }
    _write_json(manifest_path, manifest)

    try:
        real_table, real_result = evaluate_dataset(
            adapter, real, dataset="real", annotations=real_labels
        )
        generated_table, generated_result = evaluate_dataset(
            adapter, generated, dataset="generated"
        )
        cell_metrics = pd.concat([real_table, generated_table], ignore_index=True)
        summary_metrics = summarize_metrics(cell_metrics)
        cell_metrics.to_csv(output_dir / "cell_metrics.csv", index=False)
        summary_metrics.to_csv(output_dir / "summary_metrics.csv", index=False)

        acceleration_gene_summary(
            {"real": real_result, "generated": generated_result}, genes
        ).to_csv(output_dir / "acceleration_gene_summary.csv", index=False)

        jacobian_indices = {
            "real": _choose_indices(len(real), args.jacobian_cells, seed + 11),
            "generated": _choose_indices(len(generated), args.jacobian_cells, seed + 12),
        }
        pd.concat(
            [
                pd.DataFrame({"dataset": dataset, "source_index": indices})
                for dataset, indices in jacobian_indices.items()
            ],
            ignore_index=True,
        ).to_csv(output_dir / "jacobian_cell_selection.csv", index=False)
        jacobian_aggregates = aggregate_jacobians(
            adapter,
            {
                "real": real[jacobian_indices["real"]],
                "generated": generated[jacobian_indices["generated"]],
            },
            method=args.jacobian_method,
        )
        save_jacobian_aggregates(
            jacobian_aggregates, output_dir / "jacobian_aggregates.npz"
        )
        jacobian_gene_summary(jacobian_aggregates, genes).to_csv(
            output_dir / "jacobian_gene_summary.csv", index=False
        )
        jacobian_top_interactions(
            jacobian_aggregates, genes, top_n=args.top_interactions
        ).to_csv(output_dir / "jacobian_top_interactions.csv", index=False)

        sensitivity_status: dict[str, Any]
        if args.sensitivity_cells > 0:
            sensitivity_indices = {
                "real": _choose_indices(len(real), args.sensitivity_cells, seed + 21),
                "generated": _choose_indices(
                    len(generated), args.sensitivity_cells, seed + 22
                ),
            }
            sensitivity_selected = {
                "real": real[sensitivity_indices["real"]],
                "generated": generated[sensitivity_indices["generated"]],
            }
            sensitivity = sensitivity_aggregates(
                adapter,
                sensitivity_selected,
                max_dimension=args.sensitivity_max_dim,
            )
            save_jacobian_aggregates(
                sensitivity, output_dir / "sensitivity_aggregates.npz"
            )
            sensitivity_genes = jacobian_gene_summary(sensitivity, genes).rename(
                columns=lambda name: name.replace("influence", "sensitivity")
            )
            sensitivity_genes.to_csv(
                output_dir / "sensitivity_gene_summary.csv", index=False
            )
            sensitivity_interactions = jacobian_top_interactions(
                sensitivity, genes, top_n=args.top_interactions
            ).rename(
                columns={
                    "mean_jacobian": "mean_sensitivity",
                    "mean_absolute_jacobian": "mean_absolute_sensitivity",
                }
            )
            sensitivity_interactions.to_csv(
                output_dir / "sensitivity_top_interactions.csv", index=False
            )
            sensitivity_status = {
                "status": "completed",
                "cells_per_dataset": {
                    key: len(value) for key, value in sensitivity_selected.items()
                },
            }
        else:
            sensitivity_status = {
                "status": "not_requested",
                "reason": (
                    "Dynamo sensitivity requires dense inversion of I-J; use "
                    "--sensitivity-cells and raise --sensitivity-max-dim explicitly if appropriate"
                ),
            }

        pca, real_pca, generated_pca = _fit_pca(
            real, generated, args.pca_components, seed
        )
        coordinates = {"real": real_pca, "generated": generated_pca}
        projected_velocity = {
            "real": real_result.velocity @ pca.components_.T,
            "generated": generated_result.velocity @ pca.components_.T,
        }
        for dataset, value in projected_velocity.items():
            assert_finite(f"{dataset} PCA-projected velocity", value)

        fixed_seeds, fixed_seed_sources = representative_real_cells(
            real,
            real_pca,
            n_seeds=args.fixed_point_seeds,
            seed=seed,
            annotations=real_labels if label_column else None,
        )
        domain = np.column_stack((np.min(real, axis=0), np.max(real, axis=0))).astype(
            np.float64
        )
        if args.fixed_point_seeds > 0:
            fixed = find_fixed_points(
                adapter,
                fixed_seeds,
                fixed_seed_sources,
                domain,
                residual_tolerance=args.fixed_point_residual_tol,
                redundant_tolerance=args.fixed_point_redundant_tol,
                max_iterations=args.fixed_point_maxiter,
                full_eigen_max_dim=args.full_eigen_max_dim,
                leading_eigenvalues=args.leading_eigenvalues,
                eigen_tolerance=args.eigen_tolerance,
                stability_tolerance=args.stability_tolerance,
            )
            fixed.attempts.to_csv(output_dir / "fixed_point_attempts.csv", index=False)
        else:
            fixed = None

        fixed_pca: np.ndarray | None = None
        fixed_status: dict[str, Any]
        if fixed is not None and len(fixed.points):
            fixed_pca = pca.transform(fixed.points.astype(np.float32))
            assert_finite("fixed-point PCA coordinates", fixed_pca)
            coordinate_frame = pd.DataFrame(
                fixed.points,
                columns=[f"x__{gene}" for gene in genes],
            )
            point_frame = pd.concat(
                [fixed.metadata.reset_index(drop=True), coordinate_frame], axis=1
            )
            point_frame["PC1"] = fixed_pca[:, 0]
            point_frame["PC2"] = fixed_pca[:, 1]
            point_frame.to_csv(output_dir / "fixed_points.csv", index=False)
            fixed.eigenvalues.to_csv(
                output_dir / "fixed_point_eigenvalues.csv", index=False
            )
            fixed_status = {
                "status": "found",
                "n_fixed_points": len(fixed.points),
                "n_seeds": len(fixed_seeds),
            }
        elif fixed is not None:
            fixed_status = {
                "status": "none_found",
                "n_fixed_points": 0,
                "n_seeds": len(fixed_seeds),
            }
        else:
            fixed_status = {"status": "not_requested", "n_fixed_points": 0}

        plot_velocity_pca(
            real_pca,
            projected_velocity["real"],
            dataset="real",
            path=output_dir / "real_velocity_pca.png",
            max_arrows=args.max_arrows,
            seed=seed,
            fixed_coordinates=fixed_pca,
        )
        plot_velocity_pca(
            generated_pca,
            projected_velocity["generated"],
            dataset="generated",
            path=output_dir / "generated_velocity_pca.png",
            max_arrows=args.max_arrows,
            seed=seed + 1,
            fixed_coordinates=fixed_pca,
        )
        plot_metric_pca(
            coordinates,
            {
                "real": real_table["divergence"].to_numpy(),
                "generated": generated_table["divergence"].to_numpy(),
            },
            metric_label="divergence tr(J)",
            path=output_dir / "divergence_pca.png",
            fixed_coordinates=fixed_pca,
        )
        plot_metric_pca(
            coordinates,
            {
                "real": real_table["acceleration_norm"].to_numpy(),
                "generated": generated_table["acceleration_norm"].to_numpy(),
            },
            metric_label="acceleration norm ||J V||",
            path=output_dir / "acceleration_norm_pca.png",
            fixed_coordinates=fixed_pca,
        )
        plot_metric_pca(
            coordinates,
            {
                "real": real_table["speed"].to_numpy(),
                "generated": generated_table["speed"].to_numpy(),
            },
            metric_label="speed ||V||",
            path=output_dir / "speed_pca.png",
            fixed_coordinates=fixed_pca,
        )
        plot_real_generated_metrics(
            cell_metrics, output_dir / "real_vs_generated_metrics.png"
        )
        _save_pca_table(
            output_dir / "pca_projection.csv",
            cell_metrics,
            coordinates,
            projected_velocity,
        )
        np.savez_compressed(
            output_dir / "pca_model.npz",
            mean=pca.mean_.astype(np.float32),
            components=pca.components_.astype(np.float32),
            explained_variance=pca.explained_variance_.astype(np.float32),
            explained_variance_ratio=pca.explained_variance_ratio_.astype(np.float32),
        )

        completed = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        manifest.update(
            {
                "status": "completed",
                "completed_at": completed,
                "source_integrity": {
                    "config_sha256": _sha256(inputs["config_path"]),
                    "checkpoint_sha256": _sha256(inputs["checkpoint"]),
                    "sample_sha256": _sha256(inputs["sample_path"]),
                },
                "methods": {
                    "velocity": "raw restored ODE branch only",
                    "jacobian": args.jacobian_method,
                    "divergence": "exact trace of original-space Hill Jacobian",
                    "acceleration": "J(x) @ V(x) in original gene space",
                    "pca": "fit on real X; velocity projected as V @ components.T",
                    "sensitivity": sensitivity_status,
                    "fixed_points": fixed_status,
                    "cosine_zero_convention": (
                        "cosine is 0 and cosine_defined=false if ||V||*||a|| is exactly zero"
                    ),
                },
                "jacobian_cells": {
                    key: len(value) for key, value in jacobian_indices.items()
                },
                "created_files": _created_files(output_dir),
            }
        )
        _write_json(manifest_path, manifest)
        return {"output_dir": str(output_dir), **manifest}
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).astimezone().isoformat(
                    timespec="seconds"
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "created_files": _created_files(output_dir),
            }
        )
        _write_json(manifest_path, manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze the raw trained ODE branch of one existing 20260803 "
            "standard_hybrid_single/hill_after_linear run without refitting."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="", help="optional checkpoint override")
    parser.add_argument("--sample-path", default="", help="optional generated NPZ override")
    parser.add_argument("--output-dir", default="", help="must not already exist")
    parser.add_argument("--max-real-cells", type=int, default=2000)
    parser.add_argument("--max-generated-cells", type=int, default=2000)
    parser.add_argument("--jacobian-cells", type=int, default=200)
    parser.add_argument(
        "--jacobian-method",
        choices=("analytical", "autograd"),
        default="analytical",
        help="exact closed-form Hill Jacobian or torch.func.jacrev",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-interactions", type=int, default=100)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--max-arrows", type=int, default=400)
    parser.add_argument("--fixed-point-seeds", type=int, default=8)
    parser.add_argument("--fixed-point-maxiter", type=int, default=50)
    parser.add_argument("--fixed-point-residual-tol", type=float, default=1e-5)
    parser.add_argument("--fixed-point-redundant-tol", type=float, default=1e-4)
    parser.add_argument("--full-eigen-max-dim", type=int, default=256)
    parser.add_argument("--leading-eigenvalues", type=int, default=6)
    parser.add_argument("--eigen-tolerance", type=float, default=1e-5)
    parser.add_argument("--stability-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--sensitivity-cells",
        type=int,
        default=0,
        help="opt in to dense Dynamo sensitivity for this many cells per dataset",
    )
    parser.add_argument("--sensitivity-max-dim", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "max_real_cells",
        "max_generated_cells",
        "jacobian_cells",
        "batch_size",
        "top_interactions",
        "pca_components",
        "max_arrows",
        "fixed_point_maxiter",
        "full_eigen_max_dim",
        "leading_eigenvalues",
        "sensitivity_max_dim",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.fixed_point_seeds < 0 or args.sensitivity_cells < 0:
        raise ValueError("fixed-point and sensitivity cell counts must be non-negative")
    for name in (
        "fixed_point_residual_tol",
        "fixed_point_redundant_tol",
        "eigen_tolerance",
        "stability_tolerance",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    result = analyze(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    if result.get("status") == "completed":
        print(f"VECTOR_FIELD_ANALYSIS_DIR='{result['output_dir']}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
