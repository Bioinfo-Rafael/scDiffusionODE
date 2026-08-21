"""End-to-end orchestration reusing the 20260817 model/UMAP implementation."""

from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
import traceback
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dynamo_bridge import (
    evaluate_vector_field,
    fit_dynamo_vector_field,
    fixed_points_from_adata,
    run_least_action_paths,
    save_dynamo_results,
    save_lap,
)
from .landscape import (
    calibrate_common_sde,
    common_bounds,
    compute_landscape_quantities,
    save_landscape,
    simulate_stationary_distribution,
)
from .plotting import plot_core_figures, plot_lap

HERE = Path(__file__).resolve().parents[2]
REPO_ROOT = HERE.parent.parent
PRIOR_DIR = REPO_ROOT / "work" / "20260817_vector_field_analysis"
EXPECTED_FAMILY = "standard_hybrid_single"
EXPECTED_ODE = "hill_after_linear"


def _load_prior_symbols():
    """Import, rather than duplicate, the audited 20260817 loading/projection code."""

    if str(PRIOR_DIR) not in sys.path:
        sys.path.insert(0, str(PRIOR_DIR))
    from analyze_vector_field import (  # type: ignore
        _discover_vector_field_inputs,
        _select_device,
        _sha256,
        _validate_target,
        build_diffusion,
        load_model,
    )
    from erythropoietic_umap import (  # type: ignore
        _gene_names,
        _load_legacy_velocity_module,
        _to_dense_float32,
    )
    from vector_field_adapter import TrainedODEVectorField  # type: ignore

    return {
        "discover": _discover_vector_field_inputs,
        "select_device": _select_device,
        "sha256": _sha256,
        "validate_target": _validate_target,
        "build_diffusion": build_diffusion,
        "load_model": load_model,
        "gene_names": _gene_names,
        "load_legacy": _load_legacy_velocity_module,
        "dense": _to_dense_float32,
        "adapter": TrainedODEVectorField,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _hash_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_.") or "model"


def discover_run_dirs(explicit: Sequence[str], runs_root: str | Path | None) -> list[Path]:
    """Resolve explicit runs or all eligible analyzed-model runs under one root."""

    if explicit and runs_root:
        raise ValueError("use either repeated --run-dir or --runs-root, not both")
    if explicit:
        candidates = [Path(value).expanduser().resolve() for value in explicit]
    else:
        if runs_root is None:
            runs_root = REPO_ROOT / "work" / "20260803_ODE_hill_exp" / "runs"
        root = Path(runs_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"runs root does not exist: {root}")
        candidates = []
        for config_path in sorted(root.rglob("exp_config.json")):
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                config.get("model_family") == EXPECTED_FAMILY
                and config.get("ode_type") == EXPECTED_ODE
            ):
                candidates.append(config_path.parent.resolve())
    unique = list(dict.fromkeys(candidates))
    if not unique:
        raise FileNotFoundError(
            "no standard_hybrid_single/hill_after_linear run was found; pass --run-dir"
        )
    missing = [str(path) for path in unique if not (path / "exp_config.json").is_file()]
    if missing:
        raise FileNotFoundError(f"run directories missing exp_config.json: {missing}")
    return unique


def _shared_erythropoietic_adata(first_inputs: Mapping[str, Any], config: Mapping[str, Any], prior):
    import scanpy as sc

    adata = sc.read_h5ad(first_inputs["data_path"])
    selection = config["selection"]
    obs_key = str(selection["obs_key"])
    celltype_key = str(selection["celltype_key"])
    if obs_key not in adata.obs or celltype_key not in adata.obs:
        raise KeyError(f"h5ad must contain obs columns {obs_key!r} and {celltype_key!r}")
    adata = adata[adata.obs[obs_key].astype(str) == str(selection["value"])].copy()
    source_cells = int(adata.n_obs)
    if source_cells == 0:
        raise ValueError("no Superclass == Erythropoietic cells were found")
    max_cells = int(config["max_cells"])
    if max_cells > 0 and adata.n_obs > max_cells:
        rng = np.random.default_rng(int(config["seed"]))
        indices = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
        adata = adata[indices].copy()
    X = prior["dense"](adata.X)
    adata.X = X
    genes = prior["gene_names"](adata)

    umap = config["umap"]
    max_components = min(int(umap["n_components"]), adata.n_obs - 1, adata.n_vars - 1)
    if max_components < 2:
        raise ValueError("too few cells/genes for the shared PCA/UMAP")
    sc.tl.pca(
        adata,
        svd_solver="arpack",
        n_comps=max_components,
        random_state=int(config["seed"]),
    )
    n_pcs = min(int(umap["n_pcs"]), max_components)
    n_neighbors = min(int(umap["n_neighbors"]), adata.n_obs - 1)
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        random_state=int(config["seed"]),
    )
    sc.tl.umap(
        adata,
        min_dist=float(umap["min_dist"]),
        random_state=int(config["seed"]),
    )
    coordinates = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise ValueError(f"shared UMAP has invalid shape/content: {coordinates.shape}")
    return adata, genes, source_cells


def _project_model_velocity(
    shared: Any,
    genes: Sequence[str],
    inputs: Mapping[str, Any],
    config: Mapping[str, Any],
    prior,
    *,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Restore one ODE, evaluate V(x), and reuse the existing scVelo projection."""

    import anndata as ad

    # Dynamo 1.4.1 asks for matplotlib>=3.7.5.  Keep the old scVelo 0.2.5
    # warning alias available if a newer matplotlib is already present.
    import matplotlib
    from matplotlib import cbook

    if not hasattr(cbook, "mplDeprecation"):
        cbook.mplDeprecation = matplotlib.MatplotlibDeprecationWarning
    import scvelo as scv

    observed_genes = prior["gene_names"](shared)
    if observed_genes != list(genes):
        raise ValueError("shared h5ad gene ordering changed before model evaluation")
    run_config = inputs["config"]
    prior["validate_target"](run_config)
    diffusion = prior["build_diffusion"](run_config)
    model = prior["load_model"](run_config, list(genes), diffusion, inputs["checkpoint"], device)
    adapter = prior["adapter"](
        model.ode_model,
        device=device,
        batch_size=int(config.get("model_batch_size", 128)),
    )
    X = prior["dense"](shared.X)
    if X.shape[1] != adapter.dimension or X.shape[1] != len(genes):
        raise ValueError(f"model/expression/gene mismatch: X={X.shape}, D={adapter.dimension}, genes={len(genes)}")
    velocity_gene = np.asarray(adapter.func(X), dtype=np.float32)
    if velocity_gene.shape != X.shape or not np.isfinite(velocity_gene).all():
        raise ValueError("gene-space model velocity is invalid or misaligned")

    shared.layers["X"] = X
    shared.layers["velocity_ode"] = velocity_gene
    for key in [key for key in list(shared.uns) if key.startswith("velocity_ode_graph")]:
        del shared.uns[key]
    shared.obsm.pop("velocity_ode_umap", None)
    prior["load_legacy"]()
    scv.tl.velocity_graph(
        shared,
        vkey="velocity_ode",
        xkey="X",
        backend="loky",
        n_jobs=int(config["umap"]["n_jobs"]),
    )
    scv.tl.velocity_embedding(shared, basis="umap", vkey="velocity_ode")
    projected_key = "velocity_ode_umap"
    if projected_key not in shared.obsm:
        raise RuntimeError(f"scVelo did not create obsm[{projected_key!r}]")
    velocity_umap = np.asarray(shared.obsm[projected_key], dtype=np.float64)
    coordinates = np.asarray(shared.obsm["X_umap"], dtype=np.float64)
    if velocity_umap.shape != coordinates.shape or not np.isfinite(velocity_umap).all():
        raise ValueError("UMAP-projected velocity is invalid or misaligned")

    obs = shared.obs.copy()
    minimal = ad.AnnData(
        X=np.zeros((shared.n_obs, 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame({"use_for_pca": [True]}, index=["placeholder"]),
    )
    minimal.obsm["X_umap"] = coordinates.copy()
    minimal.obsm["velocity_umap"] = velocity_umap.copy()
    minimal.obsp["umap_distances"] = shared.obsp["distances"].copy()
    minimal.obsp["distances"] = shared.obsp["distances"].copy()
    minimal.obsp["connectivities"] = shared.obsp["connectivities"].copy()
    minimal.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": dict(shared.uns.get("neighbors", {}).get("params", {})),
    }
    metadata = {
        "gene_velocity_shape": list(velocity_gene.shape),
        "umap_velocity_shape": list(velocity_umap.shape),
        "gene_order_sha256": _hash_strings(list(genes)),
        "cell_order_sha256": _hash_strings(shared.obs_names.astype(str).tolist()),
        "coordinates_sha256": _hash_array(coordinates),
        "velocity_umap_sha256": _hash_array(velocity_umap),
    }
    del adapter, model, diffusion, velocity_gene
    gc.collect()
    return minimal, metadata


def _save_common(shared: Any, genes: Sequence[str], source_cells: int, output_dir: Path, config):
    common_dir = output_dir / "common"
    common_dir.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray(shared.obsm["X_umap"], dtype=np.float64)
    celltype_key = str(config["selection"]["celltype_key"])
    pd.DataFrame(
        {
            "cell_id": shared.obs_names.astype(str),
            "Superclass": shared.obs["Superclass"].astype(str).to_numpy(),
            celltype_key: shared.obs[celltype_key].astype(str).to_numpy(),
            "UMAP1": coordinates[:, 0],
            "UMAP2": coordinates[:, 1],
        }
    ).to_csv(common_dir / "erythropoietic_fixed_umap.csv", index=False)
    np.savez_compressed(
        common_dir / "erythropoietic_fixed_umap.npz",
        coordinates=coordinates,
        cell_ids=np.asarray(shared.obs_names.astype(str)),
        genes=np.asarray(genes, dtype=str),
    )
    metadata = {
        "selection": "Superclass == Erythropoietic",
        "source_cells": source_cells,
        "selected_cells": int(shared.n_obs),
        "n_genes": len(genes),
        "umap_fit_count": 1,
        "cell_order_sha256": _hash_strings(shared.obs_names.astype(str).tolist()),
        "gene_order_sha256": _hash_strings(list(genes)),
        "coordinates_sha256": _hash_array(coordinates),
        "umap_settings": dict(config["umap"]),
    }
    _write_json(common_dir / "embedding_metadata.json", metadata)
    return metadata


def _model_name(inputs: Mapping[str, Any], used: set[str]) -> str:
    base = _slug(f"{inputs['config'].get('experiment', 'model')}__{inputs['run_dir'].name}")
    name = base
    counter = 2
    while name in used:
        name = f"{base}__{counter}"
        counter += 1
    used.add(name)
    return name


def run_pipeline(
    run_dirs: Sequence[Path],
    config: Mapping[str, Any],
    output_dir: str | Path,
    *,
    device_name: str = "auto",
) -> dict[str, Any]:
    prior = _load_prior_symbols()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "analysis_manifest.json"
    started = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at": started,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "config_path": config.get("_config_path"),
        "run_dirs": [str(path) for path in run_dirs],
        "output_dir": str(output_dir),
        "high_dimensional_analysis_replaced": False,
        "analysis_space": "continuous 2D UMAP vector field",
    }
    _write_json(manifest_path, manifest)
    try:
        inputs_list = [
            prior["discover"](path, checkpoint="", sample_path="") for path in run_dirs
        ]
        for inputs in inputs_list:
            prior["validate_target"](inputs["config"])
        data_paths = {str(inputs["data_path"].resolve()) for inputs in inputs_list}
        if len(data_paths) != 1:
            raise ValueError(
                "all compared models must resolve to the identical h5ad path; "
                f"found {sorted(data_paths)}"
            )
        device = prior["select_device"](device_name)
        shared, genes, source_cells = _shared_erythropoietic_adata(inputs_list[0], config, prior)
        common_metadata = _save_common(shared, genes, source_cells, output_dir, config)
        coordinate_hash = common_metadata["coordinates_sha256"]

        used_names: set[str] = set()
        prepared = []
        for inputs in inputs_list:
            name = _model_name(inputs, used_names)
            model_dir = output_dir / "models" / name
            model_dir.mkdir(parents=True, exist_ok=True)
            model_adata, projection_metadata = _project_model_velocity(
                shared,
                genes,
                inputs,
                config,
                prior,
                device=device,
            )
            if projection_metadata["coordinates_sha256"] != coordinate_hash:
                raise RuntimeError("a model changed the shared UMAP coordinates")
            velocity_umap = np.asarray(model_adata.obsm["velocity_umap"], dtype=np.float64)
            np.savez_compressed(
                model_dir / "observed_umap_velocity.npz",
                coordinates=np.asarray(model_adata.obsm["X_umap"], dtype=np.float64),
                velocity_umap=velocity_umap,
                cell_ids=np.asarray(model_adata.obs_names.astype(str)),
            )
            pd.DataFrame(
                {
                    "cell_id": model_adata.obs_names.astype(str),
                    "UMAP1": model_adata.obsm["X_umap"][:, 0],
                    "UMAP2": model_adata.obsm["X_umap"][:, 1],
                    "velocity_UMAP1": velocity_umap[:, 0],
                    "velocity_UMAP2": velocity_umap[:, 1],
                }
            ).to_csv(model_dir / "observed_umap_velocity.csv", index=False)
            fit_metadata = fit_dynamo_vector_field(model_adata, config["dynamo"])
            save_dynamo_results(model_adata, model_dir, fit_metadata)
            vecfld = model_adata.uns["VecFld_umap"]
            prepared.append(
                {
                    "name": name,
                    "dir": model_dir,
                    "adata": model_adata,
                    "vecfld": vecfld,
                    "inputs": inputs,
                    "projection": projection_metadata,
                    "fit": fit_metadata,
                }
            )

        coordinates = np.asarray(shared.obsm["X_umap"], dtype=np.float64)
        calibration = calibrate_common_sde(
            coordinates,
            [np.asarray(item["adata"].obsm["velocity_umap"]) for item in prepared],
            config["sde"],
        )
        bounds = common_bounds(coordinates, config["sde"])
        calibration["bounds"] = bounds.tolist()
        _write_json(output_dir / "common" / "sde_calibration.json", calibration)

        model_summaries = []
        celltype_key = str(config["selection"]["celltype_key"])
        labels = shared.obs[celltype_key].astype(str).to_numpy()
        for item in prepared:
            model_dir = item["dir"]
            vecfld = item["vecfld"]

            def field(points, state=vecfld):
                return evaluate_vector_field(
                    points, state, expected_version=str(config["dynamo"]["version"])
                )

            num_tra, total_Fx, total_Fy, simulation_metadata = simulate_stationary_distribution(
                field,
                bounds,
                config["sde"],
                calibration,
                # Common random numbers make Monte Carlo comparisons paired and
                # preserve literally identical stochastic settings across models.
                seed=int(config["seed"]),
                checkpoint_dir=model_dir / "simulation",
            )
            result = compute_landscape_quantities(
                field,
                bounds,
                num_tra,
                total_Fx,
                total_Fy,
                diffusion=float(calibration["D"]),
                potential_cap_above_max=float(config["sde"]["potential_cap_above_max"]),
            )
            save_landscape(result, model_dir)
            fixed_points, fixed_types = fixed_points_from_adata(item["adata"])
            figures = plot_core_figures(
                result,
                coordinates,
                labels,
                fixed_points,
                fixed_types,
                model_dir,
                config["plot"],
            )
            lap_paths, lap_status = run_least_action_paths(
                item["adata"],
                celltype_key=celltype_key,
                config=config["lap"],
                diffusion=float(calibration["D"]),
            )
            save_lap(lap_paths, lap_status, model_dir)
            if lap_paths:
                plot_lap(
                    result,
                    coordinates,
                    labels,
                    lap_paths,
                    model_dir / "10_least_action_paths.png",
                    config["plot"],
                )
                figures.append("10_least_action_paths.png")
            model_manifest = {
                "name": item["name"],
                "run_dir": str(item["inputs"]["run_dir"]),
                "checkpoint": str(item["inputs"]["checkpoint"]),
                "checkpoint_sha256": prior["sha256"](item["inputs"]["checkpoint"]),
                "data_path": str(item["inputs"]["data_path"]),
                "projection": item["projection"],
                "dynamo": item["fit"],
                "simulation": simulation_metadata,
                "landscape": result.summary,
                "lap": lap_status,
                "figures": figures,
                "shared_sde": calibration,
                "vector_field_distinction": {
                    "input": "V(x) in original gene-expression space",
                    "projected": "per-cell UMAP velocity on one fixed embedding",
                    "reconstructed": "F_UMAP(z) fit by Dynamo SparseVFC",
                    "curl_and_landscape_space": "2D UMAP only",
                },
            }
            _write_json(model_dir / "model_manifest.json", model_manifest)
            model_summaries.append(
                {
                    "model": item["name"],
                    "run_dir": str(item["inputs"]["run_dir"]),
                    "visited_fraction": result.summary["visited_fraction"],
                    "fixed_points": len(fixed_points),
                    "lap_pairs": len(lap_paths),
                }
            )
        pd.DataFrame(model_summaries).to_csv(output_dir / "model_comparison_summary.csv", index=False)
        completed = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        manifest.update(
            {
                "status": "completed",
                "completed_at": completed,
                "common_embedding": common_metadata,
                "shared_sde": calibration,
                "models": model_summaries,
                "model_count": len(model_summaries),
            }
        )
        _write_json(manifest_path, manifest)
        return manifest
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        _write_json(manifest_path, manifest)
        raise


def default_output_dir(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
    return HERE / "outputs" / f"{mode}__{stamp}"


__all__ = ["default_output_dir", "discover_run_dirs", "run_pipeline"]
