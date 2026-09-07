#!/usr/bin/env python3
"""Synthetic end-to-end smoke test for Dynamo, simulation, figures, and LAP."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from noeqthermo.config import load_config
from noeqthermo.dynamo_bridge import (
    evaluate_vector_field,
    fit_dynamo_vector_field,
    fixed_points_from_adata,
    run_least_action_paths,
    save_dynamo_results,
    save_lap,
)
from noeqthermo.landscape import (
    calibrate_common_sde,
    common_bounds,
    compute_landscape_quantities,
    save_landscape,
    simulate_stationary_distribution,
)
from noeqthermo.plotting import plot_core_figures, plot_lap


def main() -> int:
    import anndata as ad

    config = load_config("smoke")
    rng = np.random.default_rng(int(config["seed"]))
    left = rng.normal(loc=(-1.2, 0.0), scale=(0.35, 0.5), size=(70, 2))
    right = rng.normal(loc=(1.2, 0.0), scale=(0.35, 0.5), size=(70, 2))
    coordinates = np.vstack((left, right))
    labels = np.array(["Early erythroid"] * len(left) + ["Late erythroid"] * len(right))
    velocity = np.column_stack(
        (
            coordinates[:, 0] - coordinates[:, 0] ** 3 - 0.25 * coordinates[:, 1],
            0.25 * coordinates[:, 0] - 0.8 * coordinates[:, 1],
        )
    )
    adata = ad.AnnData(
        X=np.zeros((len(coordinates), 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"Superclass": "Erythropoietic", "celltype": labels},
            index=[f"cell_{index}" for index in range(len(coordinates))],
        ),
        var=pd.DataFrame({"use_for_pca": [True]}, index=["placeholder"]),
    )
    adata.obsm["X_umap"] = coordinates
    adata.obsm["velocity_umap"] = velocity
    distances = cdist(coordinates, coordinates)
    np.fill_diagonal(distances, 0.0)
    graph = csr_matrix(distances)
    adata.obsp["umap_distances"] = graph
    adata.obsp["distances"] = graph
    adata.obsp["connectivities"] = csr_matrix((distances > 0).astype(float))
    adata.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"method": "synthetic_complete_graph"},
    }

    with tempfile.TemporaryDirectory(prefix="noeq-smoke-") as temporary:
        output = Path(temporary)
        fit = fit_dynamo_vector_field(adata, config["dynamo"])
        save_dynamo_results(adata, output, fit)
        vecfld = adata.uns["VecFld_umap"]

        def field(points):
            return evaluate_vector_field(points, vecfld)

        probe = field(np.array([[0.0, 0.0], [0.5, 0.25]]))
        if probe.shape != (2, 2) or not np.isfinite(probe).all():
            raise RuntimeError("arbitrary-coordinate Dynamo evaluation failed")
        calibrated = calibrate_common_sde(coordinates, [velocity], config["sde"])
        bounds = common_bounds(coordinates, config["sde"])
        num, fx, fy, simulation = simulate_stationary_distribution(
            field,
            bounds,
            config["sde"],
            calibrated,
            seed=int(config["seed"]),
            checkpoint_dir=output / "simulation",
        )
        result = compute_landscape_quantities(
            field,
            bounds,
            num,
            fx,
            fy,
            diffusion=float(calibrated["D"]),
        )
        save_landscape(result, output)
        points, types = fixed_points_from_adata(adata)
        figures = plot_core_figures(
            result, coordinates, labels, points, types, output, config["plot"]
        )
        lap_config = dict(config["lap"])
        lap_config["max_pairs"] = 1
        paths, lap_status = run_least_action_paths(
            adata,
            celltype_key="celltype",
            config=lap_config,
            diffusion=float(calibrated["D"]),
        )
        save_lap(paths, lap_status, output)
        if paths:
            plot_lap(result, coordinates, labels, paths, output / "10_least_action_paths.png", config["plot"])
            figures.append("10_least_action_paths.png")
        expected = {
            "dynamo_vecfld_umap.npz",
            "landscape_flux_arrays.npz",
            "01_umap_vector_field.png",
            "09_landscape_flux_overlay.png",
            "least_action_metadata.json",
        }
        missing = sorted(name for name in expected if not (output / name).is_file())
        if missing:
            raise RuntimeError(f"synthetic smoke outputs missing: {missing}")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "temporary_output": str(output),
                    "dynamo": fit,
                    "simulation": simulation,
                    "landscape": result.summary,
                    "lap": lap_status,
                    "figures": figures,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
