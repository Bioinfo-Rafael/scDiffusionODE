"""Thin bridge to the exact Dynamo APIs used in the two released studies."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def import_dynamo(expected_version: str):
    try:
        import dynamo as dyn
    except ImportError as exc:
        raise ImportError(
            "dynamo-release is required; install work/20260821_noeqThermo/requirements.txt"
        ) from exc
    observed = str(getattr(dyn, "__version__", "unknown"))
    if observed != str(expected_version):
        raise RuntimeError(
            f"Dynamo version mismatch: expected {expected_version}, observed {observed}. "
            "The analysis pins the paper-era API for reproducibility."
        )
    return dyn


def fit_dynamo_vector_field(adata: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the paper pipeline: VectorField -> topography -> curl."""

    dyn = import_dynamo(str(config["version"]))
    dyn.dynamo_logger.main_silence()
    restart_num = int(config["restart_num"])
    restart_seeds = [int(value) for value in config["restart_seeds"]]
    if len(restart_seeds) != restart_num:
        raise ValueError("dynamo.restart_seeds length must equal restart_num")
    dyn.vf.VectorField(
        adata,
        basis="umap",
        M=int(config["M"]),
        grid_num=int(config["grid_num"]),
        min_vel_corr=float(config["min_velocity_correlation"]),
        restart_num=restart_num,
        restart_seed=restart_seeds,
        map_topography=False,
        pot_curl_div=False,
        cores=int(config["cores"]),
    )
    if "VecFld_umap" not in adata.uns:
        raise RuntimeError("dyn.vf.VectorField did not create adata.uns['VecFld_umap']")
    dyn.vf.topography(adata, n=int(config["topography_samples"]), basis="umap")
    dyn.vf.curl(adata, basis="umap")
    if "curl_umap" not in adata.obs:
        raise RuntimeError("dyn.vf.curl did not create adata.obs['curl_umap']")
    vecfld = adata.uns["VecFld_umap"]
    probes = np.vstack(
        (
            np.asarray(adata.obsm["X_umap"][:3], dtype=np.float64),
            np.mean(np.asarray(adata.obsm["X_umap"], dtype=np.float64), axis=0, keepdims=True),
        )
    )
    evaluated = evaluate_vector_field(probes, vecfld, expected_version=str(config["version"]))
    if evaluated.shape != probes.shape or not np.isfinite(evaluated).all():
        raise RuntimeError("analytical Dynamo field failed arbitrary-coordinate evaluation")
    return {
        "dynamo_version": str(config["version"]),
        "n_control_points": int(np.asarray(vecfld["X_ctrl"]).shape[0]),
        "n_valid_cells": int(np.asarray(vecfld["valid_ind"]).size),
        "probe_coordinates": probes.tolist(),
        "probe_velocity": evaluated.tolist(),
    }


def evaluate_vector_field(
    points: np.ndarray,
    vecfld: Mapping[str, Any],
    *,
    expected_version: str = "1.4.1",
) -> np.ndarray:
    """Use Dynamo's released analytical SparseVFC function, as the papers do."""

    dyn = import_dynamo(expected_version)
    points = np.asarray(points, dtype=np.float64)
    was_vector = points.ndim == 1
    points = np.atleast_2d(points)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Dynamo field input must be finite and shaped (n, 2)")
    values = np.asarray(dyn.vf.utils.vector_field_function(points, vecfld), dtype=np.float64)
    if values.shape != points.shape or not np.isfinite(values).all():
        raise FloatingPointError("Dynamo analytical field returned invalid values")
    return values[0] if was_vector else values


def fixed_points_from_adata(adata: Any) -> tuple[np.ndarray, np.ndarray]:
    vecfld = adata.uns["VecFld_umap"]
    points = np.asarray(vecfld.get("Xss", np.empty((0, 2))), dtype=np.float64)
    types = np.asarray(vecfld.get("ftype", np.empty(0)), dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    points = np.atleast_2d(points)
    if points.shape[1] != 2 or len(types) != len(points):
        raise ValueError("Dynamo fixed-point coordinates/types are inconsistent")
    return points, types


def save_dynamo_results(adata: Any, output_dir: str | Path, fit_metadata: Mapping[str, Any]) -> None:
    """Save portable arrays instead of pickling Dynamo runtime objects."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vecfld = adata.uns["VecFld_umap"]
    array_keys = [
        "X",
        "Y",
        "V",
        "X_ctrl",
        "C",
        "P",
        "VFCIndex",
        "valid_ind",
        "grid",
        "grid_V",
        "Xss",
        "ftype",
        "confidence",
    ]
    arrays = {
        key: np.asarray(vecfld[key])
        for key in array_keys
        if key in vecfld and vecfld[key] is not None and not isinstance(vecfld[key], dict)
    }
    np.savez_compressed(output_dir / "dynamo_vecfld_umap.npz", **arrays)
    scalar_metadata = {
        key: value
        for key, value in vecfld.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }
    scalar_metadata["fit_validation"] = dict(fit_metadata)
    (output_dir / "dynamo_vecfld_metadata.json").write_text(
        json.dumps(scalar_metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    points, types = fixed_points_from_adata(adata)
    pd.DataFrame(
        {
            "fixed_point": np.arange(len(points), dtype=int),
            "UMAP1": points[:, 0] if len(points) else [],
            "UMAP2": points[:, 1] if len(points) else [],
            "ftype": types,
            "classification": [
                {-1: "attractor", 0: "saddle", 1: "repellor"}.get(int(value), "unknown")
                if np.isfinite(value)
                else "unknown"
                for value in types
            ],
        }
    ).to_csv(output_dir / "fixed_points_umap.csv", index=False)
    pd.DataFrame(
        {
            "cell_id": adata.obs_names.astype(str),
            "curl_umap_dynamo": np.asarray(adata.obs["curl_umap"], dtype=np.float64),
        }
    ).to_csv(output_dir / "curl_umap_cells.csv", index=False)


def _select_endpoint_states(
    adata: Any,
    celltype_key: str,
    config: Mapping[str, Any],
) -> tuple[list[str], dict[str, str]]:
    labels = adata.obs[celltype_key].astype(str)
    minimum = int(config["minimum_cells_per_state"])
    counts = labels.value_counts()
    valid = counts[counts >= minimum].index.tolist()
    requested = [str(value) for value in config.get("endpoint_labels", [])]
    if requested:
        missing = sorted(set(requested).difference(valid))
        if missing:
            raise ValueError(f"configured LAP endpoint labels are absent or too small: {missing}")
        states = requested
    else:
        states = valid[: int(config["max_states"])]
    representatives: dict[str, str] = {}
    coordinates = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
    for state in states:
        indices = np.flatnonzero(labels.to_numpy() == state)
        centroid = coordinates[indices].mean(axis=0)
        representative = indices[np.argmin(np.linalg.norm(coordinates[indices] - centroid, axis=1))]
        representatives[state] = str(adata.obs_names[representative])
    return states, representatives


def run_least_action_paths(
    adata: Any,
    *,
    celltype_key: str,
    config: Mapping[str, Any],
    diffusion: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run ordered biologically labeled state pairs, or explicitly skip them."""

    if not bool(config.get("enabled", True)) or int(config["max_pairs"]) == 0:
        return [], {"status": "not_requested"}
    dyn = import_dynamo("1.4.1")
    states, representatives = _select_endpoint_states(adata, celltype_key, config)
    if len(states) < 2:
        return [], {
            "status": "skipped",
            "reason": "fewer than two sufficiently populated Erythropoietic celltype states",
            "states": states,
        }
    pairs = [(start, end) for start in states for end in states if start != end]
    pairs = pairs[: int(config["max_pairs"])]
    paths: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for pair_index, (start, end) in enumerate(pairs):
        add_key = f"LAP_umap_pair_{pair_index}"
        try:
            dyn.pd.least_action(
                adata,
                [representatives[start]],
                [representatives[end]],
                basis="umap",
                adj_key="umap_distances",
                min_lap_t=bool(config["minimize_time"]),
                n_points=int(config["n_points"]),
                D=float(diffusion) * float(config["D_scale"]),
                add_key=add_key,
                EM_steps=int(config["EM_steps"]),
            )
            record = adata.uns[add_key]
            prediction = np.asarray(record["prediction"][0], dtype=np.float64)
            action = np.asarray(record["action"][0], dtype=np.float64)
            if prediction.ndim != 2 or prediction.shape[1] != 2 or not np.isfinite(prediction).all():
                raise ValueError("Dynamo LAP prediction is invalid")
            paths.append(
                {
                    "start_label": start,
                    "end_label": end,
                    "start_cell": representatives[start],
                    "end_cell": representatives[end],
                    "prediction": prediction,
                    "action": action,
                    "time": np.asarray(record["t"][0], dtype=np.float64),
                    "mfpt": float(np.asarray(record["mftp"][0]).reshape(-1)[0]),
                }
            )
        # Dynamo may raise several numerical/scipy/network exceptions. Record a
        # failed biological pair without erasing already completed pairs.
        except Exception as exc:  # noqa: BLE001
            failures.append({"start": start, "end": end, "error": f"{type(exc).__name__}: {exc}"})
    status = {
        "status": "completed" if paths else "failed",
        "states": states,
        "representative_cells": representatives,
        "requested_pairs": len(pairs),
        "completed_pairs": len(paths),
        "failures": failures,
        "direction_note": "ordered label pairs; no developmental direction was fabricated",
    }
    return paths, status


def save_lap(paths: Sequence[Mapping[str, Any]], status: Mapping[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    serializable = []
    rows = []
    for path_index, path in enumerate(paths):
        serializable.append(
            {
                key: value
                for key, value in path.items()
                if key not in {"prediction", "action", "time"}
            }
        )
        prediction = np.asarray(path["prediction"])
        action = np.asarray(path["action"])
        time = np.asarray(path["time"])
        for point_index, point in enumerate(prediction):
            rows.append(
                {
                    "path": path_index,
                    "start_label": path["start_label"],
                    "end_label": path["end_label"],
                    "point": point_index,
                    "UMAP1": point[0],
                    "UMAP2": point[1],
                    "action": action[point_index] if point_index < len(action) else np.nan,
                    "time": time[point_index] if point_index < len(time) else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "least_action_paths.csv", index=False)
    (output_dir / "least_action_metadata.json").write_text(
        json.dumps({"status": dict(status), "paths": serializable}, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


__all__ = [
    "evaluate_vector_field",
    "fit_dynamo_vector_field",
    "fixed_points_from_adata",
    "import_dynamo",
    "run_least_action_paths",
    "save_dynamo_results",
    "save_lap",
]
