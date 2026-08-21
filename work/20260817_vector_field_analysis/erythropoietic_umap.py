"""Erythropoietic UMAP views of original-space ODE vector-field metrics."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
LEGACY_VIZ_DIR = REPO_ROOT / "work" / "20260609_Hybrid5x3" / "viz"
LEGACY_VELOCITY_PATH = LEGACY_VIZ_DIR / "plot_velocity_umap.py"
SUPERCLASS = "Erythropoietic"
METRIC_COLUMNS = {
    "vf_speed": ("speed", "5_speed_umap.png", "speed ||V||", "viridis"),
    "vf_divergence": (
        "divergence",
        "6_divergence_umap.png",
        "divergence tr(J)",
        "coolwarm",
    ),
    "vf_acceleration_norm": (
        "acceleration_norm",
        "7_acceleration_norm_umap.png",
        "acceleration norm ||J V||",
        "viridis",
    ),
    "vf_cosine_velocity_acceleration": (
        "cosine_velocity_acceleration",
        "8_cosine_velocity_acceleration_umap.png",
        "cosine(V, J V)",
        "coolwarm",
    ),
}
EXPECTED_PNGS = tuple(value[1] for value in METRIC_COLUMNS.values()) + (
    "1_velocity_stream.png",
    "2_velocity_arrow.png",
    "3_velocity_stream_lineage.png",
    "4_velocity_arrow_lineage.png",
)


def _load_legacy_velocity_module():
    """Load the 20260609 implementation so its plotting helpers are reused."""

    module_name = "_scdiffusion_20260609_plot_velocity_umap"
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not LEGACY_VELOCITY_PATH.is_file():
        raise FileNotFoundError(f"missing legacy UMAP implementation: {LEGACY_VELOCITY_PATH}")
    legacy_dir = str(LEGACY_VIZ_DIR)
    if legacy_dir not in sys.path:
        sys.path.insert(0, legacy_dir)
    spec = importlib.util.spec_from_file_location(module_name, LEGACY_VELOCITY_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import legacy UMAP implementation: {LEGACY_VELOCITY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _to_dense_float32(value: Any) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    elif hasattr(value, "todense"):
        value = value.todense()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or not bool(np.isfinite(result).all()):
        raise ValueError("Erythropoietic expression matrix must be finite and two-dimensional")
    return result


def _gene_names(adata: Any) -> list[str]:
    if "gene_name" in adata.var:
        return [str(value) for value in adata.var["gene_name"].tolist()]
    return [str(value) for value in adata.var_names.tolist()]


def _plot_metric_umaps(adata_sub: Any, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc

    for obs_key, (_, filename, title, color_map) in METRIC_COLUMNS.items():
        sc.pl.umap(
            adata_sub,
            color=obs_key,
            color_map=color_map,
            show=False,
            title=f"Erythropoietic: {title} (original expression space)",
        )
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


def run_erythropoietic_umap(
    *,
    data_dir: str | Path,
    adapter: Any,
    expected_genes: Sequence[str],
    analysis_output: str | Path,
    max_cells: int = 0,
    seed: int = 1234,
    n_jobs: int = 32,
) -> dict[str, Any]:
    """Build one shared Erythropoietic UMAP and plot velocity plus four metrics."""

    import scanpy as sc
    import scvelo as scv

    from dynamo_analysis import evaluate_dataset

    scv.set_figure_params(transparent=False)

    # Keep the 20260609 loading and Superclass-selection path deliberately literal.
    adata = sc.read_h5ad(data_dir)
    if "Superclass" not in adata.obs:
        raise KeyError("real h5ad is missing required obs column 'Superclass'")
    if "celltype" not in adata.obs:
        raise KeyError("real h5ad is missing required obs column 'celltype'")
    observed_genes = _gene_names(adata)
    if observed_genes != [str(value) for value in expected_genes]:
        raise ValueError("real h5ad variable order does not match the restored ODE gene order")

    adata_sub = adata[adata.obs["Superclass"] == SUPERCLASS].copy()
    source_cell_count = int(adata_sub.n_obs)
    if source_cell_count == 0:
        raise ValueError("real h5ad contains no Superclass == 'Erythropoietic' cells")
    if max_cells > 0 and adata_sub.n_obs > max_cells:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.choice(adata_sub.n_obs, max_cells, replace=False))
        adata_sub = adata_sub[positions].copy()

    cell_ids = np.asarray(adata_sub.obs_names.astype(str), dtype=object)
    X = _to_dense_float32(adata_sub.X)
    if X.shape[1] != int(adapter.dimension):
        raise ValueError(
            f"Erythropoietic/model dimension mismatch: X={X.shape}, D={adapter.dimension}"
        )
    adata_sub.X = X

    metric_table, vector_field = evaluate_dataset(
        adapter,
        X,
        dataset="erythropoietic",
        annotations=adata_sub.obs["celltype"].astype(str).to_numpy(),
    )
    velocity = np.asarray(vector_field.velocity, dtype=np.float32)
    if velocity.shape != adata_sub.X.shape:
        raise ValueError(
            f"velocity_ode shape {velocity.shape} != adata_sub.X shape {adata_sub.X.shape}"
        )
    if len(metric_table) != adata_sub.n_obs or not np.array_equal(
        cell_ids, np.asarray(adata_sub.obs_names.astype(str), dtype=object)
    ):
        raise ValueError("Erythropoietic cells and vector-field metric rows are misaligned")

    adata_sub.layers["X"] = adata_sub.X.copy()
    adata_sub.layers["velocity_ode"] = velocity
    for obs_key, (metric_key, _, _, _) in METRIC_COLUMNS.items():
        values = metric_table[metric_key].to_numpy(dtype=np.float64)
        if values.shape != (adata_sub.n_obs,):
            raise ValueError(f"{metric_key} does not have one value per Erythropoietic cell")
        adata_sub.obs[obs_key] = values

    output_dir = Path(analysis_output) / "umap_by_superclass" / SUPERCLASS
    output_dir.mkdir(parents=True, exist_ok=True)

    # This is the single UMAP fit shared by all eight output figures.
    sc.tl.pca(adata_sub, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata_sub, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_sub)

    # Import here so the legacy scVelo compatibility shims cover graph construction
    # without affecting original-space vector-field evaluation above.
    legacy = _load_legacy_velocity_module()
    scv.tl.velocity_graph(
        adata_sub,
        vkey="velocity_ode",
        xkey="X",
        backend="loky",
        n_jobs=n_jobs,
    )
    scv.tl.velocity_embedding(adata_sub, basis="umap", vkey="velocity_ode")

    palette = adata_sub.uns.get("celltype_colors", None)
    legacy._plot_stream_arrow(
        adata_sub,
        str(output_dir),
        "celltype",
        list(palette) if palette is not None else None,
        "1_velocity_stream.png",
        "2_velocity_arrow.png",
    )
    legacy.apply_lineage_palette(adata_sub, SUPERCLASS, celltype_col="celltype")
    lineage_palette = adata_sub.uns.get("celltype_colors", None)
    legacy._plot_stream_arrow(
        adata_sub,
        str(output_dir),
        "celltype",
        list(lineage_palette) if lineage_palette is not None else None,
        "3_velocity_stream_lineage.png",
        "4_velocity_arrow_lineage.png",
    )
    _plot_metric_umaps(adata_sub, output_dir)

    coordinates = np.asarray(adata_sub.obsm["X_umap"], dtype=np.float64)
    if coordinates.shape != (adata_sub.n_obs, 2):
        raise ValueError(f"unexpected X_umap shape: {coordinates.shape}")
    csv_table = pd.DataFrame(
        {
            "cell/index": cell_ids,
            "celltype": adata_sub.obs["celltype"].astype(str).to_numpy(),
            "Superclass": adata_sub.obs["Superclass"].astype(str).to_numpy(),
            "UMAP1": coordinates[:, 0],
            "UMAP2": coordinates[:, 1],
            "speed": adata_sub.obs["vf_speed"].to_numpy(),
            "divergence": adata_sub.obs["vf_divergence"].to_numpy(),
            "acceleration_norm": adata_sub.obs["vf_acceleration_norm"].to_numpy(),
            "cosine_velocity_acceleration": adata_sub.obs[
                "vf_cosine_velocity_acceleration"
            ].to_numpy(),
        }
    )
    csv_path = output_dir / "erythropoietic_umap_metrics.csv"
    csv_table.to_csv(csv_path, index=False)

    missing = sorted(name for name in EXPECTED_PNGS if not (output_dir / name).is_file())
    if missing:
        raise RuntimeError(f"Erythropoietic UMAP did not create expected PNGs: {missing}")
    return {
        "superclass": SUPERCLASS,
        "source_cell_count": source_cell_count,
        "n_cells": int(adata_sub.n_obs),
        "subsampled": int(adata_sub.n_obs) != source_cell_count,
        "umap_fit_count": 1,
        "velocity_ode_shape": list(velocity.shape),
        "metric_lengths": {
            metric_key: int(len(metric_table))
            for metric_key, _, _, _ in METRIC_COLUMNS.values()
        },
        "output_dir": str(output_dir),
        "metrics_csv": str(csv_path),
        "pngs": sorted(EXPECTED_PNGS),
    }


__all__ = ["run_erythropoietic_umap"]
