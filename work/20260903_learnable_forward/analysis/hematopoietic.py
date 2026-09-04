"""Historical hematopoietic UMAP and scVelo pipeline for learned drift."""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from sampling.artifacts import load_sample_archive
from scripts.common import REPO_ROOT, validate_run_directory, write_json
from .common import dense_float32, load_final_ema, metadata_matches_final_ema


DEFAULT_HEMATOPOIETIC_SUPERCLASSES = ("Erythropoietic", "Immune")
HISTORICAL_PALETTE_SOURCE = (
    REPO_ROOT
    / "work/20260215_embryonic/20260224_084326_Lamda5/velocity_by_Superclass.py"
)


def resolve_superclass_column(obs, requested: str = "") -> str:
    if requested:
        if requested not in obs.columns:
            raise KeyError(f"requested superclass column is absent: {requested}")
        return requested
    for candidate in ("Superclass", "superclass"):
        if candidate in obs.columns:
            return candidate
    matches = [name for name in obs.columns if str(name).lower() == "superclass"]
    if len(matches) == 1:
        return str(matches[0])
    raise KeyError("no unambiguous Superclass column found")


def resolve_superclass_selection(
    available_values: Sequence[str], requested: Sequence[str] | None = None
) -> tuple[str, ...]:
    available = {str(value) for value in available_values}
    if requested:
        selected = tuple(dict.fromkeys(str(value) for value in requested))
        missing = [value for value in selected if value not in available]
        if missing:
            raise ValueError("requested superclass values are absent: " + ", ".join(missing))
        return selected
    missing = [
        value for value in DEFAULT_HEMATOPOIETIC_SUPERCLASSES if value not in available
    ]
    if missing:
        raise ValueError(
            "default Erythropoietic + Immune values are absent: " + ", ".join(missing)
        )
    return DEFAULT_HEMATOPOIETIC_SUPERCLASSES


def select_hematopoietic_subset(
    adata,
    *,
    superclass_column: str = "",
    superclasses: Sequence[str] | None = None,
    celltype_column: str = "celltype",
    max_cells: int = 0,
    seed: int = 1234,
):
    column = resolve_superclass_column(adata.obs, superclass_column)
    available = sorted(adata.obs[column].dropna().astype(str).unique().tolist())
    selected = resolve_superclass_selection(available, superclasses)
    if celltype_column not in adata.obs.columns:
        raise KeyError(f"missing celltype column: {celltype_column}")
    mask = adata.obs[column].astype(str).isin(selected).to_numpy()
    indices = np.flatnonzero(mask)
    if int(max_cells) > 0 and len(indices) > int(max_cells):
        rng = np.random.default_rng(int(seed))
        indices = np.sort(rng.choice(indices, int(max_cells), replace=False))
    subset = adata[indices].copy()
    if subset.n_obs == 0:
        raise ValueError("hematopoietic selection produced zero cells")
    subset.obs[column] = subset.obs[column].astype("category")
    subset.obs[column] = subset.obs[column].cat.remove_unused_categories()
    subset.obs[celltype_column] = subset.obs[celltype_column].astype("category")
    subset.obs[celltype_column] = subset.obs[celltype_column].cat.remove_unused_categories()
    return subset, {
        "selected_superclass_column": column,
        "available_superclasses": available,
        "selected_superclasses": list(selected),
        "selected_cell_count": int(subset.n_obs),
        "selected_source_indices": indices.tolist(),
        "celltype_column": celltype_column,
        "celltype_counts": {
            str(key): int(value)
            for key, value in subset.obs[celltype_column].astype(str).value_counts().items()
        },
    }


def _historical_lineages():
    spec = importlib.util.spec_from_file_location(
        "historical_velocity_by_superclass_20260224", HISTORICAL_PALETTE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load historical palette: {HISTORICAL_PALETTE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LINEAGES


def apply_historical_palette(
    adata,
    *,
    selected_superclasses: Sequence[str],
    celltype_column: str = "celltype",
) -> dict[str, str]:
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt

    lineages = _historical_lineages()
    adata.obs[celltype_column] = adata.obs[celltype_column].astype("category")
    adata.obs[celltype_column] = adata.obs[celltype_column].cat.remove_unused_categories()
    present = [str(value) for value in adata.obs[celltype_column].cat.categories]
    mapping: dict[str, str] = {}
    ordered: list[str] = []
    for superclass in selected_superclasses:
        for _name, sequence, cmap_name in lineages.get(str(superclass), []):
            selected = [name for name in sequence if name in present and name not in ordered]
            cmap = plt.get_cmap(cmap_name)
            shades = np.linspace(0.25, 0.9, max(len(selected), 1))
            for name, shade in zip(selected, shades):
                mapping[name] = colors.to_hex(cmap(shade))
            ordered.extend(selected)
    fallback = plt.get_cmap("tab20").colors
    leftovers = [name for name in present if name not in ordered]
    for index, name in enumerate(sorted(leftovers)):
        mapping[name] = colors.to_hex(fallback[index % len(fallback)])
    category_order = ordered + sorted(leftovers)
    adata.obs[celltype_column] = adata.obs[celltype_column].cat.reorder_categories(
        category_order, ordered=True
    )
    adata.uns[f"{celltype_column}_colors"] = np.asarray(
        [mapping[name] for name in category_order], dtype=str
    )
    return mapping


def compute_common_umap(
    adata,
    *,
    pca_components: int = 50,
    neighbors: int = 15,
    neighbor_pcs: int = 40,
    seed: int = 1234,
) -> dict:
    import scanpy as sc

    if adata.n_obs < 3 or adata.n_vars < 2:
        raise ValueError("PCA/UMAP requires at least 3 cells and 2 genes")
    components = min(int(pca_components), adata.n_obs - 1, adata.n_vars - 1)
    if components < 2:
        raise ValueError("insufficient dimensions for PCA")
    used_neighbors = min(int(neighbors), adata.n_obs - 1)
    used_pcs = min(int(neighbor_pcs), components)
    sc.tl.pca(
        adata,
        svd_solver="arpack",
        n_comps=components,
        random_state=int(seed),
    )
    sc.pp.neighbors(adata, n_neighbors=used_neighbors, n_pcs=used_pcs)
    sc.tl.umap(adata, random_state=int(seed))
    coordinates = np.asarray(adata.obsm["X_umap"])
    if coordinates.shape != (adata.n_obs, 2) or not np.isfinite(coordinates).all():
        raise RuntimeError("invalid UMAP coordinates")
    return {
        "pca": {"n_comps": components, "svd_solver": "arpack"},
        "neighbors": {"n_neighbors": used_neighbors, "n_pcs": used_pcs},
        "umap": {"random_state": int(seed)},
        "input_preprocessing": "saved training X; no renormalization/log/scale",
    }


def _resolve_current_sample(run: Path, checkpoint: Path, gene_hash: str):
    matches = []
    for path in (run / "samples").glob("*.npz"):
        sidecar = path.with_suffix(".json")
        if not sidecar.is_file():
            continue
        with sidecar.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            Path(metadata.get("checkpoint_path", "")).expanduser().resolve()
            == checkpoint.resolve()
            and metadata.get("gene_order_sha256") == gene_hash
            and metadata.get("status") == "completed"
        ):
            matches.append(path.resolve())
    if not matches:
        raise FileNotFoundError("no completed sample matches the final EMA checkpoint")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def _build_sampling_anndata(real_subset, generated: np.ndarray, celltype_column: str):
    import anndata as ad

    real = real_subset.copy()
    real.obs["sampling_origin"] = "Real hematopoietic reference"
    real.obs["sampling_celltype"] = real.obs[celltype_column].astype(str)
    generated_adata = ad.AnnData(
        X=np.asarray(generated, dtype=np.float32),
        obs=pd.DataFrame(
            {
                "sampling_origin": ["Generated (unconditional)"] * len(generated),
                "sampling_celltype": ["Generated (unconditional)"] * len(generated),
            }
        ),
        var=real.var.copy(),
    )
    generated_adata.var_names = real.var_names.copy()
    real.obs_names = [f"real_hema_{index}" for index in range(real.n_obs)]
    generated_adata.obs_names = [f"generated_{index}" for index in range(len(generated))]
    combined = ad.concat([real, generated_adata], axis=0, join="inner", merge="same")
    if not np.array_equal(combined.var_names, real.var_names):
        raise RuntimeError("AnnData concat changed gene order")
    combined.obs["sampling_origin"] = combined.obs["sampling_origin"].astype("category")
    combined.obs["sampling_celltype"] = combined.obs["sampling_celltype"].astype("category")
    return combined


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def _plot_sampling_umaps(adata, output: Path, palette: Mapping[str, str]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coordinates = np.asarray(adata.obsm["X_umap"])
    origin = adata.obs["sampling_origin"].astype(str).to_numpy()
    generated = origin == "Generated (unconditional)"
    real = ~generated
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coordinates[real, 0], coordinates[real, 1], s=4, alpha=0.45, c="#bdbdbd", linewidths=0, label="Real hematopoietic reference")
    ax.scatter(coordinates[generated, 0], coordinates[generated, 1], s=8, alpha=0.65, c="#d62728", linewidths=0, label="Generated (unconditional)")
    ax.set(title="Real hematopoietic reference vs generated", xlabel="UMAP1", ylabel="UMAP2")
    ax.legend(frameon=False)
    _save_figure(fig, output / "real_vs_generated.png")

    fig, ax = plt.subplots(figsize=(11, 8))
    celltypes = adata.obs["sampling_celltype"].astype(str).to_numpy()
    for name in sorted(set(celltypes[real])):
        selected = real & (celltypes == name)
        ax.scatter(coordinates[selected, 0], coordinates[selected, 1], s=4, alpha=0.65, c=palette.get(name, "#808080"), linewidths=0, label=name)
    ax.scatter(coordinates[generated, 0], coordinates[generated, 1], s=9, alpha=0.7, c="#111111", linewidths=0, label="Generated (unconditional)", zorder=20)
    ax.set(title="Real celltypes + Generated (unconditional)", xlabel="UMAP1", ylabel="UMAP2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2)
    _save_figure(fig, output / "celltypes_plus_generated.png")


def analyze_generated_umap(
    run_dir,
    *,
    device: str = "auto",
    ema_rate=None,
    max_cells: int = 0,
    seed: int = 1234,
    force: bool = False,
) -> dict:
    run = validate_run_directory(run_dir)
    output = run / "analysis" / "hematopoietic_umap"
    metadata_path = output / "metadata.json"
    if metadata_path.is_file() and not force:
        with metadata_path.open(encoding="utf-8") as handle:
            if metadata_matches_final_ema(
                json.load(handle), run, ema_rate=ema_rate
            ):
                return {"status": "skipped_completed", "metadata": str(metadata_path)}
    loaded = load_final_ema(run, device=device, ema_rate=ema_rate)
    config = loaded["config"]
    celltype = str(config.get("celltype_column", "celltype"))
    real, subset_metadata = select_hematopoietic_subset(
        loaded["adata"],
        celltype_column=celltype,
        max_cells=max_cells,
        seed=seed,
    )
    layer = config.get("data_layer", config.get("layer"))
    # Generated cells live in exactly the representation used by training.
    # Put the same saved matrix into real.X before concatenation; do not
    # normalize, log-transform, or scale either group.
    real.X = dense_float32(real.layers[layer] if layer else real.X)
    palette = apply_historical_palette(
        real,
        selected_superclasses=subset_metadata["selected_superclasses"],
        celltype_column=celltype,
    )
    sample_path = _resolve_current_sample(
        run, loaded["checkpoint"], loaded["gene_order_sha256"]
    )
    generated, sample_genes, sample_metadata = load_sample_archive(sample_path)
    if sample_genes != loaded["genes"]:
        raise ValueError("sample gene order differs from training data")
    combined = _build_sampling_anndata(real, generated, celltype)
    embedding = compute_common_umap(combined, seed=seed)
    output.mkdir(parents=True, exist_ok=True)
    _plot_sampling_umaps(combined, output, palette)
    coordinates = np.asarray(combined.obsm["X_umap"])
    pd.DataFrame(
        {
            "obs_name": combined.obs_names.astype(str),
            "sampling_origin": combined.obs["sampling_origin"].astype(str),
            "sampling_celltype": combined.obs["sampling_celltype"].astype(str),
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
        }
    ).to_csv(output / "coordinates.csv", index=False)
    metadata = {
        "status": "completed",
        "checkpoint": str(loaded["checkpoint"]),
        "checkpoint_step": loaded["checkpoint_step"],
        "ema_rate": loaded["ema_rate"],
        "sample_path": str(sample_path),
        "sample_provenance": sample_metadata,
        "generated_label": "Generated (unconditional)",
        "generated_celltype_inference": False,
        "gene_order_sha256": loaded["gene_order_sha256"],
        "embedding": embedding,
        **subset_metadata,
    }
    write_json(metadata_path, metadata)
    adata_file = getattr(loaded["adata"], "file", None)
    if adata_file is not None:
        adata_file.close()
    return {"status": "completed", "metadata": str(metadata_path)}


def compute_velocity_embeddings(adata, vkey: str, *, n_jobs: int = 32) -> None:
    import scvelo as scv

    original_umap = np.asarray(adata.obsm["X_umap"]).copy()
    graph_kwargs = {"vkey": vkey, "xkey": "X", "backend": "loky", "n_jobs": int(n_jobs)}
    if str(getattr(scv, "__version__", "")) == "0.2.5":
        module = importlib.import_module("scvelo.tools.velocity_graph")
        original_parallelize = module.parallelize

        def list_parallelize(*args, **kwargs):
            kwargs["as_array"] = False
            return original_parallelize(*args, **kwargs)

        module.parallelize = list_parallelize
        try:
            scv.tl.velocity_graph(adata, **graph_kwargs)
        finally:
            module.parallelize = original_parallelize
    else:
        scv.tl.velocity_graph(adata, **graph_kwargs)
    scv.tl.velocity_embedding(adata, basis="umap", vkey=vkey)
    if not np.array_equal(original_umap, np.asarray(adata.obsm["X_umap"])):
        raise RuntimeError("scVelo changed the shared real-cell UMAP")


def _velocity_plot(adata, *, vkey: str, kind: str, title: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scvelo as scv

    fig, ax = plt.subplots(figsize=(10, 8))
    common = {
        "basis": "umap",
        "vkey": vkey,
        "color": "celltype",
        "palette": list(adata.uns.get("celltype_colors", [])) or None,
        "show": False,
        "legend_loc": "right margin",
        "size": 5,
        "alpha": 1,
        "ax": ax,
        "title": f"{title} ({kind})",
    }
    grid_neighbors = max(1, int(adata.n_obs / 50))
    def draw() -> None:
        if kind == "stream":
            scv.pl.velocity_embedding_stream(
                adata, n_neighbors=grid_neighbors, **common
            )
        elif kind == "arrow":
            scv.pl.velocity_embedding(
                adata, arrow_length=3, arrow_size=2, **common
            )
        elif kind == "grid":
            scv.pl.velocity_embedding_grid(
                adata, n_neighbors=grid_neighbors, **common
            )
        else:
            raise ValueError(f"unknown velocity plot kind: {kind}")

    if str(getattr(scv, "__version__", "")) == "0.2.5":
        # Match the established 20260830 compatibility shim: pandas removed
        # assignment to Categorical.categories, which scVelo 0.2.5 uses only
        # in its legacy legend helper.
        from matplotlib.lines import Line2D

        scatter_module = importlib.import_module("scvelo.plotting.scatter")
        original_set_legend = scatter_module.set_legend
        scatter_module.set_legend = lambda *args, **kwargs: None
        try:
            draw()
        finally:
            scatter_module.set_legend = original_set_legend
        categories = [str(value) for value in adata.obs["celltype"].cat.categories]
        colors = list(adata.uns.get("celltype_colors", []))
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=5,
                color=color,
                label=name,
            )
            for name, color in zip(categories, colors)
        ]
        if handles:
            ax.legend(
                handles=handles,
                frameon=False,
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
            )
    else:
        draw()
    _save_figure(fig, output)


def analyze_drift_velocity(
    run_dir,
    *,
    device: str = "auto",
    ema_rate=None,
    max_cells: int = 0,
    seed: int = 1234,
    n_jobs: int = 32,
    force: bool = False,
) -> dict:
    run = validate_run_directory(run_dir)
    output = run / "analysis" / "drift_velocity"
    metadata_path = output / "metadata.json"
    if metadata_path.is_file() and not force:
        with metadata_path.open(encoding="utf-8") as handle:
            if metadata_matches_final_ema(
                json.load(handle), run, ema_rate=ema_rate
            ):
                return {"status": "skipped_completed", "metadata": str(metadata_path)}
    loaded = load_final_ema(run, device=device, ema_rate=ema_rate)
    config = loaded["config"]
    celltype = str(config.get("celltype_column", "celltype"))
    real, subset_metadata = select_hematopoietic_subset(
        loaded["adata"],
        celltype_column=celltype,
        max_cells=max_cells,
        seed=seed,
    )
    if celltype != "celltype":
        real.obs["celltype"] = real.obs[celltype].copy()
    apply_historical_palette(
        real,
        selected_superclasses=subset_metadata["selected_superclasses"],
        celltype_column="celltype",
    )
    layer = config.get("data_layer", config.get("layer"))
    training_x = dense_float32(real.layers[layer] if layer else real.X)
    real.X = training_x.copy()
    embedding = compute_common_umap(real, seed=seed)
    process = loaded["components"].model.forward_process
    parameter = next(process.parameters())
    with torch.no_grad():
        drift = process.drift(
            torch.from_numpy(training_x).to(
                device=parameter.device, dtype=parameter.dtype
            )
        ).detach().cpu().float().numpy()
    if drift.shape != training_x.shape or not np.isfinite(drift).all():
        raise RuntimeError("learned forward drift has invalid shape or values")
    vkey = "learned_forward_drift"
    real.layers[vkey] = drift
    real.layers["X"] = training_x.copy()
    compute_velocity_embeddings(real, vkey, n_jobs=n_jobs)
    output.mkdir(parents=True, exist_ok=True)
    title = "Learned forward-diffusion drift field; not RNA velocity"
    for kind, filename in (
        ("stream", "velocity_stream.png"),
        ("arrow", "velocity_arrow.png"),
        ("grid", "velocity_grid.png"),
    ):
        _velocity_plot(real, vkey=vkey, kind=kind, title=title, output=output / filename)
    h5ad_path = output / "learned_forward_drift.h5ad"
    real.write_h5ad(h5ad_path, compression="gzip")
    metadata = {
        "status": "completed",
        "checkpoint": str(loaded["checkpoint"]),
        "checkpoint_step": loaded["checkpoint_step"],
        "ema_rate": loaded["ema_rate"],
        "field_name": "Learned forward-diffusion drift field",
        "field_formula": (
            "-(Q+D)x" if config["forward_model"] == "stationary_qd" else "Wx+b"
        ),
        "direction_reversed": False,
        "rna_velocity_claim": False,
        "embedding": embedding,
        "scvelo": {
            "velocity_graph": {"vkey": vkey, "xkey": "X", "backend": "loky", "n_jobs": int(n_jobs)},
            "velocity_embedding": {"basis": "umap", "vkey": vkey},
            "plots": ["stream", "arrow", "grid"],
        },
        **subset_metadata,
    }
    write_json(metadata_path, metadata)
    adata_file = getattr(loaded["adata"], "file", None)
    if adata_file is not None:
        adata_file.close()
    return {"status": "completed", "metadata": str(metadata_path)}


__all__ = [
    "DEFAULT_HEMATOPOIETIC_SUPERCLASSES",
    "analyze_drift_velocity",
    "analyze_generated_umap",
    "compute_common_umap",
    "compute_velocity_embeddings",
    "resolve_superclass_selection",
    "select_hematopoietic_subset",
]
