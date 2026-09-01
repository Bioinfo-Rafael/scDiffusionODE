"""Historical Scanpy/scVelo embedding and plotting pipeline."""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_umaps(
    adata,
    output_dir,
    *,
    celltype_palette: Mapping[str, str],
) -> list[Path]:
    output = Path(output_dir)
    coordinates = np.asarray(adata.obsm["X_umap"])
    origin = adata.obs["sampling_origin"].astype(str).to_numpy()
    generated = origin == "Generated (unconditional)"
    real = ~generated
    created: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        coordinates[real, 0], coordinates[real, 1], s=4, alpha=0.45,
        c="#bdbdbd", linewidths=0, label="Real hematopoietic reference",
    )
    ax.scatter(
        coordinates[generated, 0], coordinates[generated, 1], s=8, alpha=0.65,
        c="#d62728", linewidths=0, label="Generated (unconditional)",
    )
    ax.set_title(
        "Real hematopoietic reference vs generated\n"
        "Generated cells have no intrinsic hematopoietic label"
    )
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.legend(frameon=False)
    path = output / "01_sampling_umap_real_hema_vs_generated.png"
    _save(fig, path); created.append(path)

    fig, ax = plt.subplots(figsize=(11, 8))
    celltypes = adata.obs["sampling_celltype"].astype(str).to_numpy()
    for name in sorted(set(celltypes[real])):
        selected = real & (celltypes == name)
        ax.scatter(
            coordinates[selected, 0], coordinates[selected, 1], s=4, alpha=0.65,
            c=celltype_palette.get(name, "#808080"), linewidths=0, label=name,
        )
    ax.scatter(
        coordinates[generated, 0], coordinates[generated, 1], s=9, alpha=0.7,
        c="#111111", linewidths=0, label="Generated (unconditional)", zorder=20,
    )
    ax.set_title(
        "Hematopoietic celltype reference + generated\n"
        "Generated is an unconditional category, not a biological annotation"
    )
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", markerscale=2)
    path = output / "02_sampling_umap_celltypes_plus_generated.png"
    _save(fig, path); created.append(path)
    return created


def compute_velocity_embeddings(adata, vkeys: Sequence[str], *, n_jobs: int = 32):
    import scvelo as scv

    if "X_umap" not in adata.obsm:
        raise KeyError("shared UMAP must be computed before velocity embeddings")
    original_umap = np.asarray(adata.obsm["X_umap"]).copy()
    if "X" not in adata.layers:
        adata.layers["X"] = adata.X.copy()
    for vkey in vkeys:
        if vkey not in adata.layers or adata.layers[vkey].shape != adata.shape:
            raise ValueError(f"velocity layer shape mismatch: {vkey}")
        graph_kwargs = dict(
            vkey=vkey, xkey="X", backend="loky", n_jobs=int(n_jobs)
        )
        if str(getattr(scv, "__version__", "")) == "0.2.5":
            # scVelo 0.2.5 asks NumPy to make a homogeneous array from its
            # variable-length parallel graph chunks. NumPy >=1.24 correctly
            # rejects that ragged conversion. Preserve the list expected by
            # scVelo's following zip(*res), scoped to this one call only.
            graph_module = importlib.import_module("scvelo.tools.velocity_graph")
            original_parallelize = graph_module.parallelize

            def list_parallelize(*args, **kwargs):
                kwargs["as_array"] = False
                return original_parallelize(*args, **kwargs)

            graph_module.parallelize = list_parallelize
            try:
                scv.tl.velocity_graph(adata, **graph_kwargs)
            finally:
                graph_module.parallelize = original_parallelize
        else:
            scv.tl.velocity_graph(adata, **graph_kwargs)
        scv.tl.velocity_embedding(adata, basis="umap", vkey=vkey)
        if not np.array_equal(original_umap, np.asarray(adata.obsm["X_umap"])):
            raise RuntimeError("velocity computation changed the shared UMAP coordinates")
        embedded_key = f"{vkey}_umap"
        if embedded_key not in adata.obsm:
            raise RuntimeError(f"scVelo did not create {embedded_key}")
        if np.asarray(adata.obsm[embedded_key]).shape != (adata.n_obs, 2):
            raise RuntimeError(f"invalid velocity embedding shape for {vkey}")
    return original_umap


def _velocity_plot(
    adata,
    *,
    vkey: str,
    kind: str,
    title: str,
    palette,
    ax,
):
    import scvelo as scv

    common = dict(
        basis="umap", vkey=vkey, color="celltype", palette=palette,
        show=False, legend_loc="right margin", size=5, alpha=1, ax=ax,
        title=title,
    )
    grid_neighbors = max(1, int(adata.n_obs / 50))

    def draw():
        if kind == "stream":
            scv.pl.velocity_embedding_stream(adata, n_neighbors=grid_neighbors, **common)
        elif kind == "arrow":
            scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, **common)
        elif kind == "grid":
            scv.pl.velocity_embedding_grid(adata, n_neighbors=grid_neighbors, **common)
        else:
            raise ValueError(f"unknown velocity plot kind: {kind}")

    if str(getattr(scv, "__version__", "")) == "0.2.5":
        # pandas removed assignment to Categorical.categories, which scVelo
        # 0.2.5 performs only while constructing its legend. Keep the colored
        # scatter, suppress that legacy helper locally, then draw the same
        # category legend with supported Matplotlib primitives.
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
            Line2D([0], [0], marker="o", linestyle="", markersize=5, color=color, label=name)
            for name, color in zip(categories, colors)
        ]
        if handles:
            ax.legend(handles=handles, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        draw()


def plot_velocity_triplet(
    adata,
    output_dir,
    *,
    vkey: str,
    title: str,
    filenames: Sequence[str],
) -> list[Path]:
    if len(filenames) != 3:
        raise ValueError("stream/arrow/grid require exactly three filenames")
    palette = list(adata.uns.get("celltype_colors", [])) or None
    created = []
    for kind, filename in zip(("stream", "arrow", "grid"), filenames):
        fig, ax = plt.subplots(figsize=(10, 8))
        _velocity_plot(
            adata, vkey=vkey, kind=kind, title=f"{title} ({kind})",
            palette=palette, ax=ax,
        )
        path = Path(output_dir) / filename
        _save(fig, path); created.append(path)
    return created


def plot_side_by_side_stream(
    adata,
    output_path,
    *,
    ode_vkey: str,
    cell_vkey: str,
    timestep: int,
) -> Path:
    palette = list(adata.uns.get("celltype_colors", [])) or None
    coordinates = np.asarray(adata.obsm["X_umap"])
    margin = np.maximum(np.ptp(coordinates, axis=0) * 0.03, 0.1)
    xlim = (coordinates[:, 0].min() - margin[0], coordinates[:, 0].max() + margin[0])
    ylim = (coordinates[:, 1].min() - margin[1], coordinates[:, 1].max() + margin[1])
    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    _velocity_plot(
        adata, vkey=ode_vkey, kind="stream", title="ODE field (dx/dt)",
        palette=palette, ax=axes[0],
    )
    _velocity_plot(
        adata, vkey=cell_vkey, kind="stream",
        title=f"CellUnet diffusion-output field (t={timestep})",
        palette=palette, ax=axes[1],
    )
    for ax in axes:
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    fig.suptitle("Shared hematopoietic UMAP; fields are semantically distinct")
    path = Path(output_path)
    _save(fig, path)
    return path


def try_plot_paga(adata, output_path, *, vkey: str, title: str) -> tuple[Path | None, str]:
    import scvelo as scv

    try:
        if adata.obs["celltype"].nunique() < 2 or adata.n_obs < 15:
            return None, "skipped: insufficient cells or celltypes"
        if "distances" in adata.obsp:
            adata.uns.setdefault("neighbors", {})["distances"] = adata.obsp["distances"]
            adata.uns["neighbors"]["connectivities"] = adata.obsp["connectivities"]
        scv.tl.paga(adata, groups="celltype", vkey=vkey)
        fig, ax = plt.subplots(figsize=(11, 8))
        scv.pl.paga(
            adata, basis="umap", color="celltype",
            palette=list(adata.uns.get("celltype_colors", [])) or None,
            node_size_scale=0.6, min_edge_width=0.5, max_edge_width=2,
            threshold=0.05, alpha=0.8, show=False, ax=ax, title=title,
        )
        path = Path(output_path)
        _save(fig, path)
        return path, "created"
    except Exception as error:  # PAGA is explicitly optional.
        message = f"PAGA warning for {vkey}: {type(error).__name__}: {error}"
        warnings.warn(message)
        plt.close("all")
        return None, message


__all__ = [
    "compute_velocity_embeddings",
    "plot_sampling_umaps",
    "plot_side_by_side_stream",
    "plot_velocity_triplet",
    "try_plot_paga",
]
