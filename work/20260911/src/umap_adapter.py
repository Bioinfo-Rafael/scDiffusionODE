"""Snapshot ndarray -> unchanged 20260830 AnnData/PCA/neighbors/UMAP helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from .artifacts import load_trajectory, write_json
from .source_imports import umap_core


def select_snapshots(
    table: list[dict],
    *,
    snapshots: Sequence[int] | None = None,
    reverse_steps: Sequence[int] | None = None,
) -> list[int]:
    if (snapshots is None) == (reverse_steps is None):
        raise ValueError("explicitly specify exactly one of snapshots or reverse_steps")
    values = list(snapshots if snapshots is not None else reverse_steps)
    if not values or len(set(values)) != len(values):
        raise ValueError("snapshot selection must be nonempty and contain no duplicates")
    lookup = {row["reverse_step"]: row["snapshot_index"] for row in table}
    if snapshots is None:
        if any(value not in lookup for value in values):
            raise ValueError(f"reverse steps must be saved steps: {list(lookup)}")
        values = [lookup[value] for value in values]
    if any(type(value) is not int or not 0 <= value < len(table) for value in values):
        raise ValueError(f"snapshot indices must be integers in [0,{len(table) - 1}]")
    return values


def independent_embeddings(
    real, predictions: np.ndarray, *, seed: int, consume: Callable, core=None
) -> None:
    core = umap_core() if core is None else core
    for k in range(predictions.shape[1]):
        # Exactly one snapshot per fresh AnnData / PCA / neighbors / UMAP fit.
        combined = core.build_sampling_anndata(real, predictions[:, k, :])
        parameters = core.compute_common_umap(combined, seed=seed)
        consume(k, combined, parameters)


def selected_joint_embedding(
    real,
    predictions: np.ndarray,
    table: list[dict],
    selected: Sequence[int],
    *,
    seed: int,
    core=None,
):
    core = umap_core() if core is None else core
    selected = select_snapshots(table, snapshots=selected)
    generated = np.concatenate([predictions[:, k, :] for k in selected], axis=0)
    combined = core.build_sampling_anndata(real, generated)
    combined.obs["snapshot_index"] = np.concatenate(
        (np.full(real.n_obs, -1), np.repeat(selected, len(predictions)))
    )
    combined.obs["trajectory_id"] = np.concatenate(
        (np.full(real.n_obs, -1), np.tile(np.arange(len(predictions)), len(selected)))
    )
    steps = [table[k]["reverse_step"] for k in selected]
    combined.obs["reverse_step"] = np.concatenate(
        (np.full(real.n_obs, -1), np.repeat(steps, len(predictions)))
    )
    parameters = core.compute_common_umap(combined, seed=seed)
    return combined, parameters


def _load_real(meta: dict, predictions: np.ndarray, data_path: Path | None):
    import scanpy as sc

    core = umap_core()
    path = Path(data_path if data_path is not None else meta["data_path"]).expanduser().resolve()
    adata = sc.read_h5ad(path)
    genes = core.gene_names_from_adata(adata)
    if core.gene_order_hash(genes) != meta["gene_order_hash"]:
        raise ValueError("real data gene order hash differs from saved sampling gene order")
    core.assert_gene_alignment(
        genes,
        adata.X,
        predictions[:, 0, :],
        model_genes=meta["gene_names"],
        generated_genes=meta["gene_names"],
        sample_created_from_run_config=True,
    )
    real, selection = core.select_hematopoietic_subset(adata, superclasses=("Erythropoietic",))
    return real, {"data_path": str(path), **selection}, core


def _plot_embedding(
    adata,
    path: Path,
    title: str,
    *,
    selected: list[int] | None = None,
    table: list[dict] | None = None,
) -> None:
    # Existing plot_sampling_umaps hard-codes titles and two output filenames.
    # Only the scatter presentation is adapted for snapshot titles/Stage 2 colors.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xy = np.asarray(adata.obsm["X_umap"])
    generated = adata.obs["sampling_origin"].astype(str).to_numpy() == "Generated (unconditional)"
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        xy[~generated, 0],
        xy[~generated, 1],
        s=4,
        alpha=0.4,
        c="#bdbdbd",
        linewidths=0,
        label=f"Real: Superclass == Erythropoietic (n={(~generated).sum()})",
    )
    if selected is None:
        ax.scatter(
            xy[generated, 0],
            xy[generated, 1],
            s=8,
            alpha=0.65,
            c="#d62728",
            linewidths=0,
            label=f"Generated: unconditional pred_xstart (n={generated.sum()})",
        )
    else:
        colors = plt.get_cmap("turbo")(np.linspace(0.1, 0.9, len(selected)))
        for k, color in zip(selected, colors):
            mask = generated & (adata.obs["snapshot_index"].to_numpy() == k)
            row = table[k]
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=8,
                alpha=0.6,
                color=color,
                linewidths=0,
                label=f"Generated snapshot {k}: step {row['reverse_step']}, t={row['diffusion_t']} (n={mask.sum()})",
            )
    ax.set(title=title, xlabel="UMAP1", ylabel="UMAP2")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _coordinates(adata, path: Path) -> None:

    frame = adata.obs.copy()
    frame["UMAP1"], frame["UMAP2"] = np.asarray(adata.obsm["X_umap"]).T
    frame.index.name = "cell_id"
    frame.to_csv(path)


def plot_independent(result: Path, *, data_path: Path | None = None) -> dict:
    predictions, table, meta, _ = load_trajectory(result)
    real, selection, core = _load_real(meta, predictions, data_path)
    output = result / "umap_independent"
    output.mkdir(exist_ok=False)
    fits = []

    def consume(k, combined, parameters):
        row = table[k]
        stem = f"snapshot_{k:02d}_step_{row['reverse_step']:04d}_t{row['diffusion_t']:03d}"
        title = f"{meta['experiment']}\nSnapshot {k:02d} | reverse step {row['reverse_step']} | diffusion t={row['diffusion_t']}"
        _plot_embedding(combined, output / f"{stem}.png", title)
        _coordinates(combined, output / f"{stem}.csv")
        fits.append(
            {
                **row,
                "parameters": parameters,
                "figure": f"{stem}.png",
                "generated_cells": len(predictions),
            }
        )
        print(f"Independent UMAP {k + 1}/{len(table)}", flush=True)

    independent_embeddings(real, predictions, seed=meta["seed"], consume=consume, core=core)
    _contact_sheet(output, [fit["figure"] for fit in fits])
    info = {
        "stage": 1,
        "fit_count": len(fits),
        "mode": "one independent fit per snapshot",
        "representation": "pred_xstart",
        "seed": meta["seed"],
        "real_selection": selection,
        "fits": fits,
    }
    write_json(output / "metadata.json", info)
    return info


def _contact_sheet(output: Path, names: list[str]) -> None:
    from PIL import Image, ImageOps

    width, height, columns = 480, 320, 4
    rows = (len(names) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * width, rows * height), "white")
    for i, name in enumerate(names):
        with Image.open(output / name) as image:
            thumbnail = ImageOps.contain(image.convert("RGB"), (width, height))
            sheet.paste(thumbnail, ((i % columns) * width, (i // columns) * height))
    sheet.save(output / "umap_independent_contact_sheet.png")


def plot_selected(
    result: Path,
    *,
    snapshots: Sequence[int] | None = None,
    reverse_steps: Sequence[int] | None = None,
    data_path: Path | None = None,
) -> dict:
    predictions, table, meta, _ = load_trajectory(result)
    selected = select_snapshots(table, snapshots=snapshots, reverse_steps=reverse_steps)
    real, selection, core = _load_real(meta, predictions, data_path)
    output = result / "umap_selected_joint" / ("snapshots_" + "_".join(map(str, selected)))
    output.mkdir(parents=True, exist_ok=False)
    combined, parameters = selected_joint_embedding(
        real, predictions, table, selected, seed=meta["seed"], core=core
    )
    _plot_embedding(
        combined,
        output / "selected_joint_umap.png",
        f"{meta['experiment']}\nUser-selected clean predictions: one joint UMAP",
        selected=selected,
        table=table,
    )
    _coordinates(combined, output / "coordinates.csv")
    info = {
        "stage": 2,
        "fit_count": 1,
        "selected": [table[k] for k in selected],
        "real_selection": selection,
        "parameters": parameters,
        "representation": "pred_xstart",
    }
    write_json(output / "metadata.json", info)
    return info
