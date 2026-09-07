"""Single-run and partial 12-run orchestration for hematopoietic visualization."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from analysis.runner import create_diffusion, git_commit, load_checkpoint, select_device
from models import build_model_from_config
from scripts.common import (
    EXPERIMENT_ORDER,
    RUNS_ROOT,
    choose_sampling_checkpoint,
    read_json,
    safe_component,
    validate_config,
    write_json,
)

from .core import (
    assert_gene_alignment,
    build_sampling_anndata,
    compute_common_umap,
    compute_vector_fields,
    dense_float32,
    file_sha256,
    gene_names_from_adata,
    select_hematopoietic_subset,
)
from .palette import HISTORICAL_PALETTE_SOURCE, apply_combined_hematopoietic_palette
from .plotting import (
    compute_velocity_embeddings,
    plot_sampling_umaps,
    plot_side_by_side_stream,
    plot_velocity_triplet,
    try_plot_paga,
)


@dataclass(frozen=True)
class HematopoieticVizOptions:
    timesteps: tuple[int, ...] = (0,)
    batch_size: int = 128
    seed: int = 1234
    noise_seed: int = 1234
    device: str = "auto"
    n_jobs: int = 32
    pca_components: int = 50
    neighbors: int = 15
    neighbor_pcs: int = 40
    superclass_column: str = ""
    superclasses: tuple[str, ...] = ()
    celltype_column: str = "celltype"
    sample_path: str = ""
    save_h5ad: bool = True
    paga: bool = True
    force: bool = False


def parse_timesteps(spec: str) -> tuple[int, ...]:
    values = tuple(dict.fromkeys(int(value.strip()) for value in spec.split(",") if value.strip()))
    if not values or min(values) < 0:
        raise ValueError("timesteps must be a non-empty comma-separated list of nonnegative integers")
    return values


def _sample_candidates(run: Path, explicit: str = "") -> list[Path]:
    sample_root = (run / "samples").resolve()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        path.relative_to(sample_root)
        candidates = [path]
    else:
        candidates = sorted(sample_root.glob("*.npz")) if sample_root.is_dir() else []
    return [path for path in candidates if path.is_file() and path.with_suffix(".json").is_file()]


def resolve_current_sample(run_dir, config: dict, explicit: str = "") -> tuple[Path, Path, dict]:
    """Resolve a sample whose sidecar points to the current sampling checkpoint."""

    run = Path(run_dir).resolve()
    expected_checkpoint = choose_sampling_checkpoint(run, config["ema_rate"]).resolve()
    matches = []
    for sample in _sample_candidates(run, explicit):
        sidecar_path = sample.with_suffix(".json")
        sidecar = read_json(sidecar_path)
        recorded_sample = Path(sidecar.get("sample_path", "")).expanduser().resolve()
        checkpoint = Path(sidecar.get("checkpoint", "")).expanduser().resolve()
        if recorded_sample != sample.resolve():
            continue
        try:
            checkpoint.relative_to((run / "checkpoints").resolve())
        except ValueError:
            continue
        if checkpoint == expected_checkpoint:
            matches.append((sample, checkpoint, sidecar))
    if not matches:
        raise FileNotFoundError(
            f"no existing sample sidecar matches current checkpoint {expected_checkpoint}"
        )
    return max(matches, key=lambda item: item[0].stat().st_mtime_ns)


def _paths(run: Path) -> dict[str, Path]:
    root = run / "hematopoietic_viz"
    paths = {
        "root": root,
        "figures": root / "figures",
        "csv": root / "csv",
        "h5ad": root / "h5ad",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _sampling_coordinates_frame(adata) -> pd.DataFrame:
    coordinates = np.asarray(adata.obsm["X_umap"])
    return pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "sampling_origin": adata.obs["sampling_origin"].astype(str).to_numpy(),
        "sampling_celltype": adata.obs["sampling_celltype"].astype(str).to_numpy(),
        "umap_1": coordinates[:, 0],
        "umap_2": coordinates[:, 1],
    })


def _vector_coordinates_frame(adata, vkeys: Sequence[str]) -> pd.DataFrame:
    coordinates = np.asarray(adata.obsm["X_umap"])
    values = {
        "obs_name": adata.obs_names.astype(str),
        "celltype": adata.obs["celltype"].astype(str).to_numpy(),
        "umap_1": coordinates[:, 0],
        "umap_2": coordinates[:, 1],
    }
    for vkey in vkeys:
        embedded = np.asarray(adata.obsm[f"{vkey}_umap"])
        values[f"{vkey}_umap_1"] = embedded[:, 0]
        values[f"{vkey}_umap_2"] = embedded[:, 1]
    return pd.DataFrame(values)


def _jsonable_options(options: HematopoieticVizOptions) -> dict:
    value = asdict(options)
    value["timesteps"] = list(options.timesteps)
    value["superclasses"] = list(options.superclasses)
    return value


def _completion_matches(metadata: dict, sample_path: Path, checkpoint: Path) -> bool:
    if metadata.get("status") != "completed":
        return False
    try:
        recorded_sample = Path(metadata["sample_path"]).expanduser().resolve()
        recorded_checkpoint = Path(metadata["checkpoint"]).expanduser().resolve()
    except (KeyError, TypeError):
        return False
    return recorded_sample == sample_path.resolve() and recorded_checkpoint == checkpoint.resolve()


def run_visualization(run_dir, options: HematopoieticVizOptions) -> dict:
    import scanpy as sc

    run = Path(run_dir).resolve()
    config = read_json(run / "exp_config.json")
    validate_config(config)
    paths = _paths(run)
    metadata_path = paths["root"] / "metadata.json"
    sample_path, checkpoint, sample_sidecar = resolve_current_sample(
        run, config, options.sample_path
    )
    if metadata_path.is_file() and not options.force:
        existing = read_json(metadata_path)
        if _completion_matches(existing, sample_path, checkpoint):
            return {
                "status": "skipped_completed",
                "run_dir": str(run),
                "metadata": str(metadata_path),
            }
    sample_hash_before = file_sha256(sample_path)
    with np.load(sample_path, allow_pickle=False) as archive:
        if "cell_gen" not in archive:
            raise KeyError(f"sample archive lacks cell_gen: {sample_path}")
        generated = dense_float32(archive["cell_gen"])

    data_path = Path(config["data_dir"]).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"exp_config.json data_dir does not exist: {data_path}")
    adata = sc.read_h5ad(data_path)
    obs_inventory = {
        "columns": [str(value) for value in adata.obs.columns],
        "Superclass_unique": (
            sorted(adata.obs["Superclass"].dropna().astype(str).unique().tolist())
            if "Superclass" in adata.obs else []
        ),
        "superclass_unique": (
            sorted(adata.obs["superclass"].dropna().astype(str).unique().tolist())
            if "superclass" in adata.obs else []
        ),
        "celltype_unique": (
            sorted(adata.obs[options.celltype_column].dropna().astype(str).unique().tolist())
            if options.celltype_column in adata.obs else []
        ),
    }
    genes = gene_names_from_adata(adata)
    real_subset, subset_metadata = select_hematopoietic_subset(
        adata,
        superclass_column=options.superclass_column,
        superclasses=options.superclasses,
        celltype_column=options.celltype_column,
    )
    # The rest of the pipeline consistently uses the conventional plotting key.
    if options.celltype_column != "celltype":
        real_subset.obs["celltype"] = real_subset.obs[options.celltype_column].copy()
    real_x = dense_float32(real_subset.X)

    diffusion = create_diffusion(config)
    if not options.timesteps or min(options.timesteps) < 0 or max(options.timesteps) >= diffusion.num_timesteps:
        raise ValueError(f"timesteps must be within [0,{diffusion.num_timesteps - 1}]")
    device = select_device(options.device)
    model = build_model_from_config(config, genes, diffusion.num_timesteps, device)
    load_checkpoint(model, checkpoint, device)
    alignment = assert_gene_alignment(
        genes,
        adata.X,
        generated,
        model_genes=genes,
        sample_created_from_run_config=(
            sample_path.is_relative_to((run / "samples").resolve())
            and checkpoint.is_relative_to((run / "checkpoints").resolve())
            and Path(sample_sidecar["sample_path"]).expanduser().resolve() == sample_path
        ),
    )
    pd.DataFrame({
        "gene_index": np.arange(len(genes), dtype=np.int64),
        "gene_name": genes,
        "var_name": adata.var_names.astype(str),
    }).to_csv(paths["csv"] / "gene_order.csv", index=False)
    del adata

    apply_combined_hematopoietic_palette(
        real_subset,
        selected_superclasses=subset_metadata["selected_superclasses"],
        celltype_column="celltype",
    )
    palette = {
        str(name): str(color)
        for name, color in zip(
            real_subset.obs["celltype"].cat.categories,
            real_subset.uns["celltype_colors"],
        )
    }
    sampling_adata = build_sampling_anndata(real_subset, generated)
    sampling_embedding = compute_common_umap(
        sampling_adata,
        pca_components=options.pca_components,
        neighbors=options.neighbors,
        neighbor_pcs=options.neighbor_pcs,
        seed=options.seed,
    )
    sampling_figures = plot_sampling_umaps(
        sampling_adata, paths["figures"], celltype_palette=palette
    )
    _sampling_coordinates_frame(sampling_adata).to_csv(
        paths["csv"] / "sampling_umap_coordinates.csv", index=False
    )
    h5ad_paths = []
    if options.save_h5ad:
        sampling_h5ad = paths["h5ad"] / "sampling_umap.h5ad"
        sampling_adata.write_h5ad(sampling_h5ad, compression="gzip")
        h5ad_paths.append(str(sampling_h5ad))
    del sampling_adata, generated

    vector_adata = real_subset.copy()
    vector_embedding = compute_common_umap(
        vector_adata,
        pca_components=options.pca_components,
        neighbors=options.neighbors,
        neighbor_pcs=options.neighbor_pcs,
        seed=options.seed,
    )
    shared_umap_before = np.asarray(vector_adata.obsm["X_umap"]).copy()
    ode_field, cell_fields, field_audit = compute_vector_fields(
        model,
        diffusion,
        real_x,
        options.timesteps,
        batch_size=options.batch_size,
        noise_seed=options.noise_seed,
        device=device,
    )
    ode_vkey = "velocity_ode_20260830"
    vector_adata.layers[ode_vkey] = ode_field
    cell_vkeys = {}
    for timestep, field in cell_fields.items():
        key = f"velocity_cellunet_t{timestep:03d}_20260830"
        vector_adata.layers[key] = field
        cell_vkeys[timestep] = key
    vkeys = [ode_vkey, *(cell_vkeys[timestep] for timestep in options.timesteps)]
    shared_umap_returned = compute_velocity_embeddings(
        vector_adata, vkeys, n_jobs=options.n_jobs
    )
    if not np.array_equal(shared_umap_before, shared_umap_returned):
        raise RuntimeError("ODE and CellUnet did not use a common UMAP")

    vector_figures = plot_velocity_triplet(
        vector_adata,
        paths["figures"],
        vkey=ode_vkey,
        title=f"ODE field | {config['ode_type']} | lambda={config['cell_ode_reg_lambda_20260830']}",
        filenames=(
            "10_ode_velocity_stream.png",
            "11_ode_velocity_arrow.png",
            "12_ode_velocity_grid.png",
        ),
    )
    for timestep in options.timesteps:
        vector_figures.extend(plot_velocity_triplet(
            vector_adata,
            paths["figures"],
            vkey=cell_vkeys[timestep],
            title=(
                f"CellUnet diffusion-output field | diffusion_timestep={timestep} | "
                "not biological dx/dt"
            ),
            filenames=(
                f"20_cellunet_t{timestep:03d}_velocity_stream.png",
                f"21_cellunet_t{timestep:03d}_velocity_arrow.png",
                f"22_cellunet_t{timestep:03d}_velocity_grid.png",
            ),
        ))
    primary_timestep = options.timesteps[0]
    side_by_side = plot_side_by_side_stream(
        vector_adata,
        paths["figures"] / "30_ode_vs_cellunet_stream.png",
        ode_vkey=ode_vkey,
        cell_vkey=cell_vkeys[primary_timestep],
        timestep=primary_timestep,
    )
    vector_figures.append(side_by_side)

    paga_results = {}
    if options.paga:
        path, message = try_plot_paga(
            vector_adata,
            paths["figures"] / "13_ode_paga.png",
            vkey=ode_vkey,
            title="ODE velocity PAGA",
        )
        paga_results[ode_vkey] = {"status": message, "path": str(path) if path else ""}
        for timestep in options.timesteps:
            key = cell_vkeys[timestep]
            path, message = try_plot_paga(
                vector_adata,
                paths["figures"] / f"23_cellunet_t{timestep:03d}_paga.png",
                vkey=key,
                title=f"CellUnet diffusion-output PAGA (t={timestep})",
            )
            paga_results[key] = {"status": message, "path": str(path) if path else ""}

    _vector_coordinates_frame(vector_adata, vkeys).to_csv(
        paths["csv"] / "vector_field_umap_coordinates.csv", index=False
    )
    if options.save_h5ad:
        vector_h5ad = paths["h5ad"] / "hematopoietic_vector_fields.h5ad"
        vector_adata.write_h5ad(vector_h5ad, compression="gzip")
        h5ad_paths.append(str(vector_h5ad))

    sample_hash_after = file_sha256(sample_path)
    if sample_hash_before != sample_hash_after:
        raise RuntimeError("visualization modified the existing sample archive")
    alpha_bar_t0 = float(diffusion.alphas_cumprod[0])
    t0_is_exact = bool(alpha_bar_t0 == 1.0)
    config_path = run / "exp_config.json"
    sample_sidecar_path = sample_path.with_suffix(".json")
    sample_code_path = Path(__file__).resolve().parent.parent / "scripts/sample.py"
    metadata = {
        "status": "completed",
        "run_dir": str(run),
        "checkpoint": str(checkpoint),
        "exp_config_path": str(config_path),
        "exp_config_sha256": file_sha256(config_path),
        "sample_path": str(sample_path),
        "sample_sidecar_path": str(sample_sidecar_path),
        "sample_sidecar_sha256": file_sha256(sample_sidecar_path),
        "sample_code_path": str(sample_code_path),
        "sample_code_sha256": file_sha256(sample_code_path),
        "sample_sha256_before": sample_hash_before,
        "sample_sha256_after": sample_hash_after,
        "sample_is_unconditional": True,
        "generated_label_warning": (
            "Generated cells are unconditional samples and are not intrinsically "
            "hematopoietic-labeled."
        ),
        "data_path": str(data_path),
        **subset_metadata,
        **alignment,
        "gene_order_csv": str(paths["csv"] / "gene_order.csv"),
        "obs_inventory": obs_inventory,
        "sampling_embedding_parameters": sampling_embedding,
        "vector_embedding_parameters": vector_embedding,
        "shared_vector_umap_asserted": True,
        "vector_field_names": {
            "ode": ode_vkey,
            "cellunet": {str(key): value for key, value in cell_vkeys.items()},
        },
        "diffusion_timesteps": list(options.timesteps),
        "noise_seed": int(options.noise_seed),
        "diffusion_model_mean_type": str(diffusion.model_mean_type.name),
        "t0_alpha_cumprod": alpha_bar_t0,
        "t0_is_exact_clean_input": t0_is_exact,
        "t0_definition": (
            "q_sample(real_X, t=0, fixed_noise); t=0 is exact clean X"
            if t0_is_exact else
            "q_sample(real_X, t=0, fixed_noise); current t=0 is not exact clean X"
        ),
        "ode_time_dependence": "none for all four 20260830 single ODE implementations",
        "cellunet_semantics": (
            "diffusion target prediction (EPSILON in current config), not biological dx/dt"
        ),
        "field_audit": field_audit,
        "ode_type": config["ode_type"],
        "cell_ode_reg_lambda_20260830": float(config["cell_ode_reg_lambda_20260830"]),
        "umap_shared_between_ode_and_cellunet": True,
        "scvelo": {
            "velocity_graph": {"xkey": "X", "backend": "loky", "n_jobs": int(options.n_jobs)},
            "velocity_embedding": {"basis": "umap"},
            "plots": ["stream", "arrow", "grid"],
            "paga": paga_results,
        },
        "historical_palette_source": str(HISTORICAL_PALETTE_SOURCE),
        "figures": [str(path) for path in (*sampling_figures, *vector_figures)],
        "h5ad_outputs": h5ad_paths,
        "options": _jsonable_options(options),
        "git_commit": git_commit(),
    }
    write_json(metadata_path, metadata)
    return {
        "status": "completed",
        "run_dir": str(run),
        "metadata": str(metadata_path),
        "selected_cells": int(real_subset.n_obs),
        "figures": len(sampling_figures) + len(vector_figures),
    }


def run_all_available(
    batch_id: str,
    options: HematopoieticVizOptions,
    *,
    execute: Callable[[Path, HematopoieticVizOptions], dict] = run_visualization,
) -> dict:
    batch = safe_component(batch_id, "batch id")
    results = []
    for experiment in EXPERIMENT_ORDER:
        run = (RUNS_ROOT / experiment / batch).resolve()
        if not (run / "exp_config.json").is_file():
            message = "unfinished: missing run config"
            warnings.warn(f"{experiment}: {message}")
            results.append({"experiment": experiment, "status": "skipped_unfinished", "reason": message})
            continue
        config = read_json(run / "exp_config.json")
        try:
            sample_path, checkpoint, _sidecar = resolve_current_sample(
                run, config, options.sample_path
            )
        except (FileNotFoundError, ValueError) as error:
            warnings.warn(f"{experiment}: sample unavailable; skipped: {error}")
            results.append({
                "experiment": experiment,
                "status": "skipped_unfinished",
                "reason": str(error),
            })
            continue
        metadata = run / "hematopoietic_viz" / "metadata.json"
        if (
            metadata.is_file()
            and not options.force
            and _completion_matches(read_json(metadata), sample_path, checkpoint)
        ):
            results.append({"experiment": experiment, "status": "skipped_completed"})
            continue
        try:
            result = execute(run, replace(options, sample_path=""))
            results.append({"experiment": experiment, **result})
        except Exception as error:
            warnings.warn(f"{experiment}: visualization failed: {type(error).__name__}: {error}")
            results.append({
                "experiment": experiment,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            })
    summary = {
        "batch_id": batch,
        "results": results,
        "completed": sum(row["status"] == "completed" for row in results),
        "skipped_completed": sum(row["status"] == "skipped_completed" for row in results),
        "skipped_unfinished": sum(row["status"] == "skipped_unfinished" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
    }
    destination = RUNS_ROOT / "_hematopoietic_viz_batches" / batch / "summary.json"
    write_json(destination, summary)
    summary["summary_path"] = str(destination.resolve())
    return summary


__all__ = [
    "HematopoieticVizOptions",
    "parse_timesteps",
    "resolve_current_sample",
    "run_all_available",
    "run_visualization",
]
