"""Data selection, exact alignment checks, UMAP, and model field evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from analysis.gradients import parameter_fingerprint


DEFAULT_HEMATOPOIETIC_SUPERCLASSES = ("Erythropoietic", "Immune")


def dense_float32(matrix) -> np.ndarray:
    value = matrix.toarray() if sp.issparse(matrix) else np.asarray(matrix)
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("expression/field matrix must be finite and two-dimensional")
    return value


def gene_names_from_adata(adata) -> list[str]:
    if "gene_name" not in adata.var.columns:
        raise KeyError("AnnData .var must contain gene_name")
    genes = [str(value) for value in adata.var["gene_name"].tolist()]
    if len(genes) != adata.n_vars or len(set(genes)) != len(genes):
        raise ValueError("gene_name must be unique and one-to-one with AnnData columns")
    return genes


def gene_order_hash(genes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for index, gene in enumerate(genes):
        digest.update(str(index).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(gene).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


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
    raise KeyError("no unambiguous superclass/Superclass column found")


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
    if "Hematopoietic" in available:
        return ("Hematopoietic",)
    missing = [value for value in DEFAULT_HEMATOPOIETIC_SUPERCLASSES if value not in available]
    if missing:
        raise ValueError(
            "default hematopoietic superclass values are absent: " + ", ".join(missing)
        )
    return DEFAULT_HEMATOPOIETIC_SUPERCLASSES


def select_hematopoietic_subset(
    adata,
    *,
    superclass_column: str = "",
    superclasses: Sequence[str] | None = None,
    celltype_column: str = "celltype",
):
    column = resolve_superclass_column(adata.obs, superclass_column)
    available = sorted(adata.obs[column].dropna().astype(str).unique().tolist())
    selected = resolve_superclass_selection(available, superclasses)
    if celltype_column not in adata.obs.columns:
        raise KeyError(f"missing celltype column: {celltype_column}")
    mask = adata.obs[column].astype(str).isin(selected).to_numpy()
    subset = adata[mask].copy()
    if subset.n_obs == 0:
        raise ValueError("hematopoietic selection produced zero cells")
    observed = set(subset.obs[column].astype(str).unique())
    if not observed.issubset(set(selected)):
        raise AssertionError("non-selected superclass leaked into subset")
    subset.obs[column] = subset.obs[column].astype("category")
    subset.obs[column] = subset.obs[column].cat.remove_unused_categories()
    subset.obs[celltype_column] = subset.obs[celltype_column].astype("category")
    subset.obs[celltype_column] = subset.obs[celltype_column].cat.remove_unused_categories()
    metadata = {
        "selected_superclass_column": column,
        "available_superclasses": available,
        "selected_superclasses": list(selected),
        "selected_cell_count": int(subset.n_obs),
        "selected_superclass_counts": {
            str(key): int(value)
            for key, value in subset.obs[column].astype(str).value_counts().items()
        },
        "celltype_column": celltype_column,
        "celltype_counts": {
            str(key): int(value)
            for key, value in subset.obs[celltype_column].astype(str).value_counts().items()
        },
    }
    return subset, metadata


def assert_gene_alignment(
    real_genes: Sequence[str],
    real_matrix,
    generated_matrix,
    *,
    model_genes: Sequence[str],
    sample_created_from_run_config: bool,
    generated_genes: Sequence[str] | None = None,
) -> dict:
    """Assert the strongest ordering contract available in current sample artifacts.

    Current ``sample.py`` stores only ``cell_gen``. It constructs both the model and
    sample width from the ordered ``exp_config.json -> data_dir -> var['gene_name']``
    list. The provenance assertion below is therefore required in addition to width.
    """

    genes = [str(value) for value in real_genes]
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("real gene ordering must be non-empty and unique")
    real_shape = tuple(real_matrix.shape)
    generated_shape = tuple(generated_matrix.shape)
    expected = len(genes)
    if len(real_shape) != 2 or real_shape[1] != expected:
        raise ValueError("real X is not aligned with var['gene_name']")
    if len(generated_shape) != 2 or generated_shape[1] != expected:
        raise ValueError("generated cell_gen width does not match ordered real genes")
    if [str(value) for value in model_genes] != genes:
        raise ValueError("model gene names/order do not exactly match ordered real genes")
    generated_names_embedded = generated_genes is not None
    if generated_names_embedded and [str(value) for value in generated_genes] != genes:
        raise ValueError("generated gene names/order do not exactly match ordered real genes")
    if not sample_created_from_run_config:
        raise ValueError("sample provenance does not establish use of this run config")
    return {
        "gene_count": expected,
        "gene_order_hash": gene_order_hash(genes),
        "real_X_columns_match_gene_order": True,
        "generated_width_matches_gene_order": True,
        "model_gene_names_and_order_match": True,
        "model_width_matches_gene_order": True,
        "sample_created_from_same_run_config": True,
        "sample_contains_embedded_gene_names": generated_names_embedded,
        "generated_gene_order_verification": (
            "embedded names matched exactly" if generated_names_embedded else
            "same-run/config/checkpoint provenance plus exact width; current sample.py does not embed names"
        ),
        "ordering_contract": "sample.py uses ordered data_dir var['gene_name'] for model/sample width",
    }


def build_sampling_anndata(real_subset, generated: np.ndarray, *, celltype_column="celltype"):
    import anndata as ad

    generated = dense_float32(generated)
    if generated.shape[1] != real_subset.n_vars:
        raise ValueError("generated sample gene dimension mismatch")
    real = real_subset.copy()
    real.obs["sampling_origin"] = "Real hematopoietic reference"
    real.obs["sampling_celltype"] = real.obs[celltype_column].astype(str)
    generated_obs = pd.DataFrame({
        "sampling_origin": ["Generated (unconditional)"] * generated.shape[0],
        "sampling_celltype": ["Generated (unconditional)"] * generated.shape[0],
    })
    generated_adata = ad.AnnData(
        X=generated,
        obs=generated_obs,
        var=real.var.copy(),
    )
    generated_adata.var_names = real.var_names.copy()
    if not np.array_equal(real.var_names.to_numpy(), generated_adata.var_names.to_numpy()):
        raise AssertionError("real/generated AnnData variable ordering changed before concat")
    real.obs_names = [f"real_hema_{index}" for index in range(real.n_obs)]
    generated_adata.obs_names = [f"generated_{index}" for index in range(generated_adata.n_obs)]
    # anndata 0.8 supports inner/outer joins only. Exact equality is asserted
    # immediately above, so this inner join cannot reorder or drop variables.
    combined = ad.concat([real, generated_adata], axis=0, join="inner", merge="same")
    if not np.array_equal(real.var_names.to_numpy(), combined.var_names.to_numpy()):
        raise AssertionError("AnnData concat changed the exact variable ordering")
    combined.obs["sampling_origin"] = combined.obs["sampling_origin"].astype("category")
    combined.obs["sampling_celltype"] = combined.obs["sampling_celltype"].astype("category")
    return combined


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
    # X is the exact already-preprocessed training representation. Do not
    # normalize/log/scale again; only dimensionality reduction is performed.
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
        "input_preprocessing": "reuse saved training X; no new normalize/log1p/scale",
    }


def compute_vector_fields(
    model,
    diffusion,
    real_x,
    timesteps: Sequence[int],
    *,
    batch_size: int,
    noise_seed: int,
    device,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict]:
    """Evaluate ODE on real X and CellUnet on exact q_sample(X,t) inputs."""

    matrix = dense_float32(real_x)
    requested = tuple(dict.fromkeys(int(value) for value in timesteps))
    if not requested:
        raise ValueError("at least one diffusion timestep is required")
    if min(requested) < 0 or max(requested) >= diffusion.num_timesteps:
        raise ValueError("diffusion timestep out of range")
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    before = parameter_fingerprint(model)
    was_training = model.training
    model.eval()
    ode_parts: list[np.ndarray] = []
    cell_parts: dict[int, list[np.ndarray]] = {value: [] for value in requested}
    try:
        with torch.no_grad():
            for start in range(0, matrix.shape[0], int(batch_size)):
                clean = torch.from_numpy(matrix[start : start + int(batch_size)]).to(device)
                output = model.ode_model(clean, None)
                if tuple(output.shape) != tuple(clean.shape):
                    raise RuntimeError("ODE field shape mismatch")
                ode_parts.append(output.detach().cpu().float().numpy())
            for timestep in requested:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(noise_seed))
                for start in range(0, matrix.shape[0], int(batch_size)):
                    clean = torch.from_numpy(matrix[start : start + int(batch_size)]).to(device)
                    noise = torch.randn(
                        clean.shape, generator=generator, dtype=torch.float64
                    ).to(device)
                    t = torch.full(
                        (clean.shape[0],), timestep, dtype=torch.long, device=device
                    )
                    noisy = diffusion.q_sample(x_start=clean, t=t, noise=noise)
                    scaled_t = diffusion._scale_timesteps(t).unsqueeze(1)
                    output = model.ml_model(noisy, scaled_t)
                    if tuple(output.shape) != tuple(clean.shape):
                        raise RuntimeError("CellUnet field shape mismatch")
                    cell_parts[timestep].append(output.detach().cpu().float().numpy())
    finally:
        model.train(was_training)
    after = parameter_fingerprint(model)
    if before != after:
        raise RuntimeError("post-hoc visualization mutated model parameters/buffers")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("post-hoc visualization populated parameter gradients")
    ode = np.concatenate(ode_parts, axis=0).astype(np.float32, copy=False)
    cells = {
        timestep: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for timestep, parts in cell_parts.items()
    }
    expected_shape = matrix.shape
    if ode.shape != expected_shape or any(value.shape != expected_shape for value in cells.values()):
        raise RuntimeError("vector field output shape mismatch")
    if not np.isfinite(ode).all() or any(not np.isfinite(value).all() for value in cells.values()):
        raise FloatingPointError("non-finite model field output")
    return ode, cells, {
        "optimizer_constructed": False,
        "optimizer_step_performed": False,
        "model_fingerprint_before": before,
        "model_fingerprint_after": after,
        "noise_seed": int(noise_seed),
        "noise_policy": "same fixed CPU RNG stream regenerated for every timestep",
        "cell_input_policy": "diffusion.q_sample(x_start=real_X, t=t, fixed_noise)",
    }


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DEFAULT_HEMATOPOIETIC_SUPERCLASSES",
    "assert_gene_alignment",
    "build_sampling_anndata",
    "compute_common_umap",
    "compute_vector_fields",
    "dense_float32",
    "file_sha256",
    "gene_names_from_adata",
    "gene_order_hash",
    "resolve_superclass_column",
    "resolve_superclass_selection",
    "select_hematopoietic_subset",
]
