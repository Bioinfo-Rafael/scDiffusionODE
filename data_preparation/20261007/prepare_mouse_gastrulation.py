#!/usr/bin/env python3
"""Prepare all Mouse Gastrulation cells/genes for scDiffusionODE.

Run with the existing scdiffusion_jupyter environment (Scanpy 1.9.3,
AnnData 0.8.0). Defaults and relative --input/--output paths are resolved
relative to this script, never the current working directory::

    python /path/to/prepare_mouse_gastrulation.py
    python /path/to/prepare_mouse_gastrulation.py --self-test

Only spliced and unspliced are independently normalized to 1e4; X becomes
a copy of normalized spliced. No log transform, smoothing, filtering, or
embedding computation is performed. Use preprocess=False with the shared
cell loader (as work/20260830/scripts/train.py already does). The ODE
optimizer's load_cells reads X directly and uses var['gene_name'] for GRN
matching. Superclass is neither required by those paths nor synthesized.

The full sparse AnnData must fit in RAM. Validation scans bounded sparse
blocks; the preprocessing object is released before reloading the saved
file. Existing outputs are always refused. A temporary file in the output
directory is validated before being published without overwriting a file.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


HERE = Path(__file__).resolve().parent
EXPECTED_SHAPE = (89267, 53801)
LAYERS = ("spliced", "unspliced")
TARGET_SUM = 1e4
BLOCK_ROWS = 256
# Float32 normalization/reduction error: at most 0.101 counts at target 1e4.
SUM_RTOL = 1e-5
SUM_ATOL = 1e-3
PROVENANCE_KEY = "mouse_gastrulation_preprocessing"


def require(condition, message):
    """Keep validation active even when Python is run with -O."""
    if not condition:
        raise ValueError(message)


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else HERE / path).resolve()


def valid_gene_names(values):
    names = pd.Index(values)
    require(not names.isna().any(), "gene_name/var_names contain missing values")
    names = names.astype(str)
    require(not names.str.fullmatch(r"\s*").any(), "gene_name/var_names contain blank names")
    require(names.is_unique, "gene_name/var_names must be unique; no names will be renamed")
    return names


def matrix_stats(matrix, label):
    """Validate all stored values; only row-sum vectors become dense."""
    require(sparse.issparse(matrix), f"{label}: expected a sparse matrix")
    matrix = matrix.tocsr(copy=False)
    totals = np.empty(matrix.shape[0], dtype=np.float64)
    minimum, maximum, nonzero = np.inf, -np.inf, 0
    for start in range(0, matrix.shape[0], BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, matrix.shape[0])
        block = matrix[start:stop]
        values = block.data
        require(np.isfinite(values).all(), f"{label}: NaN or Inf at rows {start}:{stop}")
        require(not (values < 0).any(), f"{label}: negative values at rows {start}:{stop}")
        if values.size:
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
            nonzero += int(np.count_nonzero(values))
        # Cast/reduce only this block, not the entire multi-GB sparse matrix.
        totals[start:stop] = np.asarray(block.sum(axis=1, dtype=np.float64)).ravel()
    require(np.isfinite(totals).all(), f"{label}: nonfinite library sizes")
    elements = matrix.shape[0] * matrix.shape[1]
    if nonzero < elements:
        minimum = min(minimum, 0.0)
        maximum = max(maximum, 0.0)
    stats = {
        "shape": list(matrix.shape), "dtype": str(matrix.dtype),
        "storage": matrix.format, "stored_entries": int(matrix.nnz),
        "nonzero_entries": nonzero, "min": minimum, "max": maximum,
        "mean": float(totals.sum() / elements),
        "library_min": float(totals.min()), "library_max": float(totals.max()),
        "library_mean": float(totals.mean()), "library_median": float(np.median(totals)),
        "zero_library_cells": int(np.count_nonzero(totals == 0)),
    }
    print(f"{label}: {json.dumps(stats, sort_keys=True)}", flush=True)
    return totals, stats


def equal_sparse(left, right, label):
    require(sparse.issparse(left) and sparse.issparse(right), f"{label}: sparse required")
    require(left.shape == right.shape, f"{label}: shapes differ")
    for start in range(0, left.shape[0], BLOCK_ROWS):
        difference = left[start:start + BLOCK_ROWS] - right[start:start + BLOCK_ROWS]
        require(not np.any(difference.data != 0), f"{label}: mismatch near row {start}")


def check_library(totals, zero_before, label):
    require(np.array_equal(totals == 0, zero_before), f"{label}: zero-library cells changed")
    positive = totals[~zero_before]
    require(np.allclose(positive, TARGET_SUM, rtol=SUM_RTOL, atol=SUM_ATOL),
            f"{label}: nonzero library sizes do not sum to {TARGET_SUM}")
    error = float(np.max(np.abs(positive - TARGET_SUM))) if positive.size else 0.0
    print(f"PASS {label}: zero cells={int(zero_before.sum())}; "
          f"maximum nonzero row-sum error={error:.8g}", flush=True)


def validate_saved(adata, expected_shape, obs_before, var_before, obsm_before, zeros):
    require(adata.shape == expected_shape, f"Saved shape {adata.shape} != {expected_shape}")
    require(adata.obs_names.equals(obs_before.index), "Saved obs_names/order changed")
    require(adata.var_names.equals(var_before.index), "Saved var_names/order changed")
    require("gene_name" in adata.var, "Saved var lacks gene_name")
    genes = valid_gene_names(adata.var["gene_name"])
    require(genes.equals(adata.var_names.astype(str)), "Saved gene_name differs from var_names")
    # AnnData 0.8 may encode repeated strings as categorical on write. Compare
    # their values exactly while permitting that lossless storage conversion.
    pd.testing.assert_frame_equal(adata.obs, obs_before, check_dtype=False,
                                  check_categorical=False, check_exact=True)
    pd.testing.assert_frame_equal(adata.var.loc[:, var_before.columns], var_before,
                                  check_dtype=False, check_categorical=False, check_exact=True)
    require(set(adata.var.columns) == set(var_before.columns) | {"gene_name"},
            "Unexpected var columns after saving")
    require(set(adata.obsm) == set(obsm_before), "Saved obsm keys changed")
    for key, original in obsm_before.items():
        if sparse.issparse(original):
            equal_sparse(adata.obsm[key], original, f"obsm/{key}")
        elif isinstance(original, pd.DataFrame):
            pd.testing.assert_frame_equal(adata.obsm[key], original, check_exact=True)
        else:
            np.testing.assert_array_equal(adata.obsm[key], original)
    print("PASS shape, cell/gene order, gene_name, obs (including celltype/stage), "
          "original var metadata and saved embeddings", flush=True)
    for name in ("X",) + LAYERS:
        matrix = adata.X if name == "X" else adata.layers[name]
        require(sparse.issparse(matrix), f"Saved {name} is not sparse")
        require(matrix.dtype == np.float32, f"Saved {name} is not float32")
        totals, _ = matrix_stats(matrix, f"reloaded/{name}")
        if name in LAYERS:
            check_library(totals, zeros[name], name)
    equal_sparse(adata.X, adata.layers["spliced"], "X == spliced")
    print("PASS X == spliced (all elements); X/spliced/unspliced are sparse float32, "
          "finite and nonnegative", flush=True)


def prepare(input_path, output_path, *, expected_shape=EXPECTED_SHAPE):
    """File pipeline; expected_shape can be changed only through the Python API/tests."""
    input_path, output_path = resolve_path(input_path), resolve_path(output_path)
    require(input_path != output_path, "Input and output must be different files")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_path}; choose a new --output")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    print(f"Input: {input_path}\nOutput: {output_path}", flush=True)
    adata = sc.read_h5ad(input_path)
    require(adata.shape == expected_shape, f"Input shape {adata.shape} != {expected_shape}")
    require(all(key in adata.obs for key in ("celltype", "stage")),
            "Input obs must already contain celltype and stage")
    require(all(key in adata.layers for key in LAYERS), "Input needs spliced and unspliced")
    require(PROVENANCE_KEY not in adata.uns, "Input was already processed by this script")
    genes = valid_gene_names(adata.var_names)
    if "gene_name" in adata.var:
        require(valid_gene_names(adata.var["gene_name"]).equals(genes),
                "Existing gene_name conflicts with var_names; metadata will not be overwritten")
    obs_before, var_before = adata.obs.copy(), adata.var.copy()
    # Keep only small metadata references, not another copy of expression matrices.
    obsm_before = dict(adata.obsm)
    adata.var["gene_name"] = genes.to_numpy()

    before, after, zeros = {}, {}, {}
    _, before["X"] = matrix_stats(adata.X, "before/X")
    # Release the redundant raw X before making the required normalized X copy.
    # AnnData 0.8 supports X=None while retaining its obs/var dimensions.
    adata.X = None
    for layer in LAYERS:
        require(sparse.issparse(adata.layers[layer]), f"{layer}: expected sparse counts")
        adata.layers[layer] = adata.layers[layer].tocsr(copy=False)
        totals, before[layer] = matrix_stats(adata.layers[layer], f"before/{layer}")
        zeros[layer] = totals == 0
        adata.layers[layer] = adata.layers[layer].astype(np.float32, copy=False)
        # Separate calls are essential: each layer supplies its own library sizes.
        sc.pp.normalize_total(adata, target_sum=TARGET_SUM, layer=layer, inplace=True)
        totals, after[layer] = matrix_stats(adata.layers[layer], f"after/{layer}")
        check_library(totals, zeros[layer], layer)
    adata.X = adata.layers["spliced"].copy()
    after["X"] = after["spliced"].copy()
    print(f"after/X: {json.dumps(after['X'], sort_keys=True)}", flush=True)
    adata.uns[PROVENANCE_KEY] = {
        "script": Path(__file__).name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": input_path.name, "input_path": str(input_path),
        "output_file": output_path.name, "output_path": str(output_path),
        "normalization": "scanpy.pp.normalize_total",
        "normalized_layers": list(LAYERS), "target_sum": TARGET_SUM,
        "library_sizes": "computed independently for each layer across all genes",
        "X_source": "copy of independently normalized layers['spliced']",
        "gene_name_source": "var_names (gene symbols, no ID conversion)",
        "log1p": False, "scale": False, "z_score": False,
        "smoothing": False, "moments": False, "gene_filtering": False,
        "cell_filtering": False, "hvg_selection": False,
        "zero_library_policy": "retain cells as all-zero rows independently per layer",
        "output_dtype": "float32", "additional_raw_count_layers": False,
        "preserved_obsm_keys": list(obsm_before),
        "embedding_provenance": (
            "All original obsm (including PCA/UMAP) and obs coordinates are retained. "
            "They were not recomputed from the normalized output X. Their original "
            "input matrix and upstream preprocessing are unknown."
        ),
        "statistics_before": before, "statistics_after": after,
        "row_sum_validation_rtol": SUM_RTOL, "row_sum_validation_atol": SUM_ATOL,
        "versions": {key: version(key) for key in ("scanpy", "anndata", "numpy", "scipy", "pandas", "h5py")},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix=f".{output_path.stem}.", suffix=".h5ad",
                                    dir=output_path.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        print(f"Writing temporary H5AD: {temporary}", flush=True)
        adata.write_h5ad(temporary)
        del adata
        gc.collect()
        print("Reloading saved H5AD for validation", flush=True)
        saved = sc.read_h5ad(temporary)
        validate_saved(saved, expected_shape, obs_before, var_before, obsm_before, zeros)
        del saved
        gc.collect()
        # Same-directory hard link publishes the validated inode atomically and
        # fails if another process created the output in the meantime.
        os.link(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    size = output_path.stat().st_size
    print(f"PASS all validation checks\nOutput: {output_path}\n"
          f"File size: {size:,} bytes ({size / 1024**3:.3f} GiB)", flush=True)
    return output_path


def self_test():
    """Exercise the same file pipeline with independent zero rows and known values."""
    from anndata import AnnData
    from unittest.mock import patch

    spliced = sparse.csr_matrix([[1, 3, 0], [0, 0, 0], [0, 2, 2], [0, 0, 0]], dtype=np.float32)
    unspliced = sparse.csc_matrix([[9, 0, 1], [1, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    obs = pd.DataFrame({
        "celltype": pd.Categorical(["B", "A", "B", "A"], categories=["B", "A"], ordered=True),
        "stage": pd.Categorical(["E7.5", "E6.5", "E7.5", "E6.5"]),
        "sample": [3, 1, 4, 2], "umapX": [0.1, 0.2, np.nan, 0.4],
    }, index=["cell_z", "cell_a", "cell_y", "cell_b"])
    var = pd.DataFrame({"Accession": ["ENSMUSG3", "ENSMUSG1", "ENSMUSG2"]},
                       index=["Sox17", "Xkr4", "Gm37180"])
    adata = AnnData(X=spliced.copy(), obs=obs, var=var, dtype=np.float32)
    adata.layers["spliced"], adata.layers["unspliced"] = spliced, unspliced
    adata.obsm["X_pca"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    adata.obsm["X_umap"] = adata.obsm["X_pca"] + 10
    adata.varm["original"] = np.arange(6).reshape(3, 2)
    adata.obsp["original"] = sparse.eye(4, format="csr", dtype=np.float32)
    adata.uns["original"] = {"note": "preserve this metadata"}
    wanted = {
        "spliced": sparse.csr_matrix([[2500, 7500, 0], [0, 0, 0], [0, 5000, 5000], [0, 0, 0]], dtype=np.float32),
        "unspliced": sparse.csr_matrix([[9000, 0, 1000], [5000, 5000, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory(prefix="mouse_gastrulation_test_") as directory:
        source, target = Path(directory) / "input.h5ad", Path(directory) / "output.h5ad"
        adata.write_h5ad(source)
        original_bytes = source.read_bytes()
        # Fail on any sparse-to-dense conversion throughout preparation/reload.
        with patch.object(sparse.csr_matrix, "toarray", side_effect=AssertionError("dense conversion")), \
             patch.object(sparse.csc_matrix, "toarray", side_effect=AssertionError("dense conversion")):
            prepare(source, target, expected_shape=(4, 3))
        result = sc.read_h5ad(target)
        for layer in LAYERS:
            difference = result.layers[layer] - wanted[layer]
            require(np.allclose(difference.data, 0, rtol=0, atol=SUM_ATOL),
                    f"Synthetic {layer}: independently normalized values differ")
        np.testing.assert_array_equal(result.varm["original"], adata.varm["original"])
        equal_sparse(result.obsp["original"], adata.obsp["original"], "preserved obsp")
        require(result.uns["original"] == adata.uns["original"], "Original uns changed")
        require(set(result.layers) == set(LAYERS), "Unexpected added layers")
        require("Superclass" not in result.obs and result.raw is None, "Invented labels/raw")
        history = result.uns[PROVENANCE_KEY]
        require(history["target_sum"] == TARGET_SUM and not history["log1p"]
                and not history["smoothing"], "Missing preprocessing provenance")
        require(source.read_bytes() == original_bytes, "Input file was modified")
        output_bytes = target.read_bytes()
        try:
            prepare(source, target, expected_shape=(4, 3))
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing output was not refused")
        require(target.read_bytes() == output_bytes, "Existing output changed")
    print("SELF-TEST PASS: independent normalization, different zero-row masks, "
          "CSR/CSC and float/integer inputs, sparse-only processing, metadata/provenance, "
          "save/reload, input preservation and overwrite refusal", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/gastrulation_full.h5ad",
                        help="Input h5ad; relative paths are based on the script directory")
    parser.add_argument("--output", default="data/MouseGastrulation.h5ad",
                        help="New output h5ad; relative paths are based on the script directory")
    parser.add_argument("--self-test", action="store_true", help="Test synthetic data in a temporary directory only")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
    else:
        prepare(args.input, args.output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
