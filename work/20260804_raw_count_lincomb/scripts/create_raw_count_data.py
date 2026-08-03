#!/usr/bin/env python3
"""Download source data and save 1024 HVGs with raw counts in X."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


SUITE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = SUITE_ROOT / "data"
FILE_ID = "1cProQIHN_auFCowchTcW9-wd1PEPSDjF"
SOURCE_NAME = "Embryonic_raw_data.h5ad"
OUTPUT_NAME = "Embryonic_raw_count.h5ad"


def matrix_audit(matrix: Any, chunk_rows: int = 4096) -> dict:
    n_rows, n_cols = matrix.shape; minimum = np.inf; maximum = -np.inf; noninteger = 0
    for start in range(0, n_rows, chunk_rows):
        value = matrix[start:min(start + chunk_rows, n_rows)]
        data = value.data if sparse.issparse(value) else np.asarray(value).reshape(-1)
        if data.size and not np.isfinite(data).all(): raise FloatingPointError("raw candidate has NaN/Inf")
        if data.size:
            minimum = min(minimum, float(data.min())); maximum = max(maximum, float(data.max()))
            noninteger += int(np.count_nonzero(np.abs(data - np.rint(data)) > 1e-6))
    return {"shape": [n_rows, n_cols], "min": float(minimum if np.isfinite(minimum) else 0),
            "max": float(maximum if np.isfinite(maximum) else 0), "noninteger_count": noninteger,
            "nonnegative": minimum >= 0, "integer_like": noninteger == 0,
            "sparse": bool(sparse.issparse(matrix))}


def rename_columns(adata):
    if adata.obs.shape[1] < 2 or adata.var.shape[1] < 3:
        raise ValueError("Notebook positional rename requires >=2 obs and >=3 var columns")
    obs = list(adata.obs.columns); obs[1] = "celltype"; adata.obs.columns = obs
    var = list(adata.var.columns); var[2] = "gene_name"; adata.var.columns = var
    if adata.var["gene_name"].astype(str).duplicated().any(): raise ValueError("gene_name duplicates")


def build(source_path: Path, output_path: Path, metadata_dir: Path, raw_source: str = "X") -> dict:
    original = ad.read_h5ad(source_path)
    rename_columns(original)
    if raw_source == "X": matrix = original.X
    elif raw_source.startswith("layer:"):
        name = raw_source.split(":", 1)[1]
        if name not in original.layers: raise KeyError(name)
        matrix = original.layers[name]
        original.X = matrix.copy()
    else: raise ValueError("--raw-source must be X or layer:NAME; adata.raw is never inferred")
    before = matrix_audit(original.X)
    if not before["nonnegative"] or not before["integer_like"]:
        layer_audits = {name: matrix_audit(value) for name, value in original.layers.items()}
        raise ValueError(f"selected source is not raw count; layer audits={layer_audits}")
    sc.pp.filter_cells(original, min_genes=10); sc.pp.filter_genes(original, min_cells=3)
    filtered_shape = list(original.shape); filtered_audit = matrix_audit(original.X)
    raw_filtered = original.copy()
    hvg_work = original.copy()
    sc.pp.normalize_total(hvg_work, target_sum=1e4); sc.pp.log1p(hvg_work)
    sc.pp.highly_variable_genes(hvg_work, n_top_genes=1024, subset=False, inplace=True)
    selected = np.flatnonzero(hvg_work.var["highly_variable"].to_numpy())
    if len(selected) != 1024 or len(set(selected.tolist())) != 1024:
        raise RuntimeError(f"expected 1024 unique HVGs, got {len(selected)}")
    final = raw_filtered[:, selected].copy()
    final.var["raw_count_hvg_order"] = np.arange(1, 1025)
    final.uns["raw_count_processing"] = {
        "source_file": str(source_path), "raw_source": raw_source,
        "filter_cells_min_genes": 10, "filter_genes_min_cells": 3,
        "hvg_n_top_genes": 1024, "hvg_flavor": "scanpy_default",
        "hvg_only_normalize_total": 10000.0, "hvg_only_log1p": True,
        "final_X_normalized": False, "final_X_log1p": False, "final_X_scaled": False,
    }
    final_audit = matrix_audit(final.X)
    if not final_audit["nonnegative"] or not final_audit["integer_like"] or final.n_vars != 1024:
        raise RuntimeError(f"final raw-count audit failed: {final_audit}")
    output_path.parent.mkdir(parents=True, exist_ok=True); metadata_dir.mkdir(parents=True, exist_ok=True)
    final.write_h5ad(output_path, compression="gzip")
    restored = ad.read_h5ad(output_path)
    restored_audit = matrix_audit(restored.X)
    genes = final.var["gene_name"].astype(str).tolist()
    restored_genes = restored.var["gene_name"].astype(str).tolist()
    if genes != restored_genes or restored_audit != final_audit: raise RuntimeError("saved data verification mismatch")
    pd.DataFrame({"gene_index": range(1024), "gene_name": genes}).to_csv(metadata_dir / "gene_order.csv", index=False)
    metadata = {"created_at": datetime.now(timezone.utc).astimezone().isoformat(),
                "drive_file_id": FILE_ID, "download_filename": SOURCE_NAME,
                "output_filename": OUTPUT_NAME, "output_path": str(output_path),
                "shape_before_filter": before["shape"], "shape_after_filter": filtered_shape,
                "shape_after_hvg": list(final.shape), "source_audit": before,
                "filtered_audit": filtered_audit, "final_audit": final_audit,
                "raw_source": raw_source, "gene_order_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
                "processing": final.uns["raw_count_processing"]}
    (metadata_dir / "data_creation.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--data-dir", default=str(DATA_ROOT)); parser.add_argument("--raw-source", default="X")
    parser.add_argument("--skip-download", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(); root = Path(args.data_dir).expanduser().resolve()
    source, output = root / SOURCE_NAME, root / OUTPUT_NAME
    if args.dry_run:
        print(json.dumps({"file_id": FILE_ID, "download": str(source), "output": str(output), "raw_source": args.raw_source}, indent=2)); return
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        import gdown
        if source.exists(): raise FileExistsError(f"refusing to overwrite download: {source}")
        gdown.download(url=f"https://drive.google.com/uc?id={FILE_ID}", output=str(source), quiet=False)
    if not source.is_file(): raise FileNotFoundError(source)
    if output.exists(): raise FileExistsError(f"refusing to overwrite final data: {output}")
    print(json.dumps(build(source, output, root / "metadata", args.raw_source), indent=2, default=str))


if __name__ == "__main__": main()
