#!/usr/bin/env python3
"""Create an independent erythroid reference without changing the training h5ad."""
from __future__ import annotations
import argparse
import json
import os
import tempfile
from pathlib import Path
import config as C
if __name__ == '__main__':
    C.launch_isolated(__file__)

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse
from common import (annotations, check_environment, digest_ids, environment, finite,
                    ids, require, reproducible, sha256)


def make_reference(source, *, expected_full=C.FULL_CELLS, expected_ery=C.ERYTHROID_CELLS):
    require(source.n_obs == expected_full, f'expected {expected_full} training cells')
    ids(source.obs_names, 'source cell'); ids(source.var_names, 'source gene')
    require('celltype' in source.obs and not source.obs.celltype.isna().any(), 'celltype missing')
    meta = source.uns.get('mouse_gastrulation_preprocessing', {})
    require(meta.get('target_sum') == 1e4 and meta.get('normalization') == 'scanpy.pp.normalize_total'
            and meta.get('log1p') == False, 'source normalization provenance missing/incompatible')
    finite(source.X, 'source X')
    require('spliced' in source.layers, 'spliced: missing layer')
    for start in range(0, source.n_obs, 256):
        delta = sparse.csr_matrix(source.X[start:start + 256]) - sparse.csr_matrix(
            source.layers['spliced'][start:start + 256])
        require(delta.nnz == 0, 'source X does not equal normalized spliced')
    for key in ('spliced', 'unspliced'):
        require(key in source.layers, f'{key}: missing layer')
        mat = source.layers[key]
        finite(mat, key)
        values = mat.data if sparse.issparse(mat) else mat
        require(np.min(values, initial=0) >= 0, f'{key}: negative expression')
        totals = np.asarray(mat.sum(axis=1)).ravel()
        require(np.all((totals == 0) | np.isclose(totals, 1e4, rtol=1e-5, atol=1e-3)),
                f'{key}: expected independent total normalization over all source genes')
    keep = source.obs.celltype.isin(C.ERYTHROID_CELLTYPES).to_numpy()
    require(int(keep.sum()) == expected_ery, f'expected {expected_ery} erythroid cells; got {keep.sum()}')
    # Construct fresh containers so old embeddings, root labels and graphs cannot leak in.
    ref = ad.AnnData(X=source.layers['spliced'][keep].copy(),
                     obs=source.obs.loc[keep, ['celltype', 'stage']].copy(), var=source.var.copy())
    for key in ('spliced', 'unspliced'):
        ref.layers[key] = source.layers[key][keep].copy()
    ref.obs['stage_day'] = annotations(ref)
    ref.uns['benchmark'] = {
        'protocol': C.PROTOCOL, 'source_cell_ids': source.obs_names.to_numpy(dtype=str),
        'source_gene_ids_sha256': digest_ids(source.var_names), 'source_n_cells': source.n_obs,
        'source_normalization': json.dumps(dict(meta), default=lambda x: x.tolist()),
        'seed': C.SEED, 'n_pcs': C.N_PCS, 'n_neighbors': C.N_NEIGHBORS,
        'geometry_hvg_target': C.N_HVG,
        'official_hvg_status': 'not_verified; selected from normalized full-training source',
        'expression_scale': C.EXPRESSION_SCALE,
    }
    return ref


def build_geometry(ref):
    """Scanpy PCA/kNN/UMAP on an evaluation-only log1p/HVG copy; retain all genes."""
    require(min(ref.shape) > C.N_PCS, 'reference too small for fixed 30 PCs')
    reproducible()
    geom = ad.AnnData(X=ref.layers['spliced'].copy(), obs=ref.obs.copy(), var=ref.var.copy())
    sc.pp.log1p(geom)
    # Official recipe uses filter_and_normalize(min_shared_counts=20, n_top_genes=2000).
    # Raw counts were not retained; do not pretend normalized counts are raw counts.
    sc.pp.highly_variable_genes(geom, flavor='seurat', n_top_genes=min(C.N_HVG, geom.n_vars), subset=False)
    selected = geom.var['highly_variable'].to_numpy()
    require(selected.sum() > C.N_PCS, 'too few geometry HVGs for 30 PCs')
    ref.var['benchmark_geometry_hvg'] = selected
    geom = geom[:, selected].copy()
    sc.pp.pca(geom, n_comps=C.N_PCS, svd_solver='arpack', random_state=C.SEED,
              mask_var=None, zero_center=True)
    sc.pp.neighbors(geom, n_neighbors=C.N_NEIGHBORS, n_pcs=C.N_PCS,
                    use_rep='X_pca', method='umap', metric='euclidean', random_state=C.SEED)
    sc.tl.umap(geom, random_state=C.SEED, min_dist=0.5, spread=1.0, n_components=2)
    ref.obsm['X_pca'] = geom.obsm['X_pca'].copy()
    ref.obsm['X_umap'] = geom.obsm['X_umap'].copy()
    ref.uns['neighbors'] = geom.uns['neighbors'].copy()
    ref.uns['umap'] = geom.uns['umap'].copy()
    for key in ('distances', 'connectivities'):
        ref.obsp[key] = geom.obsp[key].copy()
    ref.uns['benchmark']['geometry_hvg_n'] = int(selected.sum())
    ref.uns['benchmark']['geometry_hvg_ids_sha256'] = digest_ids(ref.var_names[selected])
    return ref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=C.SOURCE)
    parser.add_argument('--output', type=Path, default=C.DATA / 'erythroid.h5ad')
    args = parser.parse_args()
    check_environment()
    require(not args.output.exists(), f'refusing to overwrite {args.output}')
    source_hash = sha256(args.input)
    source = ad.read_h5ad(args.input)
    ref = make_reference(source)
    del source
    build_geometry(ref)
    require(sha256(args.input) == source_hash, 'source file changed during preparation')
    ref.uns['benchmark'].update(source_path=str(args.input.resolve()), source_sha256=source_hash,
                                environment_json=json.dumps(environment()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Publish a complete file atomically without replacing an existing destination.
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=args.output.parent) as temp:
        staged = Path(temp) / 'erythroid.h5ad'
        ref.write_h5ad(staged, compression='gzip')
        check = ad.read_h5ad(staged, backed='r')
        try:
            require(check.shape == ref.shape and check.obs_names.equals(ref.obs_names)
                    and check.var_names.equals(ref.var_names), 'saved reference IDs/shape differ')
        finally:
            check.file.close()
        os.link(staged, args.output)
    print(f'Saved {args.output}: {ref.shape}; input SHA256 unchanged: {source_hash}')


if __name__ == '__main__':
    main()
