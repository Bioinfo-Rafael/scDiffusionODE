"""Select full-population HVGs on a log-only temporary matrix; keep linear X."""
import numpy as np
import anndata as ad
import scanpy as sc
from scipy import sparse
from common import *


def select_hvg(a, n):
    require(0 < n <= a.n_vars, 'invalid HVG count')
    geom = ad.AnnData(X=a.X.copy(), var=a.var.copy())
    sc.pp.log1p(geom)
    sc.pp.highly_variable_genes(geom, flavor='seurat', n_top_genes=n, subset=False)
    # Scanpy thresholding can select >n ties. Deterministically rank normalized
    # dispersion, breaking ties by original position, then retain source order.
    score = geom.var['dispersions_norm'].to_numpy()
    valid = np.flatnonzero(np.isfinite(score))
    require(len(valid) >= n, 'too few finite HVG dispersion scores')
    ranked = valid[np.argsort(-score[valid], kind='stable')]
    selected = np.sort(ranked[:n])
    return selected, score


def prepare(c):
    run = paths(c); out = run / 'data/training.h5ad'
    if out.exists():
        a, m = read_data(c)
        require(m['input_sha256'] == sha256(c['input']), 'source input changed')
        print('Reusing validated training data:', out)
        return out
    source_hash = sha256(c['input'])
    a = ad.read_h5ad(c['input'])
    require(a.shape == (c['expected_cells'], c['expected_genes']), 'source shape mismatch')
    ids(a.obs_names); ids(a.var_names)
    require(all(k in a.obs and not a.obs[k].isna().any() for k in ('celltype', 'stage')), 'missing annotations')
    require('gene_name' in a.var, 'gene_name missing (IDs remain var_names)')
    require(int(a.obs.celltype.isin(ERYTHROID).sum()) == c['expected_erythroid'], 'erythroid count mismatch')
    m = a.uns.get('mouse_gastrulation_preprocessing', {})
    require(m.get('target_sum') == 1e4 and m.get('log1p') == False
            and m.get('normalization') == 'scanpy.pp.normalize_total', 'normalization provenance missing')
    for key in ('spliced', 'unspliced'):
        require(key in a.layers, f'missing {key}')
        for start in range(0, a.n_obs, 256):
            x = sparse.csr_matrix(a.layers[key][start:start+256])
            require(np.isfinite(x.data).all() and (x.data >= 0).all(), 'invalid expression')
            totals = np.asarray(x.sum(axis=1)).ravel()
            require(np.all((totals == 0) | np.isclose(totals, 1e4, rtol=1e-5, atol=1e-3)), 'not normalized over all genes')
            if key == 'spliced':
                delta = sparse.csr_matrix(a.X[start:start+256]) - x
                require(not np.any(delta.data), 'X != spliced')
    selected, scores = select_hvg(a, c['n_hvg'])
    var = a.var.iloc[selected].copy()
    var['source_gene_index'] = selected
    var['hvg_dispersion_norm'] = scores[selected]
    # Fresh container avoids copying full layers/legacy embeddings or applying
    # Scanpy's implicit highly_variable mask to later PCA calls.
    var = var.drop(columns=['highly_variable'], errors='ignore')
    result = ad.AnnData(X=a.X[:, selected].copy(), obs=a.obs[['celltype', 'stage']].copy(), var=var)
    result.uns['stage1_data'] = dict(expression_scale=SCALE, preprocess=False,
                                    input_sha256=source_hash, hvg_method='seurat_log1p_copy_exact_rank')
    genes, cells = ids(result.var_names), ids(result.obs_names)
    write_h5ad(result, out)
    result.var.to_csv(run / 'data/gene_mapping.csv', index_label='gene_id')
    (run / 'data/gene_ids.txt').write_text('\n'.join(genes) + '\n')
    require(sha256(c['input']) == source_hash, 'source changed during preparation')
    write_json(run / 'data/metadata.json', dict(input=c['input'], input_sha256=source_hash,
        training_sha256=sha256(out), gene_ids=genes, cell_ids=cells,
        hvg_method='seurat on temporary log1p; rank finite dispersions_norm descending; ties source index; output source order',
        n_cells=result.n_obs, n_genes=result.n_vars, expression_scale=SCALE,
        scanpy_version=sc.__version__))
    return out


def main():
    args = parser(__doc__).parse_args()
    print(prepare(config(args.config)))

if __name__ == '__main__':
    cli(main)
