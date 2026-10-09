"""UMAP comparisons and explicitly non-biological reconstruction fields."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse

IMPORTS = importlib.import_module('work.20260911.src.source_imports')
CORE = IMPORTS.umap_core()
PLOT = IMPORTS.import_file('_1009_field_plotting', ROOT / 'work/20260830/hematopoietic_viz/plotting.py')


def geometry(a, c):
    SOURCE.seed_all(c['training']['seed'])
    return CORE.compute_common_umap(a, pca_components=c['pca_components'], neighbors=c['neighbors'],
        neighbor_pcs=c['neighbor_pcs'], seed=c['training']['seed'])


def scatter(a, key, title, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sc.pl.umap(a, color=key, title=title, ax=ax, show=False, legend_loc='right margin')
    fig.savefig(path, dpi=160, bbox_inches='tight'); plt.close(fig)


def visualize(c, scope):
    run = paths(c); a, meta = read_data(c)
    out = run / 'figures' / scope
    if scope == 'samples':
        sample_path = run / 'samples/generated.h5ad'
        gen = ad.read_h5ad(sample_path)
        smeta = json.loads((run / 'samples/metadata.json').read_text())
        require(sha256(sample_path) == smeta['output_sha256'], 'samples changed')
        require(ids(gen.var_names) == ids(a.var_names), 'sample gene order mismatch')
        # Synthetic IDs belong only to generated cells; never assign celltype.
        require(not set(gen.obs_names) & set(a.obs_names), 'generated cell ID collision')
        a.obs['origin'] = 'real'; gen.obs['origin'] = 'generated'
        gen.X = sparse.csr_matrix(gen.X)
        joint = ad.concat([a, gen], join='outer', merge='same')
        stamp = dict(scope=scope, source_hash=meta['training_sha256'],
                     generated_sha256=sha256(sample_path), n_real=a.n_obs, n_generated=gen.n_obs)
    else:
        field_path = run / 'predictions/cellunet_field.h5ad'
        fmeta = json.loads((run / 'predictions/metadata.json').read_text())
        require(sha256(field_path) == fmeta['output_sha256'], 'field file changed')
        require(fmeta['training_sha256'] == meta['training_sha256'], 'field training source mismatch')
        field = ad.read_h5ad(field_path)
        require(ids(field.obs_names) == ids(a.obs_names), 'field cell IDs/order mismatch')
        require(ids(field.var_names) == ids(a.var_names), 'field gene IDs/order mismatch')
        if scope == 'erythroid':
            keep = a.obs.celltype.isin(ERYTHROID).to_numpy()
            require(int(keep.sum()) == c['expected_erythroid'], 'erythroid count mismatch')
        else:
            keep = np.ones(a.n_obs, dtype=bool)
        a = a[keep].copy()
        a.layers['reconstruction_displacement'] = np.asarray(field.layers['reconstruction_displacement'][keep]).copy()
        del field
        SOURCE.finite('field', a.layers['reconstruction_displacement'])
        stamp = dict(scope=scope, source_hash=meta['training_sha256'], field_sha256=sha256(field_path),
                     n_cells=a.n_obs, n_genes=a.n_vars, biological_velocity=False,
                     subsampling=False, velocity_graph_approx=False, n_jobs=c['n_jobs'],
                     neighbor_search='inherited Scanpy auto/UMAP; may use NN-descent on large data',
                     cell_ids_sha256=digest_ids(a.obs_names), field_timestep=fmeta['timestep'])
    stamp['geometry_settings'] = {k: c[k] for k in ('pca_components', 'neighbors', 'neighbor_pcs')}
    marker = out / 'completed.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        require(saved['request'] == stamp, 'existing visualization provenance mismatch')
        require(all((out / name).exists() and sha256(out / name) == value
                    for name, value in saved['files'].items()), 'visualization artifact missing/changed')
        return out
    out.mkdir(parents=True, exist_ok=True)
    if scope == 'samples':
        details = geometry(joint, c)
        scatter(joint, 'origin', 'Real vs generated — joint UMAP', out / 'real_vs_generated.png')
        real = joint[joint.obs.origin == 'real'].copy()
        scatter(real, 'celltype', 'Real celltypes — joint UMAP coordinates', out / 'real_celltype.png')
        joint.write_h5ad(out / 'embedding.h5ad')
    else:
        # Recomputed independently for full and erythroid; no benchmark geometry.
        details = geometry(a, c)
        scatter(a, 'celltype', f'{scope}: celltype', out / 'celltype.png')
        scatter(a, 'stage', f'{scope}: stage', out / 'stage.png')
        a.layers['spliced'] = a.X.copy()
        a.var['reconstruction_displacement_genes'] = True
        PLOT.compute_velocity_embeddings(a, ['reconstruction_displacement'], n_jobs=c['n_jobs'])
        SOURCE.finite('projected field', a.obsm['reconstruction_displacement_umap'])
        PLOT.plot_velocity_triplet(a, out, vkey='reconstruction_displacement',
            title=f'{scope}: reconstruction displacement (NOT RNA velocity), t={fmeta["timestep"]}',
            filenames=['field_stream.png', 'field_arrow.png', 'field_grid.png'])
        a.uns['field_visualization'] = dict(definition='x_start_prediction_minus_clean_X',
                                           biological_velocity=False, benchmark_geometry=False)
        a.write_h5ad(out / 'embedding.h5ad')
    files = {p.name: sha256(p) for p in out.iterdir() if p.suffix in ('.png', '.h5ad')}
    write_json(marker, dict(request=stamp, geometry=details, files=files,
        scanpy_version=sc.__version__, scvelo_version=importlib.import_module('scvelo').__version__))
    return out


def main():
    p = parser(__doc__); p.add_argument('--scope', required=True, choices=['samples', 'all', 'erythroid'])
    args = p.parse_args(); print(visualize(config(args.config), args.scope))

if __name__ == '__main__':
    cli(main)
