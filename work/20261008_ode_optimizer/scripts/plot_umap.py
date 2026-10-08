#!/usr/bin/env python3
"""Fit one shared UMAP to ALL Erythropoietic Real cells + 24 saved states."""
import argparse
import numpy as np
from common import *


# Same selection as 20260911/src/umap_adapter.py and
# 20260913_2step/common.py::load_real(erythropoietic=True).
# Kept local to avoid their process-global `analysis` import collisions.
REFERENCE_SUPERCLASS = "Erythropoietic"
SELECTION_VERSION = "all_erythropoietic_v1"


def select_real_reference(adata):
    columns = list(adata.obs.columns)
    column = next((name for name in ("Superclass", "superclass") if name in columns), None)
    if column is None:
        matches = [name for name in columns if str(name).lower() == "superclass"]
        if len(matches) != 1:
            raise KeyError("No unambiguous Superclass/superclass column in real data")
        column = matches[0]
    if "celltype" not in columns:
        raise KeyError("Missing celltype column in real data")
    available = sorted(adata.obs[column].dropna().astype(str).unique().tolist())
    mask = adata.obs[column].astype(str).eq(REFERENCE_SUPERCLASS).to_numpy()
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError(f"No {REFERENCE_SUPERCLASS} cells in {column}; available={available}")
    genes = list(map(str, adata.var["gene_name"]))
    if len(genes) != adata.n_vars or len(set(genes)) != len(genes):
        raise ValueError("gene_name must be unique and aligned to X")
    # Subset BEFORE densifying; use every selected cell, no 3000-cell cap.
    source = adata.X[indices]
    x = source.toarray() if hasattr(source, "toarray") else np.asarray(source)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Nonfinite or invalid selected real X")
    obs = adata.obs.iloc[indices]
    metadata = dict(selection_version=SELECTION_VERSION,
        selected_superclass_column=column, selected_superclasses=[REFERENCE_SUPERCLASS],
        available_superclasses=available, total_dataset_cell_count=int(adata.n_obs),
        selected_cell_count=len(indices), subsampling=False,
        celltype_counts={str(k):int(v) for k,v in obs["celltype"].astype(str).value_counts().items()})
    return x, genes, indices, np.asarray(obs.index, dtype=str), metadata


def load_real_reference(c):
    import scanpy as sc
    adata = sc.read_h5ad(c["data_dir"])
    return select_real_reference(adata)


def draw_panel(ax, real, generated, title, limits):
    ax.scatter(real[:,0],real[:,1],s=3,c='#a0a0a0',alpha=.45,label=f'Real Erythropoietic (n={len(real):,})',linewidths=0,rasterized=True)
    ax.scatter(generated[:,0],generated[:,1],s=3,c='#d65f28',alpha=.5,label=f'Generated (n={len(generated):,})',linewidths=0,rasterized=True)
    ax.set(title=title,xlabel='UMAP 1',ylabel='UMAP 2',xlim=limits[:2],ylim=limits[2:])


def fit_shared(real, generated, c):
    # Same preprocessing and settings as 20260830/hematopoietic_viz/core.py
    # compute_common_umap. Copy only this small routine to avoid legacy absolute imports.
    import anndata as ad
    import scanpy as sc
    names=list(generated)
    arrays=[real]+[generated[n] for n in names]
    combined=ad.AnnData(np.concatenate(arrays).astype(np.float32,copy=False))
    components=min(c['pca_components'],combined.n_obs-1,combined.n_vars-1)
    sc.tl.pca(combined,svd_solver='arpack',n_comps=components,random_state=c['seed'])
    sc.pp.neighbors(combined,n_neighbors=min(c['neighbors'],combined.n_obs-1),
                    n_pcs=min(c['neighbor_pcs'],components),random_state=c['seed'])
    sc.tl.umap(combined,random_state=c['seed'])
    xy=np.asarray(combined.obsm['X_umap'])
    if xy.shape!=(sum(map(len,arrays)),2) or not np.isfinite(xy).all():
        raise ValueError('Invalid shared UMAP coordinates')
    parts=np.split(xy,np.cumsum([len(a) for a in arrays])[:-1])
    return parts[0],dict(zip(names,parts[1:]))


def render_all(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=inside(folder)
    meta=read_json(folder/'embedding.json')
    if meta.get('settings',{}).get('selection_version')!=SELECTION_VERSION:
        raise ValueError('Legacy all-cell embedding: rerun with --refit first')
    if meta['coordinate_sha256']!=sha256(folder/'coordinates.npz'):
        raise ValueError('Coordinate hash differs from embedding metadata')
    archive=np.load(folder/'coordinates.npz',allow_pickle=False)
    real=archive['real']; limits=archive['limits']
    for name,value in CONDITIONS.items():
        for step in STEPS:
            fig,ax=plt.subplots(figsize=(6,5))
            draw_panel(ax,real,archive[f'{name}__{step}'],f'{name} (lambda={value:g}) | sampling s={step}',limits)
            ax.legend(markerscale=3); fig.tight_layout()
            fig.savefig(folder/f'umap_{name}_s{step:04d}.png',dpi=150); plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--campaign',required=True)
    p.add_argument('--refit',action='store_true'); p.add_argument('--redraw-only',action='store_true')
    a=p.parse_args(argv); campaign=inside(a.campaign); folder=campaign/'umap'
    if a.redraw_only:
        render_all(folder); return
    runs=campaign_runs(campaign); c=read_json(runs[0]/'exp_config.json')
    source={str(r/'samples'/f'step_{s:04d}.npz'):sha256(r/'samples'/f'step_{s:04d}.npz') for r in runs for s in STEPS}
    source['real_data_sha256']=sha256(c['data_dir'])
    training_steps={legacy('analysis.gradients').checkpoint_training_step(checkpoint(r)) for r in runs}
    if len(training_steps)!=1:
        raise ValueError('UMAP comparison requires equal checkpoint training steps')
    for run in runs:
        info=read_json(run/'samples/sampling.json')
        if info['checkpoint_sha256']!=sha256(checkpoint(run)):
            raise ValueError('Saved samples do not match the current checkpoint; resample with --force')
    settings={k:c[k] for k in ('pca_components','neighbors','neighbor_pcs','seed')}
    settings.update(selection_version=SELECTION_VERSION,real_superclasses=[REFERENCE_SUPERCLASS],
                    real_cell_limit=None,generated_selection='all_saved_cells')
    if (folder/'coordinates.npz').exists() and not a.refit:
        old=read_json(folder/'embedding.json')
        if old['sources']!=source or old['settings']!=settings:
            raise ValueError('UMAP inputs changed: use --refit')
        render_all(folder); return
    x,genes,indices,real_names,selection=load_real_reference(c)
    print(f'Real selection: {selection}',flush=True)
    generated={}; initial=None; reference_sampling=None
    for run in runs:
        sampling=read_json(run/'samples/sampling.json')
        protocol={k:sampling[k] for k in ('seed','num_samples','sample_batch_size','rng_protocol','gene_names','device')}
        if reference_sampling is None: reference_sampling=protocol
        if protocol!=reference_sampling or sampling['gene_names']!=genes:
            raise ValueError('Sampling conditions/gene order differ')
        for s in STEPS:
            with np.load(run/'samples'/f'step_{s:04d}.npz') as data:
                arr=data['cell_gen']; idx=data['cell_index']
                if not np.array_equal(idx,np.arange(len(arr))): raise ValueError('Cell indices differ')
                if arr.shape!=(sampling['num_samples'],len(genes)) or not np.isfinite(arr).all():
                    raise ValueError(f'Invalid sample shape/values: {run.name}, s={s}')
                generated[f'{run.name}__{s}']=arr
                if s==0:
                    if initial is not None and not np.array_equal(arr,initial): raise ValueError('Initial noise differs')
                    initial=arr
    print(f'Joint fit: {len(x)} Real Erythropoietic cells + {sum(map(len,generated.values()))} generated states',flush=True)
    real,coords=fit_shared(x,generated,c)
    all_xy=np.concatenate([real]+list(coords.values()))
    lo=all_xy.min(0); hi=all_xy.max(0); pad=np.maximum((hi-lo)*.04,.1)
    limits=np.array([lo[0]-pad[0],hi[0]+pad[0],lo[1]-pad[1],hi[1]+pad[1]])
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'coordinates.npz',real=real,real_indices=indices,real_cell_names=real_names,
                        generated_indices=np.arange(c['num_samples']),limits=limits,**coords)
    write_json(folder/'embedding.json',dict(sources=source,settings=settings,fit_count=1,
        coordinate_system='one joint fit: ALL Real Erythropoietic + 4 conditions x 6 states',
        real_selection=selection,generated_counts={key:len(value) for key,value in generated.items()},
        preprocessing='saved training X; no normalization/log1p/scaling',
        coordinate_sha256=sha256(folder/'coordinates.npz'),sampling_protocol=reference_sampling))
    render_all(folder)

if __name__=='__main__': main()
