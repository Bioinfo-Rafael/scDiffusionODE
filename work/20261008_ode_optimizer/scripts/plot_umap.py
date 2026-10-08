#!/usr/bin/env python3
"""Fit one shared PCA/neighbors/UMAP to Real + all 24 generated states."""
import argparse
import numpy as np
from common import *


def draw_panel(ax, real, generated, title, limits):
    ax.scatter(real[:,0],real[:,1],s=3,c='#a0a0a0',alpha=.45,label='Real',rasterized=True)
    ax.scatter(generated[:,0],generated[:,1],s=3,c='#d65f28',alpha=.5,label='Generated',rasterized=True)
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
    folder=inside(folder); archive=np.load(folder/'coordinates.npz',allow_pickle=False)
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
    settings={k:c[k] for k in ('umap_real_cells','pca_components','neighbors','neighbor_pcs','seed')}
    if (folder/'coordinates.npz').exists() and not a.refit:
        old=read_json(folder/'embedding.json')
        if old['sources']!=source or old['settings']!=settings:
            raise ValueError('UMAP inputs changed: use --refit')
        render_all(folder); return
    x,genes,_=load_cells(c)
    indices=np.sort(np.random.default_rng(c['seed']).choice(len(x),min(c['umap_real_cells'],len(x)),replace=False))
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
                generated[f'{run.name}__{s}']=arr
                if s==0:
                    if initial is not None and not np.array_equal(arr,initial): raise ValueError('Initial noise differs')
                    initial=arr
    real,coords=fit_shared(x[indices],generated,c)
    all_xy=np.concatenate([real]+list(coords.values()))
    lo=all_xy.min(0); hi=all_xy.max(0); pad=np.maximum((hi-lo)*.04,.1)
    limits=np.array([lo[0]-pad[0],hi[0]+pad[0],lo[1]-pad[1],hi[1]+pad[1]])
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'coordinates.npz',real=real,real_indices=indices,
                        generated_indices=np.arange(c['num_samples']),limits=limits,**coords)
    write_json(folder/'embedding.json',dict(sources=source,settings=settings,fit_count=1,
        coordinate_system='one joint fit: Real + 4 conditions x 6 states',
        preprocessing='saved training X; no normalization/log1p/scaling',
        coordinate_sha256=sha256(folder/'coordinates.npz'),sampling_protocol=reference_sampling))
    render_all(folder)

if __name__=='__main__': main()
