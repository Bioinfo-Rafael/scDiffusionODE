"""Replot saved joint UMAPs using Harmony on Real/Generated PCA batches."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import anndata as ad
import harmonypy as hm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc


def replot(path, seed=1234):
    if importlib.metadata.version('harmonypy') != '0.0.10':
        raise ValueError('Use harmonypy==0.0.10 for reproducible Harmony settings')
    path=Path(path);out=path.parent
    a=ad.read_h5ad(path)
    batch='sampling_origin'
    if batch not in a.obs or a.obs[batch].nunique()!=2:
        raise ValueError(f'{path}: expected Real and Generated batches')
    if 'X_pca' not in a.obsm:
        raise ValueError(f'{path}: original joint PCA missing')
    pca=np.asarray(a.obsm['X_pca'])
    clusters=max(2,min(100,round(a.n_obs/30)))
    settings=dict(theta=2.,lamb=1.,sigma=.1,nclust=clusters,
                  max_iter_harmony=10,max_iter_kmeans=20,
                  epsilon_cluster=1e-5,epsilon_harmony=1e-4,
                  block_size=.05,tau=0,random_state=seed)
    result=hm.run_harmony(pca,a.obs,batch,**settings)
    corrected=np.asarray(result.Z_corr).T
    if corrected.shape!=pca.shape or not np.isfinite(corrected).all():
        raise ValueError('Invalid Harmony PCA')
    a.obsm['X_umap_uncorrected']=a.obsm['X_umap'].copy()
    a.obsm['X_pca_harmony']=corrected
    pcs=min(40,pca.shape[1]);neighbors=min(15,a.n_obs-1)
    sc.pp.neighbors(a,use_rep='X_pca_harmony',n_pcs=pcs,
                    n_neighbors=neighbors,random_state=seed)
    sc.tl.umap(a,random_state=seed,min_dist=.5,spread=1.,n_components=2)
    stamp=dict(batch_key=batch,batches=a.obs[batch].value_counts().to_dict(),
               harmony=settings,source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
               n_pcs=pcs,n_neighbors=neighbors,min_dist=.5,spread=1.,seed=seed,
               versions={p:importlib.metadata.version(p) for p in ('harmonypy','scanpy','anndata')},
               interpretation='Real/Generated difference is corrected for visualization; not evidence of generative fidelity')
    a.uns['harmony_visualization']=stamp
    a.write_h5ad(out/'joint_umap_harmony.h5ad')
    (out/'umap_harmony_metadata.json').write_text(json.dumps(stamp,indent=2)+'\n')
    xy=np.asarray(a.obsm['X_umap']);generated=a.obs[batch].astype(str).str.startswith('Generated').to_numpy()
    fig,ax=plt.subplots(figsize=(8,7))
    ax.scatter(*xy[~generated].T,s=3,alpha=.4,label='Real')
    ax.scatter(*xy[generated].T,s=4,alpha=.6,label='Generated')
    ax.legend();ax.set(title='Joint UMAP + Harmony (batch: Real / Generated)',xlabel='UMAP1',ylabel='UMAP2')
    fig.savefig(out/'joint_umap_harmony.png',dpi=160);plt.close(fig)
    print(out/'joint_umap_harmony.png',flush=True)
    return a


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--condition-set',choices=['pilot','ode_sgd','model_expansion'],default='pilot')
    args=p.parse_args()
    # Require the complete pilot before changing any plots.
    suite=Path(__file__).resolve().parents[1]
    expected={f.stem for f in (suite/'configs'/args.condition_set).glob('*.json')}
    files=[args.run_root/name/'figures/generative/joint_umap.h5ad' for name in sorted(expected)]
    files=[f for f in files if f.is_file()]
    found={f.parents[2].name for f in files}
    if expected-found:raise FileNotFoundError(f'Missing {args.condition_set} joint UMAPs: {sorted(expected-found)}')
    for path in files:replot(path)
    from combine_harmony_pngs import combine
    combine(args.run_root,args.condition_set)
