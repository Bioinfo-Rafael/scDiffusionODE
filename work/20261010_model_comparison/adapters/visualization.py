"""Imported generative diagnostics plus plots of existing benchmark artifacts."""
from __future__ import annotations
from ..common import *


def training_plots(run, model=None):
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);out=confined(run/'figures/training');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in sorted((run/'logs').glob('loss_*.jsonl')):
        if '.rolled_back.' in path.name:continue
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    for path in sorted((run/'logs').glob('loss_*.csv')):
        rows.extend(pd.read_csv(path).to_dict('records'))
    for path in sorted((run/'logs').glob('baseline_diagnostics_*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    if rows:
        frame=pd.DataFrame(rows).sort_values('step').drop_duplicates('step',keep='last')
        frame.to_csv(out/'history.csv',index=False)
        numeric=[col for col in frame.select_dtypes(include='number') if col!='step']
        for start in range(0,len(numeric),6):
            fig,axes=plt.subplots(3,2,figsize=(12,9),layout='constrained')
            for ax,col in zip(axes.flat,numeric[start:start+6]):
                ax.plot(frame.step,frame[col]);ax.set(title=col,xlabel='optimizer update')
            fig.savefig(out/f'components_{start:02}.png',dpi=140);plt.close(fig)
    if model is not None and hasattr(model,'ode_model'):
        rows=[]
        for name,p in model.ode_model.named_parameters():
            # Float64 histogram edges also resolve nearly constant float32 parameters.
            values=p.detach().cpu().double().numpy().ravel()
            rows.append(dict(parameter=name,count=len(values),mean=float(values.mean()),std=float(values.std()),min=float(values.min()),max=float(values.max())))
            fig,ax=plt.subplots();ax.hist(values,bins=60);ax.set_title(name)
            fig.savefig(out/('parameter_'+name.replace('.','_')+'.png'),dpi=120);plt.close(fig)
        pd.DataFrame(rows).to_csv(out/'parameter_statistics.csv',index=False)
    return out


def generative(c,run,a,*,limit=3000):
    import anndata as ad
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .data import dense_rows
    out=confined(run/'figures/generative');out.mkdir(parents=True,exist_ok=True)
    generated=ad.read_h5ad(run/'samples/generated.h5ad')
    stamp=read_json(run/'samples/metadata.json')
    require(sha256(run/'samples/generated.h5ad')==stamp['output_sha256'],'samples changed')
    require(ids(generated.var_names)==ids(a.var_names),'sample gene order mismatch')
    rng=np.random.default_rng(c['seed']);chosen=np.sort(rng.choice(a.n_obs,min(limit,a.n_obs),replace=False))
    real=a[chosen].copy();matrix=dense_rows(real.X);gen=dense_rows(generated.X)
    source=module('work.20260915_x0predict.analysis.distributions')
    report=dict(real=source.diversity(matrix,pairs=10000,seed=c['seed']),
        generated=source.diversity(gen,pairs=10000,seed=c['seed']),
        wasserstein=source.sliced_wasserstein(matrix,gen,points=min(2048,len(matrix),len(gen)),seed=c['seed']),
        real_gene_variance=float(matrix.var(0).mean()),generated_gene_variance=float(gen.var(0).mean()),
        generated_near_constant_gene_fraction=float((gen.var(0)<1e-8).mean()),
        metric_space='original linear expression gene space; UMAP distances are not quantitative errors',
        real_cell_ids=ids(real.obs_names))
    write_json(out/'collapse_diagnostics.json',report)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].hist(matrix.ravel(),bins=100,alpha=.5,label='real',density=True)
    axes[0].hist(gen.ravel(),bins=100,alpha=.5,label='generated',density=True);axes[0].legend()
    axes[1].scatter(matrix.mean(0),gen.mean(0),s=4);axes[1].set(xlabel='real gene mean',ylabel='generated gene mean')
    axes[2].scatter(matrix.var(0),gen.var(0),s=4);axes[2].set(xlabel='real gene variance',ylabel='generated gene variance')
    fig.savefig(out/'expression_distribution.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # 9/11's scoped import adapter loads 8/30's exact geometry implementation.
    core=module('work.20260911.src.source_imports').umap_core()
    combined=core.build_sampling_anndata(real,gen)
    geometry=core.compute_common_umap(combined,seed=c['seed'])
    xy=np.asarray(combined.obsm['X_umap']);n=len(real)
    fig,ax=plt.subplots(figsize=(8,7));ax.scatter(*xy[:n].T,s=3,alpha=.4,label='Real')
    ax.scatter(*xy[n:].T,s=4,alpha=.6,label='Generated');ax.legend();ax.set_title('Joint UMAP; descriptive geometry')
    fig.savefig(out/'joint_umap.png',dpi=160);plt.close(fig)
    combined.write_h5ad(out/'joint_umap.h5ad');write_json(out/'umap_metadata.json',geometry)
    return out


def velocity_worker(run):
    """Run using the existing evaluation Python. Plot official fold geometry/velocities."""
    import anndata as ad
    import numpy as np
    import scanpy as sc
    import scvelo as scv
    import pickle
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    run=Path(run);created=[]
    for result in sorted((run/'metrics').glob('*')):
        if not result.is_dir() or not (result/'metadata.json').exists():continue
        require(read_json(result/'metadata.json')['status']=='complete','incomplete benchmark cannot be visualized as complete')
        for fold in range(3):
            path=result/f'fold_{fold}/processed/adata_run_candidate_full.h5ad'
            a=ad.read_h5ad(path)
            key='candidate_velocity'
            # The processed h5ad is saved BEFORE upstream postprocess. Its official
            # embedding is in the locally generated pickle, not in that h5ad.
            with (result/f'fold_{fold}/postprocess/candidate_full.pkl').open('rb') as handle:
                post=pickle.load(handle)
            require(key in a.layers and post['cell_label'].index.equals(a.obs_names),
                    'official output cell order mismatch')
            require(post['velocity_emb'].shape==(a.n_obs,2) and np.isfinite(post['velocity_emb']).all(),
                    'official velocity projection absent/nonfinite')
            require(np.array_equal(post['exp_emb'],a.obsm['X_umap']),'official geometry differs')
            a.obsm[key+'_umap']=post['velocity_emb']
            out=confined(run/'figures/velocity'/result.name/f'fold_{fold}');out.mkdir(parents=True,exist_ok=True)
            for label in ('celltype','stage'):
                fig,ax=plt.subplots(figsize=(9,7));sc.pl.umap(a,color=label,ax=ax,show=False)
                fig.savefig(out/(label+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
                fig,ax=plt.subplots(figsize=(9,7))
                scv.pl.velocity_embedding_stream(a,basis='umap',vkey=key,color=label,ax=ax,show=False,
                                                X=post['exp_emb'],V=post['velocity_emb'])
                fig.savefig(out/('stream_'+label+'.png'),dpi=160,bbox_inches='tight');plt.close(fig)
            values=np.asarray(a.layers[key]);finite=np.isfinite(values).all(1)
            norms=np.linalg.norm(values,axis=1)
            write_json(out/'diagnostics.json',dict(nonfinite_cell_fraction=float((~finite).mean()),
                near_zero_cell_fraction=float((norms<1e-8).mean()),near_zero_threshold=1e-8,
                geometry='official normalized benchmark fold; no joint fit between folds'))
            fig,ax=plt.subplots();ax.hist(norms[finite],bins=60);ax.set_xlabel('Velocity proxy L2 norm')
            fig.savefig(out/'norms.png',dpi=160);plt.close(fig);created.append(str(out))
    require(created,'no completed three-fold benchmark artifacts; run evaluate first')
    return created


def visualize(c,run,*,data_dir=None,scope='all',checkpoint=None):
    from .data import read_data
    from .checkpoint import load
    outputs=[]
    if scope in ('all','training','samples'):
        a,meta=read_data(c,data_dir)
        if scope in ('all','training'):
            model,_,_=load(c,run,meta,value=checkpoint)
            outputs.append(training_plots(run,model))
        if scope in ('all','samples'):outputs.append(generative(c,run,a))
    if scope in ('all','velocity'):
        python=ROOT/'data_preparation/20261007/data/benchmark/.venv-eval/bin/python'
        require(python.is_file(),'existing benchmark environment missing; see benchmark/setup_env.sh')
        subprocess.run([str(python),'-m','work.20261010_model_comparison.cli','velocity-worker','--run',str(run)],cwd=ROOT,check=True)
        outputs.append(run/'figures/velocity')
    return outputs
