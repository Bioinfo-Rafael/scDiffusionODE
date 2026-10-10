"""Retain official fold values/mean/std and keep protocol/output categories distinct."""
from __future__ import annotations
from ..common import *
from .benchmark import METRICS


def collect(run_root,extra=()):
    import pandas as pd
    import numpy as np
    rows=[];pending=[]
    entries=[]
    for run in sorted(Path(run_root).glob('*')):
        if not run.is_dir():continue
        cpath=run/'effective_config.json'
        if not cpath.exists():continue
        cfg=read_json(cpath)
        results=[p.parent for p in (run/'metrics').glob('*/metadata.json')]
        if not results:pending.append(dict(condition=run.name,status='not evaluated'))
        entries += [(run.name,p.name,p,cfg) for p in results]
    entries += [('external',Path(p).name,Path(p),{}) for p in extra]
    for condition,field,p,cfg in entries:
        meta=read_json(p/'metadata.json')
        if meta.get('status')!='complete':
            pending.append(dict(condition=condition,field=field,status=meta.get('status','unknown')));continue
        folds=pd.read_csv(p/'metrics_per_fold.csv');summary=pd.read_csv(p/'metrics_summary.csv')
        require(len(folds)==3 and set(folds.fold)=={0,1,2},'not three unique folds')
        require(np.isfinite(folds[list(METRICS)].to_numpy()).all(),'nonfinite official metric')
        require(set(summary.metric)==set(METRICS),'missing official summary metrics')
        row=dict(condition=condition,field=field,category=field.rsplit('_t',1)[0],
            prediction=cfg.get('prediction','scVelo stochastic / external'),family=cfg.get('family','external'),
            inference_protocol=meta['inference_protocol'],profile=meta['profile'],
            reference_manifest_sha256=meta.get('reference_manifest_sha256',''),
            protocol_note='full population learned once; folds only extract/evaluate' if meta['inference_protocol']=='full-trained' else 'raw erythroid; fold-specific preprocessing/inference; different protocol',
            result_path=str(p))
        for metric in METRICS:
            s=summary[summary.metric==metric].iloc[0]
            require(int(s.ddof)==0 and int(s.n_folds)==3,'unexpected official summary convention')
            row[metric+'_mean']=float(s['mean']);row[metric+'_std']=float(s['std'])
            for fold in range(3):row[f'{metric}_fold{fold}']=float(folds.loc[folds.fold==fold,metric].iloc[0])
        rows.append(row)
    return rows,pending


def summarize(run_root,*,output=None,extra=()):
    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=confined(output or Path(run_root)/'comparison');out.mkdir(parents=True,exist_ok=True)
    rows,pending=collect(run_root,extra)
    # Include all pilot conditions, including never-trained ones.
    found={r['condition'] for r in rows}|{r['condition'] for r in pending}
    pending += [dict(condition=n,status='not run') for n in conditions() if n not in found]
    write_json(out/'status.json',dict(completed_output_rows=len(rows),pending=pending))
    frame=pd.DataFrame(rows);frame.to_csv(out/'comparison.csv',index=False)
    if not rows:return out
    # Never put different protocols/reference geometries in a single ranking plot.
    for i,((protocol,profile,reference),part) in enumerate(frame.groupby(['inference_protocol','profile','reference_manifest_sha256'])):
        labels=part.condition+' / '+part.field;y=np.arange(len(part))
        fig,axes=plt.subplots(1,4,figsize=(18,max(4,len(part)*.25)),layout='constrained')
        for ax,metric in zip(axes,METRICS):
            ax.errorbar(part[metric+'_mean'],y,xerr=part[metric+'_std'],fmt='o')
            for fold in range(3):ax.scatter(part[f'{metric}_fold{fold}'],y,s=12,alpha=.5)
            ax.set_yticks(y,labels if ax is axes[0] else []);ax.set_title(metric)
        fig.suptitle(f'{protocol} / {profile}; points=folds, bars=population SD')
        fig.savefig(out/f'metrics_{i}.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots(figsize=(7,max(4,len(part)*.25)),layout='constrained')
        im=ax.imshow(part[[m+'_mean' for m in METRICS]].to_numpy(),aspect='auto')
        ax.set_xticks(range(4),METRICS);ax.set_yticks(y,labels);fig.colorbar(im,ax=ax,label='raw metric mean (no aggregate score)')
        fig.savefig(out/f'heatmap_{i}.png',dpi=160);plt.close(fig)
        fig,ax=plt.subplots();ax.scatter(part.cbdir_mean,part.tsc_mean,c=part.icvcoh_mean)
        ax.set(xlabel='CBDir',ylabel='TSC',title=f'{protocol} / {profile}; color=ICVCoh')
        fig.savefig(out/f'scatter_{i}.png',dpi=160);plt.close(fig)
    return out
