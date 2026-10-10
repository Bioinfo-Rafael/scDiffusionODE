"""Clean-X branch exports. `ds_dt` in the benchmark contract is an explicit proxy assumption."""
from __future__ import annotations
import numpy as np
import torch
from ..common import *
from .checkpoint import load
from .data import read_data, dense_rows, write_h5ad
from .model import branches


available = branch_names


def definition(c, field):
    return dict(ode_raw='F(clean_X,t); t is ignored by autonomous fields',
                cellunet_raw='C(clean_X,t)',hybrid_raw='H(clean_X,t); composition='+c['hybrid'])[field]


def export(c, run, *, data_dir=None, checkpoint=None, target='auto', fields=None):
    import anndata as ad
    run=paths_at(run);a,meta=read_data(c,data_dir)
    model,ck,path=load(c,run,meta,value=checkpoint,target=device(target))
    dev=next(model.parameters()).device
    selected=fields or available(c)
    require(set(selected)<=set(available(c)), 'branch unavailable for this model')
    outputs=[]
    for field in selected:
        output=run/'predictions'/f'{field}_t{c["field_timestep"]}.h5ad'
        sidecar=output.with_suffix('.json')
        stamp=dict(field_source=field,definition=definition(c,field),condition=c['condition'],
            branch=field,model_class=type(model).__module__+'.'+type(model).__name__,
            diffusion_timestep=c['field_timestep'],input_expression_scale=SCALE,
            input_policy='clean X; no q_sample, subtraction, integration or output rescaling',
            checkpoint_path=str(path),checkpoint_sha256=sha256(path),weights='ema',
            training_sha256=meta['training_sha256'],input_sha256=meta['input_sha256'],training_target=c['prediction'],
            branch_training_role=('consistency to CellUNet plus soft penalty; lambda='+str(c['consistency_lambda'])
                if c['family']=='consistency' and field=='ode_raw' else
                'component of combined diffusion predictor' if c['family'] in ('joint','two_step') and field!='hybrid_raw' else
                'diffusion predictor'),
            composition=c['hybrid'],
            cell_ids_sha256=digest_ids(a.obs_names),gene_ids_sha256=digest_ids(a.var_names),
            biological_assumption='experimental proxy ds/dt := raw output; START_X/EPSILON loss does not establish ds/dt',
            output_category=('autonomous_field' if c['ode'] in ('simple_softplus','hill_after_linear','centered_signed_hill','shifted_hill_rho','geneode') else 'conditioned_field') if field=='ode_raw' else 'denoiser_proxy')
        if output.exists():
            old=read_json(sidecar)
            require(all(old.get(k)==v for k,v in stamp.items()), 'existing velocity provenance differs')
            require(old['output_sha256']==sha256(output), 'velocity artifact changed')
            outputs.append(output);continue
        tmp=output.with_suffix('.working.npy')
        require(not tmp.exists(),f'incomplete export exists; inspect/remove {tmp} before retry')
        arr=np.lib.format.open_memmap(tmp,mode='w+',dtype='float32',shape=a.shape)
        with torch.no_grad():
            for start in range(0,a.n_obs,c['inference_batch_size']):
                x=torch.from_numpy(dense_rows(a.X[start:start+c['inference_batch_size']])).to(dev)
                t=torch.full((len(x),1),c['field_timestep'],device=dev,dtype=torch.long)
                if field=='ode_raw':v=model.ode_model(x,t)
                elif field=='cellunet_raw':v=(model if c['family']=='baseline' else model.ml_model)(x,t)
                else:v=model(x,t)
                require(tuple(v.shape)==tuple(x.shape),'output shape mismatch')
                arr[start:start+len(x)]=v.cpu().float().numpy()
        arr.flush()
        finite=np.isfinite(arr).all(axis=1)
        norms=np.linalg.norm(np.asarray(arr),axis=1)
        diagnostics=dict(nonfinite_cell_fraction=float((~finite).mean()),near_zero_cell_fraction=float((norms<1e-8).mean()),
                         near_zero_threshold=1e-8,norm_quantiles=[float(v) for v in np.quantile(norms[finite],[0,.25,.5,.75,1])] if finite.any() else [])
        write_json(output.with_name(output.stem+'_diagnostics.json'),diagnostics)
        require(finite.all(),'nonfinite velocity: diagnostic retained; no benchmark artifact published')
        pred=ad.AnnData(X=None,obs=a.obs.copy(),var=a.var.copy())
        pred.layers['velocity']=np.asarray(arr)
        pred.uns['benchmark_velocity']=dict(definition='ds_dt',expression_scale=SCALE,time_direction='forward',
            training_n_cells=a.n_obs,time_unit='arbitrary proxy time',checkpoint=str(path),
            inference_description=stamp['biological_assumption']+'; '+stamp['definition'])
        pred.uns['model_comparison']=stamp
        write_h5ad(pred,output)
        write_json(sidecar,dict(stamp,output_sha256=sha256(output),cell_ids=meta['cell_ids'],gene_ids=meta['gene_ids']))
        del pred,arr;tmp.unlink()
        outputs.append(output)
    (run/'predictions/gene_ids.txt').write_text('\n'.join(meta['gene_ids'])+'\n')
    return outputs


def sample(c, run, *, data_dir=None, checkpoint=None, target='auto', count=None):
    import anndata as ad
    import pandas as pd
    run=paths_at(run);a,meta=read_data(c,data_dir)
    model,ck,path=load(c,run,meta,value=checkpoint,target=device(target))
    n=c['num_samples'] if count is None else count
    require(n>0,'positive sample count required')
    out=run/'samples/generated.h5ad';stamp=dict(checkpoint_sha256=sha256(path),weights='ema',
        count=n,batch_size=c['sample_batch_size'],seed=c['seed'],prediction=c['prediction'],
        method='native ancestral DDPM',steps=1000,nw=.5,clip_denoised=False,
        gene_ids=meta['gene_ids'],training_sha256=meta['training_sha256'])
    if out.exists():
        old=read_json(run/'samples/metadata.json')
        require(all(old.get(k)==v for k,v in stamp.items()) and old['output_sha256']==sha256(out),'sample provenance mismatch')
        return out
    source=module('work.20260915_x0predict.common');source.seed_all(c['seed'])
    d=diffusion(c);samples=np.empty((n,a.n_vars),dtype=np.float32)
    with torch.no_grad():
        for start in range(0,n,c['sample_batch_size']):
            size=min(c['sample_batch_size'],n-start)
            for result in d.p_sample_loop_progressive(model,(size,a.n_vars),clip_denoised=False,
                                                      device=next(model.parameters()).device,start_time=1000,nw=.5):
                source.finite('ancestral state',result['sample'])
            samples[start:start+size]=result['sample'].cpu().float().numpy()
            print(f'Generated {start+size}/{n}',flush=True)
    result=ad.AnnData(X=samples,obs=pd.DataFrame(index=[f'generated_{i}' for i in range(n)]),var=a.var.copy())
    result.uns['sampling']=dict(method=stamp['method'],prediction=c['prediction'],weights='ema')
    write_h5ad(result,out);write_json(run/'samples/metadata.json',dict(stamp,output_sha256=sha256(out)))
    return out
