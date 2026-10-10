"""Family loss adapters with sparse IO; reuse 10/09 baseline train() unchanged."""
from __future__ import annotations
import csv
import torch
from types import SimpleNamespace
from ..common import *
from . import checkpoint as CP
from .data import read_data, mask_for
from .legacy import load_script
from .model import build, branches, single
from .objectives import loss, sampler


def baseline(c, run, a, meta, *, target, stop, resume, reuse=True):
    legacy=module('work.20261009_newBenchmark.common')
    cfg=CP.baseline_training_config(c)
    source_ck=root_path(c['legacy_run'])/'checkpoints/latest.json'
    if reuse and not resume and not (run/'checkpoints/latest.json').exists() and source_ck.exists():
        p=CP.resolve(root_path(c['legacy_run']))
        state,stamp=CP.stage1_state(p,c,meta)
        write_json(run/'checkpoints/latest.json',dict(checkpoint=str(p),sha256=stamp['sha256'],step=30000,reused=True))
        write_json(run/'reuse.json',stamp)
        return p
    legacy_c=dict(training=cfg,runs=str(run),source_hashes=legacy.config()['source_hashes'],
                  expected_cells=c['expected_cells'],n_hvg=c['n_hvg'])
    # The original function expects metadata on disk; only IO globals are rebound.
    (run/'data').mkdir(exist_ok=True)
    write_json(run/'data/metadata.json',meta)
    def load_old(_c,value=None,target='cpu',weights='ema'):
        m,ck,p=CP.load(c,run,meta,value=value,weights=weights,target=target)
        return m,ck,a,p
    source=load_script('train',paths=lambda _:run,read_data=lambda _:(a,meta),
        write_json=write_json,load_checkpoint=load_old)
    # Observe the unchanged legacy loss/update through hooks; never alter gradients.
    start=CP.read(CP.resolve(run,resume))['step'] if resume else 0
    state=dict(step=start);handles=[]
    diagnostics=run/'logs'/f'baseline_diagnostics_{start:06d}_{uuid.uuid4().hex[:8]}.jsonl'
    def observed_loss(*args,**kwargs):
        value,parts=legacy.OBJECTIVES.training_loss(*args,**kwargs)
        state['components']=parts
        return value,parts
    def observed_optimizer(model,config):
        opt=legacy.MODELS.optimizer_for(model,config)
        def forward_hook(_model,_inputs,output):
            state['cellunet_raw_training_norm']=float(output.detach().float().norm(dim=1).mean())
        handles.append(model.register_forward_hook(forward_hook))
        def before_step(_opt,_args,_kwargs):
            state['step']+=1
            row=dict(step=state['step'],**state['components'],
                cellunet_raw_training_norm=state['cellunet_raw_training_norm'],
                cell_gradient_norm=sum(float(p.grad.detach().double().square().sum()) for p in model.parameters() if p.grad is not None)**.5,
                lr_used=opt.param_groups[0]['lr'])
            with diagnostics.open('a') as log:log.write(json.dumps(row,allow_nan=False)+'\n')
        handles.append(opt.register_step_pre_hook(before_step))
        return opt
    source.MODELS=SimpleNamespace(build_model=legacy.MODELS.build_model,
        optimizer_for=observed_optimizer,update_ema=legacy.MODELS.update_ema)
    source.OBJECTIVES=SimpleNamespace(training_loss=observed_loss,timestep_sampler=legacy.OBJECTIVES.timestep_sampler)
    try:
        return source.train(legacy_c,target_device=str(target),steps=stop,
                            resume=CP.resolve(run,resume) if resume else None)
    finally:
        for handle in handles:handle.remove()


def train(c, run, *, data_dir=None, target='auto', stop=None, resume=None, stage1=None, reuse=True):
    run=paths_at(run);dev=device(target)
    a,meta=read_data(c,data_dir);prov=provenance(c,meta)
    stop=c['total_steps'] if stop is None else stop
    require(1<=stop<=c['total_steps'], 'stop is absolute update count, within configured budget')
    if (run/'effective_config.json').exists():
        require(read_json(run/'effective_config.json')==c,'run config changed')
    if (run/'provenance.json').exists():
        require(read_json(run/'provenance.json')==prov,'run provenance/source/data changed')
    write_json(run/'effective_config.json',c)
    write_json(run/'provenance.json',prov)
    if c['family']=='baseline':
        p=baseline(c,run,a,meta,target=dev,stop=stop,resume=resume,reuse=reuse)
        # Legacy train writes its own envelope at effective_config.json; preserve public config.
        write_json(run/'effective_config.json',c)
        return p
    legacy=load_script('train');common=module('work.20260915_x0predict.common')
    common.seed_all(c['seed'])
    stage_stamp=None
    if resume:
        model,ck,_=CP.load(c,run,meta,value=resume,weights='raw',target=dev)
        start=ck['step'];stage_stamp=ck['stage1']
        ema={k:v.to(dev) for k,v in ck['ema'].items()}
        batches=legacy.Batches(a,c['batch_size'],c['seed'],ck['batches'],preprocess=False)
    else:
        require(not (run/'checkpoints/latest.json').exists(),'checkpoint exists; use --resume')
        mask=mask_for(meta['gene_ids'],meta)
        model=build(c,meta['gene_ids'],mask=mask,meta=meta)
        if c['family']=='two_step':
            baseline_run=run.parent/c['stage1_condition']
            stage_path=CP.resolve(baseline_run,stage1)
            state,stage_stamp=CP.stage1_state(stage_path,dict(c,total_steps=30000),meta)
            shared=run.parent/'shared_stage1.json'
            if shared.exists():
                require(read_json(shared)==stage_stamp,'all Stage2 conditions must share the same final Stage1 EMA')
            else:
                write_json(shared,stage_stamp)
            if c['freeze_cellunet']:single.freeze_from_stage1(model,state)
            else:model.ml_model.load_state_dict(state,strict=True)
        model.to(dev)
        ema={k:v.detach().clone() for k,v in model.state_dict().items()}
        start=0;batches=legacy.Batches(a,c['batch_size'],c['seed'],preprocess=False)
    require(start<stop,'checkpoint already reached stop step')
    if c['freeze_cellunet']:
        opt=single.optimizer_for(model,c)
    else:
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],weight_decay=c['weight_decay'])
    if resume:
        opt.load_state_dict(ck['optimizer']);legacy.restore_rng(ck['rng'])
        # Archive uncommitted rows after a crash before appending resumed updates.
        for log in (run/'logs').glob('loss_*.jsonl'):
            if '.rolled_back.' in log.name:continue
            lines=log.read_text().splitlines()
            keep=[line for line in lines if json.loads(line)['step']<=start]
            tail=[line for line in lines if json.loads(line)['step']>start]
            if tail:
                log.with_suffix('.rolled_back.jsonl').write_text('\n'.join(tail)+'\n')
                log.write_text('\n'.join(keep)+'\n' if keep else '')
    else:
        # Separate model initialization from data/timestep/noise streams across architectures.
        common.seed_all(c['seed'])
    model.train();d=diffusion(c);sampling=sampler(c,d)
    if c['freeze_cellunet']:single.assert_frozen(model,opt,stage_stamp['state_hash'])
    logfile=run/'logs'/f'loss_{start:06d}_{uuid.uuid4().hex[:8]}.jsonl'
    with logfile.open('x') as f:
        for index in range(start,stop):
            x=common.finite('batch',batches.next().to(dev));t,w=sampling.sample(len(x),dev)
            opt.zero_grad(set_to_none=True)
            value,components=loss(c,model,d,x,t,w)
            common.finite('loss',value).backward()
            for p in model.parameters():
                if p.grad is not None:common.finite('gradient',p.grad)
            norms={}
            for key,branch in [('cell',getattr(model,'ml_model',None)),('ode',model.ode_model)]:
                if branch is not None:
                    norms[key+'_gradient_norm']=sum(float(p.grad.detach().double().square().sum()) for p in branch.parameters() if p.grad is not None)**.5
            lr_used=opt.param_groups[0]['lr'];opt.step();single.update_ema(ema,model,c['ema_rate'])
            # 9/16 uses total_steps=10000 but retains lr_anneal_steps=30000.
            for group in opt.param_groups:group['lr']=c['lr']*(1-index/c['lr_anneal_steps'])
            step=index+1
            row=dict(step=step,lr_used=lr_used,**components,**norms)
            if step%c['log_interval']==0 or step==stop:
                # Diagnostics on clean X and fixed t are separate from training loss.
                # eval mode avoids changing dynamic regularization caches or RNG streams.
                model.eval()
                with torch.no_grad():
                    outputs=branches(c,model,x.float(),torch.full((len(x),1),c['field_timestep'],device=dev))
                    row.update({k+'_clean_norm':float(v.float().norm(dim=1).mean()) for k,v in outputs.items()})
                    if c['hybrid'] in ('blend','additive'):
                        b=model.branch_outputs(x.float(),torch.full((len(x),1),c['field_timestep'],device=dev))
                        row.update({k+'_norm':float(b[k].norm(dim=1).mean()) for k in ('ode_contribution','ml_contribution')})
                model.train()
                print(f"{c['condition']} step={step}/{c['total_steps']} loss={float(value.detach()):.6g}",flush=True)
            f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
            if step%c['save_interval']==0 or step==stop:
                if c['freeze_cellunet']:single.assert_frozen(model,opt,stage_stamp['state_hash'])
                for v in model.state_dict().values():common.finite('checkpoint',v)
                path=CP.save(c,run,meta,model,opt,ema,step,batches,legacy.rng_state(),prov,stage_stamp)
    return path
