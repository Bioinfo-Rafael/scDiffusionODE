#!/usr/bin/env python3
"""Capture six states of ONE unchanged 1000-update ancestral trajectory."""
import argparse
import numpy as np
from common import *


def capture_trajectory(diffusion, model, shape, *, device, clip_denoised=False):
    import torch
    if diffusion.num_timesteps != 1000:
        raise ValueError('Exactly 1000 reverse updates are required')
    noise=torch.randn(*shape,device=device)
    states={0:noise.detach().cpu().numpy().copy()}
    count=0
    for count,out in enumerate(diffusion.p_sample_loop_progressive(
            model,shape,noise=noise,device=device,start_time=1000,
            clip_denoised=clip_denoised),start=1):
        if count in STEPS:
            states[count]=out['sample'].detach().cpu().numpy().copy()
    if count!=1000 or set(states)!=set(STEPS):
        raise RuntimeError('Sampler did not yield the required trajectory')
    return states


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--run-dir',required=True)
    p.add_argument('--force',action='store_true'); a=p.parse_args(argv)
    import torch
    run=inside(a.run_dir); c=validate(read_json(run/'exp_config.json'))
    expected=read_json(run/'input_fingerprints.json')
    if any(sha256(c[k])!=value for k,value in expected.items()):
        raise ValueError('Training input data/GRN changed')
    cp=checkpoint(run); output=run/'samples'; output.mkdir(parents=True,exist_ok=True)
    if (output/'sampling.json').exists() and not a.force:
        m=read_json(output/'sampling.json')
        if m.get('prediction_target','epsilon')==prediction_target(c) and m['checkpoint_sha256']==sha256(cp) and all((output/f'step_{s:04d}.npz').exists() for s in STEPS):
            return
        raise FileExistsError('Sampling inputs changed; use --force')
    _,genes,_=load_cells(c); device=device_for(c)
    diffusion=diffusion_for(c); model=build(c,genes,device)
    model.load_state_dict(torch_load(cp),strict=True); model.eval()
    # Reseed AFTER model construction/loading. Each condition consumes exactly
    # the same Gaussian sequence and uses the same cell/chunk ordering.
    seed_all(c['sampling_seed'])
    calls=[]; hook=model.ode_model.register_forward_hook(lambda *args:calls.append(1))
    chunks={s:[] for s in STEPS}
    try:
        with torch.no_grad():
            for start in range(0,c['num_samples'],c['sample_batch_size']):
                n=min(c['sample_batch_size'],c['num_samples']-start)
                states=capture_trajectory(diffusion,model,(n,len(genes)),device=device,
                                          clip_denoised=c['clip_denoised'])
                for s in STEPS: chunks[s].append(states[s])
    finally: hook.remove()
    if calls: raise RuntimeError('ODE must not run during sampling')
    definition='s = completed reverse diffusion updates; s=0 initial Gaussian; s=1000 final sample after t=0'
    metadata=dict(prediction_target=prediction_target(c),checkpoint=str(cp),checkpoint_sha256=sha256(cp),seed=c['sampling_seed'],
        sampling_steps=list(STEPS),step_definition=definition,num_samples=c['num_samples'],
        sample_batch_size=c['sample_batch_size'],sampler='p_sample_loop_progressive',
        rng_protocol='reseed after model load; Gaussian initial then 1000 ancestral draws per chunk',
        ode_forward_calls=0,condition=c['experiment'],gene_names=genes,device=str(device))
    for s in STEPS:
        array=np.concatenate(chunks[s]).astype(np.float32,copy=False)
        if not np.isfinite(array).all(): raise FloatingPointError(f'Nonfinite samples at s={s}')
        np.savez_compressed(output/f'step_{s:04d}.npz',cell_gen=array,cell_index=np.arange(len(array)),
            reverse_updates=s,checkpoint=str(cp),seed=c['sampling_seed'],step_definition=definition)
    write_json(output/'sampling.json',metadata)

if __name__=='__main__': main()
