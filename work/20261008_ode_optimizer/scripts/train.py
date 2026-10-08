#!/usr/bin/env python3
"""Train exactly one of the four conditions; never invoked implicitly by import."""
import argparse
from common import *

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True); p.add_argument('--run-dir',required=True)
    p.add_argument('--resume',nargs='?',const='auto',default='')
    p.add_argument('--stop-after',type=int,help='Testing/interruption boundary; does not change the LR horizon')
    a=p.parse_args(argv)
    import torch
    from guided_diffusion import logger, dist_util
    from training.data import StatefulBatches
    from training.train_loop import TrainLoop
    c=validate(read_json(a.config)); run=inside(a.run_dir)
    stored=run/'exp_config.json'
    if stored.exists() and read_json(stored)!=c: raise ValueError('Existing run config differs')
    if (run/'checkpoints/latest.json').exists() and not a.resume:
        raise FileExistsError('Use --resume for an existing training run')
    write_json(stored,c)
    seed_all(c['seed']); device=device_for(c); dist_util.dev=lambda:device
    logger.configure(dir=str(run/'logs/train'))
    x,genes,labels=load_cells(c)
    fingerprints={k:sha256(c[k]) for k in ('data_dir','edge_tsv_path')}
    provenance=run/'input_fingerprints.json'
    if provenance.exists() and read_json(provenance)!=fingerprints:
        raise ValueError('Input data/GRN changed since the run was created')
    write_json(provenance,fingerprints)
    diffusion=diffusion_for(c)
    seed_all(c['seed'])  # isolate initialization from data loading/import side effects
    model=build(c,genes,device)
    initial_hash=legacy('analysis.gradients').parameter_fingerprint(model)
    write_json(run/'model_info.json',dict(initial_parameter_sha256=initial_hash,
        denoising_output='ml_model_only',ode=model.ode_model.get_model_info(),
        optimizer={'cell':'AdamW','ode':'SGD','momentum':0,'weight_decay':c['weight_decay']},
        data_order='seeded private torch.Generator; shuffle/drop_last; checkpointed permutation and cursor'))
    data=StatefulBatches(x,labels,c['batch_size'],c['seed'])
    resume=True if a.resume=='auto' else a.resume or False
    TrainLoop(model=model,diffusion=diffusion,data=data,config=c,run_dir=run,resume=resume).run_loop(a.stop_after)

if __name__=='__main__': main()
