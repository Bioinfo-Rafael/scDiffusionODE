"""Trace the unchanged DDPM sampler without publishing generated cells."""
import argparse
from collections import deque
import importlib
import json
from pathlib import Path
import sys
import uuid

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))


def tensor_stats(value):
    import torch
    finite=torch.isfinite(value)
    selected=value.detach()[finite].double()
    return dict(nonfinite_elements=int((~finite).sum()),elements=value.numel(),
                min=float(selected.min()) if selected.numel() else None,
                max=float(selected.max()) if selected.numel() else None,
                max_abs=float(selected.abs().max()) if selected.numel() else None)


def trace_sampling(diffusion, model, *, count, batch_size, genes, steps, emit):
    import torch
    recent=deque(maxlen=16)
    with torch.no_grad():
        for start in range(0,count,batch_size):
            for index,result in enumerate(diffusion.p_sample_loop_progressive(
                    model,(min(batch_size,count-start),genes),clip_denoised=False,
                    device=next(model.parameters()).device,start_time=steps,nw=.5)):
                row=dict(batch_start=start,timestep=steps-1-index,
                         sample=tensor_stats(result['sample']),
                         pred_xstart=tensor_stats(result['pred_xstart']))
                emit(row);recent.append(row)
                if any(row[key]['nonfinite_elements'] for key in ('sample','pred_xstart')):
                    return dict(status='failed_nonfinite',first_failure=row,recent_steps=list(recent))
    return dict(status='finite_for_requested_count',count=count,recent_steps=list(recent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',default='B06_lowrank')
    p.add_argument('--run-root',type=Path,default=HERE/'runs/model_expansion_20261012')
    p.add_argument('--data-dir',type=Path,default=HERE/'runs/data')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='cuda')
    p.add_argument('--count',type=int,help='default: original num_samples; preserve original batch size')
    args=p.parse_args()
    C=importlib.import_module('work.20261010_model_comparison.common')
    D=importlib.import_module('work.20261010_model_comparison.adapters.data')
    K=importlib.import_module('work.20261010_model_comparison.adapters.checkpoint')
    source=importlib.import_module('work.20260915_x0predict.common')
    c=C.config(args.condition);run=C.run_dir(c,args.run_root)
    a,meta=D.read_data(c,args.data_dir)
    model,_,checkpoint=K.load(c,run,meta,target=C.device(args.device))
    count=c['num_samples'] if args.count is None else args.count
    C.require(count>0,'positive count required')
    directory=C.confined(run/'diagnostics');directory.mkdir(parents=True,exist_ok=True)
    token=uuid.uuid4().hex[:12];trace=directory/f'sampling_{token}.jsonl'
    source.seed_all(c['seed'])
    diffusion=C.diffusion(c)
    with trace.open('x') as handle:
        def emit(row):
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
        report=trace_sampling(diffusion,model,count=count,batch_size=c['sample_batch_size'],
                              genes=a.n_vars,steps=c['diffusion_steps'],emit=emit)
    report.update(condition=c['condition'],checkpoint=str(checkpoint),checkpoint_sha256=C.sha256(checkpoint),
                  seed=c['seed'],batch_size=c['sample_batch_size'],trace=str(trace),
                  sampler='unchanged native ancestral DDPM; EMA; clip_denoised=False; nw=0.5')
    output=directory/f'sampling_{token}.json';C.write_json(output,report)
    print(json.dumps(report,indent=2,allow_nan=False));print(output)


if __name__=='__main__':main()
