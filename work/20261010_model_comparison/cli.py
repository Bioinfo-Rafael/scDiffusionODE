"""Unified commands; scientific imports are lazy, so dry-run needs only Python."""
from __future__ import annotations
import argparse
import importlib.util
import shlex
import traceback
from .common import *


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['list','preflight','prepare','train','sample','export_velocity','evaluate','visualize','summarize','launch','velocity-worker'])
    p.add_argument('--condition',default='A01_cellunet_startx',help='condition ID or JSON path')
    p.add_argument('--all',action='store_true',help='all 17 pilot conditions')
    p.add_argument('--extended',action='store_true',help='list/include extended conditions')
    p.add_argument('--run-root',type=Path,default=HERE/'runs/default')
    p.add_argument('--data-dir',type=Path,default=HERE/'runs/data')
    p.add_argument('--run',type=Path,help='internal plotting worker directory')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--stop-after',type=int,help='absolute update count, not added steps')
    p.add_argument('--resume',nargs='?',const='latest')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--stage1-checkpoint',type=Path)
    p.add_argument('--no-reuse',action='store_true',help='do not auto-reuse a legacy A01 checkpoint')
    p.add_argument('--count',type=int,help='sampling smoke override')
    p.add_argument('--field',action='append',choices=['ode_raw','cellunet_raw','hybrid_raw'])
    p.add_argument('--reference',type=Path)
    p.add_argument('--prepare-reference',action='store_true')
    p.add_argument('--prediction',type=Path,help='evaluate an existing full-trained prediction only')
    p.add_argument('--genes',type=Path)
    p.add_argument('--label',help='new evaluation label for existing external prediction')
    p.add_argument('--scope',choices=['all','training','samples','velocity'],default='all')
    p.add_argument('--extra-result',type=Path,action='append',default=[],help='external official result directory (e.g. scVelo)')
    p.add_argument('--stages',nargs='+',choices=['train','sample','export_velocity','evaluate','visualize'],default=['train','sample','export_velocity','evaluate','visualize'])
    return p


def preflight():
    from .adapters.benchmark import DEFAULT_REFERENCE
    baseline=read_json(HERE/'audit/protected_files.json')
    changed=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    report=dict(protected_file_count=len(baseline),protected_changed=changed,
        pilot_conditions=len(conditions()),extended_conditions=len(conditions(True)),
        source_commit=(HERE/'audit/git_commit.txt').read_text().strip(),
        source_input_exists=root_path(config('A01_cellunet_startx')['input']).is_file(),
        legacy_training_exists=(ROOT/'work/20261009_newBenchmark/runs/data/training.h5ad').is_file(),
        benchmark_python_exists=(ROOT/'data_preparation/20261007/data/benchmark/.venv-eval/bin/python').is_file(),
        normalized_reference_exists=(DEFAULT_REFERENCE/'manifest.json').exists(),
        dependencies={n:bool(importlib.util.find_spec(n)) for n in ['torch','anndata','scanpy','blobfile','pandas','pytest']},
        training_started=False)
    for n in conditions()+conditions(True):config(n)
    require(not changed,f'protected files changed: {changed}')
    return report


def dispatch(action,c,run,args):
    if action=='prepare':
        return module(__package__+'.adapters.data').prepare(c,data_dir=args.data_dir)
    if action=='train':
        return module(__package__+'.adapters.training').train(c,run,data_dir=args.data_dir,target=args.device,
            stop=args.stop_after,resume=args.resume,stage1=args.stage1_checkpoint,reuse=not args.no_reuse)
    if action in ('sample','export_velocity'):
        v=module(__package__+'.adapters.velocity')
        kwargs=dict(data_dir=args.data_dir,checkpoint=args.checkpoint,target=args.device)
        return v.sample(c,run,count=args.count,**kwargs) if action=='sample' else v.export(c,run,fields=args.field,**kwargs)
    if action=='evaluate':
        return module(__package__+'.adapters.benchmark').evaluate(c,run,fields=args.field,
            reference=args.reference,prepare=args.prepare_reference,prediction=args.prediction,
            genes=args.genes,label=args.label,dry_run=args.dry_run)
    if action=='visualize':
        return module(__package__+'.adapters.visualization').visualize(c,run,data_dir=args.data_dir,scope=args.scope,checkpoint=args.checkpoint)
    raise ValueError(action)


def main(argv=None):
    args=parser().parse_args(argv)
    if args.action=='list':
        names=conditions()+(conditions(True) if args.extended else [])
        for name in names:
            c=config(name);print(f"{name:48} {c['family']:12} {str(c['ode']):26} {c['hybrid']:22} {c['prediction']} {c['total_steps']} updates")
        return
    if args.action=='preflight':
        print(json.dumps(preflight(),indent=2));return
    if args.action=='summarize':
        print(module(__package__+'.adapters.summary').summarize(args.run_root,extra=args.extra_result));return
    if args.action=='velocity-worker':
        require(args.run is not None,'--run required')
        print(module(__package__+'.adapters.visualization').velocity_worker(confined(args.run)));return
    names=conditions()+(conditions(True) if args.extended else []) if args.all else [args.condition]
    for name in names:
        c=config(name);run=run_dir(c,args.run_root)
        actions=args.stages if args.action=='launch' else [args.action]
        for action in actions:
            if args.dry_run:
                if action=='evaluate':
                    commands=dispatch(action,c,run,args)
                    print('\n'.join(shlex.join(cmd) for cmd in commands))
                else:
                    print(json.dumps(dict(action=action,run=str(run),data_dir=str(args.data_dir),config=c,
                        checkpoint=str(args.checkpoint) if args.checkpoint else 'latest',
                        stage1='shared final A01 EMA' if c['family']=='two_step' else None),sort_keys=True))
                continue
            if args.action=='launch' and action=='train' and (run/'checkpoints/latest.json').exists():
                from .adapters.checkpoint import resolve, read
                from .adapters.data import read_data
                from .adapters.checkpoint import load
                _,meta=read_data(c,args.data_dir)
                _,ck,_=load(c,run,meta)
                if ck['step']==c['total_steps']:
                    print(f'{name}: reuse completed training');continue
                if not args.resume:raise ValueError(f'{name}: partial training; use --resume')
            try:
                print(dispatch(action,c,run,args))
            except Exception:
                write_json(run/'logs'/f'failure_{action}_{uuid.uuid4().hex[:8]}.json',dict(action=action,traceback=traceback.format_exc()))
                raise

if __name__=='__main__':main()
