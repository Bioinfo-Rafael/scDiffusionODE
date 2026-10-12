"""Run only D01-D04 with CellUNet AdamW / ODE SGD in a separate campaign."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,default=HERE/'runs/d_ode_sgd_30000')
    p.add_argument('--device',default='cuda',choices=['cuda','cpu','auto'])
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--plots-only',action='store_true',help='finish plots from saved samples/evaluations; no training or sampling')
    p.add_argument('--harmony-python',help='Python with harmonypy==0.0.10 (default: runs/.venv-harmony if present)')
    p.add_argument('--stages',nargs='+',choices=['train','sample','export_velocity','evaluate','visualize'],
                   default=['train','sample','export_velocity','evaluate','visualize'])
    args=p.parse_args();run=args.run_root.resolve()
    if not run.is_relative_to(HERE/'runs') or run==HERE/'runs':
        raise ValueError('Choose a campaign directory below this suite runs/')
    if run==HERE/'runs/default':raise ValueError('Use a separate campaign for D ODE-SGD')
    configs=sorted((HERE/'configs/ode_sgd').glob('D*.json'))
    for path in configs:
        c=json.loads(path.read_text());old=run/c['condition']/'effective_config.json'
        if old.exists() and json.loads(old.read_text()).get('ode_optimizer')!='SGD':
            raise ValueError(f'Existing AdamW campaign: {old}; choose a new run-root')
    base=[sys.executable,'-m','work.20261010_model_comparison.cli']
    def execute(command):
        print(shlex.join(command),flush=True)
        if not args.dry_run:subprocess.run(command,cwd=ROOT,check=True)
    finish=[sys.executable,str(HERE/'scripts/finish_ode_sgd_plots.py'),'--run-root',str(run)]
    if args.harmony_python:finish+=['--harmony-python',args.harmony_python]
    if args.plots_only:
        execute(finish);return
    if 'visualize' in args.stages:execute(finish+['--check-only'])
    execute(base+['prepare'])
    for path in configs:
        c=json.loads(path.read_text())
        command=base+['launch','--condition',str(path),'--run-root',str(run),'--device',args.device,
                      '--prepare-reference','--stages',*args.stages]
        if args.resume and (run/c['condition']/'checkpoints/latest.json').exists():
            command+=['--resume']
        execute(command)
    if 'visualize' in args.stages:execute(finish)
    else:execute(base+['summarize','--run-root',str(run)])


if __name__=='__main__':main()
