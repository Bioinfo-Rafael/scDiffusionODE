"""Run the 32 additional model conditions in their own campaign directory."""
from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
C = importlib.import_module('work.20261010_model_comparison.common')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=HERE/'runs/model_expansion_20261012')
    p.add_argument('--data-dir', type=Path, default=HERE/'runs/data')
    p.add_argument('--stage1-checkpoint', type=Path,
                   help='shared final A01 EMA bundle; otherwise resolve runs/default/A01_cellunet_startx')
    p.add_argument('--device', default='cuda', choices=['cuda','cpu','auto'])
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--plots-only', action='store_true', help='recreate all plots from saved checkpoints, samples and evaluations')
    p.add_argument('--harmony-python', help='Python with harmonypy==0.0.10; defaults to runs/.venv-harmony if present')
    p.add_argument('--stages', nargs='+', choices=['train','sample','export_velocity','evaluate','visualize'],
                   default=['train','sample','export_velocity','evaluate','visualize'])
    args = p.parse_args(argv)
    if args.plots_only:
        args.stages = ['visualize']
    run = args.run_root.resolve()
    manifest = C.expansion_campaign(run)
    data = args.data_dir.resolve()
    C.require(not run.is_relative_to(data) and not data.is_relative_to(run),
              'Campaign and shared data directories must be separate')
    configs = manifest['configs']
    base = [sys.executable, '-m', 'work.20261010_model_comparison.cli']
    finish = [sys.executable,str(HERE/'scripts/finish_comparison_plots.py'),
              '--run-root',str(run),'--data-dir',str(data),'--condition-set','model_expansion']
    if args.harmony_python:
        finish += ['--harmony-python',args.harmony_python]
    stage = args.stage1_checkpoint.resolve() if args.stage1_checkpoint else None
    if 'train' in args.stages:
        if args.dry_run:
            stage = stage or HERE/'runs/default/A01_cellunet_startx/checkpoints/model030000.pt'
        else:
            CP = importlib.import_module('work.20261010_model_comparison.adapters.checkpoint')
            stage = CP.resolve(HERE/'runs/default/A01_cellunet_startx', stage)
            C.require(stage.is_file(), 'shared final A01 checkpoint missing; pass --stage1-checkpoint')

    def execute(command):
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    if 'visualize' in args.stages:
        execute(finish + ['--check-only'])
    if 'train' in args.stages:
        execute(base + ['prepare', '--data-dir', str(data)])
        if not args.dry_run:
            D = importlib.import_module('work.20261010_model_comparison.adapters.data')
            _, meta = D.read_data(C.config('A01_cellunet_startx'), data)
            _, stamp = CP.stage1_state(stage, C.config('A01_cellunet_startx'), meta)
            shared = run/'shared_stage1.json'
            if shared.exists():
                C.require(C.read_json(shared) == stamp, 'shared A01 checkpoint changed')
            else:
                C.write_json(run/'campaign.json', manifest)
                C.write_json(shared, stamp)
    if not args.dry_run:
        C.write_json(run/'campaign.json', manifest)
    for c in configs:
        command = base + ['launch', '--condition', c['condition'], '--run-root', str(run),
                          '--data-dir', str(data), '--device', args.device,
                          '--prepare-reference', '--stages', *args.stages]
        if c['family'] == 'two_step' and stage is not None:
            command += ['--stage1-checkpoint', str(stage)]
        if args.resume and (run/c['condition']/'checkpoints/latest.json').exists():
            command += ['--resume']
        execute(command)
    if 'visualize' in args.stages:
        execute(finish)
    else:
        execute(base + ['summarize', '--run-root', str(run)])


if __name__ == '__main__':
    main()
