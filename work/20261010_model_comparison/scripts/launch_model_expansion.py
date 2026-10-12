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
    p.add_argument('--retry-failed-sampling', action='store_true',
                   help='retry a recorded nonfinite sampling failure for the same checkpoint')
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
            if stage is None and (run/'shared_stage1.json').exists():
                stage = Path(C.read_json(run/'shared_stage1.json')['checkpoint'])
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
    progress_path = run/'campaign_progress.json'
    progress = C.read_json(progress_path) if progress_path.exists() else {}

    def record(name, action, status, **details):
        progress.setdefault(name, {})[action] = dict(status=status, **details)
        if not args.dry_run:
            C.write_json(progress_path, progress)

    for c in configs:
        name = c['condition']; folder = run/name
        command = base + ['launch', '--condition', name, '--run-root', str(run),
                          '--data-dir', str(data), '--device', args.device, '--prepare-reference']
        for action in args.stages:
            scopes = ['training','samples','velocity'] if action=='visualize' else [None]
            for scope in scopes:
                key = f'visualize_{scope}' if scope else action
                if scope=='samples' and not args.dry_run and not (folder/'samples/generated.h5ad').exists():
                    record(name,key,'unavailable',reason='no finite generated samples')
                    continue
                failure = folder/'samples/nonfinite_failure.json'
                pointer = folder/'checkpoints/latest.json'
                checkpoint_hash = C.read_json(pointer).get('sha256') if pointer.exists() else None
                if action=='sample' and failure.exists() and not args.retry_failed_sampling and not args.dry_run:
                    previous = C.read_json(failure)
                    if previous.get('status')=='failed_nonfinite' and checkpoint_hash and previous.get('checkpoint_sha256')==checkpoint_hash and not (folder/'samples/generated.h5ad').exists():
                        record(name,key,'failed_nonfinite',failure_log=previous['failure_log'],reused_failure=True)
                        print(f'{name}: recorded nonfinite generation; continue velocity evaluation',flush=True)
                        continue
                call = command + ['--stages',action]
                if scope:call += ['--scope',scope]
                if action=='train':
                    if c['family']=='two_step' and stage is not None:
                        call += ['--stage1-checkpoint',str(stage)]
                    if args.resume and pointer.exists():call += ['--resume']
                before = set((folder/'logs').glob('failure_sample_*.json')) if action=='sample' else set()
                try:
                    execute(call)
                except subprocess.CalledProcessError:
                    # Only this exact numerical sampling failure is recoverable.
                    # Source/checkpoint/data errors and other stage failures still stop.
                    new_logs = set((folder/'logs').glob('failure_sample_*.json')) - before
                    matching = [p for p in new_logs if any(
                        line.strip()=='FloatingPointError: ancestral state contains NaN/Inf'
                        for line in C.read_json(p).get('traceback','').splitlines())]
                    if action!='sample' or len(matching)!=1:
                        record(name,key,'failed');raise
                    info = dict(status='failed_nonfinite',condition=name,checkpoint_sha256=checkpoint_hash,
                                failure_log=str(matching[0]),sampler='native ancestral DDPM; clip_denoised=False')
                    C.write_json(failure,info)
                    record(name,key,'failed_nonfinite',failure_log=str(matching[0]))
                    print(f'{name}: nonfinite generation recorded; continue velocity evaluation',flush=True)
                else:
                    record(name,key,'complete' if not args.dry_run else 'planned')
                    if action=='sample' and failure.exists() and not args.dry_run:
                        previous=C.read_json(failure);previous['status']='resolved'
                        C.write_json(failure,previous)
    if 'visualize' in args.stages:
        execute(finish + ['--allow-missing'])
    else:
        execute(base + ['summarize', '--run-root', str(run)])
    if not args.dry_run:
        failed = sorted(name for name, stages in progress.items()
                        if stages.get('sample',{}).get('status')=='failed_nonfinite')
        if failed:
            print('Campaign continued with unavailable generated samples: '+', '.join(failed)+
                  '; see campaign_progress.json. These conditions did not pass generation.',flush=True)


if __name__ == '__main__':
    main()
