"""Finish comparison metrics and normal/Harmony UMAPs for a complete condition set."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--data-dir', type=Path, default=HERE/'runs/data')
    p.add_argument('--condition-set', choices=['pilot','ode_sgd','model_expansion'], default='model_expansion')
    isolated = HERE/'runs/.venv-harmony/bin/python'
    p.add_argument('--harmony-python', default=str(isolated) if isolated.exists() else sys.executable)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--allow-missing', action='store_true', help='mark unavailable generation plots instead of requiring every condition')
    args = p.parse_args(argv)
    run = args.run_root.resolve()
    def call(command):
        subprocess.run(command, cwd=ROOT, check=True)
    call([args.harmony_python, '-c',
          'import scanpy, anndata, PIL, harmonypy; import importlib.metadata as m; '
          'assert m.version("harmonypy")=="0.0.10", "Install harmonypy==0.0.10 in the plotting environment"'])
    if args.check_only:
        return
    base = [sys.executable, '-m', 'work.20261010_model_comparison.cli']
    unavailable=[]
    for cfg in sorted((HERE/'configs'/args.condition_set).glob('*.json')):
        folder = run/cfg.stem/'figures/generative'
        if not all((folder/name).is_file() for name in ('joint_umap.png','joint_umap.h5ad')):
            if args.allow_missing and not (run/cfg.stem/'samples/generated.h5ad').exists():
                unavailable.append(cfg.stem)
                continue
            call(base + ['visualize','--condition',str(cfg),'--run-root',str(run),
                         '--data-dir',str(args.data_dir.resolve()),'--scope','samples'])
    call(base + ['summarize','--run-root',str(run)])
    csv=run/'comparison/comparison.csv'
    if not args.allow_missing or (csv.exists() and len(csv.read_text().splitlines())>1):
        call([sys.executable,str(HERE/'scripts/replot_comparison.py'),str(csv)])
    optional=['--allow-missing'] if args.allow_missing else []
    call([args.harmony_python,str(HERE/'scripts/replot_harmony.py'),
          '--run-root',str(run),'--condition-set',args.condition_set] + optional)
    call([args.harmony_python,str(HERE/'scripts/combine_harmony_pngs.py'),
          '--run-root',str(run),'--condition-set',args.condition_set,'--normal'] + optional)
    status=run/'comparison/generative_plot_status.json'
    status.parent.mkdir(parents=True,exist_ok=True)
    status.write_text(json.dumps(dict(unavailable_conditions=unavailable,
                     reason='no finite generated samples; see campaign_progress.json and condition logs'),indent=2)+'\n')


if __name__ == '__main__':
    main()
