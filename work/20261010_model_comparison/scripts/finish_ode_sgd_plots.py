"""Finish D's existing run: normal/Harmony UMAPs, PNG montages, metric plots."""
import argparse
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root',type=Path,required=True)
    isolated=HERE/'runs/.venv-harmony/bin/python'
    p.add_argument('--harmony-python',default=str(isolated) if isolated.exists() else sys.executable)
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();run=args.run_root.resolve()
    def call(command):subprocess.run(command,cwd=ROOT,check=True)
    call([args.harmony_python,'-c',
          'import scanpy, anndata, PIL, harmonypy; import importlib.metadata as m; '
          'assert m.version("harmonypy")=="0.0.10", "Install harmonypy==0.0.10 in the plotting environment"'])
    if args.check_only:return
    base=[sys.executable,'-m','work.20261010_model_comparison.cli']
    # Existing normal UMAPs are preserved; missing ones use saved generated cells.
    for cfg in sorted((HERE/'configs/ode_sgd').glob('D*.json')):
        folder=run/cfg.stem/'figures/generative'
        if not all((folder/name).is_file() for name in ('joint_umap.png','joint_umap.h5ad')):
            call(base+['visualize','--condition',str(cfg),'--run-root',str(run),'--scope','samples'])
    call(base+['summarize','--run-root',str(run)])
    call([sys.executable,str(HERE/'scripts/replot_comparison.py'),str(run/'comparison/comparison.csv')])
    call([args.harmony_python,str(HERE/'scripts/replot_harmony.py'),'--run-root',str(run),'--condition-set','ode_sgd'])
    call([args.harmony_python,str(HERE/'scripts/combine_harmony_pngs.py'),'--run-root',str(run),'--condition-set','ode_sgd','--normal'])


if __name__=='__main__':main()
