#!/usr/bin/env python3
"""Sequential train/sample/analyze/UMAP pipeline; dry-run never writes files."""
import argparse
import shlex
import subprocess
from common import *


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--batch-id',required=True)
    p.add_argument('--condition',choices=list(CONDITIONS),action='append')
    p.add_argument('--stage',choices=['all','train','sample','analysis','umap','comparison'],default='all')
    p.add_argument('--set',action='append',default=[],dest='overrides')
    p.add_argument('--resume',action='store_true'); p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    import re
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',a.batch_id): p.error('Invalid batch ID')
    names=list(dict.fromkeys(a.condition or CONDITIONS)); campaign=SUITE/'runs'/a.batch_id
    configs={n:config_for(n,a.overrides) for n in names}
    for n in names:
        existing=campaign/n/'exp_config.json'
        if existing.exists() and not a.overrides:
            configs[n]=validate(read_json(existing))
    signatures=[{k:v for k,v in c.items() if k not in ('experiment','cell_ode_reg_lambda_20260830')} for c in configs.values()]
    if any(s!=signatures[0] for s in signatures[1:]):
        raise ValueError('All conditions must have identical settings apart from lambda/name')
    commands=[]
    def command(script,*args): commands.append([sys.executable,'-B',str(SUITE/'scripts'/script),*map(str,args)])
    if a.stage in ('all','train'):
        for n in names:
            command('train.py','--config',campaign/n/'exp_config.json','--run-dir',campaign/n,*(['--resume'] if a.resume else []))
    if a.stage in ('all','sample'):
        for n in names: command('sample.py','--run-dir',campaign/n)
    if a.stage in ('all','analysis'):
        if len(names)==4: command('analyze.py','--campaign',campaign)
        else:
            for n in names: command('analyze.py','--run-dir',campaign/n)
    if a.stage in ('all','umap'):
        if len(names)==4 or all((campaign/n/'samples/sampling.json').exists() for n in CONDITIONS):
            command('plot_umap.py','--campaign',campaign)
        elif a.stage=='umap': p.error('Shared UMAP requires samples from all four conditions')
    if a.stage in ('all','comparison'):
        if len(names)==4 or (campaign/'umap/coordinates.npz').exists():
            command('plot_comparison.py','--campaign',campaign)
        elif a.stage=='comparison': p.error('Comparison requires saved shared UMAP coordinates')
    print(json.dumps(dict(campaign=str(campaign),conditions=names,configs=configs,
                          commands=[shlex.join(cmd) for cmd in commands]),indent=2))
    if a.dry_run: return
    for n,c in configs.items():
        path=campaign/n/'exp_config.json'
        if path.exists():
            if a.overrides and read_json(path)!=c: raise ValueError(f'Config mismatch: {path}')
        elif a.stage in ('all','train'): write_json(path,c)
        else: raise FileNotFoundError(path)
    for cmd in commands:
        subprocess.run(cmd,cwd=REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),check=True)

if __name__=='__main__': main()
