"""Capture/verify protected tracked files without changing the historical audit baseline."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--capture',type=Path);g.add_argument('--baseline',type=Path,default=HERE/'audit/protected_files.json')
    args=p.parse_args()
    if args.capture:
        target=args.capture.resolve()
        if not target.is_relative_to(HERE/'runs'):raise ValueError('capture must be in this suite runs/')
        paths=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
        values={path:digest(ROOT/path) for path in paths if not path.startswith('work/20261010_model_comparison/') and (ROOT/path).is_file()}
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('x') as f:json.dump(values,f,indent=2)
        print(f'Captured {len(values)} protected files: {target}')
    else:
        values=json.loads(args.baseline.read_text())
        changed=[path for path,h in values.items() if not (ROOT/path).is_file() or digest(ROOT/path)!=h]
        print(json.dumps(dict(checked=len(values),changed=changed),indent=2))
        if changed:raise SystemExit(1)
if __name__=='__main__':main()
