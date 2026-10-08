#!/usr/bin/env python3
"""Byte-for-byte audit of pre-existing tracked and untracked nonignored files."""
import argparse
import subprocess
from common import *

def snapshot():
    files=set(subprocess.check_output(['git','ls-files'],cwd=REPO,text=True).splitlines())
    files.update(subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=REPO,text=True).splitlines())
    return {name:sha256(REPO/name) for name in sorted(files)
            if not name.startswith('work/20261008_ode_optimizer/') and (REPO/name).is_file()}

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--record-remote-baseline',action='store_true',help='Record once before running on a different checkout; never overwrite')
    a=p.parse_args(argv)
    local=SUITE/'validation/protected_files.json'; remote=SUITE/'validation/protected_files_remote.json'
    if a.record_remote_baseline:
        if remote.exists(): raise FileExistsError(remote)
        write_json(remote,snapshot()); print('Recorded remote baseline:',remote); return
    manifest=read_json(remote if remote.exists() else local)
    current=snapshot()
    failures=[name for name,digest in manifest.items() if current.get(name)!=digest]
    added=sorted(set(current)-set(manifest))
    if failures or added: raise SystemExit('Protected files changed/added/missing:\n'+'\n'.join(failures+added))
    print(f'PASS: {len(manifest)} pre-existing files unchanged; no files added outside the suite')
if __name__=='__main__': main()
