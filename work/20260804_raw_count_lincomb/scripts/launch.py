#!/usr/bin/env python3
"""Ordered six-condition raw-count launcher."""

from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent; SUITE_ROOT=HERE.parent; REPO_ROOT=SUITE_ROOT.parent.parent
for path in (SUITE_ROOT,HERE): sys.path.insert(0,str(path))
from common import EXPERIMENT_ORDER, CONFIG_ROOT, run_dir_for, select_experiments

def commands(experiment,batch,args):
    run=run_dir_for(experiment,batch); config=CONFIG_ROOT/f"{experiment}.json"; py=sys.executable
    if args.sample_only:return [[py,str(HERE/"sample.py"),"--run-dir",str(run)]]
    if args.umap_only:return [[py,str(HERE/"umap.py"),"--run-dir",str(run)]]
    train=[py,str(HERE/"train.py"),"--config",str(config),"--run-dir",str(run)]
    if args.resume:train += ["--resume",args.resume]
    return [train,[py,str(HERE/"sample.py"),"--run-dir",str(run)],[py,str(HERE/"umap.py"),"--run-dir",str(run)]]
def main():
    p=argparse.ArgumentParser(allow_abbrev=False); p.add_argument("--batch-id",default="20260804_full_30000"); p.add_argument("--family",action="append",default=[]); p.add_argument("--ode",action="append",default=[]); p.add_argument("--experiment",action="append",nargs="+",default=[])
    mode=p.add_mutually_exclusive_group(); mode.add_argument("--sample-only",action="store_true"); mode.add_argument("--umap-only",action="store_true"); mode.add_argument("--smoke",action="store_true"); mode.add_argument("--data-only",action="store_true")
    p.add_argument("--resume",nargs="?",const="auto",default=""); p.add_argument("--dry-run",action="store_true"); args=p.parse_args()
    if args.data_only:
        cmd=[sys.executable,str(HERE/"create_raw_count_data.py")]; print(" ".join(cmd)) if args.dry_run else subprocess.run(cmd,cwd=REPO_ROOT,check=True); return
    requested=[x for group in args.experiment for x in group]; selected=select_experiments(args.family,requested)
    if args.ode: selected=tuple(x for x in selected if x.split("__",1)[1] in set(args.ode))
    if not selected: raise ValueError("filters select no experiments")
    if args.smoke:
        cmd=[sys.executable,str(SUITE_ROOT/"tests/smoke.py"),"--experiments",*selected]; print(" ".join(cmd)) if args.dry_run else subprocess.run(cmd,cwd=REPO_ROOT,check=True); return
    plan=[(exp,commands(exp,args.batch_id,args)) for exp in selected]
    if args.dry_run: print(json.dumps([{"experiment":e,"commands":[" ".join(c) for c in cs]} for e,cs in plan],indent=2)); return
    for experiment,cmds in plan:
        print(f"[{experiment}]")
        for cmd in cmds: subprocess.run(cmd,cwd=REPO_ROOT,check=True)
if __name__=="__main__":main()
