#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
suite=Path(__file__).resolve().parents[1]; repo=suite.parent.parent; baseline=json.loads((suite/"protected_baseline.json").read_text())
status=subprocess.run(["git","status","--porcelain=v1","--untracked-files=all"],cwd=repo,text=True,stdout=subprocess.PIPE,check=True).stdout.splitlines(); prefix=str(suite.relative_to(repo))+"/"; outside=[x for x in status if not x[3:].startswith(prefix)]
checks={}
for name,expected in baseline["preexisting_modified_file_sha256"].items():
    checks[name]=hashlib.sha256((repo/name).read_bytes()).hexdigest()==expected
result={"ok":sorted(outside)==sorted(baseline["preexisting_git_status"]) and all(checks.values()),"outside":outside,"hash_matches":checks}; print(json.dumps(result,indent=2))
if not result["ok"]:raise SystemExit(1)
