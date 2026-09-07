#!/usr/bin/env python3
"""Verify that this analysis changed no file outside its own suite."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SUITE_ROOT.parent.parent
BASELINE = json.loads((SUITE_ROOT / "protected_baseline.json").read_text())


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


status = subprocess.run(
    ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=REPO_ROOT,
    check=True, text=True, stdout=subprocess.PIPE,
).stdout.splitlines()
prefix = str(SUITE_ROOT.relative_to(REPO_ROOT)) + "/"
outside = [line for line in status if not line[3:].startswith(prefix)]
expected = BASELINE["preexisting_git_status"]
hashes = {
    name: digest(REPO_ROOT / name) == expected_hash
    for name, expected_hash in BASELINE["preexisting_modified_file_sha256"].items()
}
report = {
    "ok": sorted(outside) == sorted(expected) and all(hashes.values()),
    "outside_suite_status": outside, "expected_outside_status": expected,
    "preexisting_hash_matches": hashes,
}
print(json.dumps(report, indent=2, sort_keys=True))
if not report["ok"]:
    raise SystemExit(1)
