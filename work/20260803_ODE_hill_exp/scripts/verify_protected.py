#!/usr/bin/env python3
"""Fail if this suite changed anything outside its own directory.

The repository was already dirty when work began.  That one notebook is
accepted only while its byte hash remains identical to the captured baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parents[1]
BASELINE_PATH = SUITE_ROOT / "protected_baseline.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_status() -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    suite_rel = str(SUITE_ROOT.relative_to(REPO_ROOT)) + "/"
    expected_dirty = set(baseline["preexisting_git_status"])
    observed = git_status()
    outside = [line for line in observed if not line[3:].startswith(suite_rel)]
    unexpected = sorted(set(outside).difference(expected_dirty))
    missing = sorted(expected_dirty.difference(outside))

    hash_checks = {}
    for relative, expected in baseline["preexisting_modified_file_sha256"].items():
        path = REPO_ROOT / relative
        actual = sha256(path) if path.exists() else None
        hash_checks[relative] = {
            "expected": expected,
            "actual": actual,
            "matched": actual == expected,
        }

    protected_diff = subprocess.run(
        [
            "git", "diff", "--name-only", "--",
            "guided_diffusion", "ODE", "work/202606*", "work/202607*",
            "work/20260801*", "work/20260802*",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    # The notebook was dirty before this suite.  Its unchanged hash is the
    # authoritative check; it remains visible here for a transparent report.
    allowed_paths = set(baseline["preexisting_modified_file_sha256"])
    unexpected_protected_diff = sorted(set(protected_diff).difference(allowed_paths))

    ok = (
        not unexpected
        and not missing
        and not unexpected_protected_diff
        and all(item["matched"] for item in hash_checks.values())
    )
    report = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "ok": ok,
        "suite_root": str(SUITE_ROOT),
        "outside_suite_status": outside,
        "unexpected_outside_status": unexpected,
        "missing_preexisting_status": missing,
        "protected_git_diff": protected_diff,
        "unexpected_protected_diff": unexpected_protected_diff,
        "preexisting_hash_checks": hash_checks,
    }
    if args.write_report:
        output = SUITE_ROOT / "validation" / "protected_diff_report.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
