#!/usr/bin/env python3
"""Read-only run inventory; isolate source suite imports in subprocesses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from _bootstrap import ROOT

from src.artifacts import result_path, write_json
from src.model_runs import discover_suite
from src.source_imports import SUITES


def discover_all() -> dict:
    reports = {}
    for suite in SUITES:
        completed = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--suite", suite],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        reports[suite] = json.loads(completed.stdout)
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--suite", choices=list(SUITES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = discover_suite(args.suite) if args.suite else discover_all()
    if args.output:
        write_json(result_path(args.output), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
