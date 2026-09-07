#!/usr/bin/env python3
"""Run hematopoietic visualization for sample-complete runs in one batch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hematopoietic_viz.cli import add_common_arguments, options_from_args  # noqa: E402
from hematopoietic_viz.runner import run_all_available  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", required=True)
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    summary = run_all_available(args.batch_id, options_from_args(args))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
