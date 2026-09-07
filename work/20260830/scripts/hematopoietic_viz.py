#!/usr/bin/env python3
"""Run independent post-hoc hematopoietic visualization for one completed run."""

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
from hematopoietic_viz.runner import run_visualization  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sample-path", default="")
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    result = run_visualization(args.run_dir, options_from_args(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
