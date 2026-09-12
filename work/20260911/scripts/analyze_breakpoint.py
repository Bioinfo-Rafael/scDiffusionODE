#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401 -- establishes repository imports

from src.artifacts import result_path
from src.breakpoint import analyze

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(result_path(args.result_dir)), indent=2))
