#!/usr/bin/env python3
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401 -- establishes repository imports

from src.artifacts import result_path
from src.umap_adapter import plot_independent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument(
        "--data-path", type=Path, help="Explicit relocated real data; exact gene order must match"
    )
    args = parser.parse_args()
    plot_independent(result_path(args.result_dir), data_path=args.data_path)
