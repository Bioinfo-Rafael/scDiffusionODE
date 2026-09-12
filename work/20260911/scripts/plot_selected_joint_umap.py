#!/usr/bin/env python3
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401 -- establishes repository imports

from src.artifacts import result_path
from src.umap_adapter import plot_selected

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshots", type=lambda value: [int(x) for x in value.split(",")])
    selection.add_argument("--reverse-steps", type=lambda value: [int(x) for x in value.split(",")])
    args = parser.parse_args()
    plot_selected(
        result_path(args.result_dir),
        snapshots=args.snapshots,
        reverse_steps=args.reverse_steps,
        data_path=args.data_path,
    )
