#!/usr/bin/env python3
"""Create a minimal, reproducible numerical summary of generated samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run = Path(args.run_dir).resolve()
    suite = Path(__file__).resolve().parent.parent
    run.relative_to((suite / "runs").resolve())
    samples = sorted((run / "samples").glob("samples_*.npz"))
    if not samples:
        raise FileNotFoundError("no generated sample archive found")
    with np.load(samples[-1]) as archive:
        values = archive["cell_gen"]
    summary = {
        "sample_path": str(samples[-1]),
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }
    destination = run / "analysis" / "sample_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"ANALYSIS_PATH='{destination}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
