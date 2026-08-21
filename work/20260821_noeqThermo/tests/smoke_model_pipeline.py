#!/usr/bin/env python3
"""Full synthetic run fixture: checkpoint -> fixed UMAP -> Dynamo -> landscape."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent
REPO_ROOT = ANALYSIS.parent.parent
PRIOR_TESTS = REPO_ROOT / "work" / "20260817_vector_field_analysis" / "tests"
sys.path.insert(0, str(PRIOR_TESTS))

from smoke import _fixture  # type: ignore

REQUIRED = {
    "analysis_manifest.json",
    "common/erythropoietic_fixed_umap.csv",
    "common/sde_calibration.json",
    "model_comparison_summary.csv",
}


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="noeq-model-smoke-") as temporary:
        root = Path(temporary)
        run_dir, _ = _fixture(root)
        output_dir = root / "noeq_outputs"
        command = [
            sys.executable,
            str(ANALYSIS / "scripts" / "run_analysis.py"),
            "--mode",
            "smoke",
            "--run-dir",
            str(run_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(f"model pipeline smoke failed:\n{completed.stdout}")
        missing = sorted(name for name in REQUIRED if not (output_dir / name).is_file())
        model_dirs = sorted((output_dir / "models").iterdir())
        if len(model_dirs) != 1:
            raise AssertionError(f"expected one model output, found {model_dirs}")
        for name in (
            "observed_umap_velocity.npz",
            "dynamo_vecfld_umap.npz",
            "landscape_flux_arrays.npz",
            "01_umap_vector_field.png",
            "09_landscape_flux_overlay.png",
            "model_manifest.json",
        ):
            if not (model_dirs[0] / name).is_file():
                missing.append(f"models/*/{name}")
        if missing:
            raise AssertionError(f"missing smoke outputs: {missing}")
        manifest = json.loads((output_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("model_count") != 1:
            raise AssertionError(f"invalid completed manifest: {manifest}")
        embedding = manifest["common_embedding"]
        if embedding["umap_fit_count"] != 1 or embedding["selected_cells"] != 60:
            raise AssertionError(f"shared Erythropoietic UMAP invariant failed: {embedding}")
        model_manifest = json.loads((model_dirs[0] / "model_manifest.json").read_text(encoding="utf-8"))
        if model_manifest["projection"]["coordinates_sha256"] != embedding["coordinates_sha256"]:
            raise AssertionError("model did not use the common fixed UMAP coordinates")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "training_executed": False,
                    "sampling_executed": False,
                    "cells": embedding["selected_cells"],
                    "genes": embedding["n_genes"],
                    "model": model_manifest["name"],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
