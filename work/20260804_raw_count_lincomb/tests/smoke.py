#!/usr/bin/env python3
"""Short train/interruption/resume/sample/UMAP smoke tests for this suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parent.parent
SCRIPTS = SUITE_ROOT / "scripts"
for search_path in (REPO_ROOT, SUITE_ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from common import (  # noqa: E402
    EXPERIMENT_ORDER,
    experiment_config_path,
    latest_checkpoint_bundle,
    now_iso,
    read_json,
    run_dir_for,
    validate_path_component,
    write_json,
)


def _fixture(batch_id: str) -> tuple[Path, Path]:
    import anndata as ad
    import pandas as pd

    fixture_dir = SUITE_ROOT / "smoke_results" / batch_id / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    h5ad = fixture_dir / "synthetic_raw_count.h5ad"
    edges = fixture_dir / "edges.tsv"
    if not h5ad.exists():
        rng = np.random.default_rng(20260804)
        matrix = rng.poisson(2.0, size=(16, 5)).astype(np.float32)
        obs = pd.DataFrame(
            {"celltype": ["A", "B"] * 8, "Superclass": ["class-A", "class-B"] * 8},
            index=[f"cell_{index:02d}" for index in range(16)],
        )
        var = pd.DataFrame(
            {"gene_name": ["source", "target", "g2", "g3", "g4"]},
            index=[f"var_{index}" for index in range(5)],
        )
        ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(h5ad)
    if not edges.exists():
        edges.write_text("from\tto\nsource\ttarget\ntarget\tg2\ng2\tg3\ng3\tg4\n", encoding="utf-8")
    return h5ad.resolve(), edges.resolve()


def _run(command: list[str], log_path: Path, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        result = subprocess.run(
            command, cwd=REPO_ROOT, env=env,
            stdout=handle, stderr=subprocess.STDOUT, check=False,
        )
    return int(result.returncode)


def _train_command(experiment: str, run_dir: Path, data: Path, edges: Path) -> list[str]:
    overrides = {
        "data_dir": str(data), "edge_tsv_path": str(edges),
        "field_hidden": 16, "time_dim": 8,
        "cell_unet_hidden_num": [32, 24, 16, 16],
        "diffusion_steps": 20, "noise_schedule": "cosine",
        "total_steps": 4, "lr_anneal_steps": 4, "batch_size": 4,
        "save_interval": 2, "log_interval": 1,
        "num_samples": 4, "sample_batch_size": 2, "device": "cpu",
        "umap_real_cells": 4, "umap_n_neighbors": 3, "umap_n_pcs": 3,
    }
    command = [
        sys.executable, str(SCRIPTS / "train.py"),
        "--config", str(experiment_config_path(experiment)),
        "--run-dir", str(run_dir),
    ]
    for key, value in overrides.items():
        command.extend(["--set", f"{key}={json.dumps(value)}"])
    return command


def run_one(experiment: str, batch_id: str) -> dict:
    data, edges = _fixture(batch_id)
    run_dir = run_dir_for(experiment, batch_id)
    result_dir = SUITE_ROOT / "smoke_results" / batch_id / experiment
    report_path = result_dir / "report.json"
    if run_dir.exists():
        raise FileExistsError(f"choose a new --batch-id; run already exists: {run_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "experiment": experiment, "batch_id": batch_id,
        "run_dir": str(run_dir), "started_at": now_iso(), "checks": {},
    }
    write_json(report_path, report)
    train = _train_command(experiment, run_dir, data, edges)
    try:
        interrupt_env = os.environ.copy()
        interrupt_env["DIFFUSION_TRAINING_TEST"] = "1"
        if _run(train, result_dir / "train_interrupted.log", interrupt_env) == 0:
            raise AssertionError("intentional first segment unexpectedly completed")
        step2 = latest_checkpoint_bundle(run_dir, "0.9999", require_complete=True)
        if step2 is None or int(step2["step"]) != 2:
            raise AssertionError(f"expected complete step-2 checkpoint bundle, got {step2}")
        report["checks"]["interrupted_with_complete_step2_bundle"] = True

        if _run([*train, "--resume", "auto"], result_dir / "train_resumed.log", os.environ.copy()) != 0:
            raise RuntimeError("resume training failed")
        step4 = latest_checkpoint_bundle(run_dir, "0.9999", require_complete=True)
        if step4 is None or int(step4["step"]) != 4:
            raise AssertionError(f"expected complete step-4 checkpoint bundle, got {step4}")
        report["checks"]["resume_completed_at_step4"] = True

        sample_command = [
            sys.executable, str(SCRIPTS / "sample.py"), "--run-dir", str(run_dir),
            "--num-samples", "4", "--batch-size", "2", "--device", "cpu",
        ]
        if _run(sample_command, result_dir / "sample.log", os.environ.copy()) != 0:
            raise RuntimeError("sampling failed")
        manifest = read_json(run_dir / "manifest.json")
        with np.load(manifest["sample_path"], allow_pickle=False) as archive:
            generated = np.asarray(archive["cell_gen"])
        if generated.shape != (4, 5) or not np.isfinite(generated).all():
            raise AssertionError(f"invalid sample: {generated.shape}")
        report["checks"]["strict_restore_and_finite_sampling"] = True

        umap_command = [sys.executable, str(SCRIPTS / "umap.py"), "--run-dir", str(run_dir)]
        if _run(umap_command, result_dir / "umap.log", os.environ.copy()) != 0:
            raise RuntimeError("UMAP failed")
        manifest = read_json(run_dir / "manifest.json")
        figure = Path(manifest["stages"]["umap"]["figure"])
        if not figure.is_file():
            raise FileNotFoundError(figure)
        report["checks"]["raw_space_umap_created"] = True
        report.update({"status": "completed", "finished_at": now_iso(), "figure": str(figure)})
        write_json(report_path, report)
        return report
    except BaseException as exc:
        report.update({
            "status": "failed", "failed_at": now_iso(),
            "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
        })
        write_json(report_path, report)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--experiments", nargs="*", choices=EXPERIMENT_ORDER)
    parser.add_argument("--batch-id", default=f"smoke_{datetime.now():%Y%m%d_%H%M%S}")
    args = parser.parse_args()
    batch_id = validate_path_component(args.batch_id, "batch id")
    experiments = args.experiments or list(EXPERIMENT_ORDER)
    test_file = Path(__file__).with_name("test_raw_count_suite.py")
    if subprocess.run([sys.executable, "-m", "unittest", "-v", str(test_file)], cwd=REPO_ROOT).returncode:
        return 1
    reports = [run_one(experiment, batch_id) for experiment in experiments]
    print(json.dumps(reports, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
