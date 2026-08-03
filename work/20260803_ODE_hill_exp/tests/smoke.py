#!/usr/bin/env python3
"""Four-step train/resume/sample/analyze integration test for one condition."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np


SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parent.parent
SCRIPTS = SUITE_ROOT / "scripts"
for search_path in (SUITE_ROOT, SCRIPTS):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from common import (  # noqa: E402
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
    h5ad = fixture_dir / "synthetic.h5ad"
    edges = fixture_dir / "edges.tsv"
    if not h5ad.exists():
        rng = np.random.default_rng(20260803)
        matrix = rng.lognormal(mean=0.1, sigma=0.45, size=(16, 5)).astype(np.float32)
        obs = pd.DataFrame({
            "celltype": ["A", "B"] * 8,
            "Superclass": ["class-A", "class-B"] * 8,
        }, index=[f"cell_{index:02d}" for index in range(16)])
        var = pd.DataFrame({
            "gene_name": ["source", "target", "g2", "g3", "g4"],
        }, index=[f"var_{index}" for index in range(5)])
        ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(h5ad)
    if not edges.exists():
        edges.write_text(
            "from\tto\nsource\ttarget\ntarget\tg2\ng2\tg3\ng3\tg4\n",
            encoding="utf-8",
        )
    return h5ad.resolve(), edges.resolve()


def _run(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def _overrides(experiment: str, data_path: Path, edge_path: Path) -> list[str]:
    values = {
        "data_dir": str(data_path),
        "edge_tsv_path": str(edge_path),
        "field_hidden": 16,
        "time_dim": 8,
        "cell_unet_hidden_num": [32, 24, 16, 16],
        "diffusion_steps": 20,
        "noise_schedule": "cosine",
        "total_steps": 4,
        "lr_anneal_steps": 4,
        "batch_size": 4,
        "save_interval": 2,
        "log_interval": 1,
        "num_samples": 4,
        "sample_batch_size": 2,
        "analysis_max_cells": 4,
        "analysis_batch_size": 2,
        "analysis_t_values": "0,9,10,19",
        "ts_n_cells": 16,
        "device": "cpu",
    }
    if experiment.startswith("ts_soft_tau80_hybrid_lincomb__"):
        values["t_s"] = 10
    return [
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in values.items()
    ]


def run_smoke(experiment: str, batch_id: str) -> dict:
    data_path, edge_path = _fixture(batch_id)
    run_dir = run_dir_for(experiment, batch_id)
    result_dir = SUITE_ROOT / "smoke_results" / batch_id / experiment
    report_path = result_dir / "report.json"
    result_dir.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        raise FileExistsError(
            f"smoke run already exists; choose a new --batch-id: {run_dir}"
        )

    overrides = _overrides(experiment, data_path, edge_path)
    train = [
        sys.executable, str(SCRIPTS / "train.py"),
        "--config", str(experiment_config_path(experiment)),
        "--run-dir", str(run_dir),
    ]
    for value in overrides:
        train.extend(["--set", value])
    report = {
        "status": "running",
        "experiment": experiment,
        "batch_id": batch_id,
        "run_dir": str(run_dir),
        "started_at": now_iso(),
        "fixture": {"data": str(data_path), "edges": str(edge_path)},
        "checks": {},
    }
    write_json(report_path, report)
    try:
        interrupt_env = os.environ.copy()
        interrupt_env["DIFFUSION_TRAINING_TEST"] = "1"
        first_code = _run(train, result_dir / "train_interrupted.log", env=interrupt_env)
        report["checks"]["intentional_interruption_nonzero"] = first_code != 0
        if first_code == 0:
            raise AssertionError("intentional short first segment unexpectedly returned success")
        first_bundle = latest_checkpoint_bundle(run_dir, "0.9999", require_complete=True)
        if first_bundle is None or int(first_bundle["step"]) != 2:
            raise AssertionError(f"expected complete step-2 bundle, got {first_bundle}")
        report["checks"]["step2_raw_optimizer_ema_bundle"] = True

        resume = [*train, "--resume", "auto"]
        second_code = _run(resume, result_dir / "train_resumed.log", env=os.environ.copy())
        if second_code != 0:
            raise RuntimeError(f"resume training failed; see {result_dir / 'train_resumed.log'}")
        final_bundle = latest_checkpoint_bundle(run_dir, "0.9999", require_complete=True)
        if final_bundle is None or int(final_bundle["step"]) != 4:
            raise AssertionError(f"expected complete step-4 bundle, got {final_bundle}")
        report["checks"]["step4_resume_completed"] = True

        sample = [
            sys.executable, str(SCRIPTS / "sample.py"), "--run-dir", str(run_dir),
            "--num-samples", "4", "--batch-size", "2", "--device", "cpu",
        ]
        if _run(sample, result_dir / "sample.log", env=os.environ.copy()) != 0:
            raise RuntimeError(f"sampling failed; see {result_dir / 'sample.log'}")
        manifest = read_json(run_dir / "manifest.json")
        sample_path = Path(manifest["sample_path"])
        with np.load(sample_path, allow_pickle=False) as archive:
            generated = np.asarray(archive["cell_gen"])
        if generated.shape != (4, 5) or not np.isfinite(generated).all():
            raise AssertionError(f"invalid generated array: shape={generated.shape}")
        report["checks"]["strict_restore_and_finite_sampling"] = True

        analyze = [
            sys.executable, str(SCRIPTS / "analyze.py"), "--run-dir", str(run_dir),
            "--max-cells", "4", "--batch-size", "2",
            "--t-values", "0,9,10,19", "--device", "cpu",
        ]
        if _run(analyze, result_dir / "analysis.log", env=os.environ.copy()) != 0:
            raise RuntimeError(f"analysis failed; see {result_dir / 'analysis.log'}")
        manifest = read_json(run_dir / "manifest.json")
        analysis_dir = Path(manifest["analysis_path"])
        analysis_manifest = read_json(analysis_dir / "analysis_manifest.json")
        created = analysis_manifest.get("created_files", [])
        if analysis_manifest.get("status") != "completed" or not created:
            raise AssertionError("analysis manifest is incomplete")
        for relative in created:
            if not (analysis_dir / relative).is_file():
                raise FileNotFoundError(analysis_dir / relative)
        report["checks"]["finite_analysis_and_plots"] = True
        report.update({
            "status": "completed",
            "finished_at": now_iso(),
            "checkpoint": final_bundle,
            "sample_path": str(sample_path),
            "analysis_path": str(analysis_dir),
            "created_plot_count": len(created),
        })
        write_json(report_path, report)
        return report
    except BaseException as exc:
        report.update({
            "status": "failed",
            "failed_at": now_iso(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })
        write_json(report_path, report)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--batch-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    batch_id = validate_path_component(args.batch_id, "batch id")
    report = run_smoke(args.experiment, batch_id)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
