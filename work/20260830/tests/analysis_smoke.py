#!/usr/bin/env python3
"""End-to-end synthetic run smoke test for detailed analysis."""

from __future__ import annotations

import sys
import tempfile
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch

SUITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.runner import AnalysisOptions, analyze_run  # noqa: E402
from guided_diffusion.script_util import create_gaussian_diffusion  # noqa: E402
from models import build_model_from_config  # noqa: E402
from scripts.common import load_experiment_config, write_json  # noqa: E402


def main():
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as directory:
        run = Path(directory) / "01_centered_signed_hill_lambda0p1" / "smoke"
        run.mkdir(parents=True)
        data_path = Path(directory) / "cells.h5ad"
        edge_path = Path(directory) / "edges.tsv"
        edge_path.write_text("from\tto\ng0\tg1\n", encoding="utf-8")
        adata = ad.AnnData(X=(np.random.default_rng(7).random((12, 5)) + 0.1).astype(np.float32))
        adata.var["gene_name"] = [f"g{i}" for i in range(5)]
        adata.write_h5ad(data_path)

        config = load_experiment_config("01_centered_signed_hill_lambda0p1")
        config.update({
            "data_dir": str(data_path),
            "edge_tsv_path": str(edge_path),
            "cell_unet_hidden_num": [16, 12, 8, 8],
            "cell_unet_dropout": 0.0,
        })
        write_json(run / "exp_config.json", config)
        diffusion = create_gaussian_diffusion(steps=1000, noise_schedule="linear")
        model = build_model_from_config(config, [f"g{i}" for i in range(5)], 1000, "cpu")
        checkpoint_dir = run / "checkpoints/segment_000/model"
        checkpoint_dir.mkdir(parents=True)
        for step in (100, 500):
            raw = checkpoint_dir / f"model{step:06d}.pt"
            torch.save(model.state_dict(), raw)
            torch.save(model.state_dict(), checkpoint_dir / f"ema_0.9999_{step:06d}.pt")
            torch.save({}, checkpoint_dir / f"opt{step:06d}.pt")
        pd.DataFrame([
            {
                "step": step,
                "diffusion_loss": 1.0 / (index + 1),
                "ode_soft_constraint": 0.5,
                "ode_soft_constraint_weighted": 0.5,
                "cell_ode_consistency_20260830": 0.25,
                "cell_ode_consistency_weighted_20260830": 0.025,
                "total_loss": 1.0 / (index + 1) + 0.525,
            }
            for index, step in enumerate((100, 500))
        ]).to_csv(checkpoint_dir / "loss_components_20260830.csv", index=False)

        final_raw = checkpoint_dir / "model000500.pt"
        checkpoint_sha_before = hashlib.sha256(final_raw.read_bytes()).hexdigest()

        result = analyze_run(run, AnalysisOptions(
            preset="quick", cells=8, batch_size=4, timesteps=(0, 1, 999),
            gradient_timesteps=(0, 999), gradient_cells=2,
            rolling_window=2, seed=7, device="cpu", force=True,
        ))
        csv_dir = Path(result["csv_directory"])
        figure_dir = Path(result["figure_directory"])
        expected_csv = {
            "diffusion_metrics_by_timestep.csv",
            "cell_ode_metrics_by_timestep.csv",
            "loss_history.csv",
            "loss_fraction.csv",
            "gradient_metrics.csv",
        }
        expected_figures = {f"{index:02d}_" for index in range(1, 13)}
        actual_figures = {path.name[:3] for path in figure_dir.glob("*.png")}
        if not expected_csv.issubset({path.name for path in csv_dir.glob("*.csv")}):
            raise AssertionError("missing analysis CSV")
        if not expected_figures.issubset(actual_figures):
            raise AssertionError(f"missing per-run figures: {expected_figures - actual_figures}")
        gradient = pd.read_csv(csv_dir / "gradient_metrics.csv")
        if gradient.empty or gradient["optimizer_step_performed"].any():
            raise AssertionError("gradient analysis must run without optimizer steps")
        checkpoint_sha_after = hashlib.sha256(final_raw.read_bytes()).hexdigest()
        if checkpoint_sha_before != checkpoint_sha_after:
            raise AssertionError("analysis modified the checkpoint file")
        print(f"ANALYSIS_SMOKE_20260830=PASS rows={result['normal_metrics_rows']} gradient_rows={result['gradient_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
