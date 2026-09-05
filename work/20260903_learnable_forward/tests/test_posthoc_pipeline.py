#!/usr/bin/env python3
"""Focused tests for post-hoc metrics, ordering, and pipeline decisions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import unittest
import uuid

import numpy as np
import pandas as pd
import torch


SUITE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = SUITE_ROOT / "scripts"
for path in (SUITE_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.common import sample_corr, sample_mse  # noqa: E402
from analysis.diffusion_diagnostics import metric_bundle  # noqa: E402
from analysis.hematopoietic import select_hematopoietic_subset  # noqa: E402
from analysis.loss_analysis import (  # noqa: E402
    WEIGHTED_COLUMNS,
    add_rolling_statistics,
    contribution_fractions,
)
from analysis.parameter_evolution import (  # noqa: E402
    model_a_arrays,
    model_b_arrays,
    parameter_categories,
)
from diffusion.free_affine import FreeAffineForward  # noqa: E402
from diffusion.stationary_qd import StationaryQDForward  # noqa: E402
import common as script_common  # noqa: E402
from run_full_pipeline import (  # noqa: E402
    analysis_stage_complete,
    build_pipeline_plan,
    decide_training_action,
)


class PosthocPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260904)
        np.random.seed(20260904)

    def test_checkpoint_ordering_is_numeric_not_lexicographic(self):
        experiment = f"test_checkpoint_order_{uuid.uuid4().hex}"
        root = script_common.RUNS_ROOT / experiment
        run = root / "batch"
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        model_dir = run / "segments" / "segment_000" / "model"
        model_dir.mkdir(parents=True)
        for step in (100, 5, 20):
            (model_dir / f"model{step:06d}.pt").write_bytes(b"test")
        ordered = script_common.all_raw_checkpoints(run)
        self.assertEqual(
            [script_common.checkpoint_step(path) for path in ordered],
            [5, 20, 100],
        )

    def test_model_a_and_b_parameter_extraction_and_categories(self):
        mask = torch.tensor(
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
            dtype=torch.float64,
        )
        model_a = StationaryQDForward(
            3, aux_dim=2, grn_mask_target_source=mask, dtype=torch.float64
        )
        model_b = FreeAffineForward(
            3, grn_mask_target_source=mask, dtype=torch.float64
        )
        with torch.no_grad():
            model_a.raw_q_k.copy_(torch.tensor([[0., .1], [-.2, 0.]]))
            model_b.raw_w.copy_(torch.arange(9, dtype=torch.float64).reshape(3, 3) / 20)
            model_b.raw_b.copy_(torch.tensor([0.1, -0.1, 0.2]))

        arrays_a = model_a_arrays(model_a)
        arrays_b = model_b_arrays(model_b)
        np.testing.assert_allclose(arrays_a["A"], arrays_a["Q"] + arrays_a["D"])
        np.testing.assert_allclose(arrays_a["F"], -arrays_a["A"])
        np.testing.assert_allclose(arrays_b["W"], model_b.drift_matrix().detach().numpy())
        np.testing.assert_allclose(arrays_b["b"], model_b.raw_b.detach().numpy())
        categories_a, diagnostics_a = parameter_categories(model_a)
        categories_b, diagnostics_b = parameter_categories(model_b)
        self.assertIn("A Known GRN edges", categories_a)
        self.assertIn("A Unknown GRN edges", categories_a)
        self.assertIn("W Known GRN edges", categories_b)
        self.assertIn("W Unknown GRN edges", categories_b)
        self.assertIn("q_skew_symmetry_residual", diagnostics_a)
        self.assertEqual(diagnostics_b, {})

    def test_epsilon_drift_and_score_metric_equations(self):
        epsilon = torch.tensor(
            [[1.0, 2.0, 4.0], [-1.0, 0.5, 3.0]], dtype=torch.float64
        )
        prediction = epsilon + torch.tensor(
            [[0.5, -0.25, 0.0], [0.2, -0.1, 0.3]], dtype=torch.float64
        )
        drift = torch.tensor(
            [[0.4, -0.2, 0.7], [-0.3, 0.8, 0.2]], dtype=torch.float64
        )
        score_true = -0.7 * epsilon
        score_prediction = -0.7 * prediction
        weighted = torch.tensor([0.11, 0.22], dtype=torch.float64)
        result = metric_bundle(
            prediction,
            epsilon,
            drift,
            score_prediction,
            score_true,
            weighted,
        )
        torch.testing.assert_close(
            result["epsilon_pred_true_mse"], sample_mse(prediction, epsilon)
        )
        torch.testing.assert_close(
            result["epsilon_pred_true_corr"], sample_corr(prediction, epsilon)
        )
        torch.testing.assert_close(
            result["drift_true_epsilon_mse"], sample_mse(drift, epsilon)
        )
        torch.testing.assert_close(
            result["drift_true_epsilon_corr"], sample_corr(drift, epsilon)
        )
        torch.testing.assert_close(
            result["drift_epsilon_prediction_mse"], sample_mse(drift, prediction)
        )
        torch.testing.assert_close(
            result["drift_epsilon_prediction_corr"], sample_corr(drift, prediction)
        )
        torch.testing.assert_close(
            result["score_mse"], sample_mse(score_prediction, score_true)
        )
        torch.testing.assert_close(
            result["score_corr"], sample_corr(score_prediction, score_true)
        )
        torch.testing.assert_close(result["weighted_score_quadratic"], weighted)

    def test_hematopoietic_selection_is_erythropoietic_plus_immune(self):
        import anndata as ad

        adata = ad.AnnData(
            X=np.arange(24, dtype=np.float32).reshape(6, 4),
            obs=pd.DataFrame(
                {
                    "Superclass": [
                        "Erythropoietic",
                        "Immune",
                        "Neural",
                        "Immune",
                        "Erythropoietic",
                        "Mesenchymal",
                    ],
                    "celltype": ["E1", "I1", "N1", "I2", "E2", "M1"],
                },
                index=[f"c{i}" for i in range(6)],
            ),
        )
        subset, metadata = select_hematopoietic_subset(adata)
        self.assertEqual(subset.n_obs, 4)
        self.assertEqual(
            metadata["selected_superclasses"], ["Erythropoietic", "Immune"]
        )
        self.assertEqual(set(subset.obs["Superclass"].astype(str)), {"Erythropoietic", "Immune"})

    def test_loss_rolling_statistics_and_contribution_fractions(self):
        history = pd.DataFrame(
            {
                "training_step": [0, 1, 2],
                "path_final_per_dim": [1.0, 2.0, 3.0],
                "terminal_final_per_dim": [2.0, 2.0, 1.0],
                "boundary_final_per_dim": [1.0, 0.0, 1.0],
                "grn_penalty_final_weighted": [0.0, 0.0, 1.0],
            }
        )
        rolled = add_rolling_statistics(
            history, ("path_final_per_dim",), window=2
        )
        self.assertAlmostEqual(
            rolled.loc[2, "path_final_per_dim_rolling_median_w2"], 2.5
        )
        fractions = contribution_fractions(history, window=2)
        fraction_columns = [f"{name}_fraction" for name in WEIGHTED_COLUMNS]
        np.testing.assert_allclose(fractions[fraction_columns].sum(axis=1), 1.0)

    def test_pipeline_dry_run_same_batch_and_training_resume_skip(self):
        batch = f"dryrun-{uuid.uuid4().hex[:12]}"
        args = argparse.Namespace(
            batch_id=batch,
            models="stationary_qd,free_affine",
            device="cpu",
            base_config=None,
            set_values=[],
            python=sys.executable,
            analysis_stages=("loss", "parameters"),
            force_analysis=False,
        )
        plan = build_pipeline_plan(args)
        self.assertEqual([item["model"] for item in plan], ["stationary_qd", "free_affine"])
        self.assertTrue(all(Path(item["run_dir"]).name == batch for item in plan))
        self.assertTrue(all(item["training"]["action"] == "run" for item in plan))

        experiment = f"test_pipeline_state_{uuid.uuid4().hex}"
        root = script_common.RUNS_ROOT / experiment
        run = root / "batch"
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        status = run / "segments" / "segment_000" / "status.json"
        status.parent.mkdir(parents=True)
        script_common.write_json(status, {"status": "failed"})
        self.assertEqual(decide_training_action(run, 10), "run")
        model_dir = status.parent / "model"
        model_dir.mkdir()
        for name in ("model000005.pt", "opt000005.pt", "ema_0.9999_000005.pt"):
            (model_dir / name).write_bytes(b"test")
        script_common.write_json(
            run / "requested_config.json", {"config": {"ema_rate": "0.9999"}}
        )
        self.assertEqual(decide_training_action(run, 10), "resume")
        (model_dir / "model000009.pt").write_bytes(b"test")
        script_common.write_json(status, {"status": "completed"})
        self.assertEqual(decide_training_action(run, 10), "skip")

        source = run / "segments" / "segment_000" / "loss_components.csv"
        source.write_text("training_step,total_loss\n0,1\n", encoding="utf-8")
        metadata = run / "analysis" / "loss" / "metadata.json"
        script_common.write_json(
            metadata,
            {"status": "completed", "source_files": [str(source.resolve())]},
        )
        self.assertTrue(analysis_stage_complete(run, "loss"))


if __name__ == "__main__":
    unittest.main()
