from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SUITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analysis.gradients import (  # noqa: E402
    gradient_metrics_for_model,
    parameter_fingerprint,
    select_analysis_checkpoints,
)
from analysis.loss_history import load_loss_history  # noqa: E402
from analysis.metrics import (  # noqa: E402
    evaluate_diffusion_timesteps,
    safe_row_cosine,
    safe_row_nmse,
    safe_row_pearson,
)
from analysis.plotting import plot_run_figures  # noqa: E402
from analysis.runner import parse_timestep_spec, summarize_runs  # noqa: E402
from guided_diffusion.script_util import create_gaussian_diffusion  # noqa: E402
from models import build_model_from_config  # noqa: E402
from scripts.common import EXPERIMENT_ORDER, load_experiment_config, write_json  # noqa: E402


def tiny_model(experiment="01_centered_signed_hill_lambda0p1"):
    config = load_experiment_config(experiment)
    config["cell_unet_hidden_num"] = [16, 12, 8, 8]
    config["cell_unet_dropout"] = 0.0
    diffusion = create_gaussian_diffusion(steps=1000, noise_schedule="linear")
    model = build_model_from_config(
        config, [f"g{i}" for i in range(5)], 1000, mask=torch.zeros(5, 5)
    )
    metadata = {
        "experiment": config["experiment"],
        "ode_type": config["ode_type"],
        "cell_ode_reg_lambda_20260830": config["cell_ode_reg_lambda_20260830"],
        "run_directory": "synthetic",
        "checkpoint_path": "synthetic/model000100.pt",
        "checkpoint_training_step": 100,
    }
    return config, diffusion, model, metadata


class AnalysisTests(unittest.TestCase):
    def test_training_review_both_branches_are_trainable_model_parameters(self):
        _, _, model, _ = tiny_model()
        optimizer_scope = {id(parameter) for parameter in model.parameters()}
        self.assertTrue(all(
            id(parameter) in optimizer_scope and parameter.requires_grad
            for parameter in model.ml_model.parameters()
        ))
        self.assertTrue(all(
            id(parameter) in optimizer_scope and parameter.requires_grad
            for parameter in model.ode_model.parameters()
        ))

    def test_safe_metrics_constant_and_zero_vectors(self):
        constant = torch.ones(2, 4)
        zeros = torch.zeros(2, 4)
        self.assertTrue(torch.equal(safe_row_pearson(constant, constant), torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(safe_row_cosine(zeros, zeros), torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.equal(safe_row_nmse(zeros, zeros), torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(torch.isfinite(safe_row_nmse(constant, zeros)).all())

    def test_normal_metrics_use_explicit_diffusion_timestep_and_do_not_mutate(self):
        config, diffusion, model, metadata = tiny_model()
        before = parameter_fingerprint(model)
        was_training = model.training
        target, ode = evaluate_diffusion_timesteps(
            model, diffusion, torch.rand(6, 5), [0, 1, 999],
            batch_size=3, seed=10, device="cpu", metadata=metadata,
        )
        self.assertEqual(before, parameter_fingerprint(model))
        self.assertEqual(model.training, was_training)
        self.assertEqual(target["diffusion_target"].unique().tolist(), ["epsilon"])
        self.assertIn("diffusion_timestep", target)
        self.assertNotIn("step", target)
        self.assertEqual(ode.shape[0], 3)
        self.assertTrue(np.isfinite(ode.select_dtypes(include=[np.number])).all().all())

    def test_gradient_analysis_has_no_optimizer_or_parameter_update(self):
        config, diffusion, model, metadata = tiny_model()
        before = parameter_fingerprint(model)
        result = gradient_metrics_for_model(
            model, diffusion, torch.rand(2, 5), 499,
            cell_ode_lambda=config["cell_ode_reg_lambda_20260830"],
            ode_reg_lambda=config["ode_reg_lambda"], seed=4,
            device="cpu", metadata=metadata,
        )
        self.assertEqual(before, parameter_fingerprint(model))
        self.assertEqual(result["parameter_fingerprint_before"], result["parameter_fingerprint_after"])
        self.assertFalse(result["optimizer_constructed"])
        self.assertFalse(result["optimizer_step_performed"])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(np.isfinite(result["cell_gradient_cosine"]))

    def test_loss_mapping_and_training_step_name(self):
        config = load_experiment_config("03_centered_signed_hill_lambda10")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "checkpoints/segment_000/model"
            output.mkdir(parents=True)
            pd.DataFrame([{
                "step": 100,
                "diffusion_loss": 2.0,
                "ode_soft_constraint": 5.0,
                "ode_soft_constraint_weighted": 5.0,
                "cell_ode_consistency_20260830": 3.0,
                "cell_ode_consistency_weighted_20260830": 30.0,
                "total_loss": 37.0,
            }]).to_csv(output / "loss_components_20260830.csv", index=False)
            history, fractions = load_loss_history(root, config, rolling_window=2)
        self.assertEqual(float(history.loc[0, "ode_regularization_base_before_off_mask_lambda"]), 1.0)
        self.assertEqual(float(history.loc[0, "ode_regularization_raw"]), 1.0)
        self.assertEqual(float(history.loc[0, "ode_regularization_weighted_once_by_off_mask_lambda"]), 5.0)
        self.assertEqual(float(history.loc[0, "ode_regularization_weighted"]), 5.0)
        self.assertEqual(float(history.loc[0, "cell_ode_consistency_weighted_20260830"]), 30.0)
        self.assertIn("training_step", history)
        self.assertAlmostEqual(float(fractions[[
            "diffusion_contribution_fraction",
            "ode_regularization_contribution_fraction",
            "cell_ode_contribution_fraction_20260830",
        ]].iloc[0].sum()), 1.0)

    def test_loss_mapping_prefers_explicit_every_step_schema(self):
        config = load_experiment_config("01_centered_signed_hill_lambda0p1")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "checkpoints/segment_000/model"
            output.mkdir(parents=True)
            pd.DataFrame([{
                "training_step": 1,
                "diffusion_loss": 2.0,
                "ode_offmask_base_raw": 0.4,
                "ode_offmask_after_internal_lambda": 2.0,
                "ode_regularization_final_weighted": 2.0,
                "cell_ode_consistency_raw_20260830": 7.0,
                "cell_ode_consistency_sampler_weighted_20260830": 3.0,
                "cell_ode_consistency_final_weighted_20260830": 0.3,
                "total_loss": 4.3,
                "learning_rate": 1e-4,
            }]).to_csv(output / "loss_components_20260830.csv", index=False)
            history, _ = load_loss_history(root, config, rolling_window=2)
        self.assertEqual(float(history.loc[0, "ode_regularization_raw"]), 0.4)
        self.assertEqual(
            float(history.loc[0, "cell_ode_consistency_raw_20260830"]), 7.0
        )
        self.assertEqual(
            float(history.loc[0, "cell_ode_consistency_sampler_weighted_20260830"]),
            3.0,
        )
        self.assertEqual(float(history.loc[0, "learning_rate"]), 1e-4)

    def test_loss_history_handles_100k_optimizer_steps(self):
        config = load_experiment_config("01_centered_signed_hill_lambda0p1")
        count = 100000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "checkpoints/segment_000/model"
            output.mkdir(parents=True)
            values = np.linspace(1.0, 0.1, count)
            pd.DataFrame({
                "training_step": np.arange(1, count + 1),
                "diffusion_loss": values,
                "ode_offmask_base_raw": np.full(count, 0.2),
                "ode_offmask_after_internal_lambda": np.full(count, 1.0),
                "ode_regularization_final_weighted": np.full(count, 1.0),
                "cell_ode_consistency_raw_20260830": np.full(count, 0.5),
                "cell_ode_consistency_sampler_weighted_20260830": np.full(count, 0.5),
                "cell_ode_consistency_final_weighted_20260830": np.full(count, 0.05),
                "total_loss": values + 1.05,
                "learning_rate": np.linspace(1e-4, 0.0, count),
            }).to_csv(output / "loss_components_20260830.csv", index=False)
            history, fractions = load_loss_history(root, config, rolling_window=100)
        self.assertEqual(len(history), count)
        self.assertEqual(len(fractions), count)
        self.assertEqual(int(history.iloc[-1]["training_step"]), count)
        self.assertTrue(np.isfinite(history.filter(like="rolling_median").to_numpy()).all())

    def test_checkpoint_selection_and_timestep_parser(self):
        paths = [Path(f"model{step:06d}.pt") for step in (100, 300, 700, 1000)]
        selected = select_analysis_checkpoints(paths)
        self.assertEqual(selected[-1]["checkpoint_training_step"], 1000)
        self.assertTrue(any("early_10pct" in row["checkpoint_stage"] for row in selected))
        self.assertEqual(parse_timestep_spec("0,1,5-9:2", 1000), (0, 1, 5, 7, 9))

    def test_checkpoint_selection_for_100k_training(self):
        paths = [Path(f"model{step:06d}.pt") for step in range(5000, 100001, 5000)]
        selected = select_analysis_checkpoints(paths)
        self.assertEqual(
            [row["checkpoint_training_step"] for row in selected],
            [10000, 35000, 65000, 100000],
        )

    def test_canonical_lambda_mapping_is_four_by_three(self):
        observed = [
            (
                load_experiment_config(name)["ode_type"],
                load_experiment_config(name)["cell_ode_reg_lambda_20260830"],
            )
            for name in EXPERIMENT_ORDER
        ]
        self.assertEqual(len(observed), 12)
        self.assertEqual(sorted({value for _, value in observed}), [0.1, 1.0, 10.0])

    def test_twelve_condition_summary_preserves_canonical_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = []
            for index, name in enumerate(EXPERIMENT_ORDER):
                config = load_experiment_config(name)
                run = root / name / "batch"
                csv_dir = run / "analysis/detailed_20260830/csv"
                csv_dir.mkdir(parents=True)
                write_json(run / "exp_config.json", config)
                write_json(run / "analysis/detailed_20260830/analysis_config.json", {
                    "analysis_seed": 1234,
                    "dataset_path": "same.h5ad",
                    "diffusion_timesteps": [0],
                })
                np.save(run / "analysis/detailed_20260830/cell_indices.npy", np.arange(4))
                common = {
                    "experiment": name,
                    "ode_type": config["ode_type"],
                    "cell_ode_reg_lambda_20260830": config["cell_ode_reg_lambda_20260830"],
                    "run_directory": str(run),
                    "checkpoint_path": str(run / "model000100.pt"),
                    "checkpoint_training_step": 100,
                    "diffusion_timestep": 0,
                    "analyzed_cells": 4,
                    "diffusion_target": "epsilon",
                }
                pd.DataFrame([{
                    **common,
                    "cell_target_pearson_global": 0.1 + index / 100,
                    "cell_target_mse_mean": 1.0 + index,
                    "cell_target_cosine_mean": 0.2,
                }]).to_csv(csv_dir / "diffusion_metrics_by_timestep.csv", index=False)
                pd.DataFrame([{
                    **common,
                    "cell_ode_pearson_global": 0.3,
                    "cell_ode_cosine_mean": 0.4,
                    "cell_ode_mse_mean": 2.0,
                    "cell_ode_nmse_mean": 0.5,
                    "ode_cell_norm_ratio_mean": 1.0,
                }]).to_csv(csv_dir / "cell_ode_metrics_by_timestep.csv", index=False)
                pd.DataFrame([{
                    "training_step": 100,
                    "diffusion_loss_raw": 1.0,
                    "ode_regularization_weighted": 0.5,
                    "cell_ode_consistency_weighted_20260830": 0.25,
                }]).to_csv(csv_dir / "loss_history.csv", index=False)
                pd.DataFrame([{
                    "training_step": 100,
                    "diffusion_contribution_fraction": 1 / 1.75,
                    "ode_regularization_contribution_fraction": 0.5 / 1.75,
                    "cell_ode_contribution_fraction_20260830": 0.25 / 1.75,
                }]).to_csv(csv_dir / "loss_fraction.csv", index=False)
                pd.DataFrame([{
                    "cell_gradient_cosine": 0.1,
                    "cell_gradient_norm_ratio": 0.2,
                    "ode_gradient_cosine": -0.1,
                    "ode_gradient_norm_ratio": 0.3,
                }]).to_csv(csv_dir / "gradient_metrics.csv", index=False)
                runs.append(run)
            output = root / "summary"
            result = summarize_runs(runs, output, require_all=True)
            summary = pd.read_csv(output / "csv/condition_summary.csv")
            self.assertEqual(result["conditions"], 12)
            self.assertEqual(summary["experiment"].tolist(), list(EXPERIMENT_ORDER))
            for number in range(13, 17):
                self.assertTrue(any((output / "figures").glob(f"{number:02d}_*.png")))

    def test_plot_names_mark_zoom_exclusion_and_log_is_zero_safe(self):
        config, diffusion, model, metadata = tiny_model()
        target, ode = evaluate_diffusion_timesteps(
            model, diffusion, torch.zeros(4, 5), [0, 1, 999],
            batch_size=2, seed=1, device="cpu", metadata=metadata,
        )
        history = pd.DataFrame({
            "training_step": [0, 100],
            "diffusion_loss_raw": [1.0, 0.0],
            "ode_regularization_raw": [2.0, 0.0],
            "ode_regularization_weighted": [2.0, 0.0],
            "cell_ode_consistency_raw_20260830": [3.0, 0.0],
            "cell_ode_consistency_weighted_20260830": [0.3, 0.0],
            "total_loss": [3.3, 0.0],
        })
        for column in (
            "diffusion_loss_raw", "ode_regularization_raw",
            "cell_ode_consistency_raw_20260830", "ode_regularization_weighted",
            "cell_ode_consistency_weighted_20260830", "total_loss",
        ):
            history[f"{column}_rolling_median_w2"] = history[column]
            history[f"{column}_rolling_q25_w2"] = history[column]
            history[f"{column}_rolling_q75_w2"] = history[column]
        fraction = pd.DataFrame({
            "training_step": [0, 100],
            "diffusion_contribution_fraction": [0.3, 0.0],
            "ode_regularization_contribution_fraction": [0.6, 0.0],
            "cell_ode_contribution_fraction_20260830": [0.1, 0.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            created = plot_run_figures(
                target, ode, history, fraction, pd.DataFrame(), directory,
                rolling_window=2, zoom_timestep_min=1,
            )
            names = {path.name for path in created}
            self.assertIn("06_cell_ode_metrics_zoom_tge1.png", names)
            self.assertIn("07_cell_ode_metrics_log.png", names)
            self.assertTrue(all(path.stat().st_size > 0 for path in created))

    def test_analysis_does_not_modify_training_files(self):
        expected = {
            "guided_diffusion/gaussian_diffusion.py": "eeb83640dc140f91e3519976fa5cf031076d7713d049d2a1b9debc8b0390b9ab",
            "guided_diffusion/train_util.py": "8aba336eee5240a788d5c2de46092a91d1c5f21a0e215e46106e5b28fe710c19",
            "work/20260830/training/train_loop_20260830.py": "95c1cc9cd39eda5db2d54d635e69b0383bc83193d7e5083174d9e430985ebc06",
            "work/20260830/scripts/train.py": "5fc976c5a4b23a67bf5d4c6389968e2e0b25ceb78b1d22544e9feb000f51f5db",
            "work/20260830/scripts/sample.py": "e313f830131dded178bad9a34229c80f3109503f545a8bb92ba1310d5e68ea8c",
        }
        for relative, digest in expected.items():
            actual = hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, digest)


if __name__ == "__main__":
    unittest.main()
