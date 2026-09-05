#!/usr/bin/env python3
"""Dynamic dimension normalization, common GRN, and loss-schema tests."""

from __future__ import annotations

import inspect
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from diffusion import objectives  # noqa: E402
from diffusion.free_affine import FreeAffineForward  # noqa: E402
from diffusion.stationary_qd import StationaryQDForward  # noqa: E402
from diffusion.training_diffusion import LearnableForwardTrainingDiffusion  # noqa: E402
from models.factory import build_experiment_components  # noqa: E402
from training.loss_logging import LossComponentWriter, validate_loss_record  # noqa: E402


class TinyDenoiser(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x, t, **kwargs):
        del kwargs
        return self.linear(x) + 0.001 * t.to(x.dtype)


def config(dim: int, family: str) -> dict:
    return {
        "input_dim": dim,
        "forward_model": family,
        "forward_dtype": "float64",
        "loss_mode": "paper_elbo",
        "normalize_elbo_by_dimension": True,
        "noise_schedule": "cosine",
        "diffusion_steps": 12,
        "microbatch": -1,
        "schedule_sampler": "batch_shared_physical_uniform",
        "use_fp16": False,
        "timestep_respacing": "",
        "weight_decay": 0.0,
        "covariance_jitter": 0.0,
        "aux_dim": min(2, dim - 1),
        "d_diagonal_floor": 0.0,
        "initial_d_diagonal": 0.5,
        "use_grn_mask": False,
        "allow_self_edges": True,
        "grn_penalty_weight": 0.0,
        "grn_penalty_norm": "l1",
    }


class NormalizationGRNLoggingTests(unittest.TestCase):
    def test_per_dimension_helper_multiple_dimensions(self):
        for dimension in (2, 3, 7):
            values = torch.arange(1, 5, dtype=torch.float64) * dimension
            torch.testing.assert_close(
                objectives.per_dimension(values, dimension),
                torch.arange(1, 5, dtype=torch.float64),
            )

    def test_objective_logic_does_not_hardcode_gene_dimension(self):
        sources = (
            inspect.getsource(objectives),
            inspect.getsource(LearnableForwardTrainingDiffusion),
        )
        self.assertNotIn("1024", "\n".join(sources))

    def test_complete_elbo_uses_one_dynamic_dimension_factor(self):
        torch.manual_seed(11)
        for dimension in (2, 3, 7):
            for family in ("stationary_qd", "free_affine"):
                components = build_experiment_components(
                    config(dimension, family),
                    [f"g{i}" for i in range(dimension)],
                    torch.device("cpu"),
                    denoiser=TinyDenoiser(dimension),
                )
                x = torch.randn(4, dimension)
                t = torch.full((4,), 5.2)
                losses = components.diffusion.training_losses(components.model, x, t)
                duration = components.time_map.duration
                torch.testing.assert_close(
                    losses["path_after_duration"],
                    duration * losses["path_loss_raw"],
                )
                torch.testing.assert_close(
                    losses["path_final_per_dim"],
                    losses["path_after_duration"] / dimension,
                )
                torch.testing.assert_close(
                    losses["terminal_final_per_dim"],
                    losses["terminal_kl_raw"] / dimension,
                )
                torch.testing.assert_close(
                    losses["boundary_final_per_dim"],
                    losses["boundary_nll_raw"] / dimension,
                )
                reconstructed = (
                    losses["path_final_per_dim"]
                    + losses["terminal_final_per_dim"]
                    + losses["boundary_final_per_dim"]
                    + losses["grn_penalty_final_weighted"]
                )
                torch.testing.assert_close(losses["total_loss"], reconstructed)

    def test_model_a_and_b_share_exact_grn_mask_and_weighting(self):
        mask = torch.tensor(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        common = dict(
            grn_mask_target_source=mask,
            allow_self_edges=True,
            grn_penalty_weight=5.0,
            grn_penalty_norm="l1",
            dtype=torch.float64,
        )
        model_a = StationaryQDForward(3, aux_dim=2, **common)
        model_b = FreeAffineForward(3, **common)
        with torch.no_grad():
            operator = model_a.stationary_operator()
            model_b.raw_w.copy_(operator + 0.5 * torch.eye(3, dtype=torch.float64))
        torch.testing.assert_close(
            model_a.grn_mask_target_source, model_b.grn_mask_target_source
        )
        torch.testing.assert_close(model_a.grn_penalty_base(), model_b.grn_penalty_base())
        torch.testing.assert_close(
            model_a.additional_regularization(), model_b.additional_regularization()
        )
        torch.testing.assert_close(
            model_b.additional_regularization(), 5.0 * model_b.grn_penalty_base()
        )

    def test_model_b_penalizes_only_w_off_mask(self):
        mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        model = FreeAffineForward(
            2,
            grn_mask_target_source=mask,
            allow_self_edges=True,
            grn_penalty_weight=3.0,
            dtype=torch.float64,
        )
        with torch.no_grad():
            model.raw_w.copy_(
                torch.tensor([[0.9, 2.0], [-4.0, 1.1]], dtype=torch.float64)
            )
            model.raw_b.copy_(torch.tensor([100.0, -100.0], dtype=torch.float64))
        expected = torch.tensor(1.0, dtype=torch.float64)  # abs(-4) / four entries
        torch.testing.assert_close(model.grn_penalty_base(), expected)
        torch.testing.assert_close(model.additional_regularization(), 3.0 * expected)

    def test_buffered_loss_record_reconstructs_total(self):
        record = {
            "training_step": 9,
            "learning_rate": 1e-4,
            "sampled_physical_time": 0.5,
            "fractional_diffusion_timestep": 4.2,
            "dimension": 7,
            "path_loss_raw": 14.0,
            "terminal_kl_raw": 3.5,
            "boundary_nll_raw": -0.7,
            "path_after_duration": 7.0,
            "path_final_per_dim": 1.0,
            "terminal_final_per_dim": 0.5,
            "boundary_final_per_dim": -0.1,
            "paper_elbo_per_dim": 1.4,
            "grn_penalty_raw": 0.04,
            "grn_penalty_weight": 5.0,
            "grn_penalty_final_weighted": 0.2,
            "plain_epsilon_mse": 0.8,
            "total_loss": 1.6,
        }
        validate_loss_record(record)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "loss_components.csv"
            writer = LossComponentWriter(path, flush_interval=2)
            writer.append(record)
            self.assertFalse(path.exists())
            writer.append({**record, "training_step": 10})
            self.assertTrue(path.is_file())
            self.assertEqual(len(path.read_text().splitlines()), 3)


if __name__ == "__main__":
    unittest.main()
