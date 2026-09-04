#!/usr/bin/env python3
"""Equation, reproducibility, and provenance tests for the custom sampler."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from diffusion.free_affine import FreeAffineForward  # noqa: E402
from diffusion.objectives import score_from_noise  # noqa: E402
from diffusion.stationary_qd import StationaryQDForward  # noqa: E402
from diffusion.time_mapping import PhysicalTimeMap  # noqa: E402
from models.wrapper import LearnableForwardModel  # noqa: E402
from sampling.artifacts import load_sample_archive, save_sample_archive  # noqa: E402
from sampling.reverse_sde import (  # noqa: E402
    boundary_decode,
    euler_maruyama_step,
    reverse_drift,
    reverse_time_indices,
    sample_reverse_sde,
)
from scripts.common import gene_order_sha256  # noqa: E402


class LinearNoisePredictor(nn.Module):
    def __init__(self, dim: int, scale: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale), dtype=torch.float64))
        self.dim = dim

    def forward(self, values, timesteps, **kwargs):
        del timesteps, kwargs
        return self.scale * values


class ReverseSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260904)
        self.dtype = torch.float64

    def _model(self, process, scale: float = 0.0):
        return LearnableForwardModel(
            LinearNoisePredictor(process.dim, scale=scale), process
        )

    def test_reverse_grid_descends_and_positive_dt_is_required(self):
        self.assertEqual(reverse_time_indices(10, 3), (9, 6, 3, 0))
        self.assertEqual(reverse_time_indices(9, 3), (8, 5, 2, 0))
        states = torch.zeros(2, 1, dtype=self.dtype)
        with self.assertRaisesRegex(ValueError, "positive"):
            euler_maruyama_step(
                states,
                states,
                torch.ones(1, 1, dtype=self.dtype),
                -0.1,
                states,
            )

    def test_score_transform_is_minus_inverse_transpose_cholesky(self):
        cholesky = torch.tensor(
            [[1.2, 0.0], [0.3, 0.7]], dtype=self.dtype
        )
        epsilon = torch.tensor([[0.4, -0.8], [-0.2, 0.5]], dtype=self.dtype)
        score = score_from_noise(cholesky, epsilon)
        expected = -torch.linalg.solve(
            cholesky.T, epsilon.T
        ).T
        torch.testing.assert_close(score, expected)

    def test_reverse_drift_sign_and_model_specific_diffusions(self):
        states = torch.tensor([[0.7, -0.2]], dtype=self.dtype)
        score = torch.tensor([[0.3, 0.4]], dtype=self.dtype)

        model_a = StationaryQDForward(2, dtype=self.dtype)
        expected_a = score @ (2.0 * model_a.d_matrix()).T
        expected_a = expected_a + states @ (
            model_a.q_matrix() + model_a.d_matrix()
        ).T
        torch.testing.assert_close(reverse_drift(model_a, states, score), expected_a)
        torch.testing.assert_close(
            model_a.diffusion_factor() @ model_a.diffusion_factor().T,
            2.0 * model_a.d_matrix(),
        )

        model_b = FreeAffineForward(2, dtype=self.dtype)
        expected_b = score - (
            states @ model_b.drift_matrix().T + model_b.drift_bias()
        )
        torch.testing.assert_close(reverse_drift(model_b, states, score), expected_b)
        torch.testing.assert_close(
            model_b.diffusion_factor(), torch.eye(2, dtype=self.dtype)
        )

    def test_standard_vp_initialization_limit_is_shared_by_both_models(self):
        model_a = StationaryQDForward(2, dtype=self.dtype)
        model_b = FreeAffineForward(2, dtype=self.dtype)
        x = torch.tensor([[0.2, -0.4]], dtype=self.dtype)
        time = 0.63
        stats_a = model_a.transition_stats(x, time)
        stats_b = model_b.transition_stats(x, time)
        torch.testing.assert_close(stats_a.mean, stats_b.mean, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(
            stats_a.covariance, stats_b.covariance, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(
            model_a.diffusion_covariance(), model_b.diffusion_covariance()
        )
        torch.testing.assert_close(model_a.drift(x), model_b.drift(x))

    def test_boundary_decoder_and_full_sampler_are_finite_reproducible_and_shaped(self):
        process = FreeAffineForward(2, dtype=self.dtype)
        model = self._model(process, scale=0.02)
        time_map = PhysicalTimeMap(
            torch.tensor([0.08, 0.18, 0.36, 0.71], dtype=self.dtype)
        )
        generator = torch.Generator(device="cpu").manual_seed(19)
        boundary = torch.tensor(
            [[0.2, -0.1], [0.5, 0.4]], dtype=self.dtype
        )
        decoded = boundary_decode(
            model,
            boundary,
            boundary_time=time_map.boundary_time,
            decoder_sampling_mode="sample",
            generator=generator,
        )
        self.assertEqual(decoded.shape, boundary.shape)
        self.assertTrue(torch.isfinite(decoded).all())

        first = sample_reverse_sde(
            model, time_map, batch_size=7, seed=123, reverse_stride=2
        )
        second = sample_reverse_sde(
            model, time_map, batch_size=7, seed=123, reverse_stride=2
        )
        self.assertEqual(first.samples.shape, (7, 2))
        self.assertEqual(first.visited_indices, (3, 1, 0))
        self.assertEqual(first.reverse_steps, 2)
        torch.testing.assert_close(first.initial_prior, second.initial_prior)
        torch.testing.assert_close(first.boundary_states, second.boundary_states)
        torch.testing.assert_close(first.samples, second.samples)
        self.assertTrue(torch.isfinite(first.samples).all())

    def test_sample_archive_preserves_gene_order_and_provenance(self):
        genes = ["GeneB", "GeneA"]
        cells = np.asarray([[0.1, 0.2], [-0.3, 0.4]], dtype=np.float32)
        metadata = {
            "checkpoint_path": "/tmp/ema_0.9999_000010.pt",
            "checkpoint_step": 10,
            "ema_rate": "0.9999",
            "forward_model": "free_affine",
            "random_seed": 42,
            "sampler_type": "custom_reverse_sde_euler_maruyama",
            "reverse_steps": 9,
            "decoder_sampling_mode": "sample",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.npz"
            archive, sidecar = save_sample_archive(path, cells, genes, metadata)
            restored, restored_genes, restored_metadata = load_sample_archive(archive)
            np.testing.assert_array_equal(restored, cells)
            self.assertEqual(restored_genes, genes)
            self.assertEqual(
                restored_metadata["gene_order_sha256"], gene_order_sha256(genes)
            )
            with sidecar.open(encoding="utf-8") as handle:
                persisted = json.load(handle)
            for key, value in metadata.items():
                self.assertEqual(persisted[key], value)

    def test_one_and_two_dimensional_euler_gaussian_mean_covariance(self):
        # One exact EM step for constant coefficients has an analytic Gaussian
        # law.  Monte Carlo checks both the positive-dtau sign and row-vector
        # diffusion orientation in dimensions one and two.
        count = 30000
        dt = 0.2
        generator = torch.Generator(device="cpu").manual_seed(77)
        for diffusion, expected_covariance in (
            (
                torch.tensor([[1.4]], dtype=self.dtype),
                dt * torch.tensor([[1.4**2]], dtype=self.dtype),
            ),
            (
                torch.tensor([[1.0, 0.0], [0.35, 0.8]], dtype=self.dtype),
                None,
            ),
        ):
            dim = diffusion.shape[0]
            states = torch.zeros(count, dim, dtype=self.dtype)
            drift = torch.full_like(states, 0.3)
            noise = torch.randn(
                count, dim, generator=generator, dtype=self.dtype
            )
            sampled = euler_maruyama_step(states, drift, diffusion, dt, noise)
            if expected_covariance is None:
                expected_covariance = dt * diffusion @ diffusion.T
            empirical_mean = sampled.mean(dim=0)
            empirical_covariance = torch.from_numpy(
                np.cov(sampled.numpy(), rowvar=False, ddof=0)
            ).reshape(dim, dim)
            torch.testing.assert_close(
                empirical_mean,
                torch.full((dim,), dt * 0.3, dtype=self.dtype),
                atol=1.5e-2,
                rtol=0,
            )
            torch.testing.assert_close(
                empirical_covariance,
                expected_covariance,
                atol=1.6e-2,
                rtol=0,
            )


if __name__ == "__main__":
    unittest.main()
