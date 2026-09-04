#!/usr/bin/env python3
"""Focused numerical checks for generic Gaussian objective primitives."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from diffusion.objectives import (  # noqa: E402
    boundary_gaussian_nll,
    score_from_noise,
    weighted_noise_quadratic,
)


class ObjectivePrimitiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260903)

    def test_isotropic_limit_is_scalar_weighted_plain_epsilon_mse(self):
        batch, dim = 4, 5
        sigma = torch.tensor(0.37, dtype=torch.float64)
        cholesky = sigma * torch.eye(dim, dtype=torch.float64)
        diffusion_covariance = torch.eye(dim, dtype=torch.float64)
        residual = torch.randn(batch, dim, dtype=torch.float64)
        exact_per_dimension = weighted_noise_quadratic(
            cholesky, diffusion_covariance, residual
        ) / dim
        expected = residual.square().mean(dim=1) / (2.0 * sigma.square())
        torch.testing.assert_close(exact_per_dimension, expected)

    def test_anisotropic_metric_is_not_plain_mse(self):
        cholesky = torch.tensor(
            [[0.7, 0.0], [0.25, 1.2]], dtype=torch.float64
        )
        residual = torch.tensor([[1.0, -0.4], [-0.2, 0.9]], dtype=torch.float64)
        exact = weighted_noise_quadratic(
            cholesky, torch.eye(2, dtype=torch.float64), residual
        )
        plain = 0.5 * residual.square().sum(dim=1)
        self.assertFalse(torch.allclose(exact, plain))

    def test_appendix_i_boundary_nll_matches_multivariate_normal(self):
        batch, dim = 3, 2
        phi = torch.tensor(
            [[0.82, 0.07], [-0.03, 0.76]], dtype=torch.float64
        )
        shift = torch.tensor([0.11, -0.06], dtype=torch.float64)
        cholesky = torch.tensor(
            [[0.31, 0.0], [0.08, 0.27]], dtype=torch.float64
        )
        covariance = cholesky @ cholesky.T
        x = torch.randn(batch, dim, dtype=torch.float64)
        noise = torch.randn_like(x)
        y = x @ phi.T + shift + noise @ cholesky.T
        predicted_noise = 0.13 * y - 0.04
        model_score = score_from_noise(cholesky, predicted_noise)

        actual = boundary_gaussian_nll(
            x_start=x,
            y_boundary=y,
            model_score=model_score,
            transition_matrix=phi,
            affine_shift=shift,
            covariance=covariance,
            cholesky=cholesky,
        )
        rhs = y - shift + model_score @ covariance.T
        decoder_mean = torch.linalg.solve(phi, rhs.T).T
        decoder_covariance = (
            torch.linalg.solve(phi, covariance)
            @ torch.linalg.inv(phi).T
        )
        reference = -torch.distributions.MultivariateNormal(
            decoder_mean, covariance_matrix=decoder_covariance
        ).log_prob(x)
        torch.testing.assert_close(actual, reference, atol=1e-11, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
