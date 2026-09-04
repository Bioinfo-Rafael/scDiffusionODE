#!/usr/bin/env python3
"""Focused tests for the dense stationary-Q/D forward process."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from diffusion.stationary_qd import StationaryQDForward  # noqa: E402


class StationaryQDForwardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260903)

    def model(self, dim=4, **kwargs):
        return StationaryQDForward(dim, dtype=torch.float64, **kwargs)

    def test_dense_parameter_count_and_exact_parameterizations(self):
        dim = 5
        model = self.model(dim)
        self.assertEqual(model.d_parameterization, "psd")
        self.assertEqual(model.raw_q_upper.numel(), dim * (dim - 1) // 2)
        self.assertEqual(model.raw_d_lower.numel(), dim * (dim + 1) // 2)
        self.assertEqual(
            model.raw_q_upper.numel() + model.raw_d_lower.numel(), dim * dim
        )

        with torch.no_grad():
            model.raw_q_upper.normal_()
            model.raw_d_lower.normal_()
            # A zero diagonal is legal in the paper-faithful PSD mode.
            diagonal = model._d_indices[0] == model._d_indices[1]
            model.raw_d_lower[diagonal] = 0.0

        q = model.q_matrix()
        c = model.d_factor()
        d = model.d_matrix()
        torch.testing.assert_close(q + q.T, torch.zeros_like(q), atol=0.0, rtol=0.0)
        torch.testing.assert_close(d, c @ c.T, atol=0.0, rtol=0.0)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(d).min()), -1e-12)
        self.assertEqual(torch.count_nonzero(torch.triu(c, diagonal=1)).item(), 0)

    def test_opt_in_spd_mode_has_strict_factor_diagonal_and_floor(self):
        floor = 0.03
        model = self.model(
            4,
            d_parameterization="spd_softplus",
            d_diagonal_floor=floor,
        )
        torch.testing.assert_close(
            model.d_matrix(),
            0.5 * torch.eye(4, dtype=torch.float64),
            atol=1e-12,
            rtol=1e-12,
        )
        with torch.no_grad():
            diagonal = model._d_indices[0] == model._d_indices[1]
            model.raw_d_lower[diagonal] = -100.0
        factor_diagonal = model.d_factor().diagonal()
        # At very negative raw values softplus can round below one ulp, but the
        # explicit positive floor still makes the factor diagonal strictly
        # positive.
        self.assertTrue(torch.all(factor_diagonal >= floor))
        self.assertTrue(torch.all(factor_diagonal > 0.0))
        self.assertGreater(float(torch.linalg.eigvalsh(model.d_matrix()).min()), 0.0)

        with self.assertRaisesRegex(ValueError, "only valid"):
            self.model(2, d_parameterization="psd", d_diagonal_floor=1e-6)
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            self.model(2, d_parameterization="spd_softplus")

    def test_standard_vp_initialization_transition_and_sample(self):
        dim = 4
        model = self.model(dim)
        alpha_bar = torch.tensor(0.37, dtype=torch.float64)
        physical_time = -torch.log(alpha_bar)
        x = torch.randn(3, dim, dtype=torch.float64)
        noise = torch.randn_like(x)

        sample, returned_noise, stats = model.q_sample(
            x, physical_time, noise, return_stats=True
        )
        expected_phi = torch.sqrt(alpha_bar) * torch.eye(
            dim, dtype=torch.float64
        )
        expected_covariance = (1.0 - alpha_bar) * torch.eye(
            dim, dtype=torch.float64
        )
        expected_sample = torch.sqrt(alpha_bar) * x + torch.sqrt(
            1.0 - alpha_bar
        ) * noise

        torch.testing.assert_close(stats.transition_matrix, expected_phi)
        torch.testing.assert_close(stats.covariance, expected_covariance)
        torch.testing.assert_close(stats.affine_shift, torch.zeros(dim, dtype=torch.float64))
        torch.testing.assert_close(stats.mean, torch.sqrt(alpha_bar) * x)
        torch.testing.assert_close(sample, expected_sample)
        self.assertIs(returned_noise, noise)

    def test_stationary_identity_covariance_and_lyapunov_residual(self):
        model = self.model(5)
        with torch.no_grad():
            model.raw_q_upper.normal_(std=0.15)
            model.raw_d_lower.normal_(std=0.1)
            diagonal = model._d_indices[0] == model._d_indices[1]
            model.raw_d_lower[diagonal] += 0.8

        q = model.q_matrix()
        d = model.d_matrix()
        drift = -(q + d)
        a = 2.0 * d
        residual = drift + drift.T + a
        torch.testing.assert_close(residual, torch.zeros_like(residual), atol=1e-13, rtol=0.0)
        torch.testing.assert_close(
            model.stationarity_residual(),
            torch.zeros_like(residual),
            atol=1e-13,
            rtol=0.0,
        )
        self.assertAlmostEqual(float(model.drift_divergence()), -float(torch.trace(d)))

        x = torch.randn(2, 5, dtype=torch.float64)
        stats = model.transition_stats(x, 0.7)
        expected = torch.eye(5, dtype=torch.float64) - stats.phi @ stats.phi.T
        torch.testing.assert_close(stats.nominal_covariance, expected)

        # If Y_0 has covariance I, applying the transition again leaves it I.
        propagated_stationary = (
            stats.phi @ stats.phi.T + stats.nominal_covariance
        )
        torch.testing.assert_close(
            propagated_stationary, torch.eye(5, dtype=torch.float64)
        )

    def test_score_space_and_noise_space_weighted_losses_match(self):
        model = self.model(4)
        with torch.no_grad():
            model.raw_q_upper.normal_(std=0.1)
            model.raw_d_lower.normal_(std=0.12)
            diagonal = model._d_indices[0] == model._d_indices[1]
            model.raw_d_lower[diagonal] += 0.7
        x = torch.randn(3, 4, dtype=torch.float64)
        stats = model.transition_stats(x, 0.45)
        residual = torch.randn_like(x)

        noise_loss = model.noise_metric_quadratic(stats, residual)
        score_difference = -model.conditional_score(stats, residual)
        # conditional_score(stats, residual) is -L^{-T} residual; its sign
        # disappears in the quadratic form.
        score_loss = 0.5 * (
            (score_difference @ stats.diffusion_covariance) * score_difference
        ).sum(-1)
        torch.testing.assert_close(noise_loss, score_loss, atol=1e-11, rtol=1e-11)

    def test_gradients_reach_dense_q_and_d_parameters(self):
        model = self.model(4, grn_penalty_weight=0.2, grn_mask_target_source=torch.eye(4))
        with torch.no_grad():
            model.raw_q_upper.normal_(std=0.08)
            model.raw_d_lower.normal_(std=0.08)
            diagonal = model._d_indices[0] == model._d_indices[1]
            model.raw_d_lower[diagonal] += math.sqrt(0.5)
        x = torch.randn(3, 4, dtype=torch.float64)
        residual = torch.randn_like(x)
        stats = model.transition_stats(x, 0.6)
        loss = (
            model.noise_metric_quadratic(stats, residual).mean()
            + 0.1 * stats.mean.square().mean()
            + model.additional_regularization()
        )
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().sum()), 0.0, name)

    def test_grn_mask_is_target_source_and_diagonal_is_allowed(self):
        mask = torch.zeros(2, 2, dtype=torch.float64)
        mask[0, 1] = 1.0  # target 0 <- source 1 is allowed.
        model = self.model(
            2,
            grn_mask_target_source=mask,
            grn_penalty_weight=3.0,
            grn_penalty_norm="l1",
        )
        with torch.no_grad():
            model.raw_q_upper.fill_(2.0)
        expected_effective_mask = torch.tensor(
            [[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64
        )
        torch.testing.assert_close(
            model.grn_mask_target_source, expected_effective_mask
        )
        # Only Q[target=1, source=0] == -2 is off mask; mean is over all 4.
        self.assertAlmostEqual(float(model.grn_penalty_base()), 0.5)
        self.assertAlmostEqual(float(model.grn_penalty()), 1.5)

    def test_psd_degeneracy_fails_cholesky_without_hidden_floor(self):
        model = self.model(3)
        with torch.no_grad():
            model.raw_d_lower.zero_()
            model.raw_q_upper.zero_()
        x = torch.randn(2, 3, dtype=torch.float64)
        stats = model.transition_stats(x, 0.5, compute_cholesky=False)
        torch.testing.assert_close(
            stats.nominal_covariance,
            torch.zeros(3, 3, dtype=torch.float64),
            atol=0.0,
            rtol=0.0,
        )
        with self.assertRaisesRegex(RuntimeError, "does not add a hidden floor"):
            model.transition_stats(x, 0.5)

    def test_explicit_covariance_jitter_is_reported_and_used(self):
        jitter = 1e-5
        model = self.model(2, initial_d_diagonal=0.0, covariance_jitter=jitter)
        x = torch.zeros(1, 2, dtype=torch.float64)
        stats = model.transition_stats(x, 0.2)
        torch.testing.assert_close(
            stats.nominal_covariance,
            torch.zeros(2, 2, dtype=torch.float64),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            stats.covariance,
            jitter * torch.eye(2, dtype=torch.float64),
            atol=0.0,
            rtol=0.0,
        )
        self.assertEqual(float(stats.covariance_jitter), jitter)

    def test_integral_series_matches_stationary_identity_value_and_gradient(self):
        model = self.model(3)
        with torch.no_grad():
            model.raw_q_upper.copy_(
                torch.tensor([0.13, -0.08, 0.05], dtype=torch.float64)
            )
            model.raw_d_lower.copy_(
                torch.tensor(
                    [0.71, 0.04, 0.68, -0.03, 0.02, 0.73],
                    dtype=torch.float64,
                )
            )
        time = torch.tensor(0.17, dtype=torch.float64)
        q = model.q_matrix()
        d = model.d_matrix()
        drift = -(q + d)
        phi = torch.matrix_exp(time * drift)
        identity_form = torch.eye(3, dtype=torch.float64) - phi @ phi.T
        identity_form = 0.5 * (identity_form + identity_form.T)
        series_form, terms = model._integral_series_covariance(
            drift, 2.0 * d, time
        )
        self.assertGreater(terms, 1)
        torch.testing.assert_close(
            series_form, identity_form, atol=2e-12, rtol=2e-12
        )

        probe = torch.tensor(
            [[0.2, -0.1, 0.05], [-0.1, 0.3, 0.07], [0.05, 0.07, -0.2]],
            dtype=torch.float64,
        )
        identity_gradients = torch.autograd.grad(
            (identity_form * probe).sum(),
            (model.raw_q_upper, model.raw_d_lower),
            retain_graph=True,
        )
        series_gradients = torch.autograd.grad(
            (series_form * probe).sum(),
            (model.raw_q_upper, model.raw_d_lower),
        )
        for actual, expected in zip(series_gradients, identity_gradients):
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)

    def test_dense_1024_float32_boundary_uses_stable_series_without_jitter(self):
        dimension = 1024
        model = StationaryQDForward(
            dimension,
            d_parameterization="psd",
            d_diagonal_floor=0.0,
            initial_d_diagonal=0.5,
            covariance_jitter=0.0,
            dtype=torch.float32,
        )
        # Mimic the dense sign-sized first Adam update that exposed the CUDA
        # cancellation failure in I-Phi@Phi.T at the 1000-step boundary.
        with torch.no_grad():
            q_index = torch.arange(model.raw_q_upper.numel())
            d_index = torch.arange(model.raw_d_lower.numel())
            model.raw_q_upper.add_(
                torch.where(q_index.remainder(2) == 0, 1e-4, -1e-4)
            )
            model.raw_d_lower.add_(
                torch.where(d_index.remainder(3) == 0, -1e-4, 1e-4)
            )
        boundary_time = 0.00010000500033334732
        stats = model.transition_stats(
            torch.zeros(1, dimension, dtype=torch.float32), boundary_time
        )
        self.assertEqual(stats.covariance_evaluation, "adaptive_integral_series")
        self.assertGreaterEqual(stats.covariance_series_terms, 2)
        self.assertEqual(float(stats.covariance_jitter), 0.0)
        self.assertTrue(torch.isfinite(stats.cholesky).all())
        self.assertGreater(float(torch.diagonal(stats.cholesky).min()), 0.0)

    def test_rejects_non_shared_time_and_bad_mask_shape(self):
        model = self.model(3)
        x = torch.randn(2, 3, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "batch-shared"):
            model.transition_stats(x, torch.tensor([0.2, 0.3]))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.model(3, grn_mask_target_source=torch.ones(2, 2))


if __name__ == "__main__":
    unittest.main()
