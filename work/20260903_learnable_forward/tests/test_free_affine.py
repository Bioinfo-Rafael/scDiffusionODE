#!/usr/bin/env python3
"""Mathematical and autograd tests for dense free-affine Model B."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

import torch


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from diffusion.free_affine import FreeAffineForward  # noqa: E402


class FreeAffineForwardTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260904)
        self.dtype = torch.float64

    def assertClose(self, actual, expected, *, atol=1e-10, rtol=1e-10):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    def _set_drift(self, process: FreeAffineForward, w, b=None) -> None:
        w = torch.as_tensor(w, dtype=self.dtype)
        with torch.no_grad():
            process.raw_w.copy_(w + 0.5 * torch.eye(process.dim, dtype=self.dtype))
            if b is not None:
                process.raw_b.copy_(torch.as_tensor(b, dtype=self.dtype))

    def test_zero_raw_parameters_match_standard_vp_transition(self):
        process = FreeAffineForward(3, dtype=self.dtype)
        x = torch.randn(4, 3, dtype=torch.float32)
        time = torch.tensor([0.8] * 4, dtype=self.dtype)

        stats = process.transition_stats(x, time)
        expected_phi = math.exp(-0.5 * 0.8) * torch.eye(3, dtype=self.dtype)
        expected_covariance = (1.0 - math.exp(-0.8)) * torch.eye(
            3, dtype=self.dtype
        )

        self.assertClose(process.raw_w, torch.zeros_like(process.raw_w))
        self.assertClose(process.raw_b, torch.zeros_like(process.raw_b))
        self.assertClose(stats.phi, expected_phi)
        self.assertClose(stats.transition_matrix, expected_phi)
        self.assertClose(stats.affine_shift, torch.zeros(3, dtype=self.dtype))
        self.assertClose(stats.covariance, expected_covariance)
        self.assertClose(stats.mean, x.to(self.dtype) @ expected_phi.T)
        self.assertClose(stats.diffusion_covariance, torch.eye(3, dtype=self.dtype))
        self.assertClose(stats.drift_divergence, torch.tensor(-1.5, dtype=self.dtype))

    def test_singular_zero_w_has_exact_transition_without_inverse(self):
        process = FreeAffineForward(3, dtype=self.dtype)
        b = torch.tensor([0.4, -0.2, 0.1], dtype=self.dtype)
        self._set_drift(process, torch.zeros(3, 3), b)
        x = torch.randn(5, 3, dtype=self.dtype)
        time = 0.7

        stats = process.transition_stats(x, time)

        self.assertClose(stats.w_matrix, torch.zeros(3, 3, dtype=self.dtype))
        self.assertClose(stats.phi, torch.eye(3, dtype=self.dtype))
        self.assertClose(stats.affine_shift, time * b)
        self.assertClose(stats.mean, x + time * b)
        self.assertClose(stats.covariance, time * torch.eye(3, dtype=self.dtype))
        self.assertClose(stats.drift_divergence, torch.zeros((), dtype=self.dtype))

    def test_diagonal_w_matches_componentwise_analytic_solution(self):
        process = FreeAffineForward(3, dtype=self.dtype)
        diagonal = torch.tensor([-0.3, 0.0, 0.4], dtype=self.dtype)
        b = torch.tensor([0.7, -0.5, 0.2], dtype=self.dtype)
        self._set_drift(process, torch.diag(diagonal), b)
        x = torch.randn(2, 3, dtype=self.dtype)
        time = 0.45

        stats = process.transition_stats(x, time)
        expected_phi_diagonal = torch.exp(time * diagonal)
        expected_shift = torch.empty_like(diagonal)
        expected_covariance_diagonal = torch.empty_like(diagonal)
        for index, value in enumerate(diagonal):
            if value == 0:
                expected_shift[index] = time * b[index]
                expected_covariance_diagonal[index] = time
            else:
                expected_shift[index] = b[index] * torch.expm1(time * value) / value
                expected_covariance_diagonal[index] = (
                    torch.expm1(2.0 * time * value) / (2.0 * value)
                )

        expected_phi = torch.diag(expected_phi_diagonal)
        self.assertClose(stats.phi, expected_phi)
        self.assertClose(stats.affine_shift, expected_shift)
        self.assertClose(stats.covariance, torch.diag(expected_covariance_diagonal))
        self.assertClose(stats.mean, x @ expected_phi.T + expected_shift)

    def test_general_nonnormal_transition_obeys_semigroup(self):
        process = FreeAffineForward(3, dtype=self.dtype)
        w = torch.tensor(
            [[-0.4, 0.7, -0.1], [0.0, -0.2, 0.5], [0.15, 0.0, -0.6]],
            dtype=self.dtype,
        )
        b = torch.tensor([0.3, -0.25, 0.1], dtype=self.dtype)
        self._set_drift(process, w, b)
        x = torch.randn(4, 3, dtype=self.dtype)
        first_time, second_time = 0.31, 0.47

        first = process.transition_stats(x, first_time, compute_cholesky=False)
        second = process.transition_stats(x, second_time, compute_cholesky=False)
        total = process.transition_stats(
            x, first_time + second_time, compute_cholesky=False
        )

        composed_phi = second.phi @ first.phi
        composed_shift = second.phi @ first.affine_shift + second.affine_shift
        composed_covariance = (
            second.phi @ first.covariance @ second.phi.T + second.covariance
        )
        self.assertClose(total.phi, composed_phi, atol=2e-10, rtol=2e-10)
        self.assertClose(total.affine_shift, composed_shift, atol=2e-10, rtol=2e-10)
        self.assertClose(
            total.covariance, composed_covariance, atol=2e-10, rtol=2e-10
        )

    def test_q_sample_score_and_noise_metric_use_same_cholesky(self):
        process = FreeAffineForward(3, dtype=self.dtype)
        w = torch.tensor(
            [[-0.8, 0.4, 0.1], [-0.2, -0.5, 0.3], [0.0, -0.1, -0.6]],
            dtype=self.dtype,
        )
        self._set_drift(process, w, [0.2, -0.1, 0.05])
        x = torch.randn(4, 3, dtype=self.dtype)
        noise = torch.randn_like(x)

        sample, returned_noise, stats = process.q_sample(
            x, 0.6, noise=noise, return_stats=True
        )
        self.assertIs(returned_noise, noise)
        self.assertClose(sample, stats.mean + noise @ stats.cholesky.T)

        expected_score = -torch.linalg.solve_triangular(
            stats.cholesky.T, noise.T, upper=True
        ).T
        self.assertClose(process.conditional_score(stats, noise), expected_score)

        residual = torch.randn_like(x)
        inverse_l = torch.linalg.inv(stats.cholesky)
        metric = inverse_l @ inverse_l.T
        expected_quadratic = 0.5 * torch.einsum(
            "bi,ij,bj->b", residual, metric, residual
        )
        self.assertClose(
            process.noise_metric_quadratic(stats, residual), expected_quadratic
        )

    def test_terminal_kl_matches_torch_distribution_and_cross_entropy_identity(self):
        process = FreeAffineForward(2, dtype=self.dtype)
        self._set_drift(
            process,
            [[-0.7, 0.35], [-0.1, -0.25]],
            [0.25, -0.4],
        )
        x = torch.randn(5, 2, dtype=self.dtype)
        stats = process.transition_stats(x, 0.9)

        q = torch.distributions.MultivariateNormal(
            stats.mean, covariance_matrix=stats.covariance
        )
        prior = torch.distributions.MultivariateNormal(
            torch.zeros_like(stats.mean),
            covariance_matrix=torch.eye(2, dtype=self.dtype),
        )
        expected_kl = torch.distributions.kl_divergence(q, prior)
        actual_kl = process.terminal_kl(stats)
        self.assertClose(actual_kl, expected_kl)

        logdet = 2.0 * torch.log(torch.diagonal(stats.cholesky)).sum()
        entropy = 0.5 * (
            2.0 * (1.0 + math.log(2.0 * math.pi)) + logdet
        )
        self.assertClose(
            process.terminal_cross_entropy(stats) - actual_kl,
            entropy.expand_as(actual_kl),
        )

    def test_reparameterized_exact_terms_give_finite_nonzero_w_and_b_gradients(self):
        process = FreeAffineForward(3, dtype=self.dtype)
        self._set_drift(
            process,
            [[-0.6, 0.25, 0.0], [-0.15, -0.4, 0.2], [0.1, 0.0, -0.5]],
            [0.12, -0.08, 0.2],
        )
        x = torch.randn(6, 3, dtype=self.dtype)
        noise = torch.randn_like(x)
        sample, _, stats = process.q_sample(x, 0.55, noise=noise, return_stats=True)

        # A deterministic differentiable stand-in for epsilon_theta makes the
        # reparameterized input path explicit while the metric and KL retain
        # their direct phi dependencies.
        epsilon_pred = 0.17 * sample + 0.09
        residual = epsilon_pred - noise
        loss = (
            process.noise_metric_quadratic(stats, residual).mean()
            + process.terminal_kl(stats).mean()
        )
        loss.backward()

        for name, gradient in (
            ("raw_w", process.raw_w.grad), ("raw_b", process.raw_b.grad)
        ):
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(float(gradient.norm()), 0.0, name)

    def test_jitter_is_explicit_and_batch_time_is_validated(self):
        process = FreeAffineForward(2, covariance_jitter=1e-5, dtype=self.dtype)
        x = torch.randn(3, 2, dtype=self.dtype)
        stats = process.transition_stats(x, torch.full((3,), 0.4, dtype=self.dtype))
        self.assertClose(
            stats.covariance - stats.nominal_covariance,
            1e-5 * torch.eye(2, dtype=self.dtype),
        )
        self.assertEqual(float(stats.covariance_jitter), 1e-5)

        with self.assertRaisesRegex(ValueError, "batch-shared"):
            process.transition_stats(x, torch.tensor([0.2, 0.3, 0.2]))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            process.transition_stats(x, -0.1)

        exact = FreeAffineForward(2, dtype=self.dtype)
        zero_stats = exact.transition_stats(x, 0.0, compute_cholesky=False)
        self.assertClose(zero_stats.covariance, torch.zeros(2, 2, dtype=self.dtype))
        with self.assertRaisesRegex(RuntimeError, "not positive definite"):
            exact.transition_stats(x, 0.0)

        jittered_zero = process.transition_stats(x, 0.0)
        self.assertClose(
            jittered_zero.covariance, 1e-5 * torch.eye(2, dtype=self.dtype)
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
