#!/usr/bin/env python3
"""Equivalence tests for the valid truncated paper ELBO.

The direct reference in this file is *not* raw Equation (7) with its lower
integration limit changed.  It includes the Appendix-I / Theorem-3 boundary
correction ``-log p_theta(x|y_delta) + log q_phi(y_delta|x)``.  Equality with
the compact objective holds after taking the Gaussian/time expectations, not
sample by sample, so this test uses analytic Gaussian expectations and a
high-order float64 Gauss--Legendre quadrature.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from diffusion.free_affine import FreeAffineForward  # noqa: E402
from diffusion.objectives import (  # noqa: E402
    boundary_gaussian_nll,
    compact_valid_truncated_loss,
    expected_forward_score_energy,
    gaussian_entropy,
    score_from_noise,
    standard_normal_cross_entropy,
    standard_normal_kl,
    valid_truncated_direct_loss,
    weighted_noise_quadratic,
)
from diffusion.stationary_qd import StationaryQDForward  # noqa: E402
from diffusion.time_mapping import PhysicalTimeMap  # noqa: E402
from diffusion.training_diffusion import (  # noqa: E402
    LearnableForwardTrainingDiffusion,
)
from models.wrapper import LearnableForwardModel  # noqa: E402



def _dense_stats(process, x, time, *, prescribed_root=False):
    stats = process.transition_stats(x, time)
    if not isinstance(process, StationaryQDForward):
        return stats
    from types import SimpleNamespace
    dense = stats.materialize_for_analysis()
    return SimpleNamespace(
        mean=stats.mean, covariance=dense["covariance"],
        cholesky=dense["noise_root"] if prescribed_root else torch.linalg.cholesky(dense["covariance"]),
        transition_matrix=dense["phi"], affine_shift=x.new_zeros(process.dim),
        diffusion_covariance=process.diffusion_covariance(), drift_divergence=stats.drift_divergence)


def _gauss_legendre_interval(
    lower: float,
    upper: float,
    *,
    order: int,
    reference: torch.Tensor,
):
    """Return differentiable-computation constants for an interval."""

    # Golub--Welsch construction, kept torch-only so the tests do not add a
    # NumPy dependency to this repository's minimal environment.
    indices = torch.arange(
        1, order, dtype=reference.dtype, device=reference.device
    )
    off_diagonal = indices / torch.sqrt(4.0 * indices.square() - 1.0)
    jacobi = reference.new_zeros((order, order))
    jacobi.diagonal(offset=1).copy_(off_diagonal)
    jacobi.diagonal(offset=-1).copy_(off_diagonal)
    nodes, eigenvectors = torch.linalg.eigh(jacobi)
    weights = 2.0 * eigenvectors[0].square()
    half_width = reference.new_tensor(0.5 * (upper - lower))
    midpoint = reference.new_tensor(0.5 * (upper + lower))
    return midpoint + half_width * nodes, half_width * weights


def _boundary_nll(
    process,
    x: torch.Tensor,
    boundary_time: float,
    boundary_noise: torch.Tensor,
) -> tuple[torch.Tensor, object]:
    """Evaluate the actual Appendix-I decoder term on fixed MC noise."""

    stats = _dense_stats(process, x, boundary_time)
    y_boundary = stats.mean + boundary_noise @ stats.cholesky.T

    # A deterministic nontrivial stand-in for a denoiser output.  The
    # equivalence under test does not require an optimal score.  Depending on
    # y_boundary deliberately retains reparameterization gradients.
    feature_mix = x.new_tensor(
        [[0.21, -0.08, 0.04], [0.03, 0.17, -0.05], [-0.07, 0.02, 0.13]]
    )[: x.shape[1], : x.shape[1]]
    predicted_noise = torch.tanh(y_boundary @ feature_mix.T + 0.11)
    model_score = score_from_noise(stats.cholesky, predicted_noise)

    nll = boundary_gaussian_nll(
        x_start=x,
        y_boundary=y_boundary,
        model_score=model_score,
        transition_matrix=stats.transition_matrix,
        affine_shift=stats.affine_shift,
        covariance=stats.covariance,
        cholesky=stats.cholesky,
    ).mean()
    return nll, stats


def _quadrature_path_averages(
    process,
    x: torch.Tensor,
    lower: float,
    upper: float,
    *,
    order: int = 48,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return uniform-time averages of the three direct-form path terms."""

    reference = next(process.parameters())
    times, integration_weights = _gauss_legendre_interval(
        lower, upper, order=order, reference=reference
    )
    duration = reference.new_tensor(upper - lower)

    mismatch_integral = reference.new_zeros(())
    forward_energy_integral = reference.new_zeros(())
    divergence_integral = reference.new_zeros(())

    fixed_residual = x.new_tensor(
        [[0.31, -0.27, 0.09], [-0.18, 0.23, 0.14], [0.07, 0.19, -0.29]]
    )[: x.shape[0], : x.shape[1]]

    for time, integration_weight in zip(times, integration_weights):
        stats = _dense_stats(process, x, time)
        # This is an arbitrary deterministic score error, shared by the direct
        # and compact forms.  Its dependence on the transition mean ensures
        # that the comparison includes pathwise forward-parameter gradients.
        residual = fixed_residual + 0.12 * torch.tanh(stats.mean)
        mismatch = weighted_noise_quadratic(
            stats.cholesky, stats.diffusion_covariance, residual
        ).mean()
        forward_energy = expected_forward_score_energy(
            stats.cholesky, stats.diffusion_covariance
        )

        mismatch_integral = (
            mismatch_integral + integration_weight * mismatch
        )
        forward_energy_integral = (
            forward_energy_integral + integration_weight * forward_energy
        )
        divergence_integral = (
            divergence_integral
            + integration_weight * stats.drift_divergence
        )

    return (
        mismatch_integral / duration,
        forward_energy_integral / duration,
        divergence_integral / duration,
    )


def _valid_direct_and_compact(process):
    """Construct both expected objectives from the same process parameters."""

    parameter = next(process.parameters())
    dim = process.dim
    x = parameter.new_tensor(
        [[0.41, -0.37, 0.22], [-0.16, 0.28, 0.35], [0.09, 0.13, -0.31]]
    )[:, :dim]
    boundary_noise = parameter.new_tensor(
        [[0.17, -0.23, 0.05], [-0.11, 0.29, 0.14], [0.21, 0.07, -0.19]]
    )[:, :dim]
    delta = 0.18
    terminal_time = 0.93
    duration = terminal_time - delta

    boundary_nll, boundary_stats = _boundary_nll(
        process, x, delta, boundary_noise
    )
    terminal_stats = _dense_stats(process, x, terminal_time)
    mismatch, forward_energy, drift_divergence = _quadrature_path_averages(
        process, x, delta, terminal_time
    )

    # The direct Theorem-3 boundary correction contains E[log q_delta].
    # For the Gaussian transition this is exactly -H(q_delta).  A single
    # sampled log-density must not be used here: entropy cancellation is an
    # expectation-level identity.
    expected_boundary_log_q = -gaussian_entropy(boundary_stats.cholesky)
    terminal_cross_entropy = standard_normal_cross_entropy(
        terminal_stats.mean, terminal_stats.covariance
    ).mean()
    terminal_kl = standard_normal_kl(
        terminal_stats.mean,
        terminal_stats.covariance,
        terminal_stats.cholesky,
    ).mean()

    direct = valid_truncated_direct_loss(
        boundary_nll=boundary_nll,
        boundary_log_q=expected_boundary_log_q,
        terminal_cross_entropy=terminal_cross_entropy,
        score_mismatch=mismatch,
        forward_energy=forward_energy,
        drift_divergence=drift_divergence,
        duration=duration,
    )
    compact = compact_valid_truncated_loss(
        boundary_nll=boundary_nll,
        terminal_kl=terminal_kl,
        score_mismatch=mismatch,
        duration=duration,
    )
    return direct, compact


class _AuditDenoiser(nn.Module):
    """Small nonlinear denoiser used to audit the production loss assembly."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.tensor(
                [[0.18, -0.07], [0.04, 0.13]], dtype=torch.float64
            )[:dim, :dim]
        )
        self.bias = nn.Parameter(
            torch.tensor([0.025, -0.035], dtype=torch.float64)[:dim]
        )
        self.time_gain = nn.Parameter(
            torch.tensor([0.045, 0.065], dtype=torch.float64)[:dim]
        )

    def forward(self, noisy, timesteps, **model_kwargs):
        del model_kwargs
        return torch.tanh(
            noisy @ self.weight.T
            + self.bias
            + torch.sin(timesteps.to(noisy.dtype) / 3.0) * self.time_gain
        )


def _audit_model(family: str):
    """Build a non-isotropic production wrapper and training facade."""

    dim = 2
    if family == "stationary_qd":
        process = StationaryQDForward(dim, aux_dim=1, dtype=torch.float64)
        with torch.no_grad():
            process.b.fill_(.23)
    elif family == "free_affine":
        process = FreeAffineForward(dim, dtype=torch.float64)
        with torch.no_grad():
            process.raw_w.copy_(
                torch.tensor(
                    [[0.055, 0.045], [-0.035, -0.025]],
                    dtype=torch.float64,
                )
            )
            process.raw_b.copy_(
                torch.tensor([0.06, -0.04], dtype=torch.float64)
            )
    else:  # pragma: no cover - a test-construction error
        raise ValueError(f"unknown family {family!r}")

    delta = 0.18
    terminal_time = 0.93
    time_map = PhysicalTimeMap(
        torch.tensor([delta, terminal_time], dtype=torch.float64)
    )
    model = LearnableForwardModel(_AuditDenoiser(dim), process)
    diffusion = LearnableForwardTrainingDiffusion(
        time_map,
        loss_mode="paper_elbo",
        normalize_elbo_by_dimension=False,
    )
    return model, diffusion, time_map


def _call_audit_denoiser(
    model: LearnableForwardModel,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the wrapper's dtype/shape call without using its helpers."""

    denoiser_dtype = next(model.denoiser.parameters()).dtype
    result = model.denoiser(
        noisy.to(dtype=denoiser_dtype), timesteps.reshape(-1, 1)
    )
    return result.to(dtype=noisy.dtype)


def _independent_score_from_noise(
    cholesky: torch.Tensor, noise_coordinate: torch.Tensor
) -> torch.Tensor:
    """Compute ``-L^{-T} epsilon`` independently of objectives.py."""

    return -torch.linalg.solve_triangular(
        cholesky.T, noise_coordinate.T, upper=True
    ).T


def _independent_boundary_terms(
    model: LearnableForwardModel,
    x: torch.Tensor,
    boundary_time: float,
    boundary_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Appendix-I boundary NLL and analytic E[log q_delta]."""

    stats = _dense_stats(model.forward_process, x, boundary_time, prescribed_root=True)
    sample = stats.mean + boundary_noise @ stats.cholesky.T
    zero_timesteps = x.new_zeros(x.shape[0])
    prediction = _call_audit_denoiser(model, sample, zero_timesteps)
    model_score = (-torch.linalg.solve(stats.cholesky.T, prediction.T).T
                   if isinstance(model.forward_process, StationaryQDForward) else
                   _independent_score_from_noise(stats.cholesky, prediction))

    rhs = (
        sample
        - stats.affine_shift.unsqueeze(0)
        + model_score @ stats.covariance.T
    )
    decoder_mean = torch.linalg.solve(stats.transition_matrix, rhs.T).T
    decoder_root = torch.linalg.solve(
        stats.transition_matrix, stats.cholesky
    )
    whitened = torch.linalg.solve(decoder_root, (x - decoder_mean).T).T
    _, decoder_logabsdet = torch.linalg.slogdet(decoder_root)
    dim = x.shape[1]
    boundary_nll = 0.5 * (
        whitened.square().sum(dim=1)
        + 2.0 * decoder_logabsdet
        + dim * torch.log(x.new_tensor(2.0 * torch.pi))
    )

    logdet_q = (2 * torch.linalg.slogdet(stats.cholesky)[1]
                if isinstance(model.forward_process, StationaryQDForward) else
                2.0 * torch.log(torch.diagonal(stats.cholesky)).sum())
    entropy_q = 0.5 * (
        logdet_q
        + dim * (1.0 + torch.log(x.new_tensor(2.0 * torch.pi)))
    )
    return boundary_nll, -entropy_q


def _independent_terminal_cross_entropy(stats) -> torch.Tensor:
    """Return ``-E_q log N(0,I)`` without calling objectives.py."""

    dim = stats.mean.shape[1]
    return 0.5 * (
        stats.mean.square().sum(dim=1)
        + torch.trace(stats.covariance)
        + dim * torch.log(stats.mean.new_tensor(2.0 * torch.pi))
    )


def _independent_path_direct_terms(
    model: LearnableForwardModel,
    x: torch.Tensor,
    physical_time: torch.Tensor,
    timesteps: torch.Tensor,
    path_noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute score mismatch, expected forward energy, and divergence."""

    stats = _dense_stats(model.forward_process, x, physical_time, prescribed_root=True)
    sample = stats.mean + path_noise @ stats.cholesky.T
    prediction = _call_audit_denoiser(model, sample, timesteps)
    model_score = (-torch.linalg.solve(stats.cholesky.T, prediction.T).T
                   if isinstance(model.forward_process, StationaryQDForward) else
                   _independent_score_from_noise(stats.cholesky, prediction))
    forward_score = (-torch.linalg.solve(stats.cholesky.T, path_noise.T).T
                     if isinstance(model.forward_process, StationaryQDForward) else
                     _independent_score_from_noise(stats.cholesky, path_noise))
    score_difference = model_score - forward_score
    mismatch = 0.5 * torch.einsum(
        "bi,ij,bj->b",
        score_difference,
        stats.diffusion_covariance,
        score_difference,
    )

    identity = torch.eye(
        x.shape[1], dtype=x.dtype, device=x.device
    )
    inverse_l = (torch.linalg.solve(stats.cholesky, identity)
                 if isinstance(model.forward_process, StationaryQDForward) else
                 torch.linalg.solve_triangular(stats.cholesky, identity, upper=False))
    metric = (
        inverse_l
        @ stats.diffusion_covariance
        @ inverse_l.T
    )
    expected_forward_energy = 0.5 * torch.trace(metric)
    return mismatch, expected_forward_energy, stats.drift_divergence


def _production_compact_and_independent_direct(family: str, *, order: int = 24):
    """Build production compact, valid direct, and invalid raw objectives."""

    model, diffusion, time_map = _audit_model(family)
    x = torch.tensor(
        [[0.42, -0.31], [-0.17, 0.36], [0.08, 0.21]],
        dtype=torch.float64,
    )
    path_noise = torch.tensor(
        [[0.29, -0.24], [-0.13, 0.32], [0.07, 0.18]],
        dtype=torch.float64,
    )
    boundary_noise = torch.tensor(
        [[-0.09, 0.27], [0.22, -0.14], [-0.19, 0.11]],
        dtype=torch.float64,
    )
    times, integration_weights = _gauss_legendre_interval(
        time_map.boundary_time,
        time_map.terminal_time,
        order=order,
        reference=next(model.parameters()),
    )
    duration = x.new_tensor(time_map.duration)

    # Integrating the actual production loss over uniform sampled time gives
    # B_NLL + KL_T + integral(mismatch ds), since training_diffusion already
    # multiplies its sampled path term by T-delta.
    production_integral = x.new_zeros(x.shape[0])
    direct_path_integral = x.new_zeros(x.shape[0])
    for physical_time, integration_weight in zip(times, integration_weights):
        fractional = time_map.physical_time_to_fractional_index(
            physical_time
        )
        timesteps = fractional.expand(x.shape[0]).clone()
        production_losses = diffusion.training_losses(
            model,
            x,
            timesteps,
            noise=path_noise,
            boundary_noise=boundary_noise,
        )
        production_integral = production_integral + (
            integration_weight / duration
        ) * production_losses["loss"]

        mismatch, forward_energy, divergence = _independent_path_direct_terms(
            model, x, physical_time, timesteps, path_noise
        )
        direct_path_integral = direct_path_integral + integration_weight * (
            mismatch - forward_energy - divergence
        )

    boundary_nll, expected_boundary_log_q = _independent_boundary_terms(
        model, x, time_map.boundary_time, boundary_noise
    )
    terminal_stats = _dense_stats(model.forward_process, x, time_map.terminal_time, prescribed_root=True)
    terminal_cross_entropy = _independent_terminal_cross_entropy(
        terminal_stats
    )

    valid_direct = (
        boundary_nll
        + expected_boundary_log_q
        + terminal_cross_entropy
        + direct_path_integral
    )
    # Negative raw truncated Eq. (7): no Appendix-I/Theorem-3 boundary
    # correction.  This must not be accepted as the production paper_elbo.
    raw_truncated = terminal_cross_entropy + direct_path_integral
    return model, production_integral, valid_direct, raw_truncated


class ValidTruncatedObjectiveEquivalenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260903)

    @staticmethod
    def _make_stationary_qd() -> StationaryQDForward:
        process = StationaryQDForward(3, aux_dim=2, dtype=torch.float64)
        with torch.no_grad():
            process.raw_q_k.copy_(torch.tensor([[0., .09], [-.04, 0.]], dtype=torch.float64))
            process.b.copy_(torch.tensor([[.26, .05], [-.03, .19]], dtype=torch.float64))
        return process

    @staticmethod
    def _make_free_affine() -> FreeAffineForward:
        process = FreeAffineForward(3, dtype=torch.float64)
        with torch.no_grad():
            process.raw_w.copy_(
                torch.tensor(
                    [
                        [0.07, 0.05, -0.02],
                        [-0.04, -0.03, 0.06],
                        [0.01, -0.05, 0.04],
                    ],
                    dtype=torch.float64,
                )
            )
            process.raw_b.copy_(
                torch.tensor([0.08, -0.05, 0.03], dtype=torch.float64)
            )
        return process

    def _assert_value_and_gradient_equivalence(self, process):
        direct, compact = _valid_direct_and_compact(process)
        torch.testing.assert_close(
            direct,
            compact,
            atol=2e-10,
            rtol=2e-10,
            msg=(
                "Theorem-3 valid direct form (including -log p + log q) "
                "must equal the compact boundary-NLL + terminal-KL + "
                "weighted-mismatch form"
            ),
        )

        parameters = tuple(process.parameters())
        direct_gradients = torch.autograd.grad(
            direct, parameters, retain_graph=True
        )
        compact_gradients = torch.autograd.grad(compact, parameters)
        for parameter, direct_gradient, compact_gradient in zip(
            parameters, direct_gradients, compact_gradients
        ):
            self.assertTrue(torch.isfinite(direct_gradient).all())
            self.assertTrue(torch.isfinite(compact_gradient).all())
            torch.testing.assert_close(
                direct_gradient,
                compact_gradient,
                atol=2e-9,
                rtol=2e-9,
                msg=f"gradient mismatch for parameter shape {tuple(parameter.shape)}",
            )

    def test_stationary_qd_valid_direct_matches_compact(self):
        self._assert_value_and_gradient_equivalence(
            self._make_stationary_qd()
        )

    def test_free_affine_valid_direct_matches_compact(self):
        self._assert_value_and_gradient_equivalence(self._make_free_affine())

    def _assert_production_matches_independent_valid_direct(self, family):
        model, production, valid_direct, raw_truncated = (
            _production_compact_and_independent_direct(family)
        )
        torch.testing.assert_close(
            production,
            valid_direct,
            atol=4e-10,
            rtol=4e-10,
            msg=(
                "production paper_elbo must equal the independently assembled "
                "Appendix-I/Theorem-3 valid direct form"
            ),
        )

        parameters = tuple(model.parameters())
        production_gradients = torch.autograd.grad(
            production.mean(), parameters, retain_graph=True
        )
        direct_gradients = torch.autograd.grad(
            valid_direct.mean(), parameters, retain_graph=True
        )
        for (name, _), production_gradient, direct_gradient in zip(
            model.named_parameters(), production_gradients, direct_gradients
        ):
            self.assertTrue(torch.isfinite(production_gradient).all(), name)
            self.assertTrue(torch.isfinite(direct_gradient).all(), name)
            torch.testing.assert_close(
                production_gradient,
                direct_gradient,
                atol=5e-9,
                rtol=5e-9,
                msg=f"production/direct gradient mismatch for {name}",
            )

        # A raw [delta,T] Eq. (7) integral lacks both pieces of the Theorem-3
        # boundary correction.  Guard against accidentally relabelling it as
        # paper_elbo in a future refactor.
        self.assertGreater(
            float((raw_truncated - production).abs().max().detach()),
            1e-3,
        )
        raw_gradients = torch.autograd.grad(raw_truncated.mean(), parameters)
        largest_gradient_difference = max(
            float((raw - expected).abs().max().detach())
            for raw, expected in zip(raw_gradients, production_gradients)
        )
        self.assertGreater(largest_gradient_difference, 1e-4)

    def test_stationary_qd_production_paper_elbo_matches_valid_direct(self):
        self._assert_production_matches_independent_valid_direct(
            "stationary_qd"
        )

    def test_free_affine_production_paper_elbo_matches_valid_direct(self):
        self._assert_production_matches_independent_valid_direct("free_affine")


if __name__ == "__main__":
    unittest.main()
