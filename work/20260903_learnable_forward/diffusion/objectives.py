"""Gaussian objectives for the dense learnable-forward experiments.

The functions in this module deliberately operate on a *shared* transition
covariance.  A training batch has independent ``x`` and Gaussian noise, but a
single physical time and hence a single ``(Sigma, L, g g^T)``.  This is the
computational contract used by the batch-shared time sampler.

No inverse is formed explicitly.  In particular, when ``Sigma = L L^T`` and
the denoiser predicts a Cholesky noise coordinate, the score-space metric is
``L^{-1} (g g^T) L^{-T}``, not generally ``Sigma^{-1}``.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


LOG_TWO_PI = math.log(2.0 * math.pi)


def _check_shared_cholesky(cholesky: torch.Tensor) -> int:
    if cholesky.ndim != 2 or cholesky.shape[0] != cholesky.shape[1]:
        raise ValueError(
            "cholesky must be one shared square [d,d] matrix, got "
            f"{tuple(cholesky.shape)}"
        )
    return int(cholesky.shape[0])


def _check_batch_vectors(values: torch.Tensor, dim: int, name: str) -> None:
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"{name} must have shape [batch,{dim}], got {tuple(values.shape)}")


def solve_lower(cholesky: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Return ``L^{-1} values`` for row-major batch vectors."""

    dim = _check_shared_cholesky(cholesky)
    _check_batch_vectors(values, dim, "values")
    return torch.linalg.solve_triangular(
        cholesky, values.transpose(0, 1), upper=False
    ).transpose(0, 1)


def solve_cholesky_transpose(
    cholesky: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Return ``L^{-T} values`` for row-major batch vectors."""

    dim = _check_shared_cholesky(cholesky)
    _check_batch_vectors(values, dim, "values")
    return torch.linalg.solve_triangular(
        cholesky.transpose(0, 1), values.transpose(0, 1), upper=True
    ).transpose(0, 1)


def score_from_noise(cholesky: torch.Tensor, noise_prediction: torch.Tensor) -> torch.Tensor:
    """Map a predicted Cholesky noise coordinate to a score."""

    return -solve_cholesky_transpose(cholesky, noise_prediction)


def weighted_noise_quadratic(
    cholesky: torch.Tensor,
    diffusion_covariance: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Compute ``0.5 * ||L^{-T} residual||^2_{g g^T}`` per example."""

    dim = _check_shared_cholesky(cholesky)
    if diffusion_covariance.shape != (dim, dim):
        raise ValueError(
            "diffusion_covariance must have shape "
            f"{(dim, dim)}, got {tuple(diffusion_covariance.shape)}"
        )
    transformed = solve_cholesky_transpose(cholesky, residual)
    return 0.5 * torch.einsum(
        "bi,ij,bj->b", transformed, diffusion_covariance, transformed
    )


def forward_score_energy(
    cholesky: torch.Tensor,
    diffusion_covariance: torch.Tensor,
    sampling_noise: torch.Tensor,
) -> torch.Tensor:
    """Compute ``0.5 * ||s_phi||^2_{g g^T}`` per sampled example."""

    return weighted_noise_quadratic(
        cholesky, diffusion_covariance, sampling_noise
    )


def expected_forward_score_energy(
    cholesky: torch.Tensor, diffusion_covariance: torch.Tensor
) -> torch.Tensor:
    """Analytic expectation of :func:`forward_score_energy` over N(0,I)."""

    dim = _check_shared_cholesky(cholesky)
    identity = torch.eye(dim, dtype=cholesky.dtype, device=cholesky.device)
    inverse_l = torch.linalg.solve_triangular(cholesky, identity, upper=False)
    metric = inverse_l @ diffusion_covariance @ inverse_l.transpose(0, 1)
    return 0.5 * torch.trace(metric)


def gaussian_entropy(cholesky: torch.Tensor) -> torch.Tensor:
    """Entropy of a Gaussian with a shared covariance ``L L^T``."""

    dim = _check_shared_cholesky(cholesky)
    diagonal = torch.diagonal(cholesky)
    if torch.any(diagonal <= 0):
        raise ValueError("cholesky diagonal must be strictly positive")
    logdet = 2.0 * torch.log(diagonal).sum()
    return 0.5 * (logdet + dim * (1.0 + LOG_TWO_PI))


def conditional_gaussian_log_prob(
    value: torch.Tensor, mean: torch.Tensor, cholesky: torch.Tensor
) -> torch.Tensor:
    """Log density for ``N(mean, L L^T)`` with one shared covariance."""

    dim = _check_shared_cholesky(cholesky)
    _check_batch_vectors(value, dim, "value")
    _check_batch_vectors(mean, dim, "mean")
    whitened = solve_lower(cholesky, value - mean)
    logdet = 2.0 * torch.log(torch.diagonal(cholesky)).sum()
    return -0.5 * (
        whitened.square().sum(dim=1) + logdet + dim * LOG_TWO_PI
    )


def standard_normal_cross_entropy(
    mean: torch.Tensor, covariance: torch.Tensor
) -> torch.Tensor:
    """``-E_q log N(0,I)`` for ``q=N(mean,covariance)`` per example."""

    if mean.ndim != 2:
        raise ValueError(f"mean must have shape [batch,d], got {tuple(mean.shape)}")
    dim = int(mean.shape[1])
    if covariance.shape != (dim, dim):
        raise ValueError(
            f"covariance must have shape {(dim, dim)}, got {tuple(covariance.shape)}"
        )
    constant = torch.trace(covariance) + dim * LOG_TWO_PI
    return 0.5 * (mean.square().sum(dim=1) + constant)


def standard_normal_kl(
    mean: torch.Tensor, covariance: torch.Tensor, cholesky: torch.Tensor
) -> torch.Tensor:
    """``KL(N(mean,covariance) || N(0,I))`` per example."""

    dim = _check_shared_cholesky(cholesky)
    _check_batch_vectors(mean, dim, "mean")
    if covariance.shape != (dim, dim):
        raise ValueError(
            f"covariance must have shape {(dim, dim)}, got {tuple(covariance.shape)}"
        )
    diagonal = torch.diagonal(cholesky)
    if torch.any(diagonal <= 0):
        raise ValueError("cholesky diagonal must be strictly positive")
    logdet = 2.0 * torch.log(diagonal).sum()
    return 0.5 * (
        mean.square().sum(dim=1) + torch.trace(covariance) - dim - logdet
    )


def boundary_gaussian_nll(
    *,
    x_start: torch.Tensor,
    y_boundary: torch.Tensor,
    model_score: torch.Tensor,
    transition_matrix: torch.Tensor,
    affine_shift: Optional[torch.Tensor],
    covariance: torch.Tensor,
    cholesky: torch.Tensor,
) -> torch.Tensor:
    """Appendix-I Gaussian decoder negative log likelihood.

    For ``q(y_delta|x)=N(Phi x+h, Sigma)``, Theorem 3 uses

    ``mu = Phi^{-1}(y_delta-h+Sigma*s_theta)`` and
    ``Cov = Phi^{-1} Sigma Phi^{-T}``.

    ``Phi^{-1} L`` is a (not necessarily triangular) square root of this
    decoder covariance, which lets us evaluate the density without explicitly
    constructing either an inverse or another Cholesky decomposition.
    """

    dim = _check_shared_cholesky(cholesky)
    for name, value in (
        ("x_start", x_start),
        ("y_boundary", y_boundary),
        ("model_score", model_score),
    ):
        _check_batch_vectors(value, dim, name)
    if transition_matrix.shape != (dim, dim):
        raise ValueError(
            "transition_matrix must have shape "
            f"{(dim, dim)}, got {tuple(transition_matrix.shape)}"
        )
    if covariance.shape != (dim, dim):
        raise ValueError(
            f"covariance must have shape {(dim, dim)}, got {tuple(covariance.shape)}"
        )
    if affine_shift is None:
        affine_shift = x_start.new_zeros(dim)
    if affine_shift.shape != (dim,):
        raise ValueError(
            f"affine_shift must have shape {(dim,)}, got {tuple(affine_shift.shape)}"
        )

    rhs = (
        y_boundary
        - affine_shift.unsqueeze(0)
        + model_score @ covariance.transpose(0, 1)
    )
    decoder_mean = torch.linalg.solve(
        transition_matrix, rhs.transpose(0, 1)
    ).transpose(0, 1)
    decoder_root = torch.linalg.solve(transition_matrix, cholesky)
    residual = x_start - decoder_mean
    whitened = torch.linalg.solve(
        decoder_root, residual.transpose(0, 1)
    ).transpose(0, 1)
    sign, logabsdet_root = torch.linalg.slogdet(decoder_root)
    if torch.any(sign == 0):
        raise RuntimeError("Appendix-I decoder covariance root is singular")
    return 0.5 * (
        whitened.square().sum(dim=1)
        + 2.0 * logabsdet_root
        + dim * LOG_TWO_PI
    )


def valid_truncated_direct_loss(
    *,
    boundary_nll: torch.Tensor,
    boundary_log_q: torch.Tensor,
    terminal_cross_entropy: torch.Tensor,
    score_mismatch: torch.Tensor,
    forward_energy: torch.Tensor,
    drift_divergence: torch.Tensor,
    duration: float | torch.Tensor,
) -> torch.Tensor:
    """Negative valid truncated ELBO in Theorem-3 direct form.

    This includes the complete boundary correction ``-log p + log q``.  It is
    intentionally *not* the raw Eq. (7) integral with its lower limit merely
    changed from zero to ``delta``.
    """

    return (
        boundary_nll
        + boundary_log_q
        + terminal_cross_entropy
        + duration * (score_mismatch - forward_energy - drift_divergence)
    )


def compact_valid_truncated_loss(
    *,
    boundary_nll: torch.Tensor,
    terminal_kl: torch.Tensor,
    score_mismatch: torch.Tensor,
    duration: float | torch.Tensor,
) -> torch.Tensor:
    """Negative valid truncated ELBO after the entropy cancellation."""

    return boundary_nll + terminal_kl + duration * score_mismatch
