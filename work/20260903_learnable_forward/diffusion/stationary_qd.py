"""Exact stationary Q/D mediated by one learned auxiliary subspace.

Only explicit analysis/GRN methods materialize gene-space matrices. Transition
statistics store d x K and K x K tensors; their noise root is generally NOT a
gene-space triangular Cholesky factor.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .grn import effective_grn_mask, off_mask_penalty, validate_penalty_norm

PARAMETERIZATION = "auxiliary_shared_subspace"
MODEL_SCHEMA_VERSION = 2


def _inverse_softplus(value):
    return value + torch.log(-torch.expm1(-value))


def _solve(matrix, rows, *, upper=None):
    flat = rows.reshape(-1, rows.shape[-1])
    result = (torch.linalg.solve(matrix, flat.T) if upper is None else
              torch.linalg.solve_triangular(matrix, flat.T, upper=upper))
    return result.T.reshape_as(rows)


@dataclass(frozen=True)
class StationaryQDTransition:
    time: torch.Tensor
    mean: torch.Tensor
    z: torch.Tensor
    q_k: torch.Tensor
    d_k: torch.Tensor
    sigma2: torch.Tensor
    phi_k: torch.Tensor
    phi_perp: torch.Tensor
    covariance_k: torch.Tensor
    variance_perp: torch.Tensor
    cholesky_k: Optional[torch.Tensor]
    mean_parallel: torch.Tensor
    mean_perp: torch.Tensor
    covariance_evaluation: str
    covariance_series_terms: int

    def split(self, rows):
        rows = rows.to(self.z)
        parallel = rows @ self.z
        return parallel, rows - parallel @ self.z.T

    def apply(self, rows, small, scalar):
        parallel, perp = self.split(rows)
        return scalar * perp + (parallel @ small.T) @ self.z.T

    def transition_mean(self, rows):
        return self.apply(rows, self.phi_k, self.phi_perp)

    def noise_transform(self, rows):
        if self.cholesky_k is None:
            raise ValueError("cholesky_k is required")
        return self.apply(rows, self.cholesky_k, self.variance_perp.sqrt())

    def score(self, rows):
        if self.cholesky_k is None:
            raise ValueError("cholesky_k is required")
        parallel, perp = self.split(rows)
        return -perp / self.variance_perp.sqrt() - _solve(
            self.cholesky_k.T, parallel, upper=True) @ self.z.T

    def covariance_logdet(self):
        return ((self.z.shape[0] - self.z.shape[1]) * self.variance_perp.log()
                + 2 * self.cholesky_k.diagonal().log().sum())

    def covariance_trace(self):
        return ((self.z.shape[0] - self.z.shape[1]) * self.variance_perp
                + self.covariance_k.trace())

    def mean_squared(self):
        return self.mean_parallel.square().sum(-1) + self.mean_perp.square().sum(-1)

    @property
    def drift_divergence(self):
        return -(self.z.shape[0] - self.z.shape[1]) * self.sigma2 - self.d_k.trace()

    def materialize_for_analysis(self):
        """Explicit dense diagnostics only; never called by production losses.

        noise_root is a covariance square root, NOT the dense Cholesky.
        """
        eye = torch.eye(self.z.shape[0], device=self.z.device, dtype=self.z.dtype)
        complement = eye - self.z @ self.z.T
        return {
            "phi": self.phi_perp * complement + self.z @ self.phi_k @ self.z.T,
            "covariance": self.variance_perp * complement + self.z @ self.covariance_k @ self.z.T,
            "noise_root": (None if self.cholesky_k is None else
                           self.variance_perp.sqrt() * complement + self.z @ self.cholesky_k @ self.z.T),
        }


class StationaryQDForward(nn.Module):
    parameterization = PARAMETERIZATION
    model_schema_version = MODEL_SCHEMA_VERSION

    def __init__(self, dim: int, *, aux_dim: int,
                 model_a_parameterization: str = PARAMETERIZATION,
                 learn_isotropic_d: bool = True, isotropic_d_init: float = 0.5,
                 isotropic_d_floor: float = 1e-6, auxiliary_b_init_scale: float = 0.0,
                 covariance_jitter: float = 0.0,
                 grn_mask_target_source=None, allow_self_edges: bool = True,
                 grn_penalty_weight: float = 0.0, grn_penalty_norm: str = "l1",
                 device=None, dtype=torch.float64):
        super().__init__()
        if int(dim) != dim or dim <= 0:
            raise ValueError("dim must be a positive integer")
        if int(aux_dim) != aux_dim or not 1 <= aux_dim < dim:
            raise ValueError("aux_dim must be an integer in [1, dim) to avoid gene-space decompositions")
        if model_a_parameterization != PARAMETERIZATION:
            raise ValueError("Model A requires auxiliary_shared_subspace; old dense checkpoints cannot resume")
        if covariance_jitter != 0:
            raise ValueError("Model A requires covariance_jitter=0; isotropic D is part of the SDE")
        if not math.isfinite(isotropic_d_floor) or isotropic_d_floor <= 0:
            raise ValueError("isotropic_d_floor must be strictly positive and finite")
        if not math.isfinite(isotropic_d_init) or isotropic_d_init <= isotropic_d_floor:
            raise ValueError("isotropic_d_init must exceed isotropic_d_floor")
        if not math.isfinite(auxiliary_b_init_scale) or auxiliary_b_init_scale < 0:
            raise ValueError("auxiliary_b_init_scale must be finite and nonnegative")
        if not math.isfinite(grn_penalty_weight) or grn_penalty_weight < 0:
            raise ValueError("grn_penalty_weight must be finite and nonnegative")
        self.dim, self.aux_dim = int(dim), int(aux_dim)
        dim, aux_dim = self.dim, self.aux_dim
        self.isotropic_d_floor = float(isotropic_d_floor)
        self.grn_penalty_weight = float(grn_penalty_weight)
        self.grn_penalty_norm = validate_penalty_norm(grn_penalty_norm)
        # Model A constrains only off-diagonal effective interactions.
        self.allow_self_edges = True
        kw = dict(device=device, dtype=dtype)
        self.raw_embedding = nn.Parameter(torch.randn(dim, aux_dim, **kw) / math.sqrt(dim))
        self.raw_q_k = nn.Parameter(torch.zeros(aux_dim, aux_dim, **kw))
        self.b = nn.Parameter(auxiliary_b_init_scale * torch.eye(aux_dim, **kw))
        rho = _inverse_softplus(torch.tensor(isotropic_d_init - isotropic_d_floor, **kw))
        if learn_isotropic_d:
            self.raw_isotropic_d = nn.Parameter(rho)
        else:
            self.register_buffer("raw_isotropic_d", rho)
        self.register_buffer("_identity_k", torch.eye(aux_dim, **kw), persistent=False)
        self.register_buffer("_schema_version", torch.tensor(MODEL_SCHEMA_VERSION, device=device))
        self.register_buffer("_parameterization_code", torch.tensor(1, device=device))
        self.register_buffer("_checkpoint_aux_dim", torch.tensor(aux_dim, device=device))
        # Persist the floor as well: loading weights must not silently change D.
        self.register_buffer("_isotropic_floor", torch.tensor(isotropic_d_floor, **kw))
        if not bool(self._isotropic_floor > 0) or not bool(torch.isfinite(self._isotropic_floor)):
            raise ValueError("isotropic_d_floor must remain positive and finite in the forward dtype")
        mask, self._has_grn_mask = effective_grn_mask(
            grn_mask_target_source, dim=dim, allow_self_edges=True, reference=self.raw_embedding)
        self.register_buffer("grn_mask_target_source", mask)

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]].update(self.provenance())

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        expected = {"_schema_version": MODEL_SCHEMA_VERSION,
                    "_parameterization_code": 1, "_checkpoint_aux_dim": self.aux_dim}
        for name, value in expected.items():
            stored = state_dict.get(prefix + name)
            if stored is None or stored.numel() != 1 or stored.item() != value:
                raise RuntimeError("Incompatible Model A checkpoint: old dense or mismatched "
                                   f"schema/parameterization/aux_dim ({prefix + name}); start a new run")
        floor = state_dict.get(prefix + "_isotropic_floor")
        if floor is None or not torch.equal(floor.to(self._isotropic_floor), self._isotropic_floor):
            raise RuntimeError("Incompatible Model A checkpoint: isotropic_d_floor mismatch")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def provenance(self):
        return dict(model_schema_version=MODEL_SCHEMA_VERSION,
                    model_a_parameterization=PARAMETERIZATION, aux_dim=self.aux_dim,
                    isotropic_d_floor=self.isotropic_d_floor,
                    full_d_cholesky=False, full_d_matrix_exponential=False,
                    approximation=False)

    def basis(self):
        z, r = torch.linalg.qr(self.raw_embedding, mode="reduced")
        if not bool(torch.isfinite(r).all()) or bool((r.diagonal() == 0).any()):
            raise RuntimeError("Model A raw embedding is non-finite or rank deficient")
        return z

    def q_auxiliary(self):
        return self.raw_q_k - self.raw_q_k.T

    def isotropic_d(self):
        return self._isotropic_floor + F.softplus(self.raw_isotropic_d)

    def d_auxiliary(self):
        return self.isotropic_d() * self._identity_k + self.b @ self.b.T

    def q_matrix(self):
        """Explicit gene-space materialization for analysis only."""
        z = self.basis()
        return z @ self.q_auxiliary() @ z.T

    def d_matrix(self):
        """Explicit gene-space materialization for analysis only."""
        zb = self.basis() @ self.b
        return self.isotropic_d() * torch.eye(self.dim, device=zb.device, dtype=zb.dtype) + zb @ zb.T

    def interaction_matrix(self):
        """GRN/analysis only; [target, source], with no mask on Z."""
        z = self.basis()
        return z @ (self.q_auxiliary() + self.b @ self.b.T) @ z.T

    def stationary_operator(self):
        return self.q_matrix() + self.d_matrix()

    def drift_matrix(self):
        return -self.stationary_operator()

    def diffusion_covariance(self):
        return 2 * self.d_matrix()

    def drift(self, states):
        z = self.basis()
        states = states.to(z)
        return -self.isotropic_d() * states - ((states @ z) @ (self.q_auxiliary() + self.b @ self.b.T).T) @ z.T

    def apply_diffusion_covariance(self, states):
        z = self.basis()
        states = states.to(z)
        return 2 * (self.isotropic_d() * states + ((states @ z) @ (self.b @ self.b.T)) @ z.T)

    def diffusion_noise(self, noise):
        # K-space eigensystem gives a symmetric diffusion root, no Cholesky.
        z = self.basis()
        parallel = noise.to(z) @ z
        perp = noise.to(z) - parallel @ z.T
        # SVD of B avoids differentiating repeated eigenvalues of D in training;
        # this method is used only in no-grad reverse sampling.
        u, values, _ = torch.linalg.svd(self.b)
        root = (u * (2 * (self.isotropic_d() + values.square())).sqrt()) @ u.T
        return (2 * self.isotropic_d()).sqrt() * perp + (parallel @ root.T) @ z.T

    def drift_divergence(self):
        return -self.dim * self.isotropic_d() - self.b.square().sum()

    def stationarity_residual(self):
        drift = self.drift_matrix()
        return drift + drift.T + self.diffusion_covariance()

    def _physical_time(self, time, *, device, dtype) -> torch.Tensor:
        value = torch.as_tensor(time, device=device, dtype=dtype)
        if value.numel() == 0:
            raise ValueError("time must not be empty")
        flat = value.reshape(-1)
        if flat.numel() > 1:
            first_detached = flat[0].detach()
            if not torch.all(flat.detach() == first_detached):
                raise ValueError(
                    "StationaryQDForward requires a batch-shared physical time"
                )
        scalar = flat[0]
        if not bool(torch.isfinite(scalar).detach().item()):
            raise ValueError("physical time must be finite")
        if bool((scalar < 0).detach().item()):
            raise ValueError("physical time must be nonnegative")
        return scalar

    def _integral_series_covariance(
        self,
        drift: torch.Tensor,
        diffusion_covariance: torch.Tensor,
        physical_time: torch.Tensor,
        *,
        max_terms: int = 64,
    ) -> Tuple[torch.Tensor, int]:
        """Evaluate the exact covariance integral without subtracting near-I matrices.

        For constant ``F=drift`` and ``a=diffusion_covariance``, define

        ``K_0=a`` and ``K_{n+1}=F K_n + K_n F^T``.  Integrating the Taylor
        series of ``exp(uF) a exp(uF^T)`` gives

        ``Sigma(s) = sum_n s^(n+1) K_n / (n+1)!``.

        The recurrence below evaluates that convergent series adaptively.  It
        has the same mathematical target as ``I-Phi Phi^T`` but avoids the
        catastrophic cancellation of two near-identity 1024 x 1024 float32
        matrices at the lower boundary.  No jitter, clipping, or change to the
        declared SDE is introduced.
        """

        term = physical_time * diffusion_covariance
        covariance = term
        finfo = torch.finfo(covariance.dtype)
        detached_drift = drift.detach().abs()
        # In the entrywise max norm,
        # ||F X + X F^T||_max <= 2 ||F||_infinity ||X||_max.
        operator_bound = (
            2.0
            * physical_time.detach().abs()
            * detached_drift.sum(dim=1).amax()
        )
        converged = False
        terms_used = 1
        for order in range(1, int(max_terms)):
            term = (physical_time / float(order + 1)) * (
                drift @ term + term @ drift.transpose(-1, -2)
            )
            covariance = covariance + term
            terms_used = order + 1
            term_scale = term.detach().abs().amax()
            covariance_scale = covariance.detach().abs().amax()
            threshold = 8.0 * finfo.eps * torch.clamp(
                covariance_scale, min=finfo.tiny
            )
            # Future term ratios are bounded by operator_bound/(n+2), and
            # decrease thereafter.  Stop only when the resulting geometric
            # tail bound is below rounding scale; a merely small current term
            # is insufficient for a highly non-normal learned drift.
            next_ratio_bound = operator_bound / float(order + 2)
            if bool((next_ratio_bound < 1.0).detach().cpu().item()):
                tail_bound = (
                    term_scale
                    * next_ratio_bound
                    / torch.clamp(1.0 - next_ratio_bound, min=finfo.eps)
                )
            else:
                tail_bound = term_scale.new_tensor(float("inf"))
            if bool((tail_bound <= threshold).detach().cpu().item()):
                converged = True
                break
        if not converged:
            raise RuntimeError(
                "stationary-Q/D covariance integral series did not converge "
                f"within {max_terms} terms; the learned drift may be too "
                "large for stable boundary evaluation; "
                + self._covariance_diagnostics(covariance, self.isotropic_d(), physical_time)
            )
        covariance = 0.5 * (
            covariance + covariance.transpose(-1, -2)
        )
        return covariance, terms_used

    @staticmethod
    def _covariance_diagnostics(covariance, sigma2, physical_time):
        with torch.no_grad():
            minimum, condition = float("nan"), float("inf")
            if bool(torch.isfinite(covariance).all()):
                try:
                    minimum = torch.linalg.eigvalsh(covariance).min().item()
                    condition = torch.linalg.cond(covariance).item()
                except RuntimeError:
                    pass  # Keep the original numerical failure and its context.
        return (f"min_eig={minimum}, condition_number={condition}, "
                f"sigma²={sigma2.item()}, physical_time={physical_time.item()}")

    def _factor_covariance(self, covariance, sigma2, physical_time):
        factor, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
        if bool((info != 0).any()) or not bool(torch.isfinite(factor).all()):
            raise RuntimeError("Model A Sigma_K Cholesky failed without jitter: "
                               + self._covariance_diagnostics(covariance, sigma2, physical_time))
        return factor

    def transition_stats(self, x, time, *, compute_cholesky=True):
        if not torch.is_tensor(x) or x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(f"x must end in dimension {self.dim}")
        z = self.basis()
        x = x.to(z)
        s = self._physical_time(time, device=z.device, dtype=z.dtype)
        if compute_cholesky and bool(s <= 0):
            raise ValueError("physical time must be strictly positive for Cholesky/score")
        sigma2, q_k, d_k = self.isotropic_d(), self.q_auxiliary(), self.d_auxiliary()
        a_k = q_k + d_k
        phi_k = torch.matrix_exp(-s * a_k)
        phi_perp = torch.exp(-sigma2 * s)
        variance_perp = -torch.expm1(-2 * sigma2 * s)
        # Evaluate the same covariance integral to rounding precision near zero.
        # The branch is parameter independent; no clipping/jitter or rank approximation.
        if bool(s.detach() <= 8 * self.aux_dim * torch.finfo(s.dtype).eps):
            covariance_k, terms = self._integral_series_covariance(-a_k, 2 * d_k, s)
            evaluation = "adaptive_integral_series"
        else:
            covariance_k = self._identity_k - phi_k @ phi_k.T
            covariance_k = (covariance_k + covariance_k.T) / 2
            terms, evaluation = 0, "stationary_identity"
        if bool(s > 0) and (not bool(torch.isfinite(variance_perp)) or not bool(variance_perp > 0)):
            raise RuntimeError("Model A complement variance is not positive and finite: "
                               + self._covariance_diagnostics(covariance_k, sigma2, s))
        cholesky = self._factor_covariance(covariance_k, sigma2, s) if compute_cholesky else None
        parallel = x @ z
        mean_perp = phi_perp * (x - parallel @ z.T)
        mean_parallel = parallel @ phi_k.T
        return StationaryQDTransition(s, mean_perp + mean_parallel @ z.T,
                                      z, q_k, d_k, sigma2, phi_k, phi_perp,
                                      covariance_k, variance_perp, cholesky,
                                      mean_parallel, mean_perp, evaluation, terms)

    @staticmethod
    def sample_from_stats(stats, noise=None):
        if noise is None:
            noise = torch.randn_like(stats.mean)
        if noise.shape != stats.mean.shape:
            raise ValueError("noise must have the same shape as the mean")
        noise = noise.to(stats.mean)
        return stats.mean + stats.noise_transform(noise), noise

    def q_sample(self, x, time, noise=None, *, return_stats=False):
        stats = self.transition_stats(x, time)
        sample, noise = self.sample_from_stats(stats, noise)
        return (sample, noise, stats) if return_stats else (sample, noise)

    @staticmethod
    def conditional_score(stats, noise):
        return stats.score(noise)

    @staticmethod
    def noise_metric_quadratic(stats, residual):
        parallel, perp = stats.split(residual)
        u = _solve(stats.cholesky_k.T, parallel, upper=True)
        return stats.sigma2 / stats.variance_perp * perp.square().sum(-1) + (u @ stats.d_k * u).sum(-1)

    @staticmethod
    def terminal_kl(stats):
        return 0.5 * (stats.mean_squared() + stats.covariance_trace()
                      - stats.covariance_logdet() - stats.z.shape[0])

    @staticmethod
    def terminal_cross_entropy(stats):
        return 0.5 * (stats.mean_squared() + stats.covariance_trace()
                      + stats.z.shape[0] * math.log(2 * math.pi))

    @staticmethod
    def boundary_mean(stats, y, score):
        yp, yo = stats.split(y)
        sp, so = stats.split(score)
        return ((yo + stats.variance_perp * so) / stats.phi_perp
                + _solve(stats.phi_k, yp + sp @ stats.covariance_k.T) @ stats.z.T)

    @staticmethod
    def boundary_noise(stats, noise):
        parallel, perp = stats.split(noise)
        return (stats.variance_perp.sqrt() / stats.phi_perp * perp
                + _solve(stats.phi_k, parallel @ stats.cholesky_k.T) @ stats.z.T)

    @staticmethod
    def boundary_nll(stats, x, y, score):
        # L^{-1}(Phi*x - y - Sigma*score) equals the decoder whitened
        # residual exactly; this avoids an ill-conditioned explicit Phi inverse.
        xp, xo = stats.split(x)
        yp, yo = stats.split(y)
        sp, so = stats.split(score)
        residual_k = xp @ stats.phi_k.T - yp - sp @ stats.covariance_k.T
        white_k = _solve(stats.cholesky_k, residual_k, upper=False)
        white_perp = (stats.phi_perp * xo - yo - stats.variance_perp * so) / stats.variance_perp.sqrt()
        # logdet Phi = -s tr(A); skew Q has zero trace.
        logdet = stats.covariance_logdet() - 2 * stats.time * stats.drift_divergence
        return 0.5 * (white_k.square().sum(-1) + white_perp.square().sum(-1)
                      + logdet + stats.z.shape[0] * math.log(2 * math.pi))

    def grn_penalty_base(self, norm=None):
        if not self._has_grn_mask:
            return self.raw_embedding.new_zeros(())
        return off_mask_penalty(self.interaction_matrix(), self.grn_mask_target_source,
                                norm=self.grn_penalty_norm if norm is None else norm)

    def grn_penalty(self, norm=None, *, weighted=True):
        base = self.grn_penalty_base(norm)
        return self.grn_penalty_weight * base if weighted else base

    def additional_regularization(self):
        return self.grn_penalty()


__all__ = ["StationaryQDForward", "StationaryQDTransition"]
