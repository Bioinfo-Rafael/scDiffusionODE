"""Dense stationary-Q/D learnable forward diffusion.

The physical-time SDE implemented here is

    dY_s = -(Q + D) Y_s ds + sqrt(2 D) dB_s,

with ``Q.T == -Q`` and ``D == C @ C.T >= 0`` by construction.  Its
stationary covariance is the identity, including when ``D`` is merely
positive semidefinite.  All matrices in this module are dense; there is no
low-rank or subspace approximation.

The default ``d_parameterization="psd"`` is the paper-faithful
parameterization: every entry of the packed lower-triangular factor ``C`` is
unconstrained, including its diagonal.  ``"spd_softplus"`` is an explicit
numerical-stability option which replaces C's diagonal by
``softplus(raw_diag) + d_diagonal_floor``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .grn import effective_grn_mask, off_mask_penalty, validate_penalty_norm

_D_PARAMETERIZATIONS = ("psd", "spd_softplus")


@dataclass(frozen=True)
class StationaryQDTransition:
    """Batch-shared transition quantities for one physical time."""

    time: torch.Tensor
    mean: torch.Tensor
    phi: torch.Tensor
    affine_shift: torch.Tensor
    covariance: torch.Tensor
    cholesky: Optional[torch.Tensor]
    q_matrix: torch.Tensor
    d_matrix: torch.Tensor
    drift_matrix: torch.Tensor
    diffusion_covariance: torch.Tensor
    drift_divergence: torch.Tensor
    nominal_covariance: torch.Tensor
    covariance_jitter: torch.Tensor
    covariance_evaluation: str
    covariance_series_terms: int

    # Concise aliases used in the mathematical derivation and by the generic
    # work-local objective code.
    @property
    def m(self) -> torch.Tensor:
        return self.mean

    @property
    def transition_matrix(self) -> torch.Tensor:
        return self.phi

    @property
    def sigma(self) -> torch.Tensor:
        return self.covariance

    @property
    def L(self) -> Optional[torch.Tensor]:
        return self.cholesky

    @property
    def g2(self) -> torch.Tensor:
        return self.diffusion_covariance

    @property
    def div_f(self) -> torch.Tensor:
        return self.drift_divergence


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """Stable inverse of softplus for strictly positive ``value``."""

    return value + torch.log(-torch.expm1(-value))


class StationaryQDForward(nn.Module):
    """Time-independent dense stationary-Q/D forward process.

    Args:
        dim: State dimension.
        d_parameterization: ``"psd"`` (default) uses ``D=C C^T`` with a
            wholly unconstrained packed-lower ``C``. ``"spd_softplus"`` is
            the opt-in stability variant with a positive transformed
            diagonal in ``C``.
        d_diagonal_floor: Added to the transformed diagonal only in
            ``"spd_softplus"`` mode. It must be strictly positive there and
            is required to be zero in paper-faithful ``"psd"`` mode.
        initial_d_diagonal: Desired scalar diagonal of D at initialization.
            The standard VP-compatible default is 1/2.
        covariance_jitter: Explicit opt-in jitter added before Cholesky. A
            nonzero value changes the sampled transition covariance and is
            therefore not paper-faithful; the nominal SDE covariance remains
            available as ``stats.nominal_covariance``.
        grn_mask_target_source: Optional dense mask in drift-matrix
            orientation ``[target, source]``. The constructor does not
            transpose it.
        allow_self_edges: If true, the diagonal is always allowed by the GRN
            penalty so stationary damping is not penalized.
        grn_penalty_weight: Multiplicative coefficient returned by
            :meth:`additional_regularization`.
        grn_penalty_norm: ``"l1"`` or ``"l2"`` full-matrix mean, matching the
            established ODE off-mask convention.
    """

    def __init__(
        self,
        dim: int,
        *,
        d_parameterization: str = "psd",
        d_diagonal_floor: float = 0.0,
        initial_d_diagonal: float = 0.5,
        covariance_jitter: float = 0.0,
        grn_mask_target_source: Optional[torch.Tensor] = None,
        allow_self_edges: bool = True,
        grn_penalty_weight: float = 0.0,
        grn_penalty_norm: str = "l1",
        device=None,
        dtype: Optional[torch.dtype] = torch.float64,
    ) -> None:
        super().__init__()
        if int(dim) != dim or int(dim) <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        self.dim = int(dim)

        mode = str(d_parameterization).lower()
        if mode not in _D_PARAMETERIZATIONS:
            raise ValueError(
                f"d_parameterization must be one of {_D_PARAMETERIZATIONS}, "
                f"got {d_parameterization!r}"
            )
        self.d_parameterization = mode
        self.d_diagonal_floor = self._finite_nonnegative(
            "d_diagonal_floor", d_diagonal_floor
        )
        self.initial_d_diagonal = self._finite_nonnegative(
            "initial_d_diagonal", initial_d_diagonal
        )
        self.covariance_jitter = self._finite_nonnegative(
            "covariance_jitter", covariance_jitter
        )
        if mode == "psd" and self.d_diagonal_floor != 0.0:
            raise ValueError(
                "d_diagonal_floor is only valid with "
                "d_parameterization='spd_softplus'; paper-faithful 'psd' "
                "uses an unconstrained C diagonal"
            )
        if mode == "spd_softplus" and self.d_diagonal_floor <= 0.0:
            raise ValueError(
                "d_diagonal_floor must be strictly positive in "
                "spd_softplus mode"
            )
        if mode == "spd_softplus" and self.initial_d_diagonal <= 0.0:
            raise ValueError(
                "initial_d_diagonal must be positive in spd_softplus mode"
            )

        self.grn_penalty_norm = validate_penalty_norm(grn_penalty_norm)
        self.grn_penalty_weight = self._finite_nonnegative(
            "grn_penalty_weight", grn_penalty_weight
        )
        self.allow_self_edges = bool(allow_self_edges)

        factory_kwargs = {"device": device, "dtype": dtype}
        q_indices = torch.triu_indices(
            self.dim, self.dim, offset=1, device=device
        )
        d_indices = torch.tril_indices(
            self.dim, self.dim, offset=0, device=device
        )
        self.register_buffer("_q_indices", q_indices, persistent=False)
        self.register_buffer("_d_indices", d_indices, persistent=False)
        self.register_buffer(
            "_identity",
            torch.eye(self.dim, **factory_kwargs),
            persistent=False,
        )

        self.raw_q_upper = nn.Parameter(
            torch.zeros(q_indices.shape[1], **factory_kwargs)
        )
        raw_d = torch.zeros(d_indices.shape[1], **factory_kwargs)
        is_diagonal = d_indices[0] == d_indices[1]
        desired_c_diagonal = math.sqrt(self.initial_d_diagonal)
        if mode == "psd":
            raw_d[is_diagonal] = desired_c_diagonal
        else:
            transformed_target = desired_c_diagonal - self.d_diagonal_floor
            if transformed_target <= 0.0:
                raise ValueError(
                    "sqrt(initial_d_diagonal) must exceed "
                    "d_diagonal_floor in spd_softplus mode"
                )
            target = raw_d.new_tensor(transformed_target)
            raw_d[is_diagonal] = _inverse_softplus(target)
        self.raw_d_lower = nn.Parameter(raw_d)

        mask, self._has_grn_mask = effective_grn_mask(
            grn_mask_target_source,
            dim=self.dim,
            allow_self_edges=self.allow_self_edges,
            reference=self.raw_q_upper,
        )
        self.register_buffer("grn_mask_target_source", mask)

    @staticmethod
    def _finite_nonnegative(name: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
        return value

    def q_matrix(self) -> torch.Tensor:
        """Return dense Q with exact skew-symmetry by construction."""

        upper = self.raw_q_upper.new_zeros((self.dim, self.dim))
        upper[self._q_indices[0], self._q_indices[1]] = self.raw_q_upper
        return upper - upper.transpose(-1, -2)

    def d_factor(self) -> torch.Tensor:
        """Return the dense lower-triangular factor C."""

        lower_values = self.raw_d_lower
        if self.d_parameterization == "spd_softplus":
            is_diagonal = self._d_indices[0] == self._d_indices[1]
            lower_values = torch.where(
                is_diagonal,
                F.softplus(lower_values) + self.d_diagonal_floor,
                lower_values,
            )
        factor = lower_values.new_zeros((self.dim, self.dim))
        factor[self._d_indices[0], self._d_indices[1]] = lower_values
        return factor

    def d_matrix(self) -> torch.Tensor:
        """Return dense D=C C^T, which is PSD in exact arithmetic."""

        factor = self.d_factor()
        return factor @ factor.transpose(-1, -2)

    def stationary_operator(self) -> torch.Tensor:
        """Return Q+D, the matrix constrained by the GRN penalty."""

        return self.q_matrix() + self.d_matrix()

    def drift_matrix(self) -> torch.Tensor:
        """Return the linear drift matrix F=-(Q+D)."""

        return -self.stationary_operator()

    def diffusion_covariance(self) -> torch.Tensor:
        """Return a=g g^T=2D without constructing a matrix square root."""

        return 2.0 * self.d_matrix()

    def diffusion_factor(self) -> torch.Tensor:
        """Return ``G=sqrt(2) C`` so that ``G G^T = 2D`` exactly."""

        return math.sqrt(2.0) * self.d_factor()

    def drift(self, states: torch.Tensor) -> torch.Tensor:
        """Evaluate row-vector forward drift ``-(Q+D)y``."""

        if states.shape[-1] != self.dim:
            raise ValueError("states have an incompatible final dimension")
        states = states.to(device=self.raw_q_upper.device, dtype=self.raw_q_upper.dtype)
        return states @ self.drift_matrix().transpose(-1, -2)

    def drift_divergence(self) -> torch.Tensor:
        """Return div_y f=-tr(D); skew-symmetry gives tr(Q)=0."""

        return -torch.trace(self.d_matrix())

    def stationarity_residual(self) -> torch.Tensor:
        """Return F + F^T + a; it is identically zero analytically."""

        drift = self.drift_matrix()
        return drift + drift.transpose(-1, -2) + self.diffusion_covariance()

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

    def _factor_covariance(
        self, nominal_covariance: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        jitter = nominal_covariance.new_tensor(self.covariance_jitter)
        if self.covariance_jitter:
            covariance = nominal_covariance + jitter * self._identity
        else:
            covariance = nominal_covariance
        cholesky, info = torch.linalg.cholesky_ex(covariance, check_errors=False)
        if bool((info != 0).detach().item()):
            raise RuntimeError(
                "stationary-Q/D transition covariance is not positive definite "
                f"(cholesky_ex info={int(info.detach().item())}, "
                f"time-dependent covariance_jitter={self.covariance_jitter}). "
                "The paper-faithful PSD parameterization deliberately does "
                "not add a hidden floor; opt into d_parameterization="
                "'spd_softplus' or an explicit covariance_jitter if required."
            )
        return covariance, cholesky, jitter

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
                "large for stable boundary evaluation"
            )
        covariance = 0.5 * (
            covariance + covariance.transpose(-1, -2)
        )
        return covariance, terms_used

    def _use_integral_series(self, physical_time: torch.Tensor) -> bool:
        """Detect the dense-float32 cancellation regime deterministically."""

        # A dense product has a worst-case rounding scale proportional to
        # d*eps.  Below this scale, I-Phi@Phi.T can lose the positive boundary
        # covariance even when D is well conditioned.  This branch depends
        # only on dimension, dtype, and physical time—not learned parameters.
        rounding_scale = 8.0 * self.dim * torch.finfo(physical_time.dtype).eps
        return bool(
            (physical_time.detach().abs() <= rounding_scale).cpu().item()
        )

    def transition_stats(
        self,
        x: torch.Tensor,
        time,
        *,
        compute_cholesky: bool = True,
    ) -> StationaryQDTransition:
        """Compute exact dense Gaussian transition statistics.

        ``time`` may be a scalar or a tensor whose entries are all identical.
        A heterogeneous batch is rejected so the expensive dense matrix
        exponential and Cholesky are computed once per batch.
        """

        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(
                f"x must end in dimension {self.dim}, got {tuple(x.shape)}"
            )
        parameter = self.raw_q_upper
        if x.device != parameter.device:
            raise ValueError(
                f"x is on {x.device}, but forward parameters are on "
                f"{parameter.device}"
            )
        physical_time = self._physical_time(
            time, device=parameter.device, dtype=parameter.dtype
        )
        x_forward = x.to(dtype=parameter.dtype)

        q = self.q_matrix()
        d = self.d_matrix()
        operator = q + d
        drift = -operator
        phi = torch.matrix_exp(-physical_time * operator)
        mean = x_forward @ phi.transpose(-1, -2)
        diffusion_covariance = 2.0 * d
        if self._use_integral_series(physical_time):
            nominal_covariance, covariance_series_terms = (
                self._integral_series_covariance(
                    drift,
                    diffusion_covariance,
                    physical_time,
                )
            )
            covariance_evaluation = "adaptive_integral_series"
        else:
            nominal_covariance = self._identity - phi @ phi.transpose(-1, -2)
            nominal_covariance = 0.5 * (
                nominal_covariance + nominal_covariance.transpose(-1, -2)
            )
            covariance_evaluation = "stationary_identity"
            covariance_series_terms = 0

        if compute_cholesky:
            covariance, cholesky, jitter = self._factor_covariance(
                nominal_covariance
            )
        else:
            covariance = nominal_covariance
            cholesky = None
            jitter = nominal_covariance.new_zeros(())

        return StationaryQDTransition(
            time=physical_time,
            mean=mean,
            phi=phi,
            affine_shift=phi.new_zeros(self.dim),
            covariance=covariance,
            cholesky=cholesky,
            q_matrix=q,
            d_matrix=d,
            drift_matrix=drift,
            diffusion_covariance=diffusion_covariance,
            drift_divergence=-torch.trace(d),
            nominal_covariance=nominal_covariance,
            covariance_jitter=jitter,
            covariance_evaluation=covariance_evaluation,
            covariance_series_terms=covariance_series_terms,
        )

    @staticmethod
    def sample_from_stats(
        stats: StationaryQDTransition,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterized sample ``mean + L noise`` from existing stats."""

        if stats.cholesky is None:
            raise ValueError("stats.cholesky is required for sampling")
        if noise is None:
            noise = torch.randn_like(stats.mean)
        else:
            if noise.shape != stats.mean.shape:
                raise ValueError(
                    f"noise must have shape {tuple(stats.mean.shape)}, "
                    f"got {tuple(noise.shape)}"
                )
            noise = noise.to(device=stats.mean.device, dtype=stats.mean.dtype)
        sample = stats.mean + noise @ stats.cholesky.transpose(-1, -2)
        return sample, noise

    def q_sample(
        self,
        x: torch.Tensor,
        time,
        noise: Optional[torch.Tensor] = None,
        *,
        return_stats: bool = False,
    ):
        """Compute stats and draw a differentiable reparameterized sample."""

        stats = self.transition_stats(x, time, compute_cholesky=True)
        sample, noise = self.sample_from_stats(stats, noise)
        if return_stats:
            return sample, noise, stats
        return sample, noise

    @staticmethod
    def conditional_score(
        stats: StationaryQDTransition, noise: torch.Tensor
    ) -> torch.Tensor:
        """Return ``-L^{-T} noise`` without forming an inverse."""

        if stats.cholesky is None:
            raise ValueError("stats.cholesky is required to compute a score")
        if noise.shape[-1] != stats.cholesky.shape[-1]:
            raise ValueError("noise has an incompatible final dimension")
        noise = noise.to(device=stats.mean.device, dtype=stats.mean.dtype)
        flat = noise.reshape(-1, noise.shape[-1])
        solved = torch.linalg.solve_triangular(
            stats.cholesky.transpose(-1, -2),
            flat.transpose(-1, -2),
            upper=True,
        ).transpose(-1, -2)
        return -solved.reshape_as(noise)

    @staticmethod
    def noise_metric_quadratic(
        stats: StationaryQDTransition, residual: torch.Tensor
    ) -> torch.Tensor:
        """Return Model-A's exact compact DSM term per sample.

        This computes

            residual^T L^{-1} D L^{-T} residual

        via triangular solves.  The factor 1/2 in score space is exactly
        cancelled by ``g g^T = 2D``.
        """

        if stats.cholesky is None:
            raise ValueError("stats.cholesky is required for the DSM metric")
        if residual.shape[-1] != stats.cholesky.shape[-1]:
            raise ValueError("residual has an incompatible final dimension")
        residual = residual.to(device=stats.mean.device, dtype=stats.mean.dtype)
        flat = residual.reshape(-1, residual.shape[-1])
        whitened = torch.linalg.solve_triangular(
            stats.cholesky.transpose(-1, -2),
            flat.transpose(-1, -2),
            upper=True,
        ).transpose(-1, -2)
        d_times = whitened @ stats.d_matrix.transpose(-1, -2)
        values = (d_times * whitened).sum(dim=-1)
        return values.reshape(residual.shape[:-1])

    @staticmethod
    def terminal_kl(stats: StationaryQDTransition) -> torch.Tensor:
        """Return analytic ``KL(N(mean, Sigma) || N(0, I))`` per sample."""

        if stats.cholesky is None:
            raise ValueError("stats.cholesky is required for terminal KL")
        diagonal = torch.diagonal(stats.cholesky, dim1=-2, dim2=-1)
        logdet = 2.0 * torch.log(diagonal).sum()
        covariance_trace = torch.trace(stats.covariance)
        mean_squared = stats.mean.square().sum(dim=-1)
        return 0.5 * (
            mean_squared + covariance_trace - logdet - stats.mean.shape[-1]
        )

    @staticmethod
    def terminal_cross_entropy(stats: StationaryQDTransition) -> torch.Tensor:
        """Return ``-E_q log N(0,I)`` per sample for direct-form checks."""

        covariance_trace = torch.trace(stats.covariance)
        mean_squared = stats.mean.square().sum(dim=-1)
        dimension = stats.mean.shape[-1]
        normalizer = dimension * math.log(2.0 * math.pi)
        return 0.5 * (mean_squared + covariance_trace + normalizer)

    def grn_penalty_base(self, norm: Optional[str] = None) -> torch.Tensor:
        """Return unweighted off-mask penalty on ``Q+D``.

        The supplied mask is already in ``[target, source]`` orientation,
        matching rows/columns of the matrix multiplying a state column.
        """

        if not self._has_grn_mask:
            return self.raw_q_upper.new_zeros(())
        selected_norm = (
            self.grn_penalty_norm
            if norm is None
            else validate_penalty_norm(norm)
        )
        return off_mask_penalty(
            self.stationary_operator(),
            self.grn_mask_target_source,
            norm=selected_norm,
        )

    def grn_penalty(
        self, norm: Optional[str] = None, *, weighted: bool = True
    ) -> torch.Tensor:
        """Return GRN penalty, weighted by its configured coefficient."""

        base = self.grn_penalty_base(norm)
        if weighted:
            return self.grn_penalty_weight * base
        return base

    def additional_regularization(self) -> torch.Tensor:
        """Generic wrapper hook for the weighted external GRN regularizer."""

        return self.grn_penalty(weighted=True)


__all__ = ["StationaryQDForward", "StationaryQDTransition"]
