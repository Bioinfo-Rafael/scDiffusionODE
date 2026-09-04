"""Dense free-affine learnable forward diffusion (Model B).

This module implements the time-independent SDE

    dY_s = (W Y_s + b) ds + dB_s,

where ``W`` and ``b`` are unconstrained learnable parameters.  The stored
matrix parameter is centred as ``W = -0.5 I + raw_W`` so that zero raw
parameters recover the standard variance-preserving transition under the
physical-time map ``s = -log(alpha_bar)``.  The centring is only an
initialization convention and does not restrict the matrices representable by
``W``.

All transition quantities are computed exactly for the dense affine system by
one augmented Van Loan matrix exponential.  There is no inverse of ``W`` and
no low-rank, subspace, diagonal, or spectral approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class FreeAffineTransition:
    """Batch-shared Gaussian transition quantities for one physical time."""

    time: torch.Tensor
    mean: torch.Tensor
    phi: torch.Tensor
    affine_shift: torch.Tensor
    covariance: torch.Tensor
    cholesky: Optional[torch.Tensor]
    w_matrix: torch.Tensor
    b_vector: torch.Tensor
    drift_matrix: torch.Tensor
    diffusion_covariance: torch.Tensor
    drift_divergence: torch.Tensor
    nominal_covariance: torch.Tensor
    covariance_jitter: torch.Tensor

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


class FreeAffineForward(nn.Module):
    """Time-independent dense free-affine forward process.

    Args:
        dim: State dimension.
        covariance_jitter: Explicit nonnegative jitter added to the nominal
            covariance before Cholesky factorization.  A nonzero value changes
            the sampled transition and is therefore an opt-in numerical
            stability option, never a hidden default.
        dtype: Parameter and dense-linear-algebra dtype.  Float64 is the
            correctness-first default; benchmark configurations may select
            float32 explicitly.
    """

    def __init__(
        self,
        dim: int,
        *,
        covariance_jitter: float = 0.0,
        device=None,
        dtype: Optional[torch.dtype] = torch.float64,
    ) -> None:
        super().__init__()
        if int(dim) != dim or int(dim) <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        self.dim = int(dim)

        covariance_jitter = float(covariance_jitter)
        if not math.isfinite(covariance_jitter) or covariance_jitter < 0.0:
            raise ValueError(
                "covariance_jitter must be finite and nonnegative, got "
                f"{covariance_jitter!r}"
            )
        self.covariance_jitter = covariance_jitter

        factory_kwargs = {"device": device, "dtype": dtype}
        self.raw_w = nn.Parameter(
            torch.zeros((self.dim, self.dim), **factory_kwargs)
        )
        self.raw_b = nn.Parameter(torch.zeros(self.dim, **factory_kwargs))
        self.register_buffer(
            "_identity",
            torch.eye(self.dim, **factory_kwargs),
            persistent=False,
        )

    def drift_matrix(self) -> torch.Tensor:
        """Return the unconstrained matrix ``W = -0.5 I + raw_W``."""

        return self.raw_w - 0.5 * self._identity

    def drift_bias(self) -> torch.Tensor:
        """Return the unconstrained affine bias ``b``."""

        return self.raw_b

    def diffusion_covariance(self) -> torch.Tensor:
        """Return ``a = g g^T = I`` for the identity-diffusion SDE."""

        return self._identity

    def drift_divergence(self) -> torch.Tensor:
        """Return ``div_y(W y + b) = tr(W)``."""

        return torch.trace(self.drift_matrix())

    def _physical_time(self, time, *, device, dtype) -> torch.Tensor:
        value = torch.as_tensor(time, device=device, dtype=dtype)
        if value.numel() == 0:
            raise ValueError("time must not be empty")
        flat = value.reshape(-1)
        if flat.numel() > 1:
            first_detached = flat[0].detach()
            if not torch.all(flat.detach() == first_detached):
                raise ValueError(
                    "FreeAffineForward requires a batch-shared physical time"
                )
        scalar = flat[0]
        if not bool(torch.isfinite(scalar).detach().item()):
            raise ValueError("physical time must be finite")
        if bool((scalar < 0).detach().item()):
            raise ValueError("physical time must be nonnegative")
        return scalar

    def _van_loan_transition(
        self, physical_time: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``Phi``, affine shift ``h``, and nominal covariance.

        For

            H = [[W, I, b], [0, -W.T, 0], [0, 0, 0]],

        write the top blocks of ``exp(s H)`` as ``[Phi, V, h]``.  Then
        ``Sigma = V @ Phi.T``.  This identity remains valid for singular and
        non-normal ``W`` and keeps gradients through ``torch.matrix_exp``.
        """

        w = self.drift_matrix()
        b_column = self.drift_bias().unsqueeze(-1)
        zeros_dd = torch.zeros_like(w)
        zeros_d1 = b_column.new_zeros((self.dim, 1))
        top = torch.cat((w, self._identity, b_column), dim=1)
        middle = torch.cat(
            (zeros_dd, -w.transpose(-1, -2), zeros_d1), dim=1
        )
        bottom = w.new_zeros((1, 2 * self.dim + 1))
        augmented = torch.cat((top, middle, bottom), dim=0)

        exponential = torch.matrix_exp(physical_time * augmented)
        phi = exponential[: self.dim, : self.dim]
        van_loan_block = exponential[: self.dim, self.dim : 2 * self.dim]
        affine_shift = exponential[: self.dim, -1]
        nominal_covariance = van_loan_block @ phi.transpose(-1, -2)
        nominal_covariance = 0.5 * (
            nominal_covariance + nominal_covariance.transpose(-1, -2)
        )
        return phi, affine_shift, nominal_covariance

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
                "free-affine transition covariance is not positive definite "
                f"(cholesky_ex info={int(info.detach().item())}, "
                f"covariance_jitter={self.covariance_jitter}). No hidden "
                "jitter or eigenvalue clipping is applied; choose a positive "
                "physical time or opt into an explicit covariance_jitter."
            )
        return covariance, cholesky, jitter

    def transition_stats(
        self,
        x: torch.Tensor,
        time,
        *,
        compute_cholesky: bool = True,
    ) -> FreeAffineTransition:
        """Compute exact dense affine-Gaussian transition statistics.

        ``time`` may be scalar or contain one repeated value.  Heterogeneous
        batch times are rejected so the augmented matrix exponential and
        Cholesky are computed once per batch.
        """

        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim < 1 or x.shape[-1] != self.dim:
            raise ValueError(
                f"x must end in dimension {self.dim}, got {tuple(x.shape)}"
            )
        if x.device != self.raw_w.device:
            raise ValueError(
                f"x is on {x.device}, but forward parameters are on "
                f"{self.raw_w.device}"
            )

        physical_time = self._physical_time(
            time, device=self.raw_w.device, dtype=self.raw_w.dtype
        )
        x_forward = x.to(dtype=self.raw_w.dtype)
        phi, affine_shift, nominal_covariance = self._van_loan_transition(
            physical_time
        )
        mean = x_forward @ phi.transpose(-1, -2) + affine_shift

        if compute_cholesky:
            covariance, cholesky, jitter = self._factor_covariance(
                nominal_covariance
            )
        else:
            covariance = nominal_covariance
            cholesky = None
            jitter = nominal_covariance.new_zeros(())

        w = self.drift_matrix()
        return FreeAffineTransition(
            time=physical_time,
            mean=mean,
            phi=phi,
            affine_shift=affine_shift,
            covariance=covariance,
            cholesky=cholesky,
            w_matrix=w,
            b_vector=self.drift_bias(),
            drift_matrix=w,
            diffusion_covariance=self.diffusion_covariance(),
            drift_divergence=torch.trace(w),
            nominal_covariance=nominal_covariance,
            covariance_jitter=jitter,
        )

    @staticmethod
    def sample_from_stats(
        stats: FreeAffineTransition,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw the differentiable sample ``mean + L noise``."""

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
        """Compute transition stats and draw a reparameterized sample."""

        stats = self.transition_stats(x, time, compute_cholesky=True)
        sample, noise = self.sample_from_stats(stats, noise)
        if return_stats:
            return sample, noise, stats
        return sample, noise

    @staticmethod
    def conditional_score(
        stats: FreeAffineTransition, noise: torch.Tensor
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
        stats: FreeAffineTransition, residual: torch.Tensor
    ) -> torch.Tensor:
        """Return Model-B's compact DSM term per sample.

        The returned value is

            0.5 * residual.T @ L^{-1} @ L^{-T} @ residual
          = 0.5 * ||L^{-T} residual||^2.
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
        values = 0.5 * whitened.square().sum(dim=-1)
        return values.reshape(residual.shape[:-1])

    @staticmethod
    def terminal_kl(stats: FreeAffineTransition) -> torch.Tensor:
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
    def terminal_cross_entropy(stats: FreeAffineTransition) -> torch.Tensor:
        """Return ``-E_q log N(0,I)`` per sample for direct-form checks."""

        covariance_trace = torch.trace(stats.covariance)
        mean_squared = stats.mean.square().sum(dim=-1)
        dimension = stats.mean.shape[-1]
        normalizer = dimension * math.log(2.0 * math.pi)
        return 0.5 * (mean_squared + covariance_trace + normalizer)

    def additional_regularization(self) -> torch.Tensor:
        """Generic wrapper hook; unconstrained Model B has no extra penalty."""

        return self.raw_w.new_zeros(())


__all__ = ["FreeAffineForward", "FreeAffineTransition"]
