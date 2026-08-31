"""Four single ODE fields local to the 2026-08-30 experiment.

Every edge matrix uses ``parameter[target, source]``.  The direct Hill fields
are adapted from the 20260817 single-ODE branch, Hill-after-linear from
20260803, and simple-softplus from ``ODE/ode_20260421_regODEMLratio.py``.
No class in this module contains a LinComb expert axis or a gate network.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


ODE_TYPES_20260830 = (
    "centered_signed_hill",
    "shifted_hill_rho",
    "hill_after_linear",
    "simple_softplus",
)


def _positive_raw(value: float, eps: float) -> float:
    shifted = float(value) - float(eps)
    if shifted <= 0:
        raise ValueError("positive initialization must exceed positive_epsilon")
    return shifted if shifted > 20 else math.log(math.expm1(shifted))


class SingleODEBase20260830(nn.Module):
    """Shared legacy-TrainLoop-compatible interface."""

    ode_type = "base"

    def __init__(
        self,
        d: int,
        *,
        mask: Optional[torch.Tensor] = None,
        soft: bool = True,
        off_mask_lambda: float = 5.0,
        positive_epsilon: float = 1e-6,
        raw_delta_init: float = 0.1,
    ):
        super().__init__()
        self.d = int(d)
        if self.d <= 0:
            raise ValueError("d must be positive")
        self.is_lincomb = False
        self.num_components = 1
        self.num_experts = 1
        self.gate_mode = None
        self.soft = bool(soft)
        self.off_mask_lambda = float(off_mask_lambda)
        self.positive_epsilon = float(positive_epsilon)
        if self.off_mask_lambda < 0 or self.positive_epsilon <= 0:
            raise ValueError("regularization must be non-negative and epsilon positive")
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.float32)
            if tuple(mask.shape) != (self.d, self.d):
                raise ValueError(f"mask must have shape {(self.d, self.d)}")
            mask = mask.contiguous()
        self.register_buffer("mask", mask)
        self.raw_delta = nn.Parameter(torch.full((self.d,), float(raw_delta_init)))

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.raw_delta) + self.positive_epsilon

    @property
    def penalty_parameter_name(self) -> str:
        raise NotImplementedError

    def penalty_parameter(self) -> torch.Tensor:
        raise NotImplementedError

    def off_mask_penalty(self, norm: str = "l1") -> torch.Tensor:
        parameter = self.penalty_parameter()
        selected = parameter if self.mask is None else parameter * (1.0 - self.mask)
        normalized = str(norm).lower()
        if normalized == "l1":
            base = selected.abs().mean()
        elif normalized == "l2":
            base = selected.square().mean()
        else:
            raise ValueError("norm must be 'l1' or 'l2'")
        # Preserve the inherited internal weighting.  TrainLoop applies its
        # separate ode_reg_lambda outside this method.
        return self.off_mask_lambda * base

    def get_model_info(self) -> dict:
        return {
            "class": type(self).__name__,
            "ode_type": self.ode_type,
            "d": self.d,
            "is_lincomb": False,
            "num_components": 1,
            "gate_mode": None,
            "mask_convention": "target_source",
            "penalty_parameter": self.penalty_parameter_name,
            "off_mask_lambda": self.off_mask_lambda,
            "penalty_reduction": "all_element_mean",
        }


class _DirectHillBase20260830(SingleODEBase20260830):
    def __init__(
        self,
        d: int,
        *,
        regulation_A_init: float = 1.0,
        theta_init: float = 1.0,
        target_chunk_size: int = 16,
        **kwargs,
    ):
        super().__init__(d, **kwargs)
        raw_a = _positive_raw(regulation_A_init, self.positive_epsilon)
        raw_theta = _positive_raw(theta_init, self.positive_epsilon)
        self.raw_A = nn.Parameter(torch.full((self.d, self.d), raw_a))
        self.raw_theta = nn.Parameter(torch.full((self.d, self.d), raw_theta))
        self.b = nn.Parameter(torch.zeros(self.d))
        self.target_chunk_size = int(target_chunk_size)
        if self.target_chunk_size <= 0:
            raise ValueError("target_chunk_size must be positive")

    @property
    def A(self) -> torch.Tensor:
        return F.softplus(self.raw_A) + self.positive_epsilon

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.raw_theta) + self.positive_epsilon

    def edge_response(self, x_pos, theta, signed):
        raise NotImplementedError

    @property
    def signed_parameter(self) -> torch.Tensor:
        raise NotImplementedError

    def _chunk_production(self, x_pos, A, theta, signed, bias):
        response = self.edge_response(
            x_pos[:, None, :], theta[None, :, :], signed[None, :, :]
        )
        return bias[None, :] + (A[None, :, :] * response).sum(dim=-1)

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        del t
        x = x.float()
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"expected x shape (B,{self.d}), got {tuple(x.shape)}")
        x_pos = x.clamp_min(self.positive_epsilon)
        A = self.A
        theta = self.theta
        signed = self.signed_parameter
        chunks = []
        for start in range(0, self.d, min(self.target_chunk_size, self.d)):
            stop = min(start + self.target_chunk_size, self.d)
            values = (
                A[start:stop, :],
                theta[start:stop, :],
                signed[start:stop, :],
                self.b[start:stop],
            )
            use_checkpoint = self.training and torch.is_grad_enabled()
            production = (
                checkpoint(
                    self._chunk_production, x_pos, *values, use_reentrant=False
                )
                if use_checkpoint
                else self._chunk_production(x_pos, *values)
            )
            chunks.append(production)
        production = torch.cat(chunks, dim=-1)
        return production - self.delta[None, :] * x

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info["target_chunk_size"] = self.target_chunk_size
        return info


class CenteredSignedHill20260830(_DirectHillBase20260830):
    ode_type = "centered_signed_hill"

    def __init__(self, d: int, *, alpha_init_std=None, **kwargs):
        super().__init__(d, **kwargs)
        std = 1.0 / math.sqrt(self.d) if alpha_init_std is None else float(alpha_init_std)
        self.alpha = nn.Parameter(torch.randn(self.d, self.d) * std)

    @property
    def signed_parameter(self):
        return self.alpha

    @property
    def penalty_parameter_name(self):
        return "alpha"

    def penalty_parameter(self):
        return self.alpha

    def edge_response(self, x_pos, theta, signed):
        return torch.tanh(signed * (torch.log(x_pos) - torch.log(theta)))


class ShiftedHillRho20260830(_DirectHillBase20260830):
    ode_type = "shifted_hill_rho"

    def __init__(self, d: int, *, hill_n: float = 2.0, rho_init_std=None, **kwargs):
        if float(hill_n) != 2.0:
            raise ValueError("hill_n is fixed to 2.0")
        super().__init__(d, **kwargs)
        self.hill_n = 2.0
        std = 1.0 / math.sqrt(self.d) if rho_init_std is None else float(rho_init_std)
        self.rho = nn.Parameter(torch.randn(self.d, self.d) * std)

    @property
    def signed_parameter(self):
        return self.rho

    @property
    def penalty_parameter_name(self):
        return "rho"

    def penalty_parameter(self):
        return self.rho

    def edge_response(self, x_pos, theta, signed):
        fraction = torch.sigmoid(self.hill_n * (torch.log(x_pos) - torch.log(theta)))
        return torch.expm1(signed) * fraction


class HillAfterLinear20260830(SingleODEBase20260830):
    ode_type = "hill_after_linear"

    def __init__(
        self,
        d: int,
        *,
        hill_n: float = 2.0,
        hill_K_init: float = 1.0,
        hill_V_init: float = 1.0,
        **kwargs,
    ):
        if float(hill_n) != 2.0:
            raise ValueError("hill_n is fixed to 2.0")
        super().__init__(d, **kwargs)
        self.hill_n = 2.0
        self.W = nn.Parameter(torch.randn(self.d, self.d) / math.sqrt(self.d))
        self.b = nn.Parameter(torch.zeros(self.d))
        self.raw_K = nn.Parameter(
            torch.full((self.d,), _positive_raw(hill_K_init, self.positive_epsilon))
        )
        self.raw_V = nn.Parameter(
            torch.full((self.d,), _positive_raw(hill_V_init, self.positive_epsilon))
        )

    @property
    def K(self):
        return F.softplus(self.raw_K) + self.positive_epsilon

    @property
    def V(self):
        return F.softplus(self.raw_V) + self.positive_epsilon

    @property
    def penalty_parameter_name(self):
        return "W"

    def penalty_parameter(self):
        return self.W

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        del t
        x = x.float()
        z = F.softplus(F.linear(x, self.W, self.b))
        fraction = z.square() / (self.K.square()[None, :] + z.square())
        return self.V[None, :] * fraction - self.delta[None, :] * x


class SimpleSoftplus20260830(SingleODEBase20260830):
    ode_type = "simple_softplus"

    def __init__(self, d: int, **kwargs):
        super().__init__(d, **kwargs)
        # This equation has the historical gamma decay instead of delta.  Do
        # not leave an unused trainable raw_delta in DDP.
        del self.raw_delta
        self.W = nn.Parameter(torch.randn(self.d, self.d) / math.sqrt(self.d))
        self.b = nn.Parameter(torch.zeros(self.d))
        # Preserve the old GeneODE raw initialization and transform.
        self.gamma = nn.Parameter(torch.full((self.d,), 0.1))
        self.register_buffer("input_scale", torch.tensor(1.0 / math.sqrt(self.d)))

    @property
    def penalty_parameter_name(self):
        return "W"

    def penalty_parameter(self):
        return self.W

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        del t
        x = x.float()
        # F.linear implements x @ W.T, hence W[target,source].
        production = F.softplus(self.input_scale * F.linear(x, self.W, self.b))
        return production - F.softplus(self.gamma)[None, :] * x


ODE_CLASS_BY_TYPE_20260830 = {
    "centered_signed_hill": CenteredSignedHill20260830,
    "shifted_hill_rho": ShiftedHillRho20260830,
    "hill_after_linear": HillAfterLinear20260830,
    "simple_softplus": SimpleSoftplus20260830,
}


__all__ = [
    "ODE_TYPES_20260830",
    "ODE_CLASS_BY_TYPE_20260830",
    "CenteredSignedHill20260830",
    "ShiftedHillRho20260830",
    "HillAfterLinear20260830",
    "SimpleSoftplus20260830",
]
