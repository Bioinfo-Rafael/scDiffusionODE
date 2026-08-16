"""Direct per-regulator Hill ODE fields for the 2026-08-16 experiments.

All edge matrices use ``[target, source]`` ordering. The diffusion state is
clamped only while evaluating log/power expressions; no latent ``Wx+b`` or
softplus transformation is applied to a regulator.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ODE.ode_20260609_mathmlp import _TimeEmb, _mlp, _prep_t


ODE_TYPES = ("centered_signed_hill", "shifted_hill_rho")


def _finite_float(name: str, value, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _inverse_softplus_for_physical_init(
    name: str, physical_value, positive_eps: float
) -> float:
    """Raw value whose ``softplus(raw) + positive_eps`` is physical_value."""

    physical_value = _finite_float(name, physical_value, positive=True)
    shifted = physical_value - positive_eps
    if shifted <= 0:
        raise ValueError(
            f"{name} ({physical_value}) must exceed positive_epsilon ({positive_eps})"
        )
    if shifted > 20.0:
        return shifted + math.log1p(-math.exp(-shifted))
    return math.log(math.expm1(shifted))


def _scalar_softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


class BaseDirectHillField(nn.Module):
    """Shared TrainLoop-compatible K-component direct-Hill field."""

    ode_type = "base"
    W_IS_EXACT = False

    def __init__(
        self,
        d: int,
        *,
        is_lincomb: bool = True,
        num_components: int = 8,
        mask: Optional[torch.Tensor] = None,
        soft: bool = True,
        time_dim: int = 64,
        hidden: int = 256,
        dropout: float = 0.0,
        gate_mode: str = "softmax",
        gate_temperature: float = 1.0,
        off_mask_lambda: float = 5.0,
        sparse_lambda: float = 0.0,
        entropy_lambda: float = 0.0,
        ratio_reg_weight: float = 0.0,
        ratio_reg_target: float = 1.0,
        ratio_reg_eps: float = 1e-8,
        positive_eps: float = 1e-6,
        raw_delta_init: float = 0.1,
        use_decay: bool = True,
        target_chunk_size: int = 16,
        regulation_A_init: float = 1.0,
        theta_init: float = 1.0,
    ):
        super().__init__()
        self.d = int(d)
        if self.d <= 0:
            raise ValueError("d must be positive")
        self.is_lincomb = bool(is_lincomb)
        self.num_components = int(num_components) if self.is_lincomb else 1
        self.num_experts = self.num_components
        if not self.is_lincomb or self.num_components != 8:
            raise ValueError("the 20260816 experiments require K=8 LinComb")
        self.time_dim = int(time_dim)
        self.hidden = int(hidden)
        self.dropout = _finite_float("dropout", dropout)
        self.gate_mode = str(gate_mode).lower()
        if self.gate_mode != "softmax":
            raise ValueError("the 20260816 experiments require gate_mode='softmax'")
        self.gate_temperature = _finite_float(
            "gate_temperature", gate_temperature, positive=True
        )
        self.off_mask_lambda = _finite_float("off_mask_lambda", off_mask_lambda)
        self.sparse_lambda = _finite_float("sparse_lambda", sparse_lambda)
        self.entropy_lambda = _finite_float("entropy_lambda", entropy_lambda)
        self.ratio_reg_weight = _finite_float("ratio_reg_weight", ratio_reg_weight)
        self.ratio_reg_target = _finite_float(
            "ratio_reg_target", ratio_reg_target, positive=True
        )
        self.ratio_reg_eps = _finite_float("ratio_reg_eps", ratio_reg_eps, positive=True)
        if min(
            self.off_mask_lambda,
            self.sparse_lambda,
            self.entropy_lambda,
            self.ratio_reg_weight,
        ) < 0:
            raise ValueError("regularization weights must be non-negative")
        if self.sparse_lambda > 0:
            raise ValueError("sparse_lambda is not defined for the softmax gate")
        self.positive_eps = _finite_float("positive_eps", positive_eps, positive=True)
        self.raw_delta_init = _finite_float("raw_delta_init", raw_delta_init)
        self.use_decay = bool(use_decay)
        self.target_chunk_size = int(target_chunk_size)
        if self.target_chunk_size <= 0:
            raise ValueError("target_chunk_size must be positive")
        self.regulation_A_init = _finite_float(
            "regulation_A_init", regulation_A_init, positive=True
        )
        self.theta_init = _finite_float("theta_init", theta_init, positive=True)

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.float32)
            if tuple(mask.shape) != (self.d, self.d):
                raise ValueError(
                    f"mask must have shape {(self.d, self.d)}, got {tuple(mask.shape)}"
                )
            self.register_buffer("mask", mask.contiguous())
        else:
            self.register_buffer("mask", None)

        self.raw_delta = nn.Parameter(torch.full((self.d,), self.raw_delta_init))
        raw_A_init = _inverse_softplus_for_physical_init(
            "regulation_A_init", self.regulation_A_init, self.positive_eps
        )
        raw_theta_init = _inverse_softplus_for_physical_init(
            "theta_init", self.theta_init, self.positive_eps
        )
        edge_shape = (self.num_components, self.d, self.d)
        self.raw_A = nn.Parameter(torch.full(edge_shape, raw_A_init))
        self.raw_theta = nn.Parameter(torch.full(edge_shape, raw_theta_init))
        self.b = nn.Parameter(torch.zeros(self.num_components, self.d))

        self.time_emb = _TimeEmb(self.time_dim)
        self.coeff_net = _mlp(
            self.d + self.time_dim, self.hidden, self.num_components, self.dropout
        )
        self.soft = bool(
            soft
            or self.off_mask_lambda > 0
            or self.entropy_lambda > 0
            or self.ratio_reg_weight > 0
        )
        self._cached_gate_reg = None
        self._cached_gate_stats = None
        self._cached_ratio_reg = None

    @property
    def A(self) -> torch.Tensor:
        """Non-negative regulation magnitude; edge sign lives elsewhere."""

        return F.softplus(self.raw_A) + self.positive_eps

    @property
    def theta(self) -> torch.Tensor:
        return F.softplus(self.raw_theta) + self.positive_eps

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.raw_delta) + self.positive_eps

    @property
    def bias(self) -> torch.Tensor:
        return self.b

    @property
    def penalty_parameter_name(self) -> str:
        raise NotImplementedError

    def penalty_parameter(self) -> torch.Tensor:
        raise NotImplementedError

    def positive_regulators(self, x: torch.Tensor) -> torch.Tensor:
        """Numerical guard only: direct regulator values clamped before log/power."""

        return x.float().clamp_min(self.positive_eps)

    def get_gate_values(self, x: torch.Tensor, t=None) -> Dict[str, torch.Tensor]:
        x = x.float()
        t_prepared = _prep_t(t, x.shape[0], x.device, x.dtype)
        logits = self.coeff_net(torch.cat([x, self.time_emb(t_prepared)], dim=-1))
        probabilities = torch.softmax(logits / self.gate_temperature, dim=-1)
        return {
            "logits": logits,
            "coefficients": probabilities,
            "probabilities": probabilities,
        }

    def effective_coefficients(self, x: torch.Tensor, t=None) -> torch.Tensor:
        return self.get_gate_values(x, t)["coefficients"]

    get_effective_coefficients = effective_coefficients

    def edge_response(
        self, x_pos: torch.Tensor, theta: torch.Tensor, signed: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def _chunk_production(
        self,
        x_pos: torch.Tensor,
        A_chunk: torch.Tensor,
        theta_chunk: torch.Tensor,
        signed_chunk: torch.Tensor,
        bias_chunk: torch.Tensor,
    ) -> torch.Tensor:
        response = self.edge_response(
            x_pos[:, None, None, :],
            theta_chunk[None, :, :, :],
            signed_chunk[None, :, :, :],
        )
        return bias_chunk[None, :, :] + (A_chunk[None, :, :, :] * response).sum(-1)

    @property
    def signed_parameter(self) -> torch.Tensor:
        raise NotImplementedError

    def component_outputs(self, x: torch.Tensor, t=None) -> torch.Tensor:
        del t
        x = x.float()
        x_pos = self.positive_regulators(x)
        chunks = []
        for start in range(0, self.d, min(self.target_chunk_size, self.d)):
            stop = min(start + self.target_chunk_size, self.d)
            values = (
                self.A[:, start:stop, :],
                self.theta[:, start:stop, :],
                self.signed_parameter[:, start:stop, :],
                self.b[:, start:stop],
            )
            use_checkpoint = self.training and torch.is_grad_enabled() and any(
                value.requires_grad for value in (x_pos, *values)
            )
            production = (
                checkpoint(self._chunk_production, x_pos, *values, use_reentrant=False)
                if use_checkpoint
                else self._chunk_production(x_pos, *values)
            )
            chunks.append(production)
        production = torch.cat(chunks, dim=-1)
        if not self.use_decay:
            return production
        return production - self.delta.view(1, 1, self.d) * x[:, None, :]

    def component_contributions(self, x: torch.Tensor, t=None) -> torch.Tensor:
        coefficients = self.effective_coefficients(x.float(), t)
        return coefficients[:, :, None] * self.component_outputs(x.float(), t)

    get_component_outputs = component_outputs
    forward_components = component_outputs

    def _cache_gate_regularization(self, probabilities: torch.Tensor) -> None:
        self._cached_gate_reg = None
        if not self.training:
            self._cached_gate_stats = None
            return
        if self.entropy_lambda > 0:
            self._cached_gate_reg = -(
                probabilities * torch.log(probabilities + self.ratio_reg_eps)
            ).sum(-1).mean()
        with torch.no_grad():
            entropy = -(
                probabilities * torch.log(probabilities + self.ratio_reg_eps)
            ).sum(-1)
            self._cached_gate_stats = {
                "mean_abs_a": float(probabilities.abs().mean()),
                "entropy": float(entropy.mean()),
            }

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        x = x.float()
        components = self.component_outputs(x, t)
        gate = self.get_gate_values(x, t)
        self._cache_gate_regularization(gate["probabilities"])
        return torch.einsum("bk,bkd->bd", gate["coefficients"], components)

    def _zero(self) -> torch.Tensor:
        return self.raw_delta.new_zeros(())

    def off_mask_penalty(self, norm: str = "l1") -> torch.Tensor:
        parameter = self.penalty_parameter()
        selected = parameter if self.mask is None else parameter * (1.0 - self.mask)
        if str(norm).lower() == "l2":
            base = selected.square().mean()
        elif str(norm).lower() == "l1":
            base = selected.abs().mean()
        else:
            raise ValueError("norm must be 'l1' or 'l2'")
        total = self.off_mask_lambda * base if self.off_mask_lambda > 0 else self._zero()
        if self._cached_gate_reg is not None and self.entropy_lambda > 0:
            total = total + self.entropy_lambda * self._cached_gate_reg
        if self._cached_ratio_reg is not None and self.ratio_reg_weight > 0:
            total = total + self.ratio_reg_weight * self._cached_ratio_reg
        return total

    def has_aux_regularization(self) -> bool:
        return bool(
            self.off_mask_lambda > 0
            or self.entropy_lambda > 0
            or self.ratio_reg_weight > 0
        )

    @torch.no_grad()
    def compute_W(self, x: torch.Tensor, t=None) -> torch.Tensor:
        coefficients = self.effective_coefficients(x.float(), t)
        return torch.einsum(
            "bk,kij->bij", coefficients, self.signed_regulatory_magnitude()
        )

    def signed_regulatory_magnitude(self) -> torch.Tensor:
        raise NotImplementedError

    def analysis_parameters(self) -> Dict[str, torch.Tensor]:
        return {
            "A": self.A,
            "theta": self.theta,
            "bias": self.b,
            "delta": self.delta,
            "signed_parameter": self.signed_parameter,
        }

    def get_model_info(self) -> dict:
        return {
            "class": type(self).__name__,
            "type": self.ode_type,
            "ode_type": self.ode_type,
            "d": self.d,
            "is_lincomb": True,
            "num_components": self.num_components,
            "K_components": self.num_components,
            "gate_mode": self.gate_mode,
            "gate_temperature": self.gate_temperature,
            "mask_convention": "target_source",
            "use_mask": self.mask is not None,
            "use_decay": self.use_decay,
            "positive_epsilon": self.positive_eps,
            "positive_input_guard": "clamp_min(x, positive_epsilon)",
            "regulator_transform": "none",
            "regulation_A_init": self.regulation_A_init,
            "theta_init": self.theta_init,
            "positive_parameterization": "softplus(raw) + positive_epsilon",
            "raw_delta_init": self.raw_delta_init,
            "delta_physical_init": _scalar_softplus(self.raw_delta_init)
            + self.positive_eps,
            "off_mask_lambda": self.off_mask_lambda,
            "penalty_parameter": self.penalty_parameter_name,
            "penalty_reduction": "all_element_mean",
            "target_chunk_size": self.target_chunk_size,
            "w_is_exact": self.W_IS_EXACT,
        }


class CenteredSignedHillField(BaseDirectHillField):
    """H_ij(x_j) = tanh(alpha_ij * log(x_j / theta_ij))."""

    ode_type = "centered_signed_hill"

    def __init__(self, d: int, *, alpha_init_std: Optional[float] = None, **kwargs):
        super().__init__(d, **kwargs)
        self.alpha_init_std = (
            1.0 / math.sqrt(self.d)
            if alpha_init_std is None
            else _finite_float("alpha_init_std", alpha_init_std, positive=True)
        )
        self.alpha = nn.Parameter(
            torch.randn(self.num_components, self.d, self.d) * self.alpha_init_std
        )

    @property
    def signed_parameter(self) -> torch.Tensor:
        return self.alpha

    @property
    def penalty_parameter_name(self) -> str:
        return "alpha"

    def penalty_parameter(self) -> torch.Tensor:
        return self.alpha

    def edge_response(self, x_pos, theta, signed):
        return torch.tanh(signed * (torch.log(x_pos) - torch.log(theta)))

    def signed_regulatory_magnitude(self) -> torch.Tensor:
        return self.A * self.alpha

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "alpha_init": f"normal(0, {self.alpha_init_std})",
            "sign_parameter": "alpha",
            "outer_half_factor": False,
        })
        return info


class ShiftedHillRhoField(BaseDirectHillField):
    """H_ij(x_j) = expm1(rho_ij) * x_j^n/(theta_ij^n+x_j^n)."""

    ode_type = "shifted_hill_rho"

    def __init__(
        self,
        d: int,
        *,
        hill_n: float = 2.0,
        rho_init_std: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(d, **kwargs)
        self.hill_n = _finite_float("hill_n", hill_n, positive=True)
        if self.hill_n != 2.0:
            raise ValueError("hill_n is fixed to the inherited value 2.0")
        self.rho_init_std = (
            1.0 / math.sqrt(self.d)
            if rho_init_std is None
            else _finite_float("rho_init_std", rho_init_std, positive=True)
        )
        self.rho = nn.Parameter(
            torch.randn(self.num_components, self.d, self.d) * self.rho_init_std
        )

    @property
    def signed_parameter(self) -> torch.Tensor:
        return self.rho

    @property
    def penalty_parameter_name(self) -> str:
        return "rho"

    def penalty_parameter(self) -> torch.Tensor:
        return self.rho

    def edge_response(self, x_pos, theta, signed):
        log_ratio = self.hill_n * (torch.log(x_pos) - torch.log(theta))
        return torch.expm1(signed) * torch.sigmoid(log_ratio)

    def signed_regulatory_magnitude(self) -> torch.Tensor:
        return self.A * torch.expm1(self.rho)

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "hill_n": self.hill_n,
            "rho_init": f"normal(0, {self.rho_init_std})",
            "sign_parameter": "rho",
            "rho_transform": "torch.expm1",
        })
        return info


CenteredSignedHillODE = CenteredSignedHillField
ShiftedHillRhoODE = ShiftedHillRhoField

__all__ = [
    "ODE_TYPES",
    "BaseDirectHillField",
    "CenteredSignedHillField",
    "CenteredSignedHillODE",
    "ShiftedHillRhoField",
    "ShiftedHillRhoODE",
]
