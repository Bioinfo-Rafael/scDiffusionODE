"""ODE fields used only by the 2026-08-03 Hill/exp comparison.

The implementation deliberately imports the established LinComb time/gate helpers
instead of copying them.  Matrix parameters in this module use the mathematical
``[target, source]`` convention.  ``factory.py`` therefore transposes the legacy
edge mask (which is returned as ``[source, target]``).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ODE.ode_20260609_mathmlp import _TimeEmb, _mlp, _prep_t


ODE_TYPES = ("hill_after_linear", "racipe", "exp")


def _finite_float(name: str, value, *, positive: bool = False) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _fixed_hill_coefficient(name: str, value) -> float:
    value = _finite_float(name, value, positive=True)
    if value != 2.0:
        raise ValueError(
            f"{name} is fixed to 2.0 in this experiment, got {value!r}"
        )
    return value


def _inverse_softplus_for_physical_init(
    name: str, physical_value, positive_eps: float
) -> float:
    """Raw value whose ``softplus(raw) + positive_eps`` is physical_value."""

    physical_value = _finite_float(name, physical_value, positive=True)
    shifted = physical_value - positive_eps
    if shifted <= 0:
        raise ValueError(
            f"{name} ({physical_value}) must be greater than positive_eps "
            f"({positive_eps})"
        )
    # log(expm1(x)) is accurate near zero; this branch avoids overflow for a
    # deliberately large configured physical initialization.
    if shifted > 20.0:
        return shifted + math.log1p(-math.exp(-shifted))
    return math.log(math.expm1(shifted))


def _scalar_softplus(value: float) -> float:
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _hill_fraction(value: torch.Tensor, threshold: torch.Tensor, n: float) -> torch.Tensor:
    """Return v**n / (K**n + v**n) without overflowing for large v/K.

    ``value`` and ``threshold`` are non-negative/positive respectively.  softplus
    can underflow to an exact zero for a very negative input, so that case is
    explicitly restored to the exact mathematical value zero.
    """

    positive = value > 0
    safe_value = torch.where(positive, value, torch.ones_like(value))
    log_ratio = n * (torch.log(safe_value) - torch.log(threshold))
    fraction = torch.sigmoid(log_ratio)
    return torch.where(positive, fraction, torch.zeros_like(fraction))


class BaseODEField(nn.Module):
    """Shared TrainLoop-compatible interface for all three ODE equations.

    A LinComb field has a time-conditioned softmax gate and ``num_components``
    component ODEs.  A single field has no ``time_emb`` or ``coeff_net`` module;
    its only component is returned directly.
    """

    ode_type = "base"
    W_IS_EXACT = False

    def __init__(
        self,
        d: int,
        *,
        is_lincomb: bool,
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
    ):
        super().__init__()
        self.d = int(d)
        if self.d <= 0:
            raise ValueError(f"d must be > 0, got {d!r}")
        self.is_lincomb = bool(is_lincomb)
        self.num_components = int(num_components) if self.is_lincomb else 1
        self.num_experts = self.num_components
        if self.num_components <= 0:
            raise ValueError("num_components must be > 0")

        self.time_dim = int(time_dim)
        self.hidden = int(hidden)
        self.dropout = _finite_float("dropout", dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout!r}")

        self.gate_mode = str(gate_mode).lower()
        if self.is_lincomb and self.gate_mode != "softmax":
            raise ValueError(
                "the 20260803 LinComb comparison requires gate_mode='softmax'"
            )
        self.gate_temperature = _finite_float(
            "gate_temperature", gate_temperature, positive=True
        )
        self.off_mask_lambda = _finite_float("off_mask_lambda", off_mask_lambda)
        self.sparse_lambda = _finite_float("sparse_lambda", sparse_lambda)
        self.entropy_lambda = _finite_float("entropy_lambda", entropy_lambda)
        if min(self.off_mask_lambda, self.sparse_lambda, self.entropy_lambda) < 0:
            raise ValueError("regularization weights must be >= 0")
        if self.is_lincomb and self.sparse_lambda > 0:
            raise ValueError("sparse_lambda is not defined for the required softmax gate")

        self.ratio_reg_weight = _finite_float("ratio_reg_weight", ratio_reg_weight)
        self.ratio_reg_target = _finite_float(
            "ratio_reg_target", ratio_reg_target, positive=True
        )
        self.ratio_reg_eps = _finite_float("ratio_reg_eps", ratio_reg_eps, positive=True)
        if self.ratio_reg_weight < 0:
            raise ValueError("ratio_reg_weight must be >= 0")
        self.positive_eps = _finite_float("positive_eps", positive_eps, positive=True)
        self.raw_delta_init = _finite_float("raw_delta_init", raw_delta_init)
        self.use_decay = bool(use_decay)

        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.float32)
            if tuple(mask.shape) != (self.d, self.d):
                raise ValueError(
                    f"mask must have shape {(self.d, self.d)}, got {tuple(mask.shape)}"
                )
            self.register_buffer("mask", mask.contiguous())
        else:
            self.register_buffer("mask", None)

        # Match the established RNA-velocity field: one raw decay vector shared
        # by every LinComb component, initialized at raw value 0.1.
        self.raw_delta = nn.Parameter(
            torch.full((self.d,), self.raw_delta_init, dtype=torch.float32)
        )

        if self.is_lincomb:
            # These are the exact helper implementations used by the 20260707 and
            # 20260802 LinComb experiments.
            self.time_emb = _TimeEmb(self.time_dim)
            self.coeff_net = _mlp(
                self.d + self.time_dim,
                self.hidden,
                self.num_components,
                self.dropout,
            )

        # Duck-typed hooks consumed by guided_diffusion.train_util.TrainLoop.
        self.soft = bool(
            soft
            or self.off_mask_lambda > 0
            or self.sparse_lambda > 0
            or self.entropy_lambda > 0
            or self.ratio_reg_weight > 0
        )
        self._cached_gate_reg = None
        self._cached_gate_stats = None
        self._cached_ratio_reg = None

    @property
    def delta(self) -> torch.Tensor:
        return F.softplus(self.raw_delta) + self.positive_eps

    def _zero(self) -> torch.Tensor:
        return self.raw_delta.new_zeros(())

    def _as_components(self, parameter: torch.Tensor) -> torch.Tensor:
        return parameter if self.is_lincomb else parameter.unsqueeze(0)

    def _component_decay(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_decay:
            return x.new_zeros((x.shape[0], self.num_components, self.d))
        decay = self.delta.view(1, 1, self.d) * x[:, None, :]
        return decay.expand(-1, self.num_components, -1)

    def get_gate_values(self, x: torch.Tensor, t=None) -> Dict[str, torch.Tensor]:
        """Return the established softmax LinComb gate representation.

        For a single ODE this returns a single exact coefficient of one without
        creating a gate network.  Consumers should use ``is_lincomb`` or the
        presence of ``coeff_net`` to decide whether a gate plot is meaningful.
        """

        x = x.float()
        if not self.is_lincomb:
            one = x.new_ones((x.shape[0], 1))
            return {"logits": x.new_zeros((x.shape[0], 1)),
                    "coefficients": one, "probabilities": one}
        t_prepared = _prep_t(t, x.shape[0], x.device, x.dtype)
        features = torch.cat([x, self.time_emb(t_prepared)], dim=-1)
        logits = self.coeff_net(features)
        probabilities = torch.softmax(logits / self.gate_temperature, dim=-1)
        return {
            "logits": logits,
            "coefficients": probabilities,
            "probabilities": probabilities,
        }

    def _coeffs(self, x: torch.Tensor, t=None) -> torch.Tensor:
        return self.get_gate_values(x, t)["coefficients"]

    def effective_coefficients(self, x: torch.Tensor, t=None) -> torch.Tensor:
        return self._coeffs(x, t)

    get_effective_coefficients = effective_coefficients

    def component_outputs(self, x: torch.Tensor, t=None) -> torch.Tensor:
        """Return equation-level component outputs with shape ``(B, C, d)``."""

        raise NotImplementedError

    def get_component_outputs(self, x: torch.Tensor, t=None) -> torch.Tensor:
        return self.component_outputs(x, t)

    forward_components = get_component_outputs

    def component_contributions(self, x: torch.Tensor, t=None) -> torch.Tensor:
        components = self.component_outputs(x, t)
        coefficients = self.effective_coefficients(x, t)
        return coefficients[:, :, None] * components

    def _cache_gate_regularization(self, gate: Dict[str, torch.Tensor]) -> None:
        self._cached_gate_reg = None
        if not self.training or not self.is_lincomb:
            self._cached_gate_stats = None
            return
        probabilities = gate["probabilities"]
        if self.entropy_lambda > 0:
            self._cached_gate_reg = -(
                probabilities * torch.log(probabilities + self.ratio_reg_eps)
            ).sum(dim=-1).mean()
        with torch.no_grad():
            entropy = -(
                probabilities * torch.log(probabilities + self.ratio_reg_eps)
            ).sum(dim=-1)
            self._cached_gate_stats = {
                "mean_abs_a": float(gate["coefficients"].abs().mean().item()),
                "entropy": float(entropy.mean().item()),
            }

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        x = x.float()
        components = self.component_outputs(x, t)
        if not self.is_lincomb:
            self._cached_gate_reg = None
            self._cached_gate_stats = None
            return components[:, 0, :]
        gate = self.get_gate_values(x, t)
        self._cache_gate_regularization(gate)
        return torch.einsum("bk,bkd->bd", gate["coefficients"], components)

    @property
    def penalty_parameter_name(self) -> str:
        raise NotImplementedError

    def penalty_parameter(self) -> torch.Tensor:
        raise NotImplementedError

    def _offmask_base(self, norm: str = "l1") -> torch.Tensor:
        parameter = self.penalty_parameter()
        if self.mask is not None:
            # Intentionally average over *all* matrix entries after masking, as in
            # the 20260707 baseline (on-mask entries become zeros in the mean).
            selected = parameter * (1.0 - self.mask)
        else:
            selected = parameter
        if str(norm).lower() == "l2":
            return selected.square().mean()
        if str(norm).lower() != "l1":
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
        return selected.abs().mean()

    def off_mask_penalty(self, norm: str = "l1") -> torch.Tensor:
        total = self._zero()
        if self.off_mask_lambda > 0:
            total = total + self.off_mask_lambda * self._offmask_base(norm)
        gate_reg = getattr(self, "_cached_gate_reg", None)
        if gate_reg is not None and self.entropy_lambda > 0:
            total = total + self.entropy_lambda * gate_reg
        ratio_reg = getattr(self, "_cached_ratio_reg", None)
        if ratio_reg is not None and self.ratio_reg_weight > 0:
            total = total + self.ratio_reg_weight * ratio_reg
        return total

    def has_aux_regularization(self) -> bool:
        return bool(
            self.off_mask_lambda > 0
            or self.entropy_lambda > 0
            or self.ratio_reg_weight > 0
        )

    def analysis_parameters(self) -> Dict[str, torch.Tensor]:
        return {"delta": self.delta, "raw_delta": self.raw_delta}

    def get_model_info(self) -> dict:
        return {
            "class": type(self).__name__,
            "type": self.ode_type,
            "ode_type": self.ode_type,
            "d": self.d,
            "is_lincomb": self.is_lincomb,
            "num_components": self.num_components,
            "K_components": self.num_components,
            "gate_mode": self.gate_mode if self.is_lincomb else None,
            "gate_temperature": self.gate_temperature if self.is_lincomb else None,
            "time_dim": self.time_dim if self.is_lincomb else None,
            "field_hidden": self.hidden if self.is_lincomb else None,
            "field_dropout": self.dropout if self.is_lincomb else None,
            "use_mask": self.mask is not None,
            "mask_convention": "target_source",
            "soft": self.soft,
            "use_decay": self.use_decay,
            "delta_shared_across_components": True,
            "positive_eps": self.positive_eps,
            "positive_epsilon": self.positive_eps,
            "raw_delta_init": self.raw_delta_init,
            "delta_physical_init": _scalar_softplus(self.raw_delta_init)
            + self.positive_eps,
            "positive_parameterization": "softplus(raw) + positive_epsilon",
            "off_mask_lambda": self.off_mask_lambda,
            "entropy_lambda": self.entropy_lambda,
            "ratio_reg_weight": self.ratio_reg_weight,
            "ratio_reg_target": self.ratio_reg_target,
            "penalty_parameter": self.penalty_parameter_name,
            "penalty_reduction": "all_element_mean",
            "w_is_exact": self.W_IS_EXACT,
        }


class HillAfterLinearField(BaseODEField):
    """softplus(Wx+b) followed by a saturating Hill production term."""

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
        super().__init__(d, **kwargs)
        self.hill_n = _fixed_hill_coefficient("hill_n", hill_n)
        self.hill_K_init = _finite_float("hill_K_init", hill_K_init, positive=True)
        self.hill_V_init = _finite_float("hill_V_init", hill_V_init, positive=True)
        raw_K_init = _inverse_softplus_for_physical_init(
            "hill_K_init", self.hill_K_init, self.positive_eps
        )
        raw_V_init = _inverse_softplus_for_physical_init(
            "hill_V_init", self.hill_V_init, self.positive_eps
        )
        self.raw_hill_K_init = raw_K_init
        self.raw_hill_V_init = raw_V_init
        component_shape = (self.num_components,) if self.is_lincomb else ()
        self.W = nn.Parameter(
            torch.randn(component_shape + (self.d, self.d)) / math.sqrt(self.d)
        )
        self.b = nn.Parameter(torch.zeros(component_shape + (self.d,)))
        self.raw_K = nn.Parameter(
            torch.full(component_shape + (self.d,), raw_K_init)
        )
        self.raw_V = nn.Parameter(
            torch.full(component_shape + (self.d,), raw_V_init)
        )

    @property
    def K(self) -> torch.Tensor:
        return F.softplus(self.raw_K) + self.positive_eps

    @property
    def V(self) -> torch.Tensor:
        return F.softplus(self.raw_V) + self.positive_eps

    @property
    def bias(self) -> torch.Tensor:
        return self.b

    @property
    def expert_W(self) -> torch.Tensor:
        if not self.is_lincomb:
            raise AttributeError("single ODE has no expert_W mixture")
        return self.W

    @property
    def expert_b(self) -> torch.Tensor:
        if not self.is_lincomb:
            raise AttributeError("single ODE has no expert_b mixture")
        return self.b

    @property
    def penalty_parameter_name(self) -> str:
        return "W"

    def penalty_parameter(self) -> torch.Tensor:
        return self.W

    def component_outputs(self, x: torch.Tensor, t=None) -> torch.Tensor:
        del t
        x = x.float()
        W = self._as_components(self.W)
        b = self._as_components(self.b)
        threshold = self._as_components(self.K)
        capacity = self._as_components(self.V)
        z = F.softplus(torch.einsum("kij,bj->bki", W, x) + b)
        saturation = _hill_fraction(z, threshold[None, :, :], self.hill_n)
        production = capacity[None, :, :] * saturation
        return production - self._component_decay(x)

    @torch.no_grad()
    def compute_W(self, x: torch.Tensor, t=None) -> torch.Tensor:
        x = x.float()
        if not self.is_lincomb:
            return self.W[None, :, :].expand(x.shape[0], -1, -1)
        coefficients = self.effective_coefficients(x, t)
        return torch.einsum("bk,kij->bij", coefficients, self.W)

    def analysis_parameters(self) -> Dict[str, torch.Tensor]:
        values = super().analysis_parameters()
        values.update({"W": self.W, "bias": self.b, "K": self.K, "V": self.V})
        return values

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "hill_n": self.hill_n,
            "hill_K_init": self.hill_K_init,
            "hill_V_init": self.hill_V_init,
            "raw_hill_K_init": self.raw_hill_K_init,
            "raw_hill_V_init": self.raw_hill_V_init,
            "W_init": "normal(0, 1/sqrt(d))",
            "bias_init": 0.0,
            "parameterization": "K[k,i], V[k,i]",
        })
        return info


class ExpField(BaseODEField):
    """RNA-velocity-style field with exp replacing only the production softplus."""

    ode_type = "exp"

    def __init__(
        self,
        d: int,
        *,
        exp_preactivation_min: float = -20.0,
        exp_preactivation_max: float = 20.0,
        **kwargs,
    ):
        super().__init__(d, **kwargs)
        self.exp_preactivation_min = _finite_float(
            "exp_preactivation_min", exp_preactivation_min
        )
        self.exp_preactivation_max = _finite_float(
            "exp_preactivation_max", exp_preactivation_max
        )
        if self.exp_preactivation_min >= self.exp_preactivation_max:
            raise ValueError("exp_preactivation_min must be < exp_preactivation_max")
        component_shape = (self.num_components,) if self.is_lincomb else ()
        self.W = nn.Parameter(
            torch.randn(component_shape + (self.d, self.d)) / math.sqrt(self.d)
        )
        self.b = nn.Parameter(torch.zeros(component_shape + (self.d,)))

    @property
    def bias(self) -> torch.Tensor:
        return self.b

    @property
    def expert_W(self) -> torch.Tensor:
        if not self.is_lincomb:
            raise AttributeError("single ODE has no expert_W mixture")
        return self.W

    @property
    def expert_b(self) -> torch.Tensor:
        if not self.is_lincomb:
            raise AttributeError("single ODE has no expert_b mixture")
        return self.b

    @property
    def penalty_parameter_name(self) -> str:
        return "W"

    def penalty_parameter(self) -> torch.Tensor:
        return self.W

    def preactivation(self, x: torch.Tensor) -> torch.Tensor:
        W = self._as_components(self.W)
        b = self._as_components(self.b)
        return torch.einsum("kij,bj->bki", W, x.float()) + b

    def component_outputs(self, x: torch.Tensor, t=None) -> torch.Tensor:
        del t
        x = x.float()
        preactivation = self.preactivation(x).clamp(
            min=self.exp_preactivation_min,
            max=self.exp_preactivation_max,
        )
        return torch.exp(preactivation) - self._component_decay(x)

    @torch.no_grad()
    def compute_W(self, x: torch.Tensor, t=None) -> torch.Tensor:
        x = x.float()
        if not self.is_lincomb:
            return self.W[None, :, :].expand(x.shape[0], -1, -1)
        coefficients = self.effective_coefficients(x, t)
        return torch.einsum("bk,kij->bij", coefficients, self.W)

    def analysis_parameters(self) -> Dict[str, torch.Tensor]:
        values = super().analysis_parameters()
        values.update({"W": self.W, "bias": self.b})
        return values

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "exp_preactivation_min": self.exp_preactivation_min,
            "exp_preactivation_max": self.exp_preactivation_max,
            "W_init": "normal(0, 1/sqrt(d))",
            "bias_init": 0.0,
        })
        return info


class RacipeField(BaseODEField):
    """RACIPE shifted-Hill product evaluated in log space.

    Thresholds have shape ``K[k,j]``: they are component/source specific and are
    shared by every target.  This avoids a second unrestricted edge-wise tensor.
    The regulatory ``r`` matrix retains full target/source resolution because it
    encodes edge direction and strength and is the sole off-mask penalty target.
    """

    ode_type = "racipe"

    def __init__(
        self,
        d: int,
        *,
        racipe_n: float = 2.0,
        racipe_input_scale: float = 1.0,
        racipe_log_production_min: float = -20.0,
        racipe_log_production_max: float = 20.0,
        racipe_target_chunk_size: int = 64,
        racipe_K_init: float = 1.0,
        racipe_G_init: float = 1.0,
        **kwargs,
    ):
        super().__init__(d, **kwargs)
        self.racipe_n = _fixed_hill_coefficient("racipe_n", racipe_n)
        self.racipe_input_scale = _finite_float(
            "racipe_input_scale", racipe_input_scale, positive=True
        )
        self.racipe_log_production_min = _finite_float(
            "racipe_log_production_min", racipe_log_production_min
        )
        self.racipe_log_production_max = _finite_float(
            "racipe_log_production_max", racipe_log_production_max
        )
        if self.racipe_log_production_min >= self.racipe_log_production_max:
            raise ValueError(
                "racipe_log_production_min must be < racipe_log_production_max"
            )
        self.racipe_target_chunk_size = int(racipe_target_chunk_size)
        if self.racipe_target_chunk_size <= 0:
            raise ValueError("racipe_target_chunk_size must be > 0")
        self.racipe_K_init = _finite_float("racipe_K_init", racipe_K_init, positive=True)
        self.racipe_G_init = _finite_float("racipe_G_init", racipe_G_init, positive=True)
        raw_K_init = _inverse_softplus_for_physical_init(
            "racipe_K_init", self.racipe_K_init, self.positive_eps
        )
        raw_G_init = _inverse_softplus_for_physical_init(
            "racipe_G_init", self.racipe_G_init, self.positive_eps
        )
        self.raw_racipe_K_init = raw_K_init
        self.raw_racipe_G_init = raw_G_init

        component_shape = (self.num_components,) if self.is_lincomb else ()
        # r=0 gives lambda=1 and an exact neutral factor at initialization.
        self.r = nn.Parameter(torch.zeros(component_shape + (self.d, self.d)))
        # K[k,j] is source-specific and shared across targets.
        self.raw_K = nn.Parameter(
            torch.full(component_shape + (self.d,), raw_K_init)
        )
        self.raw_G = nn.Parameter(
            torch.full(component_shape + (self.d,), raw_G_init)
        )

    @property
    def K(self) -> torch.Tensor:
        return F.softplus(self.raw_K) + self.positive_eps

    @property
    def G(self) -> torch.Tensor:
        return F.softplus(self.raw_G) + self.positive_eps

    @property
    def lambdas(self) -> torch.Tensor:
        return torch.exp(self.r)

    @property
    def lambda_(self) -> torch.Tensor:
        return self.lambdas

    @property
    def penalty_parameter_name(self) -> str:
        return "r"

    def penalty_parameter(self) -> torch.Tensor:
        return self.r

    def source_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x.float() / self.racipe_input_scale)

    def source_hill_values(self, x: torch.Tensor) -> torch.Tensor:
        """Return h(u_j) as ``(B, C, source)``."""

        u = self.source_inputs(x)
        threshold = self._as_components(self.K)
        return _hill_fraction(
            u[:, None, :], threshold[None, :, :], self.racipe_n
        )

    @staticmethod
    def _log_shifted_factor(h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Stable log((1-h) + h*exp(r)), exact at r=0 with live gradient.

        ``logaddexp`` avoids overflow/underflow away from neutrality.  At exactly
        ``r == 0``, ``log1p(expm1(r)*h)`` produces an exact zero while retaining
        derivative ``h`` with respect to r; using a constant-one replacement here
        would incorrectly kill the initialization gradient.
        """

        # Clamp only the logarithm inputs, then restore exact boundary formulas.
        # This prevents log(0) derivatives from contaminating an otherwise finite
        # backward pass when a Hill value saturates numerically to zero or one.
        tiny = torch.finfo(h.dtype).tiny
        log_h = torch.log(h.clamp_min(tiny))
        log_one_minus_h = torch.log((1.0 - h).clamp_min(tiny))
        stable = torch.logaddexp(log_one_minus_h, log_h + r)
        stable = torch.where(h == 0, r * 0.0, stable)
        stable = torch.where(h == 1, r, stable)

        neutral_mask = r == 0
        # Avoid evaluating expm1 on a potentially huge non-neutral r in this
        # branch; the selected r=0 entries retain their derivative.
        neutral_r = torch.where(neutral_mask, r, torch.zeros_like(r))
        neutral = torch.log1p(torch.expm1(neutral_r) * h)
        return torch.where(neutral_mask, neutral, stable)

    def shifted_hill_factors(self, x: torch.Tensor) -> torch.Tensor:
        """Return H^S with shape ``(B, C, target, source)``."""

        h = self.source_hill_values(x)[:, :, None, :]
        r = self._as_components(self.r)[None, :, :, :]
        return torch.exp(self._log_shifted_factor(h, r))

    get_shifted_hill_factors = shifted_hill_factors

    def _chunk_log_production(
        self,
        h: torch.Tensor,
        r_chunk: torch.Tensor,
        production_scale_chunk: torch.Tensor,
    ) -> torch.Tensor:
        log_factor = self._log_shifted_factor(
            h[:, :, None, :], r_chunk[None, :, :, :]
        )
        return (
            torch.log(production_scale_chunk[None, :, :])
            + log_factor.sum(dim=-1)
        )

    def log_production(self, x: torch.Tensor, *, clamp: bool = True) -> torch.Tensor:
        """Return component log production with shape ``(B, C, target)``."""

        x = x.float()
        h = self.source_hill_values(x)
        r = self._as_components(self.r)
        production_scale = self._as_components(self.G)
        chunks = []
        chunk_size = min(self.racipe_target_chunk_size, self.d)
        for start in range(0, self.d, chunk_size):
            stop = min(start + chunk_size, self.d)
            r_chunk = r[:, start:stop, :]
            production_scale_chunk = production_scale[:, start:stop]
            use_checkpoint = (
                self.training
                and torch.is_grad_enabled()
                and (h.requires_grad or r_chunk.requires_grad
                     or production_scale_chunk.requires_grad)
            )
            if use_checkpoint:
                # Saved activations are only B*C*chunk here.  The much larger
                # B*C*chunk*d factor tensor is recomputed during backward, which
                # keeps d~1024, B=128 training within a practical memory bound.
                log_value = checkpoint(
                    self._chunk_log_production,
                    h,
                    r_chunk,
                    production_scale_chunk,
                    use_reentrant=False,
                )
            else:
                log_value = self._chunk_log_production(
                    h, r_chunk, production_scale_chunk
                )
            chunks.append(log_value)
        result = torch.cat(chunks, dim=-1)
        if clamp:
            result = result.clamp(
                min=self.racipe_log_production_min,
                max=self.racipe_log_production_max,
            )
        return result

    def component_outputs(self, x: torch.Tensor, t=None) -> torch.Tensor:
        del t
        x = x.float()
        production = torch.exp(self.log_production(x, clamp=True))
        return production - self._component_decay(x)

    @torch.no_grad()
    def compute_regulatory_matrix(self, x: torch.Tensor, t=None) -> torch.Tensor:
        """Return coefficient-weighted r as an analysis proxy, never lambda."""

        x = x.float()
        if not self.is_lincomb:
            return self.r[None, :, :].expand(x.shape[0], -1, -1)
        coefficients = self.effective_coefficients(x, t)
        return torch.einsum("bk,kij->bij", coefficients, self.r)

    compute_W = compute_regulatory_matrix

    def analysis_parameters(self) -> Dict[str, torch.Tensor]:
        values = super().analysis_parameters()
        values.update({"r": self.r, "lambda": self.lambdas, "K": self.K, "G": self.G})
        return values

    def get_model_info(self) -> dict:
        info = super().get_model_info()
        info.update({
            "racipe_n": self.racipe_n,
            "racipe_input_scale": self.racipe_input_scale,
            "racipe_log_production_min": self.racipe_log_production_min,
            "racipe_log_production_max": self.racipe_log_production_max,
            "racipe_target_chunk_size": self.racipe_target_chunk_size,
            "racipe_K_init": self.racipe_K_init,
            "racipe_G_init": self.racipe_G_init,
            "raw_racipe_K_init": self.raw_racipe_K_init,
            "raw_racipe_G_init": self.raw_racipe_G_init,
            "r_init": 0.0,
            "r_neutral_at_initialization": True,
            "chunk_activation_checkpointing": True,
            "threshold_parameterization": "K[k,j], shared across targets",
            "compute_W_semantics": "coefficient-weighted r proxy",
        })
        return info


# Concise aliases used by tests/analysis and by external notebooks.
HillAfterLinearODE = HillAfterLinearField
RacipeODE = RacipeField
RACIPEField = RacipeField
ExpODE = ExpField


__all__ = [
    "ODE_TYPES",
    "BaseODEField",
    "HillAfterLinearField",
    "HillAfterLinearODE",
    "RacipeField",
    "RacipeODE",
    "RACIPEField",
    "ExpField",
    "ExpODE",
]
