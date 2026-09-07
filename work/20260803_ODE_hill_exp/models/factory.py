"""Model factory for the twelve isolated 2026-08-03 experiments."""

from __future__ import annotations

import math
from typing import Mapping, Optional

import torch

from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260609_hybrid5x3 import UnifiedODEMLHybrid
from ODE.ode_20260609_mathmlp import build_edge_mask
from ODE.ode_20260707_lincomb import LinCombOnlyDenoiser

from .ode_fields import (
    ODE_TYPES,
    ExpField,
    HillAfterLinearField,
    RacipeField,
)


MODEL_FAMILIES = (
    "ode_only_lincomb",
    "standard_hybrid_lincomb",
    "ts_soft_tau80_hybrid_lincomb",
    "standard_hybrid_single",
)


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "none", ""}:
        return False
    raise ValueError(f"cannot interpret {value!r} as bool")


def _float(config: Mapping, key: str, default) -> float:
    value = float(config.get(key, default))
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite, got {value!r}")
    return value


def _family(config: Mapping) -> str:
    family = str(config.get("model_family", "")).lower()
    if family not in MODEL_FAMILIES:
        raise ValueError(
            f"model_family must be one of {MODEL_FAMILIES}, got {family!r}"
        )
    return family


def _ode_type(config: Mapping) -> str:
    ode_type = str(config.get("ode_type", "")).lower()
    if ode_type not in ODE_TYPES:
        raise ValueError(f"ode_type must be one of {ODE_TYPES}, got {ode_type!r}")
    return ode_type


def _is_lincomb_family(family: str) -> bool:
    return family != "standard_hybrid_single"


def _build_target_source_mask(
    config: Mapping,
    gene_list,
    explicit_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Build W[target,source]-aligned mask from the legacy source,target helper."""

    if not _bool(config.get("use_mask_reg", True)):
        return None
    if explicit_mask is not None:
        mask = torch.as_tensor(explicit_mask, dtype=torch.float32)
        if tuple(mask.shape) != (len(gene_list), len(gene_list)):
            raise ValueError(
                f"explicit mask must be {(len(gene_list), len(gene_list))}, "
                f"got {tuple(mask.shape)}"
            )
        # Explicit masks are already required to follow this module's convention.
        return mask.contiguous()
    edge_tsv_path = config.get("edge_tsv_path")
    if not edge_tsv_path:
        raise ValueError("edge_tsv_path is required when use_mask_reg=true")
    # build_edge_mask returns [source,target]; equations and einsum use
    # W[target,source].  The transpose is therefore essential, not cosmetic.
    return build_edge_mask(gene_list, edge_tsv_path).T.contiguous()


def _validate_family_policy(config: Mapping, family: str) -> None:
    is_lincomb = _is_lincomb_family(family)
    if is_lincomb:
        components = int(config.get("K", 8))
        if components != 8:
            raise ValueError(f"all LinComb experiments require K=8, got {components}")
        gate_mode = str(config.get("gate_mode", "softmax")).lower()
        if gate_mode != "softmax":
            raise ValueError("all LinComb experiments require gate_mode='softmax'")

    if family == "ts_soft_tau80_hybrid_lincomb":
        regime_mode = str(config.get("regime_gate_mode", "Ts_I_vs_II_III"))
        if regime_mode.lower() != "ts_i_vs_ii_iii":
            raise ValueError(
                "ts_soft_tau80_hybrid_lincomb requires "
                "regime_gate_mode='Ts_I_vs_II_III'"
            )
        gate_type = str(config.get("regime_gate_type", "sigmoid")).lower()
        if gate_type != "sigmoid":
            raise ValueError(
                "ts_soft_tau80_hybrid_lincomb requires regime_gate_type='sigmoid'"
            )
        tau = _float(config, "gate_tau", 80.0)
        if tau != 80.0:
            raise ValueError(
                f"ts_soft_tau80_hybrid_lincomb requires gate_tau=80, got {tau}"
            )
        t_s = config.get("t_s")
        if t_s is None or (isinstance(t_s, str) and t_s.lower() == "auto"):
            raise ValueError(
                "t_s must be resolved to a numeric timestep before model construction"
            )
    else:
        regime_mode = str(config.get("regime_gate_mode", "none")).lower()
        if regime_mode != "none":
            raise ValueError(f"{family} requires regime_gate_mode='none'")

    if _bool(config.get("reverse_coef", False)):
        raise ValueError("the requested twelve experiments require reverse_coef=false")


def build_ode_from_config(
    config: Mapping,
    gene_list,
    device="cpu",
    *,
    mask: Optional[torch.Tensor] = None,
):
    """Construct one of the six equation/structure ODE fields.

    ``mask`` is an optional already-transposed ``[target,source]`` tensor intended
    for unit tests.  Normal runs build it from ``config['edge_tsv_path']`` using
    the established helper and transpose it exactly once.
    """

    family = _family(config)
    ode_type = _ode_type(config)
    _validate_family_policy(config, family)
    is_lincomb = _is_lincomb_family(family)
    edge_mask = _build_target_source_mask(config, gene_list, mask)

    common = dict(
        d=len(gene_list),
        is_lincomb=is_lincomb,
        num_components=int(config.get("K", 8)) if is_lincomb else 1,
        mask=edge_mask,
        soft=_bool(config.get("SoftReg", True)),
        time_dim=int(config.get("time_dim", 64)),
        hidden=int(config.get("field_hidden", 256)),
        dropout=_float(config, "field_dropout", 0.0),
        gate_mode=str(config.get("gate_mode", "softmax")),
        gate_temperature=_float(config, "gate_temperature", 1.0),
        off_mask_lambda=_float(config, "off_mask_lambda", 5.0),
        sparse_lambda=_float(config, "sparse_lambda", 0.0),
        entropy_lambda=_float(config, "entropy_lambda", 0.0),
        ratio_reg_weight=_float(config, "ratio_reg_weight", 0.0),
        ratio_reg_target=_float(config, "ratio_reg_target", 1.0),
        ratio_reg_eps=_float(config, "ratio_reg_eps", 1e-8),
        positive_eps=float(
            config.get("positive_epsilon", config.get("positive_eps", 1e-6))
        ),
        raw_delta_init=_float(config, "raw_delta_init", 0.1),
        use_decay=_bool(config.get("use_decay", True)),
    )

    if ode_type == "hill_after_linear":
        field = HillAfterLinearField(
            **common,
            hill_n=_float(config, "hill_n", 2.0),
            hill_K_init=_float(config, "hill_K_init", 1.0),
            hill_V_init=_float(config, "hill_V_init", 1.0),
        )
    elif ode_type == "racipe":
        field = RacipeField(
            **common,
            racipe_n=_float(config, "racipe_n", 2.0),
            racipe_input_scale=_float(config, "racipe_input_scale", 1.0),
            racipe_log_production_min=_float(
                config, "racipe_log_production_min", -20.0
            ),
            racipe_log_production_max=_float(
                config, "racipe_log_production_max", 20.0
            ),
            racipe_target_chunk_size=int(config.get("racipe_target_chunk_size", 64)),
            racipe_K_init=_float(config, "racipe_K_init", 1.0),
            racipe_G_init=_float(config, "racipe_G_init", 1.0),
        )
    else:
        field = ExpField(
            **common,
            exp_preactivation_min=_float(config, "exp_preactivation_min", -20.0),
            exp_preactivation_max=_float(config, "exp_preactivation_max", 20.0),
        )
    field.model_family = family
    return field.to(device)


class InspectableLinCombOnlyDenoiser(LinCombOnlyDenoiser):
    """Existing ODE-only wrapper plus parameter-free analysis hooks."""

    def __init__(self, field, model_family: str):
        super().__init__(field)
        self.model_family = str(model_family)

    def component_outputs(self, x, t=None):
        return self.ode_model.component_outputs(x.float(), t)

    get_component_outputs = component_outputs

    def effective_coefficients(self, x, t=None):
        return self.ode_model.effective_coefficients(x.float(), t)

    get_effective_coefficients = effective_coefficients

    def component_contributions(self, x, t=None):
        return self.ode_model.component_contributions(x.float(), t)

    def get_model_info(self):
        return {
            "class": type(self).__name__,
            "base_wrapper": "ODE.ode_20260707_lincomb.LinCombOnlyDenoiser",
            "denoiser_mode": "lincomb_only",
            "model_family": self.model_family,
            "ode_model": self.ode_model.get_model_info(),
        }


class InspectableUnifiedODEMLHybrid(UnifiedODEMLHybrid):
    """The established Hybrid implementation with read-only analysis helpers."""

    def __init__(self, *args, model_family: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_family = str(model_family)

    def ode_branch_weight(self, x: torch.Tensor, t) -> torch.Tensor:
        """Return the exact existing Hybrid ODE coefficient as ``(B,1)``."""

        if self.regime_gate_mode != "none":
            weight = self._regime_ode_weight(t, x.device, x.dtype)
        else:
            weight = self._scheduler(t, x.device, x.dtype)
            if weight.dim() == 0:
                weight = weight.view(1, 1)
            elif weight.dim() == 1:
                weight = weight[:, None]
            else:
                weight = weight.reshape(weight.shape[0], -1)
            if self.reverse_coef:
                weight = 1.0 - weight
        if weight.shape[0] == 1 and x.shape[0] > 1:
            weight = weight.expand(x.shape[0], -1)
        return weight

    def component_outputs(self, x, t=None):
        return self.ode_model.component_outputs(x.float(), t)

    get_component_outputs = component_outputs

    def effective_coefficients(self, x, t=None):
        coefficients = self.ode_model.effective_coefficients(x.float(), t)
        return coefficients * self.ode_branch_weight(x.float(), t)

    get_effective_coefficients = effective_coefficients

    def branch_outputs(self, x: torch.Tensor, t, y=None):
        """Return raw and exactly weighted Hybrid branches for diagnostics."""

        x = x.float()
        ode_raw = self.ode_model(x, t)
        ml_raw = self.ml_model(x, t, y)
        weight = self.ode_branch_weight(x, t)
        while weight.dim() < x.dim():
            weight = weight.unsqueeze(-1)

        if self.hybrid_norm_mode == "normed_learned_scale":
            eps = self.scale_eps
            ode_term = ode_raw / (ode_raw.norm(p=2, dim=-1, keepdim=True) + eps)
            ml_term = ml_raw / (ml_raw.norm(p=2, dim=-1, keepdim=True) + eps)
            scale = torch.exp(self.log_scale)
        else:
            # The twelve requested configs use hybrid_norm_mode='none'.
            ode_term, ml_term, scale = ode_raw, ml_raw, x.new_ones(())
        ode_contribution = scale * weight * ode_term
        ml_contribution = scale * (1.0 - weight) * ml_term
        return {
            "ode_raw": ode_raw,
            "ml_raw": ml_raw,
            "ode_weight": weight,
            "ml_weight": 1.0 - weight,
            "ode_contribution": ode_contribution,
            "ml_contribution": ml_contribution,
            "output": ode_contribution + ml_contribution,
        }

    get_branch_outputs = branch_outputs

    def get_model_info(self):
        info = super().get_model_info()
        info.update({
            "class": type(self).__name__,
            "base_wrapper": "ODE.ode_20260609_hybrid5x3.UnifiedODEMLHybrid",
            "model_family": self.model_family,
            "ode_model": self.ode_model.get_model_info(),
        })
        return info


def build_model_from_config(config: Mapping, gene_list, timesteps: int, device="cpu"):
    """Build one of the exact 3 ODE x 4 family experiment models."""

    family = _family(config)
    _validate_family_policy(config, family)
    field = build_ode_from_config(config, gene_list, device=device)

    if family == "ode_only_lincomb":
        return InspectableLinCombOnlyDenoiser(field, family).to(device)

    hidden_num = config.get("cell_unet_hidden_num", [2000, 1000, 500, 500])
    if not isinstance(hidden_num, (list, tuple)) or len(hidden_num) != 4:
        raise ValueError("cell_unet_hidden_num must contain exactly four positive widths")
    hidden_num = [int(width) for width in hidden_num]
    if any(width <= 0 for width in hidden_num):
        raise ValueError("cell_unet_hidden_num widths must be positive")
    ml_model = Cell_Unet(input_dim=len(gene_list), hidden_num=hidden_num).to(device)
    hybrid = InspectableUnifiedODEMLHybrid(
        ode_model=field,
        ml_model=ml_model,
        timesteps=int(timesteps),
        hybrid_norm_mode=str(config.get("hybrid_norm_mode", "none")),
        hybrid_scale_init=_float(config, "hybrid_scale_init", 1.0),
        hybrid_scale_eps=_float(config, "hybrid_scale_eps", 1e-8),
        scale_model=None,
        scale_input_source=str(config.get("scale_input_source", "ml_emb")),
        ode_input_source=str(config.get("ode_input_source", "none")),
        scale_eps=_float(config, "scale_eps", 1e-8),
        reverse_coef=_bool(config.get("reverse_coef", False)),
        regime_gate_mode=str(config.get("regime_gate_mode", "none")),
        regime_gate_type=str(config.get("regime_gate_type", "sigmoid")),
        t_s=config.get("t_s"),
        gate_tau=_float(config, "gate_tau", 80.0),
        model_family=family,
    )
    return hybrid.to(device)


# Short alias for interactive use.
build_model = build_model_from_config


__all__ = [
    "MODEL_FAMILIES",
    "InspectableLinCombOnlyDenoiser",
    "InspectableUnifiedODEMLHybrid",
    "build_ode_from_config",
    "build_model_from_config",
    "build_model",
]
