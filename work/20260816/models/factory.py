"""Model factory for the four isolated 2026-08-16 experiments."""

from __future__ import annotations

import math
from typing import Mapping, Optional

import torch

from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260609_hybrid5x3 import UnifiedODEMLHybrid
from ODE.ode_20260609_mathmlp import build_edge_mask

from .ode_fields import (
    ODE_TYPES,
    CenteredSignedHillField,
    ShiftedHillRhoField,
)


MODEL_FAMILIES = ("linear_hybrid_lincomb", "ts_soft_tau80_hybrid_lincomb")


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


def _build_target_source_mask(
    config: Mapping,
    gene_list,
    explicit_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Build A[target,source]-aligned mask from the legacy source,target helper."""

    if not _bool(config.get("use_mask_reg", True)):
        return None
    if explicit_mask is not None:
        mask = torch.as_tensor(explicit_mask, dtype=torch.float32)
        if tuple(mask.shape) != (len(gene_list), len(gene_list)):
            raise ValueError(
                f"explicit mask must be {(len(gene_list), len(gene_list))}, "
                f"got {tuple(mask.shape)}"
            )
        return mask.contiguous()
    edge_tsv_path = config.get("edge_tsv_path")
    if not edge_tsv_path:
        raise ValueError("edge_tsv_path is required when use_mask_reg=true")
    return build_edge_mask(gene_list, edge_tsv_path).T.contiguous()


def _validate_policy(config: Mapping, family: str) -> None:
    if int(config.get("K", 8)) != 8:
        raise ValueError("all 20260816 experiments require K=8")
    if str(config.get("gate_mode", "softmax")).lower() != "softmax":
        raise ValueError("all 20260816 experiments require gate_mode='softmax'")
    if _bool(config.get("reverse_coef", False)):
        raise ValueError("all 20260816 experiments require reverse_coef=false")
    regime_mode = str(config.get("regime_gate_mode", "none")).lower()
    if family == "ts_soft_tau80_hybrid_lincomb":
        if regime_mode != "ts_i_vs_ii_iii":
            raise ValueError("TS_soft_tau80 requires regime_gate_mode=Ts_I_vs_II_III")
        if str(config.get("regime_gate_type", "sigmoid")).lower() != "sigmoid":
            raise ValueError("TS_soft_tau80 requires regime_gate_type=sigmoid")
        if _float(config, "gate_tau", 80.0) != 80.0:
            raise ValueError("TS_soft_tau80 requires gate_tau=80.0")
        t_s = config.get("t_s")
        if t_s is None or str(t_s).lower() == "auto":
            raise ValueError("t_s must be numeric before model construction")
    elif regime_mode != "none":
        raise ValueError("linear Hybrid requires regime_gate_mode=none")


def build_ode_from_config(
    config: Mapping,
    gene_list,
    device="cpu",
    *,
    mask: Optional[torch.Tensor] = None,
):
    family = _family(config)
    ode_type = _ode_type(config)
    _validate_policy(config, family)
    edge_mask = _build_target_source_mask(config, gene_list, mask)
    common = dict(
        d=len(gene_list),
        is_lincomb=True,
        num_components=int(config.get("K", 8)),
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
        positive_eps=_float(config, "positive_epsilon", 1e-6),
        raw_delta_init=_float(config, "raw_delta_init", 0.1),
        use_decay=_bool(config.get("use_decay", True)),
        target_chunk_size=int(config.get("target_chunk_size", 16)),
        regulation_A_init=_float(config, "regulation_A_init", 1.0),
        theta_init=_float(config, "theta_init", 1.0),
    )
    if ode_type == "centered_signed_hill":
        field = CenteredSignedHillField(
            **common,
            alpha_init_std=config.get("alpha_init_std"),
        )
    else:
        field = ShiftedHillRhoField(
            **common,
            hill_n=_float(config, "hill_n", 2.0),
            rho_init_std=config.get("rho_init_std"),
        )
    field.model_family = family
    return field.to(device)


class InspectableUnifiedODEMLHybrid(UnifiedODEMLHybrid):
    """Established Hybrid wrapper plus exact branch-analysis hooks."""

    def __init__(self, *args, model_family: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_family = str(model_family)

    def ode_branch_weight(self, x: torch.Tensor, t) -> torch.Tensor:
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
        return self.ode_model.effective_coefficients(x.float(), t) * self.ode_branch_weight(
            x.float(), t
        )

    get_effective_coefficients = effective_coefficients

    def branch_outputs(self, x: torch.Tensor, t, y=None):
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
    family = _family(config)
    _validate_policy(config, family)
    field = build_ode_from_config(config, gene_list, device=device)
    hidden_num = config.get("cell_unet_hidden_num", [2000, 1000, 500, 500])
    if not isinstance(hidden_num, (list, tuple)) or len(hidden_num) != 4:
        raise ValueError("cell_unet_hidden_num must contain four positive widths")
    hidden_num = [int(width) for width in hidden_num]
    if any(width <= 0 for width in hidden_num):
        raise ValueError("cell_unet_hidden_num widths must be positive")
    ml_model = Cell_Unet(input_dim=len(gene_list), hidden_num=hidden_num).to(device)
    model = InspectableUnifiedODEMLHybrid(
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
    return model.to(device)


build_model = build_model_from_config

__all__ = [
    "MODEL_FAMILIES",
    "InspectableUnifiedODEMLHybrid",
    "build_ode_from_config",
    "build_model_from_config",
    "build_model",
]
