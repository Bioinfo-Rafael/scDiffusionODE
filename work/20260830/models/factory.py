"""Factory for the isolated 20260830 CellUnet + single-ODE wrapper."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260609_mathmlp import build_edge_mask

from .cellunet_ode_regularized_20260830 import CellUNetODERegularized20260830
from .ode_fields_20260830 import ODE_CLASS_BY_TYPE_20260830, ODE_TYPES_20260830


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_target_source_mask_20260830(
    config: Mapping,
    gene_list: Sequence[str],
    explicit_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    if not _bool(config.get("use_mask_reg", True)):
        return None
    if explicit_mask is not None:
        mask = torch.as_tensor(explicit_mask, dtype=torch.float32)
        if tuple(mask.shape) != (len(gene_list), len(gene_list)):
            raise ValueError("explicit mask must already be [target,source]")
        return mask.contiguous()
    path = config.get("edge_tsv_path")
    if not path:
        raise ValueError("edge_tsv_path is required when use_mask_reg=true")
    # The shared loader is legacy [source,target].  All local equations are
    # [target,source], so transpose exactly once at this boundary.
    return build_edge_mask(gene_list, path).T.contiguous()


def build_ode_from_config(
    config: Mapping,
    gene_list: Sequence[str],
    device="cpu",
    *,
    mask: Optional[torch.Tensor] = None,
):
    ode_type = str(config.get("ode_type", ""))
    if ode_type not in ODE_TYPES_20260830:
        raise ValueError(f"ode_type must be one of {ODE_TYPES_20260830}")
    if int(config.get("K", 1)) != 1 or str(config.get("gate_mode", "none")) != "none":
        raise ValueError("20260830 requires one ODE and gate_mode='none'")
    common = dict(
        d=len(gene_list),
        mask=build_target_source_mask_20260830(config, gene_list, mask),
        soft=_bool(config.get("SoftReg", True)),
        off_mask_lambda=float(config.get("off_mask_lambda", 5.0)),
        positive_epsilon=float(config.get("positive_epsilon", 1e-6)),
        raw_delta_init=float(config.get("raw_delta_init", 0.1)),
    )
    extra = {}
    if ode_type in {"centered_signed_hill", "shifted_hill_rho"}:
        extra.update(
            regulation_A_init=float(config.get("regulation_A_init", 1.0)),
            theta_init=float(config.get("theta_init", 1.0)),
            target_chunk_size=int(config.get("target_chunk_size", 16)),
        )
    if ode_type == "centered_signed_hill":
        extra["alpha_init_std"] = config.get("alpha_init_std")
    elif ode_type == "shifted_hill_rho":
        extra.update(hill_n=float(config.get("hill_n", 2.0)), rho_init_std=config.get("rho_init_std"))
    elif ode_type == "hill_after_linear":
        extra.update(
            hill_n=float(config.get("hill_n", 2.0)),
            hill_K_init=float(config.get("hill_K_init", 1.0)),
            hill_V_init=float(config.get("hill_V_init", 1.0)),
        )
    return ODE_CLASS_BY_TYPE_20260830[ode_type](**common, **extra).to(device)


def build_model_from_config(
    config: Mapping,
    gene_list: Sequence[str],
    timesteps: int,
    device="cpu",
    *,
    mask: Optional[torch.Tensor] = None,
):
    del timesteps  # CellUnet receives actual diffusion timesteps on forward.
    hidden = [int(value) for value in config.get("cell_unet_hidden_num", [2000, 1000, 500, 500])]
    if len(hidden) != 4 or any(value <= 0 for value in hidden):
        raise ValueError("cell_unet_hidden_num must contain four positive widths")
    ml_model = Cell_Unet(
        input_dim=len(gene_list),
        hidden_num=hidden,
        dropout=float(config.get("cell_unet_dropout", 0.1)),
    ).to(device)
    ode_model = build_ode_from_config(config, gene_list, device, mask=mask)
    return CellUNetODERegularized20260830(ml_model, ode_model).to(device)


__all__ = [
    "build_target_source_mask_20260830",
    "build_ode_from_config",
    "build_model_from_config",
]
