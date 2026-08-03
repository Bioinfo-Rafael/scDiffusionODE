"""Adapters around the established softplus and 20260803 Hill/exp fields."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Mapping

import torch

from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260609_hybrid5x3 import UnifiedODEMLHybrid
from ODE.ode_20260707_lincomb import build_lincomb_field_0707, LinCombOnlyDenoiser
from ODE.ode_20260609_mathmlp import build_edge_mask


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SOURCE = REPO_ROOT / "work/20260803_ODE_hill_exp/models/ode_fields.py"
spec = importlib.util.spec_from_file_location("ode_fields_20260803_raw_reuse", SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(SOURCE)
fields = importlib.util.module_from_spec(spec); spec.loader.exec_module(fields)


FAMILIES = ("ode_only_lincomb_softmax", "standard_hybrid_lincomb_softmax")
ODE_TYPES = ("softplus", "hill_after_softplus", "exp")


def _bool(value):
    return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes")


def _field_20260803(config: Mapping, genes):
    mask = build_edge_mask(genes, config["edge_tsv_path"]).T.contiguous()
    common = dict(
        d=len(genes), is_lincomb=True, num_components=8, mask=mask,
        soft=_bool(config.get("SoftReg", True)), time_dim=int(config["time_dim"]),
        hidden=int(config["field_hidden"]), dropout=float(config["field_dropout"]),
        gate_mode="softmax", gate_temperature=float(config["gate_temperature"]),
        off_mask_lambda=float(config["off_mask_lambda"]), sparse_lambda=0.0,
        entropy_lambda=0.0, ratio_reg_weight=0.0, ratio_reg_target=1.0,
        positive_eps=float(config["positive_epsilon"]),
        raw_delta_init=float(config["raw_delta_init"]), use_decay=True,
    )
    if config["ode_type"] == "hill_after_softplus":
        return fields.HillAfterLinearField(
            **common, hill_n=float(config["hill_n"]),
            hill_K_init=float(config["hill_K_init"]), hill_V_init=float(config["hill_V_init"]),
        )
    return fields.ExpField(
        **common, exp_preactivation_min=float(config["exp_preactivation_min"]),
        exp_preactivation_max=float(config["exp_preactivation_max"]),
    )


def build_model_from_config(config: Mapping, gene_list, timesteps: int, device="cpu"):
    family, ode = str(config["model_family"]), str(config["ode_type"])
    if family not in FAMILIES or ode not in ODE_TYPES:
        raise ValueError(f"unsupported family/ODE: {family}/{ode}")
    hybrid = family == "standard_hybrid_lincomb_softmax"
    if ode == "softplus":
        field = build_lincomb_field_0707(
            gene_list, config["edge_tsv_path"],
            K=8, use_mask=True, soft=True, time_dim=int(config["time_dim"]),
            hidden=int(config["field_hidden"]), dropout=float(config["field_dropout"]),
            use_decay=True, gate_mode="softmax", gate_temperature=float(config["gate_temperature"]),
            off_mask_lambda=float(config["off_mask_lambda"]), device=device,
        )
    else:
        field = _field_20260803(config, gene_list).to(device)
    field.model_family = family
    if not hybrid:
        return LinCombOnlyDenoiser(field).to(device)
    widths = [int(value) for value in config.get("cell_unet_hidden_num", [2000,1000,500,500])]
    ml = Cell_Unet(input_dim=len(gene_list), hidden_num=widths).to(device)
    return UnifiedODEMLHybrid(
        ode_model=field, ml_model=ml, timesteps=int(timesteps), hybrid_norm_mode="none",
        reverse_coef=False, regime_gate_mode="none", regime_gate_type="sigmoid",
        t_s=None, gate_tau=float(config.get("gate_tau", 80)),
    ).to(device)
