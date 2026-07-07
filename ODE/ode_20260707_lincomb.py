"""Configurable LinComb denoisers for the 2026-07-07 experiments."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260609_hybrid5x3 import UnifiedODEMLHybrid
from ODE.ode_20260609_mathmlp import LinCombField, build_edge_mask, _prep_t
from ODE.ode_20260609_scalemodel import build_scale_model


class ConfigurableLinCombField(LinCombField):
    """LinComb with raw/softmax coefficients and cached gate penalties."""

    def __init__(
        self,
        d,
        K=8,
        mask=None,
        soft=True,
        time_dim=64,
        hidden=256,
        dropout=0.0,
        use_decay=True,
        gate_mode="raw",
        gate_temperature=1.0,
        off_mask_lambda=0.0,
        sparse_lambda=0.0,
        entropy_lambda=0.0,
        gate_eps=1e-8,
    ):
        super().__init__(
            d=d, K=K, mask=mask, soft=soft, time_dim=time_dim,
            hidden=hidden, dropout=dropout, use_decay=use_decay,
        )
        self.gate_mode = str(gate_mode).lower()
        if self.gate_mode not in ("raw", "softmax"):
            raise ValueError("gate_mode must be 'raw' or 'softmax'")
        self.gate_temperature = float(gate_temperature)
        if self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be > 0")
        self.off_mask_lambda = float(off_mask_lambda)
        self.sparse_lambda = float(sparse_lambda)
        self.entropy_lambda = float(entropy_lambda)
        self.gate_eps = float(gate_eps)
        if min(self.off_mask_lambda, self.sparse_lambda, self.entropy_lambda) < 0:
            raise ValueError("regularization weights must be >= 0")
        if self.sparse_lambda > 0 and self.gate_mode != "raw":
            raise ValueError("sparse regularization requires gate_mode='raw'")
        if self.entropy_lambda > 0 and self.gate_mode != "softmax":
            raise ValueError("entropy regularization requires gate_mode='softmax'")
        # train_util checks this flag before calling off_mask_penalty().
        self.soft = bool(
            soft or self.sparse_lambda > 0 or self.entropy_lambda > 0
            or self.off_mask_lambda > 0
        )
        self._cached_gate_reg = None
        self._cached_gate_stats = None

    def _logits(self, x, t=None):
        batch = x.shape[0]
        t = _prep_t(t, batch, x.device, x.dtype)
        feat = torch.cat([x, self.time_emb(t)], dim=-1)
        return self.coeff_net(feat)

    def get_gate_values(self, x, t=None):
        logits = self._logits(x, t)
        probabilities = torch.softmax(logits / self.gate_temperature, dim=-1)
        coefficients = logits if self.gate_mode == "raw" else probabilities
        return {
            "logits": logits,
            "coefficients": coefficients,
            "probabilities": probabilities,
        }

    def _coeffs(self, x, t=None):
        return self.get_gate_values(x, t)["coefficients"]

    def forward(self, x, t=None):
        x = x.float()
        gate = self.get_gate_values(x, t)
        a = gate["coefficients"]
        fx = F.softplus(
            torch.einsum("kij,bj->bki", self.expert_W, x) + self.expert_b
        )
        out = torch.einsum("bk,bkd->bd", a, fx) - self._decay(x)

        self._cached_gate_reg = None
        if self.training:
            if self.sparse_lambda > 0:
                self._cached_gate_reg = a.abs().mean()
            elif self.entropy_lambda > 0:
                p = gate["probabilities"]
                self._cached_gate_reg = -(p * torch.log(p + self.gate_eps)).sum(-1).mean()
            with torch.no_grad():
                p = gate["probabilities"]
                self._cached_gate_stats = {
                    "mean_abs_a": float(a.abs().mean().item()),
                    "entropy": float((-(p * torch.log(p + self.gate_eps)).sum(-1)).mean().item()),
                }
        else:
            self._cached_gate_stats = None
        return out

    def off_mask_penalty(self, norm="l1"):
        total = self._zero()
        if self.off_mask_lambda > 0:
            total = total + self.off_mask_lambda * self._offmask_base(norm)
        gate_reg = getattr(self, "_cached_gate_reg", None)
        if gate_reg is not None:
            weight = self.sparse_lambda if self.sparse_lambda > 0 else self.entropy_lambda
            total = total + weight * gate_reg
        ratio = getattr(self, "_cached_ratio_reg", None)
        if ratio is not None and self.ratio_reg_weight > 0:
            total = total + self.ratio_reg_weight * ratio
        return total

    def has_aux_regularization(self):
        return bool(
            self.off_mask_lambda > 0 or self.sparse_lambda > 0
            or self.entropy_lambda > 0 or self.ratio_reg_weight > 0
        )

    def get_model_info(self):
        info = super().get_model_info()
        info.update({
            "class": type(self).__name__,
            "gate_mode": self.gate_mode,
            "gate_temperature": self.gate_temperature,
            "off_mask_lambda": self.off_mask_lambda,
            "sparse_lambda": self.sparse_lambda,
            "entropy_lambda": self.entropy_lambda,
        })
        return info


class LinCombOnlyDenoiser(nn.Module):
    """Diffusion-compatible wrapper preserving train_util's ode_model hook."""

    def __init__(self, field):
        super().__init__()
        self.ode_model = field

    def forward(self, x, t, y=None, **kwargs):
        return self.ode_model(x.float(), t)

    def get_model_info(self):
        return {
            "class": type(self).__name__,
            "denoiser_mode": "lincomb_only",
            "ode_model": self.ode_model.get_model_info(),
        }


def build_lincomb_field_0707(
    gene_list,
    edge_tsv_path,
    *,
    K=8,
    use_mask=True,
    soft=True,
    time_dim=64,
    hidden=256,
    dropout=0.0,
    use_decay=True,
    gate_mode="raw",
    gate_temperature=1.0,
    off_mask_lambda=0.0,
    sparse_lambda=0.0,
    entropy_lambda=0.0,
    ratio_reg_weight=0.0,
    ratio_reg_target=1.0,
    device="cpu",
):
    mask = build_edge_mask(gene_list, edge_tsv_path) if use_mask else None
    field = ConfigurableLinCombField(
        len(gene_list), K=K, mask=mask, soft=soft, time_dim=time_dim,
        hidden=hidden, dropout=dropout, use_decay=use_decay,
        gate_mode=gate_mode, gate_temperature=gate_temperature,
        off_mask_lambda=off_mask_lambda, sparse_lambda=sparse_lambda,
        entropy_lambda=entropy_lambda,
    )
    field.ratio_reg_weight = float(ratio_reg_weight)
    field.ratio_reg_target = float(ratio_reg_target)
    return field.to(device)


def build_denoiser_0707(
    gene_list,
    edge_tsv_path,
    timesteps,
    *,
    denoiser_mode="lincomb_only",
    hybrid_norm_mode="none",
    reverse_coef=False,
    regime_gate_mode="none",
    regime_gate_type="sigmoid",
    t_s=None,
    gate_tau=20.0,
    K=8,
    use_mask=True,
    soft=True,
    time_dim=64,
    field_hidden=256,
    field_dropout=0.0,
    use_decay=True,
    gate_mode="raw",
    gate_temperature=1.0,
    off_mask_lambda=0.0,
    sparse_lambda=0.0,
    entropy_lambda=0.0,
    ratio_reg_weight=0.0,
    ratio_reg_target=1.0,
    hybrid_scale_init=1.0,
    hybrid_scale_eps=1e-8,
    scale_model_type="none",
    scale_input_source="ml_emb",
    ode_input_source="none",
    scale_hidden=128,
    scale_eps=1e-8,
    device="cpu",
):
    field = build_lincomb_field_0707(
        gene_list, edge_tsv_path, K=K, use_mask=use_mask, soft=soft,
        time_dim=time_dim, hidden=field_hidden, dropout=field_dropout,
        use_decay=use_decay, gate_mode=gate_mode,
        gate_temperature=gate_temperature, off_mask_lambda=off_mask_lambda,
        sparse_lambda=sparse_lambda, entropy_lambda=entropy_lambda,
        ratio_reg_weight=ratio_reg_weight, ratio_reg_target=ratio_reg_target,
        device=device,
    )
    denoiser_mode = str(denoiser_mode).lower()
    if denoiser_mode == "lincomb_only":
        if str(regime_gate_mode).lower() != "none" or reverse_coef:
            raise ValueError("reverse/regime blend options require denoiser_mode='hybrid'")
        return LinCombOnlyDenoiser(field).to(device)
    if denoiser_mode != "hybrid":
        raise ValueError("denoiser_mode must be 'hybrid' or 'lincomb_only'")

    ml_model = Cell_Unet(input_dim=len(gene_list)).to(device)
    scale_model = None
    if str(hybrid_norm_mode).lower() == "scale_model":
        if str(scale_model_type).lower() == "none":
            raise ValueError("scale_model mode requires scale_model_type != 'none'")
        feature_dim = (
            int(ml_model.hidden_num[-1])
            if str(scale_input_source).lower() == "ml_emb"
            else len(gene_list)
        )
        scale_model = build_scale_model(
            scale_model_type, feature_dim=feature_dim,
            scale_hidden=scale_hidden, scale_eps=scale_eps,
        ).to(device)
    return UnifiedODEMLHybrid(
        ode_model=field,
        ml_model=ml_model,
        timesteps=timesteps,
        hybrid_norm_mode=hybrid_norm_mode,
        hybrid_scale_init=hybrid_scale_init,
        hybrid_scale_eps=hybrid_scale_eps,
        scale_model=scale_model,
        scale_input_source=scale_input_source,
        ode_input_source=ode_input_source,
        scale_eps=scale_eps,
        reverse_coef=reverse_coef,
        regime_gate_mode=regime_gate_mode,
        regime_gate_type=regime_gate_type,
        t_s=t_s,
        gate_tau=gate_tau,
    ).to(device)

