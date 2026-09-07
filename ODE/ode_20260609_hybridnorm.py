"""
ode_20260609_hybridnorm.py
==========================

ODE_ML_Hybrid の ODE/ML 出力統合を 3 モードから選べる新クラス `ODE_ML_HybridNorm`。

既存 `ODE/ode_20260421_regODEMLratio.py` は **無変更**で、その `GeneODE` を import 再利用する。
`guided_diffusion/train_util.py` も無変更（既存の off_mask_penalty hook をそのまま使う）。

3 モード（排他）:
  - ratio_reg            : out = r·ode + (1−r)·ml、log-norm-ratio penalty を cache（現行互換）
  - normed_learned_scale : ODE/ML を L2 正規化 → 共有 scalar scale 倍 → blend。ratio penalty なし
  - none                 : raw blend、penalty なし（baseline）

`reverse_coef=False` は従来の係数方向を維持し、True の場合だけ ODE/ML weight を交換する。

最小差分の鍵:
  既存 `GeneODE.off_mask_penalty()` は `_cached_ratio_reg is None` なら base のみを返す。
  本クラスが forward で mode 別に `ode_model._cached_ratio_reg` を出し分けるだけで、
  train_util も GeneODE も無変更で 3 モードが成立する。

  - ratio_reg            : training 時に ratio penalty を cache → off_mask_penalty は base+ratio
  - normed_learned_scale : 常に None を cache         → off_mask_penalty は base のみ
  - none                 : 常に None を cache         → off_mask_penalty は base のみ
"""

import math

import torch
import torch.nn as nn

# 既存の GeneODE を再利用（このファイルは ode_20260421 を一切変更しない）
from ODE.ode_20260421_regODEMLratio import GeneODE  # noqa: F401  (再エクスポート用にも公開)

_VALID_MODES = ("ratio_reg", "normed_learned_scale", "none")


class ODE_ML_HybridNorm(nn.Module):
    """ODE と ML の出力統合を 3 モードから選べる hybrid。

    既存 `ODE_ML_Hybrid` と互換の構築シグネチャ（追加引数は既定値つき）。
    `self.ode_model = ode_model` のため train_util の正則化 hook がゼロ変更で発火する。
    """

    def __init__(
        self,
        ode_model: nn.Module,
        ml_model: nn.Module,
        timesteps: int,
        hybrid_norm_mode: str = "ratio_reg",
        hybrid_scale_init: float = 1.0,
        hybrid_scale_eps: float = 1e-8,
        reverse_coef: bool = False,
    ):
        super().__init__()
        assert timesteps > 1
        mode = str(hybrid_norm_mode).lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid hybrid_norm_mode='{hybrid_norm_mode}'. choose from {_VALID_MODES}"
            )

        self.ode_model = ode_model          # ← hook が getattr(model, "ode_model") で参照
        self.ml_model = ml_model
        self.T = int(timesteps)

        self.hybrid_norm_mode = mode
        self.scale_eps = float(hybrid_scale_eps)
        self.reverse_coef = bool(reverse_coef)

        # 学習可能 scale は normed_learned_scale のときだけ生成（他 mode は param 集合を現行と同一に保つ）
        if mode == "normed_learned_scale":
            init = max(float(hybrid_scale_init), self.scale_eps)
            # 共有 scalar 1 個。exp(log_scale) で常に正・滑らか・init=1→log_scale=0
            self.log_scale = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))
        # それ以外では log_scale 属性を持たない（checkpoint 後方互換）

        # debug 用キャッシュ（副作用なし。train_util は触らない）
        self._cached_hybrid_stats = None

    # ------------------------------------------------------------------
    def _scheduler(self, t, device, dtype):
        # ODE_ML_Hybrid と同一実装（r = 1 - t/(T-1)）
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)
        t = t.to(dtype=dtype, device=device)
        return 1.0 - t / (self.T - 1)

    def _blend_terms(self, ode_term, ml_term, r):
        if self.reverse_coef:
            return (1.0 - r) * ode_term + r * ml_term
        return r * ode_term + (1.0 - r) * ml_term

    def _norm_ratio_penalty(self, ode_out, ml_out):
        # ODE_ML_Hybrid と同一実装（GeneODE の ratio_reg_* 属性を参照）
        eps = getattr(self.ode_model, "ratio_reg_eps", 1e-8)
        ode_norm = ode_out.norm(p=2, dim=-1).clamp_min(eps)
        ml_norm = ml_out.norm(p=2, dim=-1).clamp_min(eps)
        log_ratio = torch.log(ode_norm) - torch.log(ml_norm)
        target = getattr(self.ode_model, "ratio_reg_target", 1.0)
        target_log = math.log(max(target, eps))
        return ((log_ratio - target_log) ** 2).mean()

    # ------------------------------------------------------------------
    def forward(self, x, t, y=None):
        x = x.float()

        ode_out = self.ode_model(x)
        ml_out = self.ml_model(x, t, y)

        r = self._scheduler(t, device=x.device, dtype=x.dtype)
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)

        mode = self.hybrid_norm_mode
        ratio_penalty = None

        if mode == "normed_learned_scale":
            eps = self.scale_eps
            ode_unit = ode_out / (ode_out.norm(p=2, dim=-1, keepdim=True) + eps)
            ml_unit = ml_out / (ml_out.norm(p=2, dim=-1, keepdim=True) + eps)
            scale = torch.exp(self.log_scale)
            out = scale * self._blend_terms(ode_unit, ml_unit, r)
            # ratio penalty は使わない → off_mask_penalty は base のみ
            self.ode_model._cached_ratio_reg = None
        else:
            # ratio_reg / none はともに raw blend
            out = self._blend_terms(ode_out, ml_out, r)
            if mode == "ratio_reg" and self.training:
                ratio_penalty = self._norm_ratio_penalty(ode_out, ml_out)
                self.ode_model._cached_ratio_reg = ratio_penalty
            else:
                # none、または eval 時の ratio_reg
                self.ode_model._cached_ratio_reg = None

        # --- debug stats（train_util 非干渉。後から参照可）---
        if self.training:
            with torch.no_grad():
                self._cached_hybrid_stats = {
                    "hybrid_norm_mode": mode,
                    "ratio_penalty": (float(ratio_penalty.item()) if ratio_penalty is not None else None),
                    "ode_norm_mean": float(ode_out.norm(p=2, dim=-1).mean().item()),
                    "ml_norm_mean": float(ml_out.norm(p=2, dim=-1).mean().item()),
                    "scale": (float(torch.exp(self.log_scale).item()) if hasattr(self, "log_scale") else None),
                }
        else:
            self._cached_hybrid_stats = None

        return out

    # ------------------------------------------------------------------
    def get_model_info(self) -> dict:
        return {
            "class": "ODE_ML_HybridNorm",
            "hybrid_norm_mode": self.hybrid_norm_mode,
            "scale_eps": self.scale_eps,
            "has_learnable_scale": hasattr(self, "log_scale"),
            "reverse_coef": self.reverse_coef,
            "timesteps": self.T,
        }
