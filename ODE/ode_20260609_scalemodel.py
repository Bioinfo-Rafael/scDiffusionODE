"""
ode_20260609_scalemodel.py
==========================

`hybrid_norm_mode="scale_model"`（ODE/ML 出力を L2 正規化して方向を作り、その方向ベクトルに
**scalar scale** を掛ける）用の scale 予測モデル群。

- SimpleScalarScaleModel : z=(B, feature_dim) → scale=(B, 1)（正値）。
  方向は ode_unit/ml_unit が決め、scale は「大きさだけ」を調整する役割分担。
- build_scale_model       : type 文字列から instance を生成する factory。

設計方針:
  - scale は gene-wise vector ではなく **scalar (B,1)**。
  - scale は必ず正値: softplus(raw) + eps。
  - 初期状態で scale ≈ 1 になるよう、最終 bias を softplus^{-1}(1.0) 付近に初期化し、
    最終 weight を小さくして初期出力を入力に依存させない。
  - t は将来拡張（time embedding 等）のため forward(z, t) で受け取るが、simple model では未使用。
  - matplotlib 等の描画依存は持たない（model class は tensor を返すだけ）。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleScalarScaleModel(nn.Module):
    """z=(B, feature_dim) から scalar scale=(B,1)（正値）を予測する簡易 MLP。

    内部: Linear(feature_dim, scale_hidden) → SiLU → Linear(scale_hidden, 1) → softplus + eps。
    t は引数として受け取るが simple model では使わない（将来拡張用）。
    """

    def __init__(self, feature_dim: int, scale_hidden: int = 128, scale_eps: float = 1e-8):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.scale_hidden = int(scale_hidden)
        self.scale_eps = float(scale_eps)
        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, self.scale_hidden),
            nn.SiLU(),
            nn.Linear(self.scale_hidden, 1),
        )
        # 初期 scale ≈ 1: softplus(b)+eps ≈ 1 → b = softplus^{-1}(1.0) = log(e^1 - 1)
        inv_softplus_1 = math.log(math.expm1(1.0))   # ≈ 0.5413
        with torch.no_grad():
            self.net[-1].bias.fill_(inv_softplus_1)
            # 最終 weight を小さくして初期出力を入力 z に依存させない（どの z でも scale≈1）
            self.net[-1].weight.mul_(0.01)

    def forward(self, z: torch.Tensor, t=None) -> torch.Tensor:
        # t は将来拡張のため受け取るが simple model では未使用
        del t
        raw = self.net(z.float())                 # (B, 1)
        return F.softplus(raw) + self.scale_eps   # (B, 1) 正値

    def get_model_info(self) -> dict:
        return {
            "type": "simple_scalar_scale",
            "feature_dim": self.feature_dim,
            "scale_hidden": self.scale_hidden,
            "scale_eps": self.scale_eps,
            "output": "(B,1) scalar, positive",
        }


_SCALE_REGISTRY = {"simple": SimpleScalarScaleModel}


def build_scale_model(scale_model_type: str, feature_dim: int,
                      scale_hidden: int = 128, scale_eps: float = 1e-8) -> nn.Module:
    """scale_model_type からスケールモデルを構築する。

    現状 type は "simple" のみ（output dim は常に 1 の scalar）。
    """
    t = str(scale_model_type).lower()
    if t not in _SCALE_REGISTRY:
        raise ValueError(
            f"Unknown scale_model_type='{scale_model_type}'. choose from {list(_SCALE_REGISTRY)}"
        )
    return _SCALE_REGISTRY[t](feature_dim, scale_hidden=scale_hidden, scale_eps=scale_eps)
