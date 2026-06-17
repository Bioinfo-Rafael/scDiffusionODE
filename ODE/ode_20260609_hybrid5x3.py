"""
ode_20260609_hybrid5x3.py
=========================

5 ODE 枝 × 3 hybrid mode を 1 経路で扱う統合 hybrid。

既存ファイルは無編集。以下を import 再利用する:
  - GeneODE                         : ODE/ode_20260421_regODEMLratio.py
  - build_math_field / MathMLPField : ODE/ode_20260609_mathmlp.py
  - Cell_Unet                       : guided_diffusion/cell_model.py

統合の肝:
  `UnifiedODEMLHybrid.forward` は ODE 枝に **t を渡す**（ode_model(x, t)）。
  - GeneODE.forward(x, t=None) は t を無視 → そのまま動く
  - 4 math fields は t を使う → 時刻依存が保たれる
  これで GeneODE 専用だった ODE_ML_HybridNorm の「field を差すと t=0 で時刻依存が消える」問題を回避。

3 hybrid mode（ODE_ML_HybridNorm と同一仕様、scale は共有 scalar 1 個）:
  - ratio_reg            : out=r·ode+(1-r)·ml、log-norm-ratio penalty を cache
  - normed_learned_scale : ode/ml を L2 正規化 → exp(log_scale) 倍 → blend。ratio penalty なし
  - none                 : raw blend、penalty なし

train_util.py は無変更。`self.ode_model = branch` が `.soft`/`off_mask_penalty`/`_cached_ratio_reg`
を持つので既存 hook がそのまま発火する（plain Cell_Unet には ode_model が無いので不発火）。
"""

import math

import torch
import torch.nn as nn

import inspect

from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260421_regODEMLratio import GeneODE
from ODE.ode_20260609_mathmlp import build_math_field
from ODE.ode_20260609_scalemodel import build_scale_model

# ratio_reg / none / normed_learned_scale(deprecated) は既存挙動を完全維持。
# scale_model は新規追加（ode_unit/ml_unit の方向 × scale_model 予測の scalar scale）。
_VALID_MODES = ("ratio_reg", "normed_learned_scale", "none", "scale_model")
_FIELD_BRANCHES = ("lowrank", "lincomb", "matsum", "lora")
_ODE_BRANCHES = ("geneode",) + _FIELD_BRANCHES
_ALL_BRANCHES = _ODE_BRANCHES + ("plain",)
_VALID_SCALE_INPUT = ("ml_emb", "x")
_VALID_ODE_INPUT = ("none", "x_ml_emb")


def _ode_supports_ml_emb(ode_model) -> bool:
    """ode_model.forward が ml_emb キーワード（または **kwargs）を受け取れるか判定。"""
    try:
        params = inspect.signature(ode_model.forward).parameters
    except (TypeError, ValueError):
        return False
    if "ml_emb" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


# ============================================================
# 統合 hybrid
# ============================================================

class UnifiedODEMLHybrid(nn.Module):
    """ODE 枝（GeneODE or math field）と Cell_Unet を 3 mode で統合する。

    ODE_ML_HybridNorm との差は forward が `ode_model(x, t)` と t を渡す点のみ。
    """

    def __init__(
        self,
        ode_model: nn.Module,
        ml_model: nn.Module,
        timesteps: int,
        hybrid_norm_mode: str = "ratio_reg",
        hybrid_scale_init: float = 1.0,
        hybrid_scale_eps: float = 1e-8,
        scale_model: nn.Module = None,
        scale_input_source: str = "ml_emb",
        ode_input_source: str = "none",
        scale_eps: float = 1e-8,
    ):
        super().__init__()
        assert timesteps > 1
        mode = str(hybrid_norm_mode).lower()
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid hybrid_norm_mode='{hybrid_norm_mode}'. choose from {_VALID_MODES}"
            )
        self.ode_model = ode_model          # ← train_util hook が getattr(model, "ode_model") で参照
        self.ml_model = ml_model
        self.T = int(timesteps)
        self.hybrid_norm_mode = mode
        self.scale_eps = float(hybrid_scale_eps)   # normed_learned_scale 用（既存・不変）

        if mode == "normed_learned_scale":
            init = max(float(hybrid_scale_init), self.scale_eps)
            # 共有 scalar 1 個（ユーザ確定）。exp(log_scale) で常に正。init=1→log_scale=0
            self.log_scale = nn.Parameter(torch.tensor(math.log(init), dtype=torch.float32))

        # ---- scale_model mode（新規）用の設定 ----
        # scale_model は build_denoiser 側で生成して渡す（Hybrid 内では class 生成しない）。
        self.scale_model = scale_model            # nn.Module assignment で submodule 登録 → state_dict に入る
        self.scale_input_source = str(scale_input_source).lower()
        self.ode_input_source = str(ode_input_source).lower()
        self.scale_model_norm_eps = float(scale_eps)   # scale_model mode の方向正規化 eps
        if mode == "scale_model":
            if self.scale_model is None:
                raise ValueError(
                    "hybrid_norm_mode='scale_model' requires a scale_model instance "
                    "(build_denoiser に scale_model_type='simple' を指定してください)。"
                )
            if self.scale_input_source not in _VALID_SCALE_INPUT:
                raise ValueError(
                    f"Invalid scale_input_source='{scale_input_source}'. choose from {_VALID_SCALE_INPUT}"
                )
            if self.ode_input_source not in _VALID_ODE_INPUT:
                raise ValueError(
                    f"Invalid ode_input_source='{ode_input_source}'. choose from {_VALID_ODE_INPUT}"
                )

        self._cached_hybrid_stats = None

    def _scheduler(self, t, device, dtype):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)
        t = t.to(dtype=dtype, device=device)
        return 1.0 - t / (self.T - 1)

    def _norm_ratio_penalty(self, ode_out, ml_out):
        eps = getattr(self.ode_model, "ratio_reg_eps", 1e-8)
        ode_norm = ode_out.norm(p=2, dim=-1).clamp_min(eps)
        ml_norm = ml_out.norm(p=2, dim=-1).clamp_min(eps)
        log_ratio = torch.log(ode_norm) - torch.log(ml_norm)
        target = getattr(self.ode_model, "ratio_reg_target", 1.0)
        target_log = math.log(max(target, eps))
        return ((log_ratio - target_log) ** 2).mean()

    def forward(self, x, t, y=None):
        x = x.float()
        # scale_model mode だけ別経路（既存 ratio_reg/none/normed_learned_scale は下の経路を不変で通す）。
        if self.hybrid_norm_mode == "scale_model":
            return self._forward_scale_model(x, t, y)

        ode_out = self.ode_model(x, t)          # ★ t を渡す（全 5 枝対応）
        ml_out = self.ml_model(x, t, y)
        assert ode_out.shape == ml_out.shape, (
            f"shape mismatch: ode={tuple(ode_out.shape)} ml={tuple(ml_out.shape)}"
        )

        r = self._scheduler(t, device=x.device, dtype=x.dtype)
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)

        mode = self.hybrid_norm_mode
        ratio_penalty = None
        if mode == "normed_learned_scale":
            eps = self.scale_eps
            ode_unit = ode_out / (ode_out.norm(p=2, dim=-1, keepdim=True) + eps)
            ml_unit = ml_out / (ml_out.norm(p=2, dim=-1, keepdim=True) + eps)
            out = torch.exp(self.log_scale) * (r * ode_unit + (1.0 - r) * ml_unit)
            self.ode_model._cached_ratio_reg = None
        else:
            out = r * ode_out + (1.0 - r) * ml_out
            if mode == "ratio_reg" and self.training:
                ratio_penalty = self._norm_ratio_penalty(ode_out, ml_out)
                self.ode_model._cached_ratio_reg = ratio_penalty
            else:
                self.ode_model._cached_ratio_reg = None

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

    # ---- scale_model mode（新規）----
    def compute_ml_out_and_features(self, x, t, y=None):
        """ml_out と中間特徴 dict を返す（forward_with_features があればそれを使う）。"""
        x = x.float()
        if hasattr(self.ml_model, "forward_with_features"):
            return self.ml_model.forward_with_features(x, t, y)
        return self.ml_model(x, t, y), {}

    def compute_ode_out(self, x, t, ml_features=None):
        """ode_out を返す。ode_input_source='x_ml_emb' かつ ode が ml_emb 対応のときだけ ml_emb を渡す。"""
        x = x.float()
        if (self.ode_input_source == "x_ml_emb" and ml_features is not None
                and _ode_supports_ml_emb(self.ode_model)):
            return self.ode_model(x, t, ml_emb=ml_features.get("ml_emb"))
        return self.ode_model(x, t)

    def _forward_scale_model(self, x, t, y=None):
        """ode_unit/ml_unit の方向ブレンドに、scale_model が予測する scalar scale を掛ける。

        Cell_Unet は forward_with_features で 1 回だけ通し（二重 forward 禁止）、
        ml_out と ml_emb を同時に取得する。ratio penalty は使わない（_cached_ratio_reg=None）。
        """
        if self.scale_model is None:
            raise RuntimeError(
                "hybrid_norm_mode='scale_model' but scale_model is None. "
                "build_denoiser(scale_model_type='simple', ...) で生成してください。"
            )
        if not hasattr(self.ml_model, "forward_with_features"):
            raise RuntimeError(
                "scale_model mode requires ml_model.forward_with_features(); "
                f"{type(self.ml_model).__name__} にありません。"
            )

        ml_out, ml_features = self.ml_model.forward_with_features(x, t, y)
        ml_emb = ml_features["ml_emb"]

        if self.ode_input_source == "none":
            ode_out = self.ode_model(x, t)
        elif self.ode_input_source == "x_ml_emb":
            if not _ode_supports_ml_emb(self.ode_model):
                raise RuntimeError(
                    "ode_input_source='x_ml_emb' but ode_model "
                    f"({type(self.ode_model).__name__}) は ml_emb を受け取れません "
                    "（forward に ml_emb 引数がない）。ode_input_source='none' を使うか、"
                    "ml_emb 対応の ODE branch を実装してください。"
                )
            ode_out = self.ode_model(x, t, ml_emb=ml_emb)
        else:
            raise ValueError(
                f"Invalid ode_input_source='{self.ode_input_source}'. choose from {_VALID_ODE_INPUT}"
            )

        assert ode_out.shape == ml_out.shape, (
            f"shape mismatch: ode={tuple(ode_out.shape)} ml={tuple(ml_out.shape)}"
        )

        eps = self.scale_model_norm_eps
        ode_unit = ode_out / (ode_out.norm(p=2, dim=-1, keepdim=True) + eps)
        ml_unit = ml_out / (ml_out.norm(p=2, dim=-1, keepdim=True) + eps)

        r = self._scheduler(t, device=x.device, dtype=x.dtype)
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)
        base = r * ode_unit + (1.0 - r) * ml_unit          # (B, d)

        if self.scale_input_source == "ml_emb":
            scale_in = ml_emb
        elif self.scale_input_source == "x":
            scale_in = x
        else:
            raise ValueError(
                f"Invalid scale_input_source='{self.scale_input_source}'. choose from {_VALID_SCALE_INPUT}"
            )
        scale = self.scale_model(scale_in, t)              # (B, 1)
        out = scale * base                                  # broadcast (B,1)*(B,d) → (B,d)

        self.ode_model._cached_ratio_reg = None             # scale_model mode は ratio penalty なし
        if self.training:
            with torch.no_grad():
                self._cached_hybrid_stats = {
                    "hybrid_norm_mode": "scale_model",
                    "ratio_penalty": None,
                    "ode_norm_mean": float(ode_out.norm(p=2, dim=-1).mean().item()),
                    "ml_norm_mean": float(ml_out.norm(p=2, dim=-1).mean().item()),
                    "scale_mean": float(scale.mean().item()),
                    "scale_std": float(scale.std().item()),
                }
        else:
            self._cached_hybrid_stats = None
        return out

    def get_model_info(self) -> dict:
        return {
            "class": "UnifiedODEMLHybrid",
            "hybrid_norm_mode": self.hybrid_norm_mode,
            "scale_eps": self.scale_eps,
            "has_learnable_scale": hasattr(self, "log_scale"),
            "has_scale_model": self.scale_model is not None,
            "scale_input_source": self.scale_input_source,
            "ode_input_source": self.ode_input_source,
            "timesteps": self.T,
        }


# ============================================================
# factory
# ============================================================

def build_ode_branch(
    ode_branch: str,
    gene_list,
    edge_tsv_path,
    rank: int = 16,
    K: int = 8,
    soft: bool = True,
    time_dim: int = 64,
    field_hidden: int = 256,
    field_dropout: float = 0.0,
    lowrank_penalty_subsample: int = 8,
    use_decay: bool = True,
    ratio_reg_weight: float = 1.0,
    ratio_reg_target: float = 1.0,
    device="cpu",
) -> nn.Module:
    """ODE 枝（geneode / lowrank / lincomb / matsum / lora）を構築する。

    いずれも `.soft` / `off_mask_penalty(norm)` / `_cached_ratio_reg` /
    `ratio_reg_weight/target/eps` を持つ duck-type 互換。
    """
    ode_branch = str(ode_branch).lower()
    if ode_branch == "geneode":
        branch = GeneODE(
            gene_list=gene_list,
            edge_tsv_path=edge_tsv_path,
            soft=soft,
            device=device,
        )
    elif ode_branch in _FIELD_BRANCHES:
        branch = build_math_field(
            model_type=ode_branch,
            gene_list=gene_list,
            edge_tsv_path=edge_tsv_path,
            rank=rank,
            K=K,
            use_mask=soft,            # soft=True で off-mask 罰則を有効化
            soft=soft,
            time_dim=time_dim,
            hidden=field_hidden,
            dropout=field_dropout,
            lowrank_penalty_subsample=lowrank_penalty_subsample,
            use_decay=use_decay,
            device=device,
        )
    else:
        raise ValueError(f"Unknown ode_branch='{ode_branch}'. choose from {_ODE_BRANCHES}")

    branch.ratio_reg_weight = ratio_reg_weight
    branch.ratio_reg_target = ratio_reg_target
    return branch


def build_denoiser(
    ode_branch: str,
    gene_list,
    edge_tsv_path,
    timesteps: int,
    hybrid_norm_mode: str = "ratio_reg",
    *,
    rank: int = 16,
    K: int = 8,
    soft: bool = True,
    ode_reg_lambda: float = 0.0,
    time_dim: int = 64,
    field_hidden: int = 256,
    field_dropout: float = 0.0,
    lowrank_penalty_subsample: int = 8,
    use_decay: bool = True,
    ratio_reg_weight: float = 1.0,
    ratio_reg_target: float = 1.0,
    hybrid_scale_init: float = 1.0,
    hybrid_scale_eps: float = 1e-8,
    scale_model_type: str = "none",
    scale_input_source: str = "ml_emb",
    ode_input_source: str = "none",
    scale_hidden: int = 128,
    scale_eps: float = 1e-8,
    device="cpu",
) -> nn.Module:
    """TrainLoop に渡す denoising model を構築する。

    ode_branch == "plain" → Cell_Unet 単体（ODE/hybrid なし、off-mask hook 不発火）。
    それ以外               → UnifiedODEMLHybrid(ODE 枝, Cell_Unet, mode)。

    scale_model_*（既定で既存挙動不変）:
      - hybrid_norm_mode が ratio_reg / none / normed_learned_scale → scale_model は作らない。
      - hybrid_norm_mode == "scale_model" → scale_model_type=="simple" のとき SimpleScalarScaleModel を作る。
        input dim は scale_input_source="ml_emb" なら ml_model.hidden_num[-1]、"x" なら遺伝子次元 d。
    """
    ode_branch = str(ode_branch).lower()
    d = len(gene_list)
    ml_model = Cell_Unet(input_dim=d).to(device)

    if ode_branch == "plain":
        return ml_model

    if ode_branch not in _ODE_BRANCHES:
        raise ValueError(f"Unknown ode_branch='{ode_branch}'. choose from {_ALL_BRANCHES}")

    branch = build_ode_branch(
        ode_branch, gene_list, edge_tsv_path,
        rank=rank, K=K, soft=soft, time_dim=time_dim, field_hidden=field_hidden,
        field_dropout=field_dropout, lowrank_penalty_subsample=lowrank_penalty_subsample,
        use_decay=use_decay, ratio_reg_weight=ratio_reg_weight, ratio_reg_target=ratio_reg_target,
        device=device,
    )
    # field の subsample W キャッシュは罰則が有効なときだけ作る
    if hasattr(branch, "enable_offmask_cache"):
        branch.enable_offmask_cache = bool(soft and ode_reg_lambda > 0)

    # scale_model は scale_model mode のときだけ生成（他 mode は None＝既存挙動）。
    scale_model = None
    if str(hybrid_norm_mode).lower() == "scale_model":
        smt = str(scale_model_type).lower()
        if smt == "none":
            raise ValueError(
                "hybrid_norm_mode='scale_model' には scale_model_type='simple' が必要です（'none' が指定されました）。"
            )
        sis = str(scale_input_source).lower()
        if sis == "ml_emb":
            feat_dim = int(ml_model.hidden_num[-1])   # Cell_Unet encoder 最深部の次元
        elif sis == "x":
            feat_dim = d
        else:
            raise ValueError(f"Invalid scale_input_source='{scale_input_source}'. choose from {_VALID_SCALE_INPUT}")
        scale_model = build_scale_model(
            smt, feature_dim=feat_dim, scale_hidden=scale_hidden, scale_eps=scale_eps).to(device)

    hybrid = UnifiedODEMLHybrid(
        ode_model=branch,
        ml_model=ml_model,
        timesteps=timesteps,
        hybrid_norm_mode=hybrid_norm_mode,
        hybrid_scale_init=hybrid_scale_init,
        hybrid_scale_eps=hybrid_scale_eps,
        scale_model=scale_model,
        scale_input_source=scale_input_source,
        ode_input_source=ode_input_source,
        scale_eps=scale_eps,
    ).to(device)
    return hybrid
