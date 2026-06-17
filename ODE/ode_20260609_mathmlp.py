"""
ode_20260609_mathmlp.py
=======================

数理構造 + MLP の hybrid denoising field 群（4 系統）。

各 field は V(x,t) = softplus(W(x,t) x + b) の形を取り、W(x,t) を MLP で
パラメタ化する。既存の `GeneODE` / `ODE_ML_Hybrid` と **duck-type 互換** にして
あるため、`guided_diffusion/train_util.py` の正則化 hook
(`model.ode_model.soft` と `off_mask_penalty(norm)`) をゼロ変更で発火できる。

4 系統:
  - LowRankField : W(x,t) = U(x,t) V_low(x,t)^T          （U,V を MLP で生成）
  - LinCombField : V = Σ_k a_k(x,t) softplus(W_k x + b_k) （線型結合・W_k 静的）
  - MatSumField  : W(x,t) = Σ_k a_k(x,t) A_k             （行列和・A_k 静的）
  - LoRAField    : W(x,t) = W_0 + Σ_k a_k(x,t) U_k V_k^T （LoRA・全て静的 + a_k）

forward は全モデルで **batched W (B×d×d) を materialize しない**（associativity を利用）。
正則化 (off-mask 罰則) は:
  - LinComb / MatSum / LoRA : 静的パラメタ off-mask（x,t 非依存・安価）
  - LowRank                 : forward 内で batch を subsample して W を限定構築（≤数十MB）

共通 interface（GeneODE 互換）:
  forward(x, t) -> (B, d)
  compute_W(x, t) -> (B, d, d)        （可視化用、少数サンプルのみで呼ぶこと）
  off_mask_penalty(norm="l1") -> scalar
  get_model_info() -> dict
  属性: soft(bool), mask(buffer (d,d) or None), ratio_reg_weight/target/eps, _cached_ratio_reg
"""

import math
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from guided_diffusion.nn import timestep_embedding


# ============================================================
# helpers
# ============================================================

def build_edge_mask(gene_list, edge_tsv_path: Union[str, Path]) -> torch.Tensor:
    """tf_target_edges.tsv から (n, n) の制御関係 mask を作る（GeneODE と同じ規約）。

    mask[src, tgt] = 1.0 if (from=src, to=tgt) が gene_list 内に存在。
    戻り値 shape: (len(gene_list), len(gene_list))
    """
    full_genes = list(gene_list)
    full_set = set(full_genes)
    g2idx = {g: i for i, g in enumerate(full_genes)}

    df = pd.read_csv(edge_tsv_path, sep="\t", usecols=["from", "to"])
    df = df[df["from"].isin(full_set) & df["to"].isin(full_set)]

    n = len(full_genes)
    mask = torch.zeros((n, n), dtype=torch.float32)
    for _, r in df.iterrows():
        mask[g2idx[r["from"]], g2idx[r["to"]]] = 1.0
    print(f"[build_edge_mask] genes={n}, edges(in-list)={int(mask.sum().item())}")
    return mask


class _TimeEmb(nn.Module):
    """sinusoidal timestep_embedding + 小 MLP -> (B, dim)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = timestep_embedding(t, self.dim)
        return self.net(emb)


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim),
    )


def _prep_t(t, batch: int, device, dtype) -> torch.Tensor:
    """t を (B,) float tensor に正規化する。

    diffusion からは (B,1) で渡ってくる。可視化では int / 0-d も許容。
    t=None（または NaN sentinel）の場合は t=0 とみなす（GeneODE 互換の「t を渡さない」呼び出し）。
    """
    if t is None:
        return torch.zeros(batch, device=device, dtype=torch.float32)
    if not torch.is_tensor(t):
        t = torch.tensor(t, device=device, dtype=dtype)
    t = t.to(device=device)
    # NaN sentinel（t を渡さない意図）も t=0 扱いにする
    if torch.is_floating_point(t) and torch.isnan(t).all():
        return torch.zeros(batch, device=device, dtype=torch.float32)
    if t.dim() > 1:
        t = t.reshape(-1)
    if t.dim() == 0:
        t = t[None]
    if t.numel() == 1 and batch > 1:
        t = t.repeat(batch)
    return t.float()


# ============================================================
# base class
# ============================================================

class MathMLPField(nn.Module):
    """4 系統の共通基底。GeneODE と duck-type 互換にするための属性/メソッドを提供。"""

    model_type = "base"
    # compute_W が厳密な W(x,t) を返すか（lincomb は proxy なので False）
    W_IS_EXACT = True

    def __init__(
        self,
        d: int,
        mask: Optional[torch.Tensor] = None,
        soft: bool = True,
        time_dim: int = 64,
        hidden: int = 256,
        dropout: float = 0.0,
        lowrank_penalty_subsample: int = 8,
        use_decay: bool = True,
    ):
        super().__init__()
        self.d = int(d)
        self.soft = bool(soft)
        self.time_dim = int(time_dim)
        self.hidden = int(hidden)
        self.dropout = float(dropout)
        self.lowrank_penalty_subsample = int(lowrank_penalty_subsample)

        # GeneODE 風の減衰項 -softplus(gamma)*x （出力を符号付きにする）
        # gamma は W とは独立な加法項なので off-mask 罰則 / compute_W には影響しない
        self.use_decay = bool(use_decay)
        self.gamma = nn.Parameter(torch.ones(self.d) * 0.1)

        # off-mask 罰則用 mask（None なら W 全体に罰則）
        if mask is not None:
            self.register_buffer("mask", mask.float())
        else:
            self.mask = None

        # GeneODE 互換の ratio 正則化フック（ODE_ML_Hybrid 由来）
        self.ratio_reg_weight = 1.0
        self.ratio_reg_target = 1.0
        self.ratio_reg_eps = 1e-8
        self._cached_ratio_reg = None

        # LowRank が forward 内で詰める W subsample（grad 付き）
        self._cached_W_sub = None
        # LowRank の subsample W キャッシュを作るか。
        # cell_train が ode_reg_lambda>0 のときだけ True にする（無駄な構築を防ぐ）。
        self.enable_offmask_cache = True

    # ---- 共通ユーティリティ -------------------------------------------------
    def _zero(self) -> torch.Tensor:
        # gamma は全 field が必ず持つ Parameter なので device 安全
        return self.gamma.new_zeros(())

    def _decay(self, x: torch.Tensor) -> torch.Tensor:
        """GeneODE 風の減衰項 softplus(gamma)*x （(B,d)）。use_decay=False なら 0。"""
        if not self.use_decay:
            return torch.zeros_like(x)
        return F.softplus(self.gamma) * x

    def _masked_norm(self, W: torch.Tensor, norm: str) -> torch.Tensor:
        """off-mask 成分の L1/L2 平均。

        W: (..., d, d)。mask が None なら全成分を罰則。
        """
        if self.mask is not None:
            sel = (1.0 - self.mask) * W      # broadcast over leading dims
        else:
            sel = W
        if norm.lower() == "l2":
            return (sel ** 2).mean()
        return sel.abs().mean()

    def _offmask_base(self, norm: str) -> torch.Tensor:
        raise NotImplementedError

    def off_mask_penalty(self, norm: str = "l1") -> torch.Tensor:
        """train_util の hook が呼ぶ。base 罰則 + ratio aux（GeneODE と同形）。"""
        base = self._offmask_base(norm)
        aux = getattr(self, "_cached_ratio_reg", None)
        if aux is None or self.ratio_reg_weight <= 0:
            return base
        return base + self.ratio_reg_weight * aux

    def get_model_info(self) -> dict:
        return {
            "type": self.model_type,
            "d": self.d,
            "rank": getattr(self, "rank", None),
            "K": getattr(self, "K", None),
            "use_mask": self.mask is not None,
            "soft": self.soft,
            "use_decay": self.use_decay,
            "w_is_exact": self.W_IS_EXACT,
        }

    # subclass が実装（t は省略可。省略時は t=0 扱い = GeneODE 互換）
    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def compute_W(self, x: torch.Tensor, t=None) -> torch.Tensor:
        raise NotImplementedError


# ============================================================
# 1) Low Rank :  W(x,t) = U(x,t) V_low(x,t)^T
# ============================================================

class LowRankField(MathMLPField):
    model_type = "lowrank"

    def __init__(self, d: int, rank: int = 16, mask=None, soft=True,
                 time_dim=64, hidden=256, dropout=0.0, lowrank_penalty_subsample=8,
                 use_decay=True):
        super().__init__(d, mask=mask, soft=soft, time_dim=time_dim, hidden=hidden,
                         dropout=dropout, lowrank_penalty_subsample=lowrank_penalty_subsample,
                         use_decay=use_decay)
        self.rank = int(rank)
        self.time_emb = _TimeEmb(time_dim)
        feat = d + time_dim
        self.U_net = _mlp(feat, hidden, d * self.rank, dropout)       # -> (B, d*r)
        self.V_net = _mlp(feat, hidden, d * self.rank, dropout)       # -> (B, d*r)
        self.b = nn.Parameter(torch.zeros(d))

    def _factors(self, x, t=None):
        B = x.shape[0]
        t = _prep_t(t, B, x.device, x.dtype)
        feat = torch.cat([x, self.time_emb(t)], dim=-1)
        U = self.U_net(feat).view(B, self.d, self.rank)              # (B, d, r)
        V = self.V_net(feat).view(B, self.d, self.rank)              # (B, d, r)
        return U, V

    def forward(self, x, t=None):
        x = x.float()
        U, V = self._factors(x, t)
        # Wx = U (V^T x)  : (B,d,r),(B,d) -> (B,r) -> (B,d)   ※ W は作らない
        z = torch.einsum("bdr,bd->br", V, x)
        Wx = torch.einsum("bdr,br->bd", U, z)
        out = F.softplus(Wx + self.b) - self._decay(x)

        # subsample W キャッシュは「学習中 & soft & 罰則有効」のときだけ作る。
        # それ以外（eval/sampling/ode_reg_lambda=0）では None にして
        # 古い cache が残らないようにする（off_mask_penalty は None→0 を返す）。
        if self.training and self.soft and self.enable_offmask_cache:
            n_sub = min(self.lowrank_penalty_subsample, x.shape[0])
            self._cached_W_sub = torch.einsum(
                "bir,bjr->bij", U[:n_sub], V[:n_sub]
            )                                                        # grad 付き、≤ n_sub*d*d
        else:
            self._cached_W_sub = None
        return out

    def _offmask_base(self, norm: str) -> torch.Tensor:
        # W(x,t) が完全に動的なので forward の cache が無ければ罰則 0（防御）
        if self._cached_W_sub is None:
            return self.gamma.new_zeros(())
        return self._masked_norm(self._cached_W_sub, norm)

    @torch.no_grad()
    def compute_W(self, x, t=None):
        x = x.float()
        U, V = self._factors(x, t)
        return torch.einsum("bir,bjr->bij", U, V)                    # (B, d, d)


# ============================================================
# 2) 線型結合 :  V = Σ_k a_k(x,t) softplus(W_k x + b_k)
# ============================================================

class LinCombField(MathMLPField):
    model_type = "lincomb"
    # V = Σ_k a_k softplus(W_k x + b_k) は単一 softplus(W_eff x + b_eff) と等価でない。
    # compute_W は厳密な W ではなく proxy（Σ_k a_k W_k）を返す。
    W_IS_EXACT = False

    def __init__(self, d: int, K: int = 8, mask=None, soft=True,
                 time_dim=64, hidden=256, dropout=0.0, use_decay=True, **kw):
        super().__init__(d, mask=mask, soft=soft, time_dim=time_dim, hidden=hidden,
                         dropout=dropout, use_decay=use_decay)
        self.K = int(K)
        self.time_emb = _TimeEmb(time_dim)
        self.coeff_net = _mlp(d + time_dim, hidden, self.K, dropout)  # a_k(x,t) -> (B,K)
        self.expert_W = nn.Parameter(torch.randn(self.K, d, d) / math.sqrt(d))
        self.expert_b = nn.Parameter(torch.zeros(self.K, d))

    def _coeffs(self, x, t=None):
        B = x.shape[0]
        t = _prep_t(t, B, x.device, x.dtype)
        feat = torch.cat([x, self.time_emb(t)], dim=-1)
        return self.coeff_net(feat)                                   # (B, K)

    def forward(self, x, t=None):
        x = x.float()
        a = self._coeffs(x, t)                                        # (B,K)
        # f_k(x) = softplus(W_k x + b_k)  : (B,K,d)
        Fx = F.softplus(torch.einsum("kij,bj->bki", self.expert_W, x) + self.expert_b)
        out = torch.einsum("bk,bkd->bd", a, Fx) - self._decay(x)       # (B,d)
        return out

    def _offmask_base(self, norm: str) -> torch.Tensor:
        # 静的 W_k の off-mask（x,t 非依存）。罰則は厳密に expert W に対して掛かる。
        return self._masked_norm(self.expert_W, norm)

    @torch.no_grad()
    def compute_W(self, x, t=None):
        """⚠ proxy（厳密な W ではない）。

        このモデルは V = Σ_k a_k softplus(W_k x + b_k) で、単一の softplus(W_eff x + b_eff)
        とは等価でない。可視化のため preactivation の係数付き線型近似 Σ_k a_k W_k を返す。
        他モデル（lowrank/matsum/lora）の厳密な W との横比較では解釈に注意。
        W_IS_EXACT=False で識別できる。
        """
        a = self._coeffs(x.float(), t)                               # (B,K)
        return torch.einsum("bk,kij->bij", a, self.expert_W)          # (B,d,d) proxy


# ============================================================
# 3) 行列の和 :  W(x,t) = Σ_k a_k(x,t) A_k ,  V = softplus(Wx + b)
# ============================================================

class MatSumField(MathMLPField):
    model_type = "matsum"

    def __init__(self, d: int, K: int = 8, mask=None, soft=True,
                 time_dim=64, hidden=256, dropout=0.0, use_decay=True, **kw):
        super().__init__(d, mask=mask, soft=soft, time_dim=time_dim, hidden=hidden,
                         dropout=dropout, use_decay=use_decay)
        self.K = int(K)
        self.time_emb = _TimeEmb(time_dim)
        self.coeff_net = _mlp(d + time_dim, hidden, self.K, dropout)
        self.A = nn.Parameter(torch.randn(self.K, d, d) / math.sqrt(d))
        self.b = nn.Parameter(torch.zeros(d))

    def _coeffs(self, x, t=None):
        B = x.shape[0]
        t = _prep_t(t, B, x.device, x.dtype)
        feat = torch.cat([x, self.time_emb(t)], dim=-1)
        return self.coeff_net(feat)                                   # (B,K)

    def forward(self, x, t=None):
        x = x.float()
        a = self._coeffs(x, t)                                        # (B,K)
        # Wx = Σ_k a_k (A_k x) : (A_k x) -> (B,K,d), 重み和 -> (B,d)   ※ W は作らない
        AX = torch.einsum("kij,bj->bki", self.A, x)                   # (B,K,d)
        Wx = torch.einsum("bk,bkd->bd", a, AX)                        # (B,d)
        return F.softplus(Wx + self.b) - self._decay(x)

    def _offmask_base(self, norm: str) -> torch.Tensor:
        return self._masked_norm(self.A, norm)

    @torch.no_grad()
    def compute_W(self, x, t=None):
        a = self._coeffs(x.float(), t)                               # (B,K)
        return torch.einsum("bk,kij->bij", a, self.A)                 # (B,d,d)


# ============================================================
# 4) LoRA :  W(x,t) = W_0 + Σ_k a_k(x,t) U_k V_k^T
# ============================================================

class LoRAField(MathMLPField):
    model_type = "lora"

    def __init__(self, d: int, K: int = 8, rank: int = 16, mask=None, soft=True,
                 time_dim=64, hidden=256, dropout=0.0, use_decay=True, **kw):
        super().__init__(d, mask=mask, soft=soft, time_dim=time_dim, hidden=hidden,
                         dropout=dropout, use_decay=use_decay)
        self.K = int(K)
        self.rank = int(rank)
        self.time_emb = _TimeEmb(time_dim)
        self.coeff_net = _mlp(d + time_dim, hidden, self.K, dropout)
        self.W0 = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        self.U = nn.Parameter(torch.randn(self.K, d, self.rank) / math.sqrt(d))
        self.V = nn.Parameter(torch.randn(self.K, d, self.rank) / math.sqrt(d))
        self.b = nn.Parameter(torch.zeros(d))

    def _coeffs(self, x, t=None):
        B = x.shape[0]
        t = _prep_t(t, B, x.device, x.dtype)
        feat = torch.cat([x, self.time_emb(t)], dim=-1)
        return self.coeff_net(feat)                                   # (B,K)

    def forward(self, x, t=None):
        x = x.float()
        a = self._coeffs(x, t)                                        # (B,K)
        # Wx = W0 x + Σ_k a_k U_k (V_k^T x)   ※ W は作らない
        W0x = torch.einsum("ij,bj->bi", self.W0, x)                   # (B,d)
        Vx = torch.einsum("kjr,bj->bkr", self.V, x)                   # (B,K,r)
        UVx = torch.einsum("kir,bkr->bki", self.U, Vx)               # (B,K,d)
        delta = torch.einsum("bk,bki->bi", a, UVx)                    # (B,d)
        return F.softplus(W0x + delta + self.b) - self._decay(x)

    def _offmask_base(self, norm: str) -> torch.Tensor:
        # 静的 W_0 + 静的 Δ_k = U_k V_k^T （batch 無しで K 枚 materialize）
        Delta = torch.einsum("kir,kjr->kij", self.U, self.V)          # (K,d,d)
        return self._masked_norm(self.W0, norm) + self._masked_norm(Delta, norm)

    @torch.no_grad()
    def compute_W(self, x, t=None):
        a = self._coeffs(x.float(), t)                               # (B,K)
        Delta = torch.einsum("kir,kjr->kij", self.U, self.V)          # (K,d,d)
        return self.W0[None] + torch.einsum("bk,kij->bij", a, Delta)  # (B,d,d)


# ============================================================
# hybrid wrapper（ODE_ML_Hybrid の t-対応版）
# ============================================================

class MathML_Hybrid(nn.Module):
    """field（数理構造）と Cell_Unet を t-scheduler で blend する。

    ODE_ML_Hybrid との唯一の差: ode_model に t を渡す（field は W(x,t)）。
    `self.ode_model = field` なので train_util の正則化 hook がゼロ変更で発火する。
    """

    def __init__(self, field: MathMLPField, ml_model: nn.Module, timesteps: int):
        super().__init__()
        assert timesteps > 1
        self.ode_model = field        # ← hook が getattr(model, "ode_model") で参照
        self.ml_model = ml_model
        self.T = int(timesteps)

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
        field_out = self.ode_model(x, t)          # ← GeneODE 版は (x) のみ。ここだけ違う
        ml_out = self.ml_model(x, t, y)
        # blend は shape 一致が前提（field/ML とも (B,d) を返すこと）
        assert field_out.shape == ml_out.shape, (
            f"shape mismatch: field={tuple(field_out.shape)} ml={tuple(ml_out.shape)}"
        )

        if self.training:
            self.ode_model._cached_ratio_reg = self._norm_ratio_penalty(field_out, ml_out)
        else:
            self.ode_model._cached_ratio_reg = None

        r = self._scheduler(t, device=x.device, dtype=x.dtype)
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)
        return r * field_out + (1.0 - r) * ml_out


# ============================================================
# factory
# ============================================================

_FIELD_REGISTRY = {
    "lowrank": LowRankField,
    "lincomb": LinCombField,
    "matsum": MatSumField,
    "lora": LoRAField,
}


def build_math_field(
    model_type: str,
    gene_list,
    edge_tsv_path: Union[str, Path],
    rank: int = 16,
    K: int = 8,
    use_mask: bool = True,
    soft: bool = True,
    time_dim: int = 64,
    hidden: int = 256,
    dropout: float = 0.0,
    lowrank_penalty_subsample: int = 8,
    use_decay: bool = True,
    device: Union[str, torch.device] = "cpu",
) -> MathMLPField:
    """model_type に応じて field を生成して device へ載せる。

    gene_list の長さが d。use_mask=True のとき tf_target_edges.tsv から off-mask 用 mask を構築。
    """
    model_type = str(model_type).lower()
    if model_type not in _FIELD_REGISTRY:
        raise ValueError(
            f"Unknown model_type='{model_type}'. choose from {list(_FIELD_REGISTRY)}"
        )

    d = len(gene_list)
    mask = build_edge_mask(gene_list, edge_tsv_path) if use_mask else None

    common = dict(mask=mask, soft=soft, time_dim=time_dim, hidden=hidden,
                  dropout=dropout, use_decay=use_decay)
    if model_type == "lowrank":
        field = LowRankField(d, rank=rank, lowrank_penalty_subsample=lowrank_penalty_subsample, **common)
    elif model_type == "lincomb":
        field = LinCombField(d, K=K, **common)
    elif model_type == "matsum":
        field = MatSumField(d, K=K, **common)
    elif model_type == "lora":
        field = LoRAField(d, K=K, rank=rank, **common)

    return field.to(device)


# ============================================================
# checkpoint loading（strict=False の危険性に対する防御つき）
# ============================================================

def clean_state_dict(sd):
    """EMA / wrapper 階層 / prefix を剥がして素の state_dict にする。"""
    if any(k.startswith("ema") for k in sd.keys()):
        sd = sd.get("ema", sd)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    # DDP の "module." prefix を剥がす
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    return sd


def load_hybrid_state_dict(hybrid, sd, strict=True, log=print):
    """MathML_Hybrid に checkpoint を読み込む。

    strict=False の「重要 parameter が読まれないまま動く」危険を避けるため:
      - clean_state_dict で EMA/wrapper/prefix を処理
      - missing / unexpected / shape-mismatch を **すべてログ出力**
      - model_type / rank / K / dim ズレを検出（ode_model.* / ml_model.* の
        missing or shape mismatch があれば critical とみなす）
      - strict=True かつ critical があれば **例外で止める**（誤 checkpoint の暴走防止）

    間違った checkpoint を黙って読まないことが目的。比較実験では特に重要。
    """
    sd = clean_state_dict(sd)
    model_sd = hybrid.state_dict()
    model_keys = set(model_sd.keys())
    ckpt_keys = set(sd.keys())

    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    mismatch = sorted(
        k for k in (model_keys & ckpt_keys) if tuple(sd[k].shape) != tuple(model_sd[k].shape)
    )

    log(f"[load] missing_keys ({len(missing)}): {missing}")
    log(f"[load] unexpected_keys ({len(unexpected)}): {unexpected}")
    log(f"[load] shape_mismatch ({len(mismatch)}): "
        + ", ".join(f"{k} ckpt{tuple(sd[k].shape)} vs model{tuple(model_sd[k].shape)}" for k in mismatch))

    def is_core(k):
        return k.startswith("ode_model.") or k.startswith("ml_model.")

    critical = [k for k in missing if is_core(k)] + [k for k in mismatch if is_core(k)]
    if critical:
        msg = (
            "[load] CRITICAL: core parameters missing or shape-mismatched "
            f"(likely wrong model_type/rank/K/dim or checkpoint): {critical}"
        )
        if strict:
            raise RuntimeError(msg + "  -> strict=True のため中断。--strict_load False で強行可。")
        log(msg + "  -> strict=False のため警告のみで続行。")

    # shape mismatch の項目は load_state_dict が拒否するので除外して読む
    filtered = {k: v for k, v in sd.items() if not (k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape))}
    hybrid.load_state_dict(filtered, strict=False)
    return missing, unexpected, mismatch
