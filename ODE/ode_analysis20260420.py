import math
from pathlib import Path
from typing import Optional, Union, Literal

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion.nn import timestep_embedding


# ============================================================
# Shared helpers
# ============================================================


class ODETimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, t: Union[torch.Tensor, int, float]) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t)
        if t.dim() == 0:
            t = t[None]
        if t.dim() > 1:
            t = t.view(-1)
        t = t.float()
        emb = timestep_embedding(t, self.hidden_dim)
        return self.time_embed(emb)


class AlphaNet(nn.Module):
    """
    alpha(x, t) generator.

    mode='mlp'    : alpha = softmax(MLP([x, t_emb]))
    mode='linear' : alpha = softmax(Linear([x, t_emb]))

    If topk is set, only top-k logits survive before the final softmax.
    """

    def __init__(
        self,
        input_dim: int,
        num_coeffs: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        mode: Literal["mlp", "linear"] = "mlp",
        use_time_embedding: bool = True,
        time_hidden_dim: int = 64,
        topk: Optional[int] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_coeffs = int(num_coeffs)
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode)
        self.use_time_embedding = bool(use_time_embedding)
        self.time_hidden_dim = int(time_hidden_dim)
        self.topk = topk if (topk is None or int(topk) > 0) else None
        self.temperature = float(temperature)

        if self.use_time_embedding:
            self.time_embedding = ODETimeEmbedding(self.time_hidden_dim)
            feat_dim = self.input_dim + self.time_hidden_dim
        else:
            self.time_embedding = None
            feat_dim = self.input_dim

        if self.mode == "mlp":
            self.logit_net = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_coeffs),
            )
        elif self.mode == "linear":
            self.logit_net = nn.Linear(feat_dim, num_coeffs)
        else:
            raise ValueError(f"Unknown alpha mode: {self.mode}")

    def _normalize_t(self, x: torch.Tensor, t: Optional[Union[torch.Tensor, int]]) -> Optional[torch.Tensor]:
        if not self.use_time_embedding:
            return None

        if t is None:
            t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        elif not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)
        else:
            t = t.to(device=x.device)

        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        elif t.dim() > 1:
            t = t.view(-1)

        if t.numel() == 1:
            t = t.repeat(x.shape[0])

        if t.shape[0] != x.shape[0]:
            raise ValueError(f"Batch mismatch: x.shape[0]={x.shape[0]} but t.shape[0]={t.shape[0]}")
        return t

    def _apply_topk(self, logits: torch.Tensor) -> torch.Tensor:
        if self.topk is None or self.topk >= logits.shape[-1]:
            return logits
        values, indices = torch.topk(logits, k=self.topk, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(dim=-1, index=indices, src=values)
        return masked

    def forward(self, x: torch.Tensor, t: Optional[Union[torch.Tensor, int]] = None) -> torch.Tensor:
        x = x.float()
        if self.use_time_embedding:
            t = self._normalize_t(x, t)
            temb = self.time_embedding(t)
            feat = torch.cat([x, temb], dim=-1)
        else:
            feat = x

        logits = self.logit_net(feat) / max(self.temperature, 1e-8)
        logits = self._apply_topk(logits)
        alpha = F.softmax(logits, dim=-1)
        return alpha


class MaskMixin:
    def _build_mask(self, gene_list: list[str], edge_tsv_path: Union[str, Path]):
        full_genes = list(gene_list)
        full_set = set(full_genes)
        g2idx = {g: i for i, g in enumerate(full_genes)}

        df = pd.read_csv(edge_tsv_path, sep="\t", usecols=["from", "to"])
        df = df[df["from"].isin(full_set) & df["to"].isin(full_set)]

        sub_genes = [g for g in full_genes if g in set(df["from"]) | set(df["to"])]
        n = len(full_genes)
        m = len(sub_genes)
        print(f"TSVに記述されてる遺伝子の数 {m}, 全体の遺伝子の数 {n}")

        mask = torch.zeros((n, n), dtype=torch.float32)
        for _, r in df.iterrows():
            src = g2idx[r["from"]]
            tgt = g2idx[r["to"]]
            mask[src, tgt] = 1.0

        return full_genes, sub_genes, g2idx, mask

    @torch.no_grad()
    def _update_plot_W(self, W_mean_detached: torch.Tensor):
        w_new = W_mean_detached.detach().to(device=self.W.device, dtype=self.W.dtype)
        if self.W.shape != w_new.shape:
            self.W = torch.zeros_like(w_new)
            self.W.copy_(w_new)
            return
        self.W.mul_(self.w_ema).add_(w_new, alpha=1.0 - self.w_ema)

    def off_mask_penalty(self, norm: str = "l1"):
        if (not self.soft) or (self._cached_W_for_penalty is None):
            return self.b.new_zeros(())
        off = (1.0 - self.mask) * self._cached_W_for_penalty
        if norm.lower() == "l2":
            return (off ** 2).mean()
        return off.abs().mean()


# ============================================================
# 1) V = sum_k alpha_k(x, t) f_k(x), f_k(x)=softplus(xW_k + b_k)
# ============================================================


class MixtureOfSoftplusODE(nn.Module):
    def __init__(
        self,
        gene_list: list[str],
        edge_tsv_path: Union[str, Path],
        soft: bool = True,
        num_experts: int = 8,
        alpha_hidden_dim: int = 128,
        alpha_topk: Optional[int] = None,
        alpha_temperature: float = 1.0,
        use_time_embedding: bool = True,
        time_hidden_dim: int = 64,
        alpha_dropout: float = 0.0,
        w_ema: float = 0.95,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.soft = bool(soft)
        self.w_ema = float(w_ema)

        self.full_genes = list(gene_list)
        self.n = len(self.full_genes)
        self.num_experts = int(num_experts)

        full_set = set(self.full_genes)
        g2idx = {g: i for i, g in enumerate(self.full_genes)}

        df = pd.read_csv(edge_tsv_path, sep="\t", usecols=["from", "to"])
        df = df[df["from"].isin(full_set) & df["to"].isin(full_set)]

        mask = torch.zeros((self.n, self.n), dtype=torch.float32)
        for _, r in df.iterrows():
            src = g2idx[r["from"]]
            tgt = g2idx[r["to"]]
            mask[src, tgt] = 1.0

        self.register_buffer("mask", mask)
        self.register_buffer("W", torch.zeros(self.n, self.n, dtype=torch.float32))
        self._cached_W_for_penalty = None

        self.expert_W = nn.Parameter(
            torch.randn(self.num_experts, self.n, self.n) / math.sqrt(max(self.n, 1))
        )
        self.expert_b = nn.Parameter(torch.zeros(self.num_experts, self.n))
        self.gamma = nn.Parameter(torch.full((self.n,), 0.1, dtype=torch.float32))

        self.alpha_net = AlphaNet(
            input_dim=self.n,
            num_coeffs=self.num_experts,
            hidden_dim=alpha_hidden_dim,
            dropout=alpha_dropout,
            mode="mlp",
            use_time_embedding=use_time_embedding,
            time_hidden_dim=time_hidden_dim,
            topk=alpha_topk,
            temperature=alpha_temperature,
        )
        self.to(self.device)
    def _expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bn,knm->bkm", x, self.expert_W) + self.expert_b.unsqueeze(0)
        return F.softplus(logits)

    def forward(self, x: torch.Tensor, t: Optional[Union[torch.Tensor, int]] = None) -> torch.Tensor:
        x = x.float()
        expert_v = self._expert_outputs(x)      # (B, K, N)
        alpha = self.alpha_net(x, t=t)          # (B, K)
        V = torch.einsum("bk,bkn->bn", alpha, expert_v)

        W_mean = torch.einsum("k,knm->nm", alpha.mean(dim=0), self.expert_W)

        if not self.soft:
            W_mean = W_mean * self.mask

        self._cached_W_for_penalty = W_mean if self.soft else None
        self.W.copy_(W_mean.detach())

        gamma = F.softplus(self.gamma)
        dxdt = V - gamma * x
        return dxdt

    def off_mask_penalty(self, norm: str = "l1"):
        if (not self.soft) or (self._cached_W_for_penalty is None):
            return self.gamma.new_zeros(())

        off = (1.0 - self.mask) * self._cached_W_for_penalty
        if norm.lower() == "l2":
            return (off ** 2).mean()
        return off.abs().mean()


# ============================================================
# 2) W(x, t) = sum_k alpha_k(x, t) A_k
# ============================================================


class MatrixDictionaryODE(MaskMixin, nn.Module):
    def __init__(
        self,
        gene_list: list[str],
        edge_tsv_path: Union[str, Path],
        soft: bool = False,
        device: Union[str, torch.device] = "cuda",
        num_bases: int = 8,
        alpha_mode: Literal["mlp", "linear"] = "mlp",
        alpha_hidden_dim: int = 128,
        alpha_topk: Optional[int] = None,
        alpha_temperature: float = 1.0,
        use_time_embedding: bool = True,
        time_hidden_dim: int = 64,
        alpha_dropout: float = 0.0,
        w_ema: float = 0.95,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.soft = bool(soft)
        self.w_ema = float(w_ema)
        self.num_bases = int(num_bases)

        self.full_genes, self.sub_genes, self.g2idx, mask = self._build_mask(gene_list, edge_tsv_path)
        self.n = len(self.full_genes)
        self.m = len(self.sub_genes)
        self.register_buffer("mask", mask)
        self.register_buffer("W", torch.zeros(self.n, self.n, dtype=torch.float32))
        self._cached_W_for_penalty: Optional[torch.Tensor] = None

        self.A = nn.Parameter(
            torch.randn(self.num_bases, self.n, self.n) / math.sqrt(max(self.n, 1))
        )
        self.alpha_net = AlphaNet(
            input_dim=self.n,
            num_coeffs=self.num_bases,
            hidden_dim=alpha_hidden_dim,
            dropout=alpha_dropout,
            mode=alpha_mode,
            use_time_embedding=use_time_embedding,
            time_hidden_dim=time_hidden_dim,
            topk=alpha_topk,
            temperature=alpha_temperature,
        )
        self.b = nn.Parameter(torch.zeros(self.n, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full((self.n,), 0.1, dtype=torch.float32))
        self.to(self.device)

    def forward(self, x: torch.Tensor, t: Optional[Union[torch.Tensor, int]] = None) -> torch.Tensor:
        x = x.float()
        alpha = self.alpha_net(x, t=t)                # (B, K)
        W = torch.einsum("bk,knm->bnm", alpha, self.A)

        if not self.soft:
            W = W * self.mask.unsqueeze(0)

        Wx = torch.einsum("bn,bnm->bm", x, W)
        W_mean = W.mean(dim=0)
        self._cached_W_for_penalty = W_mean if self.soft else None
        self._update_plot_W(W_mean.detach())

        alpha_out = F.softplus(Wx + self.b)
        gamma = F.softplus(self.gamma)
        dxdt = alpha_out - gamma * x
        return dxdt


# ============================================================
# 3) W(x, t) = W0 + sum_k alpha_k(x, t) Delta_k, Delta_k = U_k V_k^T
# ============================================================


class LowRankResidualDictionaryODE(MaskMixin, nn.Module):
    def __init__(
        self,
        gene_list: list[str],
        edge_tsv_path: Union[str, Path],
        soft: bool = False,
        device: Union[str, torch.device] = "cuda",
        num_bases: int = 8,
        rank: int = 16,
        alpha_mode: Literal["mlp", "linear"] = "mlp",
        alpha_hidden_dim: int = 128,
        alpha_topk: Optional[int] = None,
        alpha_temperature: float = 1.0,
        use_time_embedding: bool = True,
        time_hidden_dim: int = 64,
        alpha_dropout: float = 0.0,
        learn_W0: bool = True,
        w_ema: float = 0.95,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.soft = bool(soft)
        self.w_ema = float(w_ema)
        self.num_bases = int(num_bases)
        self.rank = int(rank)

        self.full_genes, self.sub_genes, self.g2idx, mask = self._build_mask(gene_list, edge_tsv_path)
        self.n = len(self.full_genes)
        self.m = len(self.sub_genes)
        self.register_buffer("mask", mask)
        self.register_buffer("W", torch.zeros(self.n, self.n, dtype=torch.float32))
        self.register_buffer("latent_scale", torch.tensor(1.0 / math.sqrt(max(self.rank, 1))))
        self._cached_W_for_penalty: Optional[torch.Tensor] = None

        if learn_W0:
            self.W0 = nn.Parameter(torch.zeros(self.n, self.n, dtype=torch.float32))
        else:
            self.register_buffer("W0", torch.zeros(self.n, self.n, dtype=torch.float32))

        self.U = nn.Parameter(
            torch.randn(self.num_bases, self.n, self.rank) / math.sqrt(max(self.n, 1))
        )
        self.V = nn.Parameter(
            torch.randn(self.num_bases, self.n, self.rank) / math.sqrt(max(self.n, 1))
        )
        self.alpha_net = AlphaNet(
            input_dim=self.n,
            num_coeffs=self.num_bases,
            hidden_dim=alpha_hidden_dim,
            dropout=alpha_dropout,
            mode=alpha_mode,
            use_time_embedding=use_time_embedding,
            time_hidden_dim=time_hidden_dim,
            topk=alpha_topk,
            temperature=alpha_temperature,
        )
        self.b = nn.Parameter(torch.zeros(self.n, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full((self.n,), 0.1, dtype=torch.float32))
        self.to(self.device)

    def _compute_residual_Wx(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        proj = torch.einsum("bn,knr->bkr", x, self.U)
        weighted_proj = alpha.unsqueeze(-1) * proj
        residual = torch.einsum("bkr,kmr->bm", weighted_proj, self.V) * self.latent_scale
        return residual

    def _compute_delta_mean(self, alpha: torch.Tensor) -> torch.Tensor:
        alpha_mean = alpha.mean(dim=0)
        delta_mean = torch.einsum("k,knr,kmr->nm", alpha_mean, self.U, self.V) * self.latent_scale
        return delta_mean

    def forward(self, x: torch.Tensor, t: Optional[Union[torch.Tensor, int]] = None) -> torch.Tensor:
        x = x.float()
        alpha = self.alpha_net(x, t=t)
        Wx = x @ self.W0 + self._compute_residual_Wx(x, alpha)

        W_mean = self.W0 + self._compute_delta_mean(alpha)
        if not self.soft:
            W_mean = W_mean * self.mask

        self._cached_W_for_penalty = W_mean if self.soft else None
        self._update_plot_W(W_mean.detach())

        alpha_out = F.softplus(Wx + self.b)
        gamma = F.softplus(self.gamma)
        dxdt = alpha_out - gamma * x
        return dxdt


class ODE_ML_Hybrid(nn.Module):
    def __init__(
        self,
        ode_model: nn.Module,
        ml_model: Optional[nn.Module] = None,
        timesteps: int = 1000,
        norm: float = 1.0,
        use_hybrid: bool = False,
    ):
        super().__init__()
        self.ode_model = ode_model
        self.ml_model = ml_model
        self.T = timesteps
        self.norm = norm
        self.use_hybrid = bool(use_hybrid) and (ml_model is not None)

    def _scheduler(self, t: Union[torch.Tensor, int], device, dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)
        t = t.to(dtype=dtype, device=device)
        if self.T <= 1:
            return torch.ones_like(t)
        return 1.0 - t / (self.T - 1)

    def forward(self, x: torch.Tensor, t: Union[torch.Tensor, int], y=None):
        x = x.float()
        ode_out = self.ode_model(x, t)
        if not self.use_hybrid:
            return ode_out
        ml_out = self.ml_model(x, t, y)
        r = self._scheduler(t, device=x.device, dtype=x.dtype)
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)
        ode_unit = F.normalize(ode_out, p=2, dim=-1, eps=1e-8) * self.norm
        ml_unit = F.normalize(ml_out, p=2, dim=-1, eps=1e-8) * self.norm
        return r * ode_unit + (1.0 - r) * ml_unit


# ============================================================
# Factory for cell_train / cell_sample
# ============================================================


def validate_ode_args(args):
    ode_model_type = getattr(args, "ode_model_type", "lowrank_residual")
    alpha_mode = getattr(args, "alpha_mode", "mlp")
    alpha_topk = getattr(args, "alpha_topk", 0)

    if ode_model_type == "mixture_softplus" and alpha_mode != "mlp":
        raise ValueError(
            "mixture_softplus は仕様上 alpha_mode='mlp' 固定です。"
            " --alpha_mode mlp を使ってください。"
        )
    if alpha_topk is not None and int(alpha_topk) < 0:
        raise ValueError("alpha_topk は 0 以上で指定してください。0 は Top-K 無効です。")



def create_ode_model_from_args(args, gene_list: list[str], device: Union[str, torch.device]):
    validate_ode_args(args)

    ode_model_type = getattr(args, "ode_model_type", "lowrank_residual")
    alpha_topk = getattr(args, "alpha_topk", 0)
    alpha_topk = None if int(alpha_topk) <= 0 else int(alpha_topk)

    common = dict(
        gene_list=gene_list,
        device=device,
    )

    if ode_model_type == "mixture_softplus":
        return MixtureOfSoftplusODE(
            **common,
            edge_tsv_path=getattr(args, "edge_tsv_path"),
            soft=bool(getattr(args, "SoftReg", True)),
            num_experts=int(getattr(args, "num_experts", 8)),
            alpha_hidden_dim=int(getattr(args, "alpha_hidden_dim", 128)),
            alpha_topk=alpha_topk,
            alpha_temperature=float(getattr(args, "alpha_temperature", 1.0)),
            use_time_embedding=bool(getattr(args, "use_time_embedding", True)),
            time_hidden_dim=int(getattr(args, "time_hidden_dim", 64)),
            alpha_dropout=float(getattr(args, "alpha_dropout", 0.0)),
            w_ema=float(getattr(args, "w_ema", 0.95)),
        )

    if ode_model_type == "matrix_dict":
        return MatrixDictionaryODE(
            **common,
            edge_tsv_path=getattr(args, "edge_tsv_path"),
            soft=bool(getattr(args, "SoftReg", True)),
            num_bases=int(getattr(args, "num_bases", 8)),
            alpha_mode=str(getattr(args, "alpha_mode", "mlp")),
            alpha_hidden_dim=int(getattr(args, "alpha_hidden_dim", 128)),
            alpha_topk=alpha_topk,
            alpha_temperature=float(getattr(args, "alpha_temperature", 1.0)),
            use_time_embedding=bool(getattr(args, "use_time_embedding", True)),
            time_hidden_dim=int(getattr(args, "time_hidden_dim", 64)),
            alpha_dropout=float(getattr(args, "alpha_dropout", 0.0)),
            w_ema=float(getattr(args, "w_ema", 0.95)),
        )

    if ode_model_type == "lowrank_residual":
        return LowRankResidualDictionaryODE(
            **common,
            edge_tsv_path=getattr(args, "edge_tsv_path"),
            soft=bool(getattr(args, "SoftReg", True)),
            num_bases=int(getattr(args, "num_bases", 8)),
            rank=int(getattr(args, "rank", 16)),
            alpha_mode=str(getattr(args, "alpha_mode", "mlp")),
            alpha_hidden_dim=int(getattr(args, "alpha_hidden_dim", 128)),
            alpha_topk=alpha_topk,
            alpha_temperature=float(getattr(args, "alpha_temperature", 1.0)),
            use_time_embedding=bool(getattr(args, "use_time_embedding", True)),
            time_hidden_dim=int(getattr(args, "time_hidden_dim", 64)),
            alpha_dropout=float(getattr(args, "alpha_dropout", 0.0)),
            learn_W0=bool(getattr(args, "learn_W0", True)),
            w_ema=float(getattr(args, "w_ema", 0.95)),
        )

    raise ValueError(f"Unknown ode_model_type: {ode_model_type}")
