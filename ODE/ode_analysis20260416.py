"""
/home/suzuki/Projects/scDiffusion/work/20260416_Transformer
/home/suzuki/Projects/scDiffusion/work/20260420_TransformerTimeEmbedding
のモデル
"""
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from guided_diffusion.nn import timestep_embedding

class FactorMLP(nn.Module):
    """
    Shared node-wise encoder f(x_i; theta).

    Each gene/node i is represented by:
      - current expression x_i (per sample)
      - a learnable gene embedding e_i (global)

    The same MLP is applied to every node, which keeps the model permutation-aware
    with respect to the node axis while still allowing node identity through e_i.
    """

    def __init__(
        self,
        num_nodes: int,
        out_dim: int,
        hidden_dim: int = 64,
        gene_emb_dim: int = 16,
        dropout: float = 0.0,
        input_clip: float = 8.0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_dim = out_dim
        self.input_clip = float(input_clip)

        self.gene_emb = nn.Parameter(
            torch.randn(num_nodes, gene_emb_dim) / math.sqrt(max(gene_emb_dim, 1))
        )

        in_dim = gene_emb_dim + 3  # [x, x^2, sign(x)] + gene embedding
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.out_norm = nn.LayerNorm(out_dim)
        self.out_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(min=-self.input_clip, max=self.input_clip)

        bsz, n = x.shape
        if n != self.num_nodes:
            raise ValueError(f"Expected x.shape[1] == {self.num_nodes}, got {n}")

        x_feat = torch.stack([x, x * x, torch.sign(x)], dim=-1)  # (B, N, 3)
        gene_feat = self.gene_emb.unsqueeze(0).expand(bsz, -1, -1)  # (B, N, E)
        h = torch.cat([x_feat, gene_feat], dim=-1)
        h = self.net(h)
        h = self.out_norm(h)
        h = torch.tanh(h) * self.out_scale
        return h
    

class GeneODE(nn.Module):
    """
    Factorized ODE model.

    forward:
        A = f_theta(x)
        B = f_phi(x)
        W = A B^T / sqrt(rank)
        dx/dt = softplus(Wx + b) - softplus(gamma) * x

    Notes
    -----
    1) Training path computes Wx as A(B^T x), so we do not materialize per-sample W.
       This avoids the huge (batch, genes, genes) tensor.
    2) For plotting / checkpoint compatibility, self.W is kept as an EMA of the latest
       batch-mean adjacency. This preserves the old visualization contract.
    3) off_mask_penalty() uses the latest differentiable batch-mean W, so the existing
       training loop can keep calling ode_model.off_mask_penalty(...) unchanged.
    """

    def __init__(
        self,
        gene_list: list[str],
        edge_tsv_path: Union[str, Path] = "tf_target_edges.tsv",
        soft: bool = False,
        device: Union[str, torch.device] = "cuda",
        rank: int = 32,
        hidden_dim: int = 64,
        gene_emb_dim: int = 16,
        dropout: float = 0.0,
        w_ema: float = 0.95,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.soft = bool(soft)
        self.rank = int(rank)
        self.w_ema = float(w_ema)
        self.eps = float(eps)

        self.full_genes: list[str] = list(gene_list)
        full_set = set(self.full_genes)
        self.g2idx = {g: i for i, g in enumerate(self.full_genes)}

        df = pd.read_csv(edge_tsv_path, sep="\t", usecols=["from", "to"])
        df = df[df["from"].isin(full_set) & df["to"].isin(full_set)]

        # 元コード互換の集計用
        self.sub_genes = [g for g in self.full_genes if g in set(df["from"]) | set(df["to"])]
        self.m = len(self.sub_genes)   # TSVに登場する遺伝子数
        self.n = len(self.full_genes)  # 入力遺伝子数

        print(f"TSVに記述されてる遺伝子の数 {self.m}, 全体の遺伝子の数 {self.n}")

        mask = torch.zeros((self.n, self.n), dtype=torch.float32)
        for _, r in df.iterrows():
            src = self.g2idx[r["from"]]
            tgt = self.g2idx[r["to"]]
            mask[src, tgt] = 1.0

        self.register_buffer("mask", mask)
        self.register_buffer("latent_scale", torch.tensor(1.0 / math.sqrt(max(self.rank, 1))))

        # Plot/checkpoint compatibility:
        # keep a persistent adjacency-like tensor named exactly `W`.
        self.register_buffer("W", torch.zeros(self.n, self.n, dtype=torch.float32))

        # Latest differentiable batch-mean W used by off_mask_penalty().
        self._cached_W_for_penalty: Optional[torch.Tensor] = None

        self.f_theta = FactorMLP(
            num_nodes=self.n,
            out_dim=self.rank,
            hidden_dim=hidden_dim,
            gene_emb_dim=gene_emb_dim,
            dropout=dropout,
        )
        self.f_phi = FactorMLP(
            num_nodes=self.n,
            out_dim=self.rank,
            hidden_dim=hidden_dim,
            gene_emb_dim=gene_emb_dim,
            dropout=dropout,
        )

        self.b = nn.Parameter(torch.zeros(self.n, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full((self.n,), 0.1, dtype=torch.float32))
        self.to(self.device)
        print(
            f"[GeneODE-factorized] genes={self.n} edges_in_mask={int(mask.sum().item())} "
            f"rank={self.rank} soft={self.soft}"
        )

    def _compute_factors(self, x: torch.Tensor):
        A = self.f_theta(x)  # (B, N, R)
        B = self.f_phi(x)    # (B, N, R)
        return A, B

    def _compute_Wx_lowrank(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Efficiently compute x @ (A B^T) without building per-sample W.

        This matches the current implementation convention `x @ W + b`.

        x: (B, N)
        A, B: (B, N, R)
        returns: (B, N)
        """
        # u_b[r] = sum_i x_b[i] * A_b[i, r]   == x_b @ A_b
        u = torch.einsum("bn,bnr->br", x, A)
        # Wx_b[j] = sum_r u_b[r] * B_b[j, r]  == (x_b @ A_b) @ B_b^T
        Wx = torch.einsum("br,bnr->bn", u, B) * self.latent_scale
        return Wx

    def _batch_mean_W(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Mean over batch of W_b = A_b B_b^T / sqrt(rank), shape (N, N).
        Computed without allocating (B, N, N).
        """
        bsz = max(A.shape[0], 1)
        W_mean = torch.einsum("bnr,bmr->nm", A, B) / float(bsz)
        W_mean = W_mean * self.latent_scale
        return W_mean

    @torch.no_grad()
    def _update_plot_W(self, W_mean_detached: torch.Tensor):
        w_new = W_mean_detached.detach().to(device=self.W.device, dtype=self.W.dtype)

        if self.W.shape != w_new.shape:
            self.W = torch.zeros_like(w_new)
            self.W.copy_(w_new)
            return

        self.W.mul_(self.w_ema).add_(w_new, alpha=1.0 - self.w_ema)

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        x = x.float()

        A, B = self._compute_factors(x)
        Wx = self._compute_Wx_lowrank(x, A, B)

        W_mean = self._batch_mean_W(A, B)
        if not self.soft:
            # Hard-mask mode is kept for compatibility. In this mode the penalty is 0.
            # The forward path itself remains low-rank; only the exported/regularized W
            # is masked. If you need exact hard-masked dynamics per sample, that path
            # should be implemented separately with sparse edge aggregation.
            W_mean = W_mean * self.mask

        self._cached_W_for_penalty = W_mean if self.soft else None
        self._update_plot_W(W_mean.detach())

        alpha = F.softplus(Wx + self.b) #Wx + self.b #F.softplus(Wx + self.b) #F.sigmoid(Wx + self.b) 
        gamma = F.softplus(self.gamma)
        dxdt = alpha - gamma * x

        return dxdt

    def off_mask_penalty(self, norm: str = "l1"):
        """
        Regularize off-mask entries of the latest differentiable batch-mean W.
        This keeps the current training loop unchanged.
        """
        if (not self.soft) or (self._cached_W_for_penalty is None):
            return self.b.new_zeros(())

        off = (1.0 - self.mask) * self._cached_W_for_penalty
        norm = norm.lower()
        if norm == "l2":
            return (off ** 2).mean()
        return off.abs().mean()



class ODE_ML_Hybrid(nn.Module):
    """
    Backward-compatible wrapper.

    Default behavior is ODE-only (use_hybrid=False), so existing code that expects
    `model.ode_model` and calls `model(x, t, y)` keeps working with almost no edits.

    If you later want to re-enable the old hybrid behavior, pass use_hybrid=True and
    provide ml_model + timesteps.
    """

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
        else:
            ml_out = self.ml_model(x, t, y)
            r = self._scheduler(t, device=x.device, dtype=x.dtype)
            while r.dim() < x.dim():
                r = r.unsqueeze(-1)

            ode_unit = F.normalize(ode_out, p=2, dim=-1, eps=1e-8) * self.norm
            ml_unit = F.normalize(ml_out, p=2, dim=-1, eps=1e-8) * self.norm
            out = r * ode_unit + (1.0 - r) * ml_unit
            return out


def _finite_report(name, z, t=None, log_file=None):
    t_str = _format_t(t)

    if not torch.isfinite(z).all():
        z_det = z.detach()
        z_safe = torch.nan_to_num(z_det, nan=0.0, posinf=0.0, neginf=0.0)

        msg = (
            f"[BAD] {name}"
            f" t={t_str}"
            f" shape={tuple(z.shape)}"
            f" nan={torch.isnan(z_det).sum().item()}"
            f" +inf={torch.isposinf(z_det).sum().item()}"
            f" -inf={torch.isneginf(z_det).sum().item()}"
            f" min={z_safe.min().item():.3e}"
            f" max={z_safe.max().item():.3e}"
        )
        print(msg)
        if log_file is not None:
            with open(log_file, "a") as f:
                f.write(msg + "\n")
        return False

    return True


def _format_t(t):
    if t is None:
        return "NA"
    if torch.is_tensor(t):
        if t.numel() == 0:
            return "empty"
        return str(t.detach().flatten()[0].item())
    return str(t)



class ODETimeEmbedding(nn.Module):
    """
    Time embedding for ODE models.
    Keeps the same design spirit as guided_diffusion.cell_model.TimeEmbedding.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t)

        # normalize shape to (B,)
        if t.dim() == 0:
            t = t[None]
        if t.dim() > 1:
            t = t.view(-1)

        t = t.float()
        emb = timestep_embedding(t, self.hidden_dim)
        return self.time_embed(emb)

class GeneODE_Time(nn.Module):
    """
    Time-dependent factorized ODE model.

    Minimal-change version:
      - keep FactorMLP unchanged
      - inject time through a per-gene additive bias
      - preserve W / penalty / plotting compatibility
    """

    def __init__(
        self,
        gene_list: list[str],
        edge_tsv_path: Union[str, Path] = "tf_target_edges.tsv",
        soft: bool = False,
        device: Union[str, torch.device] = "cuda",
        rank: int = 32,
        hidden_dim: int = 64,
        gene_emb_dim: int = 16,
        dropout: float = 0.0,
        w_ema: float = 0.95,
        eps: float = 1e-8,
        time_hidden_dim: Optional[int] = None,
        time_scale: float = 1.0,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.soft = bool(soft)
        self.rank = int(rank)
        self.w_ema = float(w_ema)
        self.eps = float(eps)
        self.time_hidden_dim = int(time_hidden_dim or hidden_dim)
        self.time_scale = float(time_scale)

        self.full_genes: list[str] = list(gene_list)
        full_set = set(self.full_genes)
        self.g2idx = {g: i for i, g in enumerate(self.full_genes)}

        df = pd.read_csv(edge_tsv_path, sep="\t", usecols=["from", "to"])
        df = df[df["from"].isin(full_set) & df["to"].isin(full_set)]

        self.sub_genes = [g for g in self.full_genes if g in set(df["from"]) | set(df["to"])]
        self.m = len(self.sub_genes)
        self.n = len(self.full_genes)

        print(f"TSVに記述されてる遺伝子の数 {self.m}, 全体の遺伝子の数 {self.n}")

        mask = torch.zeros((self.n, self.n), dtype=torch.float32)
        for _, r in df.iterrows():
            src = self.g2idx[r["from"]]
            tgt = self.g2idx[r["to"]]
            mask[src, tgt] = 1.0

        self.register_buffer("mask", mask)
        self.register_buffer("latent_scale", torch.tensor(1.0 / math.sqrt(max(self.rank, 1))))
        self.register_buffer("W", torch.zeros(self.n, self.n, dtype=torch.float32))

        self._cached_W_for_penalty: Optional[torch.Tensor] = None

        self.f_theta = FactorMLP(
            num_nodes=self.n,
            out_dim=self.rank,
            hidden_dim=hidden_dim,
            gene_emb_dim=gene_emb_dim,
            dropout=dropout,
        )
        self.f_phi = FactorMLP(
            num_nodes=self.n,
            out_dim=self.rank,
            hidden_dim=hidden_dim,
            gene_emb_dim=gene_emb_dim,
            dropout=dropout,
        )

        # ---- time embedding modules ----
        self.time_embedding = ODETimeEmbedding(self.time_hidden_dim)

        # map global time embedding to per-gene additive bias
        self.time_to_x = nn.Sequential(
            nn.Linear(self.time_hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n),
        )

        self.b = nn.Parameter(torch.zeros(self.n, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.full((self.n,), 0.1, dtype=torch.float32))

        self.to(self.device)
        print(
            f"[GeneODE-Time] genes={self.n} edges_in_mask={int(mask.sum().item())} "
            f"rank={self.rank} soft={self.soft}"
        )

    def _compute_factors(self, x: torch.Tensor):
        A = self.f_theta(x)  # (B, N, R)
        B = self.f_phi(x)    # (B, N, R)
        return A, B

    def _compute_Wx_lowrank(self, x: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        u = torch.einsum("bn,bnr->br", x, A)
        Wx = torch.einsum("br,bnr->bn", u, B) * self.latent_scale
        return Wx

    def _batch_mean_W(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        bsz = max(A.shape[0], 1)
        W_mean = torch.einsum("bnr,bmr->nm", A, B) / float(bsz)
        W_mean = W_mean * self.latent_scale
        return W_mean

    @torch.no_grad()
    def _update_plot_W(self, W_mean_detached: torch.Tensor):
        w_new = W_mean_detached.detach().to(device=self.W.device, dtype=self.W.dtype)

        if self.W.shape != w_new.shape:
            self.W = torch.zeros_like(w_new)
            self.W.copy_(w_new)
            return

        self.W.mul_(self.w_ema).add_(w_new, alpha=1.0 - self.w_ema)

    def _time_condition(self, x: torch.Tensor, t: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        Convert timestep t into per-gene additive conditioning and add it to x.
        Returns x_tilde with same shape as x: (B, N)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=x.device)

        t = t.to(device=x.device)

        if t.dim() == 0:
            t = t.repeat(x.shape[0])
        elif t.dim() > 1:
            t = t.view(-1)

        if t.shape[0] != x.shape[0]:
            if t.numel() == 1:
                t = t.repeat(x.shape[0])
            else:
                raise ValueError(
                    f"Batch mismatch: x.shape[0]={x.shape[0]} but t.shape[0]={t.shape[0]}"
                )

        temb = self.time_embedding(t)              # (B, H)
        time_bias = self.time_to_x(temb)          # (B, N)
        x_tilde = x + self.time_scale * time_bias
        return x_tilde

    def forward(self, x: torch.Tensor, t: Optional[Union[torch.Tensor, int]] = None) -> torch.Tensor:
        x = x.float()

        # backward compatibility:
        # if t is missing, behave like a time-independent conditioned-zero version
        if t is None:
            x_in = x
        else:
            x_in = self._time_condition(x, t)

        A, B = self._compute_factors(x_in)
        Wx = self._compute_Wx_lowrank(x_in, A, B)

        W_mean = self._batch_mean_W(A, B)
        if not self.soft:
            W_mean = W_mean * self.mask

        self._cached_W_for_penalty = W_mean if self.soft else None
        self._update_plot_W(W_mean.detach())

        alpha = F.softplus(Wx + self.b)
        gamma = F.softplus(self.gamma)
        dxdt = alpha - gamma * x

        return dxdt

    def off_mask_penalty(self, norm: str = "l1"):
        if (not self.soft) or (self._cached_W_for_penalty is None):
            return self.b.new_zeros(())

        off = (1.0 - self.mask) * self._cached_W_for_penalty
        norm = norm.lower()
        if norm == "l2":
            return (off ** 2).mean()
        return off.abs().mean()