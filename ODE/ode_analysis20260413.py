# gene_ode.py  (re-write)

# from matplotlib import scale
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union
import numpy as np


class GeneODE(nn.Module):
    """
    dx/dt = exp(x @ (W ⊙ mask) + b) − γ * x
    ─────────────────────────────────────────
    • gene_list  : list[str]  ― adata.var.index 等をそのまま渡す
    • edge_tsv   : 'from' 'to' 2 列だけを持つ TSV
    • 出力形状   : (cells, len(gene_list))  ―  入力と同じ次元で返す
      └── TSV に無い遺伝子は 0 を返す
    """
    def __init__(
        self,
        gene_list: list[str], #解析で入力するデータの遺伝子リスト
        edge_tsv_path: Union[str, Path] = "tf_target_edges.tsv",
        soft = False,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.soft = soft

        # ── 1. 基本リスト・セット ───────────────────────────────
        self.full_genes: list[str] = list(gene_list)          # 外部順序そのまま
        full_set = set(self.full_genes)

        # ── 2. TSV 読み込み & フィルタ ─────────────────────────
        """
        解析で使用する遺伝子リスト中の制御関係をデータベースから抽出する
        1. データベースのtsvファイルを読み取る
        2. データベースの遺伝子リストのうち、解析で使用する遺伝子リストに含まれるものを抽出
        """
        df = pd.read_csv(edge_tsv_path, sep="\t", usecols=["from", "to"])
        df = df[df["from"].isin(full_set) & df["to"].isin(full_set)]

        # ── 3. モデル対象遺伝子（TSV ∩ gene_list） ────────────
        self.sub_genes = [g for g in self.full_genes if g in set(df["from"]) | set(df["to"])] 
        #2で選択された遺伝子のリスト。データベースと解析で使用する遺伝子リストの共通部分
        self.m = len(self.sub_genes) # ODEで使う遺伝子の数
        self.n = len(self.full_genes) #データベースの遺伝子の数
        print(f"ODEで使う遺伝子の数{self.m}, ODEで使う遺伝子の数{self.n}")
        
        # 遺伝子の制御関係を保存
        self.g2sub = {g: i for i, g in enumerate(self.sub_genes)}

        # ── 4. エッジマスク (m × m) ────────────────────────────
        mask = torch.zeros((self.n, self.n), dtype=torch.float32)
        for _, r in df.iterrows():
            src = self.g2sub[r["from"]]
            tgt = self.g2sub[r["to"]]
            mask[src, tgt] = 1.0

        # 遺伝子の制御関係が無い場所は重みを0で固定するためのmask
        self.register_buffer("mask", mask)
        self.mask = self.mask.to(self.device)

        # ── 5. 学習パラメタ ───────────────────────────────────
        # SOFT CONSTRAINT処理

        """
        self.mの大きさでパラメタを作っている→tsvファイルとadataの共通部分
        →
        """

        self.register_buffer("scale", torch.tensor(1.0 / np.sqrt(self.n), device=self.device))
        if soft == True:
            self.W = nn.Parameter(torch.randn(self.n, self.n, device=self.device) * self.scale)  # 初期値はランダム

        else:
            self.W = nn.Parameter(torch.randn(self.n, self.n, device=self.device) * self.mask * self.scale)  # 初期値は masked
        self.b = nn.Parameter(torch.zeros(self.n, device=self.device))
        self.gamma = nn.Parameter(torch.ones(self.n, device=self.device)*0.1)
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (cells, len(gene_list))  — 外部順序そのまま
        戻り値も同形状
        """
        # Ensure input is float32 to match model parameters
        x = x.float()

        # SOFT CONSTRAINT処理
        Wx = x @ (self.W if self.soft else (self.W * self.mask)) + self.b

        # #⭐️新規変更点20260413(1) expに戻す
        # alpha =  torch.exp(Wx * self.scale) #torch.nn.functional.softplus(Wx)

        # #⭐️新規変更点20260413(2) expだとinfになり発散するので、softplusに変更
        alpha = torch.nn.functional.softplus(Wx) #Wx

        gamma = torch.nn.functional.softplus(self.gamma) #γは正の値に制約
        dxdt_sub = alpha - gamma * x 

        out = torch.zeros_like(x)                              # (cells, n)
        out = dxdt_sub
        return out

    # def forward(self, x):
    #     x = x.float()
    #     _finite_report("ode_input_x", x)

    #     z = x @ (self.W if self.soft else (self.W * self.mask)) + self.b
    #     _finite_report("Wx", z)

    #     z_scaled = z * self.scale
    #     _finite_report("Wx_scaled", z_scaled)

    #     alpha = torch.exp(z_scaled)
    #     _finite_report("alpha_after_exp", alpha)

    #     gamma = F.softplus(self.gamma)
    #     dxdt = alpha - gamma * x
    #     _finite_report("dxdt", dxdt)

    #     return dxdt

    # SOFT CONSTRAINT処理
    def off_mask_penalty(self, norm: str = 'l1'):
        """
        既知エッジマスク mask の 0 の部分 (=未知エッジ部分) の W を対象にした正則化
        norm='l1' または 'l2'
        """
        off = (1.0 - self.mask) * self.W  # mask が 0 の部分のみ残す
        if norm.lower() == 'l2':
            return (off ** 2).mean()
        else:
            # L1
            return off.abs().mean() #sum→meanに変更


class ODE_ML_Hybrid(nn.Module):
    """
    out = r(t) * ODE(x) + (1-r(t)) * ML(x)
    ──────────────────────────────────────
    • ODE : GeneODE インスタンスなど (len(gene_list) 出力)
    • ML  : 任意の PyTorch NN (同じ次元を出力するもの)
    • r(t): 0 → 1 の線形スケジューラ
      └── t = 0        → r = 1   (ODE100%)
      └── t = T-1      → r = 0   (ML100%)
    """
    def __init__(self, ode_model: nn.Module, ml_model: nn.Module, timesteps: int, norm = 1):
        super().__init__()
        assert timesteps > 1, "timesteps must be >= 2"
        self.ode_model = ode_model
        self.ml_model  = ml_model
        self.T   = timesteps                       # total steps
        self.norm = norm

    # ---------------------------------------------------------------------
    def _scheduler(self, t: Union[torch.Tensor, int], device, dtype) -> torch.Tensor:
        """
        Linear r ∈ [0,1]; broadcastable to (batch, …, genes)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)
        t = t.to(dtype=dtype, device=device)
        r = 1.0 - t / (self.T - 1)
        return r

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor, t: Union[torch.Tensor, int], y=None):
        """
        x : (batch, genes)  — same dim as gene_list
        t : scalar int or (batch,) tensor of time indices 0 … T-1
        y : optional conditioning (unused in this implementation)
        """
        # Ensure input is float32 to match model parameters
        x = x.float()
        
        ode_out = self.ode_model(x)                      # (batch, genes)
        ml_out  = self.ml_model(x, t, y)                 # (batch, genes) - pass t and y to ML model

        r = self._scheduler(t, device=x.device, dtype=x.dtype)

        # ensure broadcast shape:  (batch, 1) or scalar
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)

        # 各サンプルごとにL2正規化
        ode_unit = F.normalize(ode_out, p=2, dim=-1, eps=1e-8) * self.norm
        ml_unit  = F.normalize(ml_out,  p=2, dim=-1, eps=1e-8) * self.norm

        out = (r * ode_unit + (1.0 - r) * ml_unit) 

        if torch.isnan(ode_out).any() or torch.isnan(ml_out).any() or torch.isnan(out).any():
            print(f"⚠️ NaN detected in hybrid forward | ode:{torch.isnan(ode_out).any()} ml:{torch.isnan(ml_out).any()} out:{torch.isnan(out).any()}")


        return out
    # def forward(self, x, t, y=None):
    #     x = x.float()
    #     _finite_report("hybrid_input_x", x, t)

    #     ode_out = self.ode_model(x)
    #     _finite_report("ode_out", ode_out, t)

    #     ml_out = self.ml_model(x, t, y)
    #     _finite_report("ml_out", ml_out, t)

    #     r = self._scheduler(t, device=x.device, dtype=x.dtype)
    #     while r.dim() < x.dim():
    #         r = r.unsqueeze(-1)

    #     ode_unit = F.normalize(ode_out, p=2, dim=-1, eps=1e-8)
    #     ml_unit  = F.normalize(ml_out,  p=2, dim=-1, eps=1e-8)

    #     _finite_report("ode_unit", ode_unit, t)
    #     _finite_report("ml_unit", ml_unit, t)

    #     out = r * ode_unit + (1.0 - r) * ml_unit
    #     _finite_report("hybrid_out", out, t)
    #     return out

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



import math


class SmallScaleNet(nn.Module):
    """
    小さい特徴量だけから scale を決める NN
    出力は [scale_min, scale_max] に制限
    """
    def __init__(
        self,
        hidden_dim: int = 32,
        scale_min: float = 0.5,
        scale_max: float = 8.0,
        init_scale: float = 3.0,
    ):
        super().__init__()
        assert scale_max > scale_min
        assert scale_min < init_scale < scale_max

        self.scale_min = scale_min
        self.scale_max = scale_max

        # 入力特徴:
        # [t_norm, x_mean, x_std, x_rms, ode_norm, ml_norm] の 6 次元
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # 初期出力が init_scale 付近になるように最後の bias を初期化
        p = (init_scale - scale_min) / (scale_max - scale_min)
        p = min(max(p, 1e-4), 1.0 - 1e-4)
        init_bias = math.log(p / (1.0 - p))

        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, init_bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        T: int,
        ode_out: torch.Tensor,
        ml_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        戻り値: (batch, 1)
        """
        bsz = x.shape[0]
        device = x.device
        dtype = x.dtype

        if not torch.is_tensor(t):
            t = torch.tensor([t], device=device, dtype=dtype)

        t = t.to(device=device, dtype=dtype)

        if t.dim() == 0:
            t = t.view(1).expand(bsz)
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(bsz)
        elif t.dim() == 1 and t.shape[0] == bsz:
            pass
        else:
            t = t.reshape(bsz)

        t_norm = (t / max(T - 1, 1)).unsqueeze(-1)  # (batch, 1)

        # scale 用の特徴量は detach しておくと、
        # 「scale を上げるために本体が norm をいじる」抜け道を減らせる
        xd = x.detach()
        od = ode_out.detach()
        md = ml_out.detach()

        x_mean = xd.mean(dim=-1, keepdim=True)
        x_std  = xd.std(dim=-1, keepdim=True).clamp_min(1e-6)
        x_rms  = xd.pow(2).mean(dim=-1, keepdim=True).sqrt()

        gene_dim = x.shape[-1]
        denom = math.sqrt(max(gene_dim, 1))
        ode_norm = od.norm(p=2, dim=-1, keepdim=True) / denom
        ml_norm  = md.norm(p=2, dim=-1, keepdim=True) / denom

        feat = torch.cat(
            [t_norm, x_mean, x_std, x_rms, ode_norm, ml_norm],
            dim=-1
        )  # (batch, 6)

        z = self.net(feat)
        scale = self.scale_min + (self.scale_max - self.scale_min) * torch.sigmoid(z)
        return scale


class ODE_ML_HybridLearnedScale(nn.Module):
    """
    out = scale(x,t,...) * [ r(t) * normalize(ODE(x)) + (1-r(t)) * normalize(ML(x,t)) ]
    """
    def __init__(
        self,
        ode_model: nn.Module,
        ml_model: nn.Module,
        timesteps: int,
        scale_hidden_dim: int = 32,
        scale_min: float = 0.5,
        scale_max: float = 8.0,
        init_scale: float = 3.0,
    ):
        super().__init__()
        assert timesteps > 1, "timesteps must be >= 2"
        self.ode_model = ode_model
        self.ml_model = ml_model
        self.T = timesteps

        self.scale_net = SmallScaleNet(
            hidden_dim=scale_hidden_dim,
            scale_min=scale_min,
            scale_max=scale_max,
            init_scale=init_scale,
        )

    def _scheduler(self, t: Union[torch.Tensor, int], device, dtype) -> torch.Tensor:
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)
        t = t.to(dtype=dtype, device=device)
        r = 1.0 - t / (self.T - 1)
        return r

    def forward(self, x: torch.Tensor, t: Union[torch.Tensor, int], y=None):
        x = x.float()

        ode_out = self.ode_model(x)       # (batch, genes)
        ml_out  = self.ml_model(x, t, y)  # (batch, genes)

        r = self._scheduler(t, device=x.device, dtype=x.dtype)
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)

        ode_unit = F.normalize(ode_out, p=2, dim=-1, eps=1e-8)
        ml_unit  = F.normalize(ml_out,  p=2, dim=-1, eps=1e-8)

        scale = self.scale_net(
            x=x,
            t=t,
            T=self.T,
            ode_out=ode_out,
            ml_out=ml_out,
        )  # (batch, 1)

        out = scale * (r * ode_unit + (1.0 - r) * ml_unit)

        if (
            torch.isnan(ode_out).any()
            or torch.isnan(ml_out).any()
            or torch.isnan(scale).any()
            or torch.isnan(out).any()
        ):
            print(
                f"⚠️ NaN detected | "
                f"ode:{torch.isnan(ode_out).any()} "
                f"ml:{torch.isnan(ml_out).any()} "
                f"scale:{torch.isnan(scale).any()} "
                f"out:{torch.isnan(out).any()}"
            )

        return out