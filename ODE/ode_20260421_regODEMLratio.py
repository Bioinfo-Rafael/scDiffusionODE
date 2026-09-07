# gene_ode.py  (re-write)

# from matplotlib import scale
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union
import numpy as np
import math


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
        self.register_buffer("scale", torch.tensor(1.0 / np.sqrt(self.n), device=self.device))
        if soft == True:
            self.W = nn.Parameter(torch.randn(self.n, self.n, device=self.device) * self.scale)  # 初期値はランダム

        else:
            self.W = nn.Parameter(torch.randn(self.n, self.n, device=self.device) * self.mask * self.scale)  # 初期値は masked
        self.b = nn.Parameter(torch.zeros(self.n, device=self.device))
        self.gamma = nn.Parameter(torch.ones(self.n, device=self.device)*0.1)  
        # self.beta = nn.Parameter(torch.ones(self.n, device=self.device))
    # ---------------------------------------------------------------------

        # GeneODE.__init__ の末尾あたりに追加
        self.ratio_reg_weight = 1.0 # ODE出力とML出力のノルム比を正則化する項の重み。0なら正則化なし
        self.ratio_reg_target = 1.0 # 目標とするODE出力とML出力のノルム比（ODEノルム / MLノルム）。1.0なら同程度の大きさを目指す
        self.ratio_reg_eps = 1e-8
        self._cached_ratio_reg = None


    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        """
        x : (cells, len(gene_list))  — 外部順序そのまま
        t : 省略可（デフォルト None）。GeneODE は時刻非依存なので無視する。
            t を渡す hybrid（MathML_Hybrid 等）に GeneODE をそのまま差し込めるよう、
            互換のため引数だけ受ける（既存呼び出し ode_model(x) は不変）。
        戻り値も同形状
        """
        # t は時刻非依存モデルのため未使用（互換目的の引数）
        del t
        # Ensure input is float32 to match model parameters
        x = x.float()
        
        Wx = x @ (self.W if self.soft else (self.W * self.mask)) + self.b

        alpha =  torch.nn.functional.softplus(Wx * self.scale) #Wx
        gamma = torch.nn.functional.softplus(self.gamma) 
        dxdt_sub = alpha - gamma * x 

        out = torch.zeros_like(x)                              # (cells, n)
        out = dxdt_sub
  
        return out
    
    # SOFT CONSTRAINT処理
    # 20260421に変更
    def off_mask_penalty(self, norm: str = "l1"):
        off = (1.0 - self.mask) * self.W
        if norm.lower() == "l2":
            base = (off ** 2).mean()
        else:
            base = off.abs().mean()

        aux = getattr(self, "_cached_ratio_reg", None)
        if aux is None or self.ratio_reg_weight <= 0:
            return base

        return base + self.ratio_reg_weight * aux

class ODE_ML_Hybrid(nn.Module):
    def __init__(self, ode_model: nn.Module, ml_model: nn.Module, timesteps: int):
        super().__init__()
        assert timesteps > 1
        self.ode_model = ode_model
        self.ml_model = ml_model
        self.T = timesteps

    def _scheduler(self, t, device, dtype):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=dtype, device=device)
        t = t.to(dtype=dtype, device=device)
        return 1.0 - t / (self.T - 1)

    def _norm_ratio_penalty(self, ode_out, ml_out):
        eps = getattr(self.ode_model, "ratio_reg_eps", 1e-8)

        ode_norm = ode_out.norm(p=2, dim=-1).clamp_min(eps)
        ml_norm  = ml_out.norm(p=2, dim=-1).clamp_min(eps)

        log_ratio = torch.log(ode_norm) - torch.log(ml_norm)
        target = getattr(self.ode_model, "ratio_reg_target", 1.0)
        target_log = math.log(max(target, eps))

        return ((log_ratio - target_log) ** 2).mean()

    def forward(self, x, t, y=None):
        x = x.float()

        ode_out = self.ode_model(x)
        ml_out  = self.ml_model(x, t, y)

        if self.training:
            self.ode_model._cached_ratio_reg = self._norm_ratio_penalty(ode_out, ml_out)
        else:
            self.ode_model._cached_ratio_reg = None

        r = self._scheduler(t, device=x.device, dtype=x.dtype)
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)

        out = r * ode_out + (1.0 - r) * ml_out
        return out


