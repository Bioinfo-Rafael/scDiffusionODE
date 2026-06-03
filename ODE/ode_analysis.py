# gene_ode.py  (re-write)

import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union


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
        # マッピング
        # 入力された遺伝子からODEで使用する遺伝子を選択するためのmask
        self.full_mask = torch.tensor(
            [g in self.sub_genes for g in self.full_genes],
            dtype=torch.bool,
            device=self.device,
        )
        
        # 遺伝子の制御関係を保存
        self.g2sub = {g: i for i, g in enumerate(self.sub_genes)}

        # ── 4. エッジマスク (m × m) ────────────────────────────
        mask = torch.zeros((self.m, self.m), dtype=torch.float32)
        for _, r in df.iterrows():
            src = self.g2sub[r["from"]]
            tgt = self.g2sub[r["to"]]
            mask[src, tgt] = 1.0

        # 遺伝子の制御関係が無い場所は重みを0で固定するためのmask
        self.register_buffer("mask", mask)

        # ── 5. 学習パラメタ ───────────────────────────────────
        # SOFT CONSTRAINT処理

        """
        self.mの大きさでパラメタを作っている→tsvファイルとadataの共通部分
        →
        """

        if soft == True:
            self.W = nn.Parameter(torch.randn(self.m, self.m, device=self.device))  # 初期値はランダム
            
        else:
            self.W = nn.Parameter(torch.randn(self.m, self.m, device=self.device) * self.mask)  # 初期値は masked
        self.b = nn.Parameter(torch.zeros(self.m, device=self.device))
        self.gamma = nn.Parameter(torch.ones(self.m, device=self.device))                   # broadcast OK
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (cells, len(gene_list))  — 外部順序そのまま
        戻り値も同形状
        """
        # Ensure input is float32 to match model parameters
        x = x.float()
        
        x_sub = x[:, self.full_mask]                           # (cells, m)

        # SOFT CONSTRAINT処理
        Wx = x_sub @ (self.W if self.soft else (self.W * self.mask)) + self.b          # (cells, m)
        # alpha = torch.exp(Wx) →20251105に変更。これのせいで発散する
        alpha = torch.nn.functional.softplus(Wx)
        dxdt_sub = alpha - self.gamma * x_sub                  # (cells, m)

        out = torch.zeros_like(x)                              # (cells, n)
        out[:, self.full_mask] = dxdt_sub
        return out
    
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
    def __init__(self, ode_model: nn.Module, ml_model: nn.Module, timesteps: int):
        super().__init__()
        assert timesteps > 1, "timesteps must be >= 2"
        self.ODE = ode_model
        self.ML  = ml_model
        self.T   = timesteps                       # total steps

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
        
        ode_out = self.ODE(x)                      # (batch, genes)
        ml_out  = self.ML(x, t, y)                 # (batch, genes) - pass t and y to ML model

        r = self._scheduler(t, device=x.device, dtype=x.dtype)

        # ensure broadcast shape:  (batch, 1) or scalar
        while r.dim() < x.dim():
            r = r.unsqueeze(-1)

        out = r * ode_out + (1.0 - r) * ml_out
        return out



