#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_model_io.py  (20260609)  — 旧 analyze_hybrid_unified.py をリネーム
======================================================================

モデルの **入出力評価** を担当する（旧 analyze_hybrid.py weighted + replot_unweighted を統合）。
モデル出力と「真の付加ノイズ ε」を比較する（q_sample で ε 既知）。
real vs gen の UMAP は plot_velocity_umap.py（viz/velocity）へ移設した（ここには無い）。

- --weighting {weighted, unweighted, both}:
    weighted   : ode_c = r(t)·ODE(x_t), ml_c = (1-r(t))·ML(x_t)
    unweighted : ode_c = ODE(x_t),      ml_c = ML(x_t)
  既存メトリクス（norm_ratio / alignment=<·,ε>/||ε||² / mse_norm）を t ごとに mean±std。

- 新規「次元方向相関」: 各サンプルで pearson( out(1024), ε(1024) )（次元方向）。
    corr_dim_vs_t.png    : t ごと mean±std 曲線（hybrid / ode / ml）
    corr_scatter_t{t}.png: 代表 t（0, T//2, T-1）で out vs ε の散布図（点=各次元、Pearson r 注記）

モデル復元は _restore.build_model（geneode/lowrank/lincomb/matsum/lora/plain 対応）。
plain（ode_model 無）は branch 解析を省き、hybrid(=Cell_Unet 出力) のみ評価。
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                       # viz/（_restore）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # Hybrid5x3/（run_paths）
import _restore as R  # noqa: E402
import run_paths  # noqa: E402

ROLE = "eval_io"


def flatten_norm(x, eps=1e-12):
    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).clamp_min(eps)


def flatten_dot(x, y):
    return (x.reshape(x.shape[0], -1) * y.reshape(y.shape[0], -1)).sum(dim=1)


def dim_corr(a, b, eps=1e-8):
    """各サンプルで次元方向の Pearson 相関（(B,) を返す）。"""
    a = a.reshape(a.shape[0], -1); b = b.reshape(b.shape[0], -1)
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = (a.norm(dim=1) * b.norm(dim=1)).clamp_min(eps)
    return num / den


class RunningMoments:
    def __init__(self):
        self.sum = {}; self.sumsq = {}; self.n = {}

    def update(self, values):
        for k, v in values.items():
            v = v.detach().float().reshape(-1)
            self.sum[k] = self.sum.get(k, 0.0) + v.sum().item()
            self.sumsq[k] = self.sumsq.get(k, 0.0) + (v * v).sum().item()
            self.n[k] = self.n.get(k, 0) + v.numel()

    def finalize(self):
        out = {}
        for k in self.sum:
            n = max(self.n[k], 1)
            mean = self.sum[k] / n
            var = max(self.sumsq[k] / n - mean * mean, 0.0)
            out[f"{k}_mean"] = mean
            out[f"{k}_std"] = var ** 0.5
        return out


def build_timestep_list(num_timesteps, num_t_points):
    if num_t_points <= 0 or num_t_points >= num_timesteps:
        return list(range(num_timesteps))
    v = np.unique(np.round(np.linspace(0, num_timesteps - 1, num_t_points)).astype(int))
    return v.tolist()


def metric_set(ode_c, ml_c, hybrid, noise, prefix=""):
    eps_norm = flatten_norm(noise)
    eps_sq = flatten_dot(noise, noise).clamp_min(1e-12)
    vals = {
        f"{prefix}hybrid_mse_norm": flatten_dot(hybrid - noise, hybrid - noise) / eps_sq,
        f"{prefix}hybrid_corr": dim_corr(hybrid, noise),
    }
    if ode_c is not None:
        vals.update({
            f"{prefix}ode_norm_ratio": flatten_norm(ode_c) / eps_norm,
            f"{prefix}ml_norm_ratio": flatten_norm(ml_c) / eps_norm,
            f"{prefix}ode_align": flatten_dot(ode_c, noise) / eps_sq,
            f"{prefix}ml_align": flatten_dot(ml_c, noise) / eps_sq,
            f"{prefix}ode_mse_norm": flatten_dot(ode_c - noise, ode_c - noise) / eps_sq,
            f"{prefix}ml_mse_norm": flatten_dot(ml_c - noise, ml_c - noise) / eps_sq,
            f"{prefix}ode_corr": dim_corr(ode_c, noise),
            f"{prefix}ml_corr": dim_corr(ml_c, noise),
        })
    return vals


@torch.no_grad()
def main():
    args = create_argparser().parse_args()
    args.data_dir = R.resolve_path(args.data_dir)
    args.edge_tsv_path = R.resolve_path(args.edge_tsv_path)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # --output_dir 未指定なら --model_path から {base}/viz/eval_io を導出（単体実行用）
    base_out = args.output_dir
    if not base_out:
        b = run_paths.infer_run_base(args.model_path) if args.model_path else ""
        base_out = os.path.join(b, "viz", ROLE) if b else os.getcwd()
    out_dir = os.path.join(base_out, f"{ts}_analyze")
    os.makedirs(out_dir, exist_ok=True)
    # サブディレクトリ: corr_* は corr/、alignment/mse/norm_ratio は alignment_mse/。.json/.csv は out_dir 直下。
    corr_dir = os.path.join(out_dir, "corr"); os.makedirs(corr_dir, exist_ok=True)
    align_dir = os.path.join(out_dir, "alignment_mse"); os.makedirs(align_dir, exist_ok=True)
    print(f"[analyze] output: {out_dir}")

    cfg = R.normalize_config(R.load_config(args.config))
    diffusion = R.build_diffusion(cfg["diffusion_steps"])
    gene_list = R.load_gene_list(args.data_dir)
    model = R.build_model(cfg, args.model_path, gene_list, diffusion.num_timesteps,
                          args.edge_tsv_path, device)
    has_branch = R.has_ode_branch(model)

    adata = sc.read_h5ad(args.data_dir)
    n = min(args.max_cells, adata.n_obs)
    idx = np.random.choice(adata.n_obs, n, replace=False)
    X = R.to_dense(adata.X[idx]).astype(np.float32)
    batches = [X[s:s + args.batch_size] for s in range(0, n, args.batch_size)]

    modes = ["weighted", "unweighted"] if args.weighting == "both" else [args.weighting]
    T = diffusion.num_timesteps
    # corr_dim_vs_t は 0..T-1 を 100 間隔（+終端 T-1）: T=1000 → 0,100,...,900,999
    dim_t = sorted(set(list(range(0, T, 100)) + [T - 1]))
    # corr_scatter は等間隔 5 点: T=1000 → 0,249,499,749,999
    rep_t = sorted(set(np.linspace(0, T - 1, 5).astype(int).tolist()))
    # 本ループは両グリッドの和集合を計算（corr 曲線は dim_t、散布図は rep_t を使う）
    timesteps = sorted(set(dim_t) | set(rep_t))
    scatter_store = {m: {t: {"out": [], "eps": []} for t in rep_t} for m in modes}

    rows = []
    for t in timesteps:
        mom = RunningMoments()
        for xb_np in batches:
            x0 = torch.from_numpy(xb_np).to(device).float()
            B = x0.shape[0]
            for _ in range(args.num_noise_draws):
                t_raw = torch.full((B,), int(t), device=device, dtype=torch.long)
                noise = torch.randn_like(x0)
                x_t = diffusion.q_sample(x0, t_raw, noise=noise)
                model_t = diffusion._scale_timesteps(t_raw)

                if has_branch:
                    ode_raw = model.ode_model(x_t, model_t)
                    ml_raw = model.ml_model(x_t, model_t)
                    r = model._scheduler(model_t, device=x_t.device, dtype=x_t.dtype)
                    while r.dim() < x_t.dim():
                        r = r.unsqueeze(-1)
                else:
                    ode_raw = ml_raw = r = None

                row_vals = {"effective_r": r.reshape(B, -1)[:, 0]} if has_branch else {}
                for m in modes:
                    if has_branch:
                        if m == "weighted":
                            ode_c, ml_c = r * ode_raw, (1.0 - r) * ml_raw
                        else:
                            ode_c, ml_c = ode_raw, ml_raw
                        hybrid = ode_c + ml_c
                    else:
                        ode_c = ml_c = None
                        hybrid = model(x_t, model_t)
                    pfx = f"{m}_" if len(modes) > 1 else ""
                    row_vals.update(metric_set(ode_c, ml_c, hybrid, noise, prefix=pfx))
                    if t in rep_t and len(scatter_store[m][t]["out"]) < args.scatter_rows:
                        scatter_store[m][t]["out"].append(hybrid[0].detach().cpu().numpy())
                        scatter_store[m][t]["eps"].append(noise[0].detach().cpu().numpy())
                mom.update(row_vals)
        row = {"t": int(t), "reverse_step": int(T - 1 - t)}
        row.update(mom.finalize())
        rows.append(row)
        print(f"[analyze] t={t} done")

    df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "timestep_metrics.csv"), index=False)
    df_dim = df[df["t"].isin(dim_t)]   # corr_dim_vs_t は 100 間隔グリッドのみ使用

    # ---- metric curves ----
    for m in modes:
        pfx = f"{m}_" if len(modes) > 1 else ""
        # norm_ratio / alignment / mse は alignment_mse/ サブディレクトリへ
        _plot_curve(df, [f"{pfx}ode_norm_ratio", f"{pfx}ml_norm_ratio"],
                    "norm ratio (||out||/||eps||)", os.path.join(align_dir, f"{m}_norm_ratio.png"))
        _plot_curve(df, [f"{pfx}ode_align", f"{pfx}ml_align"],
                    "alignment <out,eps>/||eps||^2", os.path.join(align_dir, f"{m}_alignment.png"))
        _plot_curve(df, [f"{pfx}hybrid_mse_norm", f"{pfx}ode_mse_norm", f"{pfx}ml_mse_norm"],
                    "MSE/||eps||^2", os.path.join(align_dir, f"{m}_mse_norm.png"))
        # 次元方向相関 曲線（0..T-1 を 100 間隔）→ corr/ サブディレクトリへ
        _plot_curve(df_dim, [f"{pfx}hybrid_corr", f"{pfx}ode_corr", f"{pfx}ml_corr"],
                    "dim-wise corr(out, eps)", os.path.join(corr_dir, f"{m}_corr_dim_vs_t.png"), std=True)
        # 散布図（等間隔 5 点 t）→ corr/ サブディレクトリへ
        for t in rep_t:
            o = scatter_store[m][t]["out"]; ev = scatter_store[m][t]["eps"]
            if not o:
                continue
            ox = np.concatenate([a.ravel() for a in o]); ey = np.concatenate([a.ravel() for a in ev])
            rr = np.corrcoef(ox, ey)[0, 1]
            plt.figure(figsize=(5.5, 5.5))
            plt.scatter(ey, ox, s=4, alpha=0.3)
            lim = [min(ey.min(), ox.min()), max(ey.max(), ox.max())]
            plt.plot(lim, lim, "r--", alpha=0.6)
            plt.xlabel("true noise eps (per dim)"); plt.ylabel("model output (per dim)")
            plt.title(f"{m}: out vs eps  t={t}  (Pearson r={rr:.3f})")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(os.path.join(corr_dir, f"{m}_corr_scatter_t{t}.png"), dpi=150); plt.close()

    # real vs gen の UMAP は plot_velocity_umap.py（viz/velocity）へ移設した（eval_io には置かない）。

    with open(os.path.join(out_dir, "analysis_config.json"), "w") as f:
        json.dump({"args": vars(args), "cfg": cfg, "has_branch": has_branch}, f, indent=2)
    print(f"[analyze] done. Output directory: {out_dir}")


def _plot_curve(df, cols, ylabel, path, std=False):
    cols = [c for c in cols if f"{c}_mean" in df.columns]
    if not cols:
        return
    plt.figure(figsize=(8, 5))
    for c in cols:
        x = df["t"].to_numpy(); y = df[f"{c}_mean"].to_numpy()
        plt.plot(x, y, label=c)
        if std and f"{c}_std" in df.columns:
            s = df[f"{c}_std"].to_numpy(); plt.fill_between(x, y - s, y + s, alpha=0.15)
    plt.xlabel("forward diffusion timestep t"); plt.ylabel(ylabel)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def create_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--data_dir", default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad")
    p.add_argument("--edge_tsv_path", default="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv")
    p.add_argument("--output_dir", default="")
    p.add_argument("--num_t_points", type=int, default=16)
    p.add_argument("--num_noise_draws", type=int, default=4)
    p.add_argument("--max_cells", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--weighting", choices=["weighted", "unweighted", "both"], default="both")
    p.add_argument("--scatter_rows", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    return p


if __name__ == "__main__":
    main()
