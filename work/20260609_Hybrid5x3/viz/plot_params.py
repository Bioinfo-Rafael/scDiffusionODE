#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_params.py  (20260609)  — 旧 visualization_analysis.py をリネーム
====================================================================

モデルの **学習パラメタ可視化** を担当する:
  - W 可視化           : W_hist_t*.png / W_heat_t*.png(フル解像度・枠線なし) / W_abs_vs_t.png / gamma_decay_hist.png
  - パラメタ分布(学習ステップ横軸): param_dist_W.png / param_dist_misc.png / scale_vs_step.png
  - t×学習ステップ の W ヒートマップ: W_meanabs_t_vs_step.png / W_offmask_t_vs_step.png

役割分割（20260609）:
  - loss 曲線      → plot_loss.py
  - 入出力評価/UMAP → eval_model_io.py
  - velocity UMAP  → plot_velocity_umap.py
モデル復元は _restore.build_model（geneode/lowrank/lincomb/matsum/lora/plain 対応）。
単体実行時、--output_dir 未指定なら --model_path から {base}/viz/params を導出する。
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                       # viz/（_restore）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # Hybrid5x3/（run_paths）
import _restore as R  # noqa: E402
import run_paths  # noqa: E402

HEAT_MAX = 256
N_VIS = 4
ROLE = "params"


# =========================================================
# 汎用
# =========================================================
def default_output_dir(args, role):
    """run_all_viz からは --output_dir 指定あり。単体実行時は --model_path から {base}/viz/{role} を導出。"""
    if args.output_dir:
        return args.output_dir
    base = run_paths.infer_run_base(args.model_path) if getattr(args, "model_path", "") else ""
    return os.path.join(base, "viz", role) if base else os.getcwd()


def setup_output_directory(output_dir, model_name):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir if output_dir else os.getcwd()
    out = os.path.join(base, f"{ts}_viz")
    c = 1
    orig = out
    while os.path.exists(out):
        out = f"{orig}_{c}"; c += 1
    os.makedirs(out, exist_ok=True)
    return out


def load_real_data(data_dir):
    """既に前処理済み（HVG/zscore）の h5ad をそのまま読む（生成サンプルと同じ空間に揃える）。"""
    adata = sc.read_h5ad(R.resolve_path(data_dir))
    label_col = R.auto_label_col(adata)
    adata.obs["final_annotation"] = (adata.obs[label_col].astype(str) if label_col else "real")
    return adata


# =========================================================
# W 可視化（visualize_W 統合）— 新モデル対応
# =========================================================
def _subidx(d):
    return np.arange(d) if d <= HEAT_MAX else np.sort(np.random.choice(d, HEAT_MAX, replace=False))


@torch.no_grad()
def _field_W(model, x, t):
    """field は compute_W、GeneODE は静的 W を batched に展開して返す（(B,d,d)）。"""
    ode = model.ode_model
    if hasattr(ode, "compute_W"):
        return ode.compute_W(x, t).detach().cpu().numpy()
    if hasattr(ode, "W"):
        W = (ode.W * ode.mask if (getattr(ode, "soft", True) is False and hasattr(ode, "mask")) else ode.W)
        return W.detach().cpu().numpy()[None]  # (1,d,d) 静的
    return None


def plot_W(model, X_real, diffusion, output_dir, device):
    if not R.has_ode_branch(model):
        print("[params] plain baseline: skip W viz")
        return
    ode = model.ode_model
    T = diffusion.num_timesteps
    n_vis = min(N_VIS, X_real.shape[0])
    x = torch.from_numpy(np.asarray(X_real[:n_vis])).float().to(device)
    mask = ode.mask.detach().cpu().numpy() if getattr(ode, "mask", None) is not None else None
    w_exact = getattr(ode, "W_IS_EXACT", True)
    proxy = "" if w_exact else "  [PROXY]"

    trend = {"t": [], "all": [], "on": [], "off": []}
    for t in sorted(set([0, T // 2, T - 1])):
        W = _field_W(model, x, t)
        if W is None:
            print("[params] no W available; skip"); return
        flat = W.reshape(-1)
        plt.figure(figsize=(7, 4))
        plt.hist(flat, bins=100, alpha=0.5, density=True, label="all")
        if mask is not None and W.shape[-2:] == mask.shape:
            on = W[..., mask == 1].reshape(-1); off = W[..., mask == 0].reshape(-1)
            if on.size: plt.hist(on, bins=100, alpha=0.5, density=True, label="on-mask")
            if off.size: plt.hist(off, bins=100, alpha=0.5, density=True, label="off-mask")
        plt.legend(); plt.title(f"W(x,t={t}){proxy}"); plt.xlabel("W")
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"W_hist_t{t}.png"), dpi=150); plt.close()

        # フル解像度 (n,n) を方眼紙塗り（枠線・目盛りなし）
        Wm = W.mean(axis=0)
        vmax = np.abs(Wm).max() + 1e-8
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(Wm, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest", aspect="equal")
        ax.set_axis_off()  # 枠線・目盛りなし
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(f"mean W heatmap t={t} ({Wm.shape[0]}x{Wm.shape[1]}){proxy}")
        fig.savefig(os.path.join(output_dir, f"W_heat_t{t}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

        trend["t"].append(t); trend["all"].append(float(np.abs(flat).mean()))
        trend["on"].append(float(np.abs(W[..., mask == 1]).mean()) if mask is not None and (mask == 1).any() else 0.0)
        trend["off"].append(float(np.abs(W[..., mask == 0]).mean()) if mask is not None and (mask == 0).any() else 0.0)

    plt.figure(figsize=(6, 4))
    plt.plot(trend["t"], trend["all"], "o-", label="all")
    if mask is not None:
        plt.plot(trend["t"], trend["on"], "s-", label="on-mask"); plt.plot(trend["t"], trend["off"], "^-", label="off-mask")
    plt.xlabel("t"); plt.ylabel("mean |W|"); plt.legend(); plt.title("|W(x,t)| vs t")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "W_abs_vs_t.png"), dpi=150); plt.close()

    if getattr(ode, "use_decay", False) and getattr(ode, "gamma", None) is not None:
        g = torch.nn.functional.softplus(ode.gamma.detach().cpu()).numpy()
        plt.figure(figsize=(6, 4)); plt.hist(g, bins=60, alpha=0.7)
        plt.xlabel("softplus(gamma)"); plt.ylabel("count"); plt.title("decay term distribution")
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, "gamma_decay_hist.png"), dpi=150); plt.close()
    print(f"[params] W viz saved (w_exact={w_exact})")


# =========================================================
# パラメタ分布（学習ステップ横軸・ヒストグラム格子）
# =========================================================
_W_PREFIX = ("W", "expertW", "A", "Delta")


def _is_w_label(lab):
    return lab.startswith(_W_PREFIX) or lab.endswith("_on") or lab.endswith("_off")


def plot_param_dist_grid(checkpoints, cfg, output_dir):
    """行=param 群, 列=学習ステップ(checkpoint) の格子で各セルにヒストグラム。

    W 系(on/off×k) は param_dist_W.png、それ以外は param_dist_misc.png、
    scalar(log_scale) は scale_vs_step.png。
    """
    if not checkpoints:
        print("[params] no checkpoints for param-dist"); return
    steps, per_step, scalars_series, union = [], [], {}, []
    for step, path in checkpoints:
        sd = R.dist_util.load_state_dict(path, map_location="cpu")
        groups, scalars = R.extract_param_groups(sd, cfg)
        steps.append(step); per_step.append(groups)
        for lab in groups:
            if lab not in union:
                union.append(lab)
        for lab, v in scalars.items():
            scalars_series.setdefault(lab, []).append((step, v))

    w_labels = [l for l in union if _is_w_label(l)]
    misc_labels = [l for l in union if not _is_w_label(l)]

    def grid(labels, fname, title):
        if not labels:
            return
        nr, nc = len(labels), len(steps)
        fig, axes = plt.subplots(nr, nc, figsize=(max(2.6 * nc, 4), max(2.0 * nr, 2.5)),
                                 squeeze=False)
        for i, lab in enumerate(labels):
            for j, g in enumerate(per_step):
                ax = axes[i][j]
                d = g.get(lab)
                if d is not None and d.size:
                    ax.hist(d, bins=60, color="steelblue", alpha=0.85)
                    ax.set_title(f"{lab} | step{steps[j]}\nμ={d.mean():.2g} σ={d.std():.2g}", fontsize=6)
                else:
                    ax.text(0.5, 0.5, "-", ha="center", va="center"); ax.set_title(f"{lab}|step{steps[j]}", fontsize=6)
                ax.tick_params(labelsize=5)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        fig.savefig(os.path.join(output_dir, fname), dpi=130); plt.close(fig)

    grid(w_labels, "param_dist_W.png", "W params (on/off × k) vs training step")
    grid(misc_labels, "param_dist_misc.png", "params (gamma/b/MLP/cellunet/other) vs training step")

    for lab, series in scalars_series.items():
        series.sort()
        xs = [s for s, _ in series]; ys = [v for _, v in series]
        plt.figure(figsize=(6, 4)); plt.plot(xs, ys, "o-")
        plt.xlabel("training step"); plt.ylabel(lab); plt.title(f"{lab} vs training step")
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scale_vs_step.png"), dpi=150); plt.close()
    print(f"[params] param dist saved (W rows={len(w_labels)}, misc rows={len(misc_labels)}, "
          f"scalars={list(scalars_series)})")


# =========================================================
# t（縦）× 学習ステップ（横）の W ヒートマップ（2 枚）
# =========================================================
def plot_W_t_vs_step(checkpoints, cfg, gene_list, diffusion, x_real, edge_tsv_path,
                     output_dir, device, n_t_grid=16, n_vis=4):
    if not checkpoints:
        print("[params] no checkpoints for W t-vs-step"); return
    T = diffusion.num_timesteps
    t_grid = np.unique(np.linspace(0, T - 1, n_t_grid).round().astype(int))
    x = torch.from_numpy(np.asarray(x_real[:n_vis])).float().to(device)
    M_all = np.full((len(t_grid), len(checkpoints)), np.nan)
    M_off = np.full_like(M_all, np.nan)
    for j, (step, path) in enumerate(checkpoints):
        model = R.build_model(cfg, path, gene_list, T, edge_tsv_path, device)
        if not R.has_ode_branch(model):
            continue
        ode = model.ode_model
        mask = ode.mask.detach().cpu().numpy() if getattr(ode, "mask", None) is not None else None
        for i, t in enumerate(t_grid):
            W = _field_W(model, x, int(t))
            if W is None:
                continue
            aW = np.abs(W)
            M_all[i, j] = float(aW.mean())
            if mask is not None and W.shape[-2:] == mask.shape and (mask == 0).any():
                M_off[i, j] = float(aW[..., mask == 0].mean())
    for M, fname, ttl in ((M_all, "W_meanabs_t_vs_step.png", "mean|W(x,t)|"),
                          (M_off, "W_offmask_t_vs_step.png", "off-mask mean|W(x,t)|")):
        if np.all(np.isnan(M)):
            continue
        plt.figure(figsize=(max(4, 0.6 * len(checkpoints) + 2), 6))
        plt.imshow(M, aspect="auto", cmap="viridis", origin="lower", interpolation="nearest")
        plt.colorbar()
        plt.xlabel("training step"); plt.ylabel("diffusion t")
        plt.xticks(range(len(checkpoints)), [s for s, _ in checkpoints], rotation=90, fontsize=7)
        plt.yticks(range(len(t_grid)), [str(int(t)) for t in t_grid], fontsize=7)
        plt.title(f"{ttl}  (t × training step)")
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, fname), dpi=150); plt.close()
    print(f"[params] W t-vs-step saved ({len(t_grid)} t × {len(checkpoints)} steps)")


# =========================================================
def main():
    args = create_argparser().parse_args()
    args.data_dir = R.resolve_path(args.data_dir)
    args.edge_tsv_path = R.resolve_path(args.edge_tsv_path)
    np.random.seed(1234); torch.manual_seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = setup_output_directory(default_output_dir(args, ROLE), args.model_name)
    print(f"[params] output: {out}")

    # real data ロード（W / param 分布で使用）
    adata_real = None
    if args.data_dir and os.path.exists(args.data_dir):
        adata_real = load_real_data(args.data_dir)

    # モデル復元（param 分布 / W 用）
    if args.model_path and os.path.exists(args.model_path):
        cfg = R.normalize_config(R.load_config(args.config))
        diffusion = R.build_diffusion(cfg["diffusion_steps"])
        ckpts = R.find_checkpoints(args.model_path, args.max_ckpts)

        # パラメタ分布（学習ステップ横軸）— gene_list/data 不要、checkpoint 群のみ
        if not args.skip_param_dist:
            plot_param_dist_grid(ckpts, cfg, out)

        gene_list = R.load_gene_list(args.data_dir) if args.data_dir else None
        if gene_list is not None:
            model = R.build_model(cfg, args.model_path, gene_list, diffusion.num_timesteps,
                                  args.edge_tsv_path, device)
            x_real = R.to_dense(adata_real.X).astype(np.float32) if adata_real is not None else None
            if adata_real is not None and not args.skip_W:
                plot_W(model, x_real, diffusion, out, device)
            if x_real is not None and not args.skip_W_t_step:
                plot_W_t_vs_step(ckpts, cfg, gene_list, diffusion, x_real,
                                 args.edge_tsv_path, out, device, args.n_t_grid)
        else:
            print("[params] no data_dir → skip W (need gene_list); param-dist done")
    print(f"[params] done. Output directory: {out}")


def create_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="")
    p.add_argument("--config", default="", help="exp/field/hybrid config json")
    p.add_argument("--data_dir", default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad")
    p.add_argument("--edge_tsv_path", default="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv")
    p.add_argument("--output_dir", default="", help="未指定なら --model_path から {base}/viz/params を導出")
    p.add_argument("--model_name", default="hybrid5x3")
    p.add_argument("--max_ckpts", type=int, default=8, help="param分布 / t×step に使う checkpoint 最大数")
    p.add_argument("--n_t_grid", type=int, default=16, help="t×step ヒートマップの t 点数")
    p.add_argument("--skip_W", action="store_true")
    p.add_argument("--skip_param_dist", action="store_true")
    p.add_argument("--skip_W_t_step", action="store_true")
    return p


if __name__ == "__main__":
    main()
