#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize W(x,t) distributions for math-MLP hybrid fields (20260609).

既存 visualization_analysis.py は触らず、別ファイルで W(x,t) の分布を可視化する。
field.compute_W(x, t) を少数サンプルで呼び、以下を出力:

  - histogram     : W 全要素 / on-mask / off-mask の重ね描き（t ごと）
  - heatmap       : W(x,t) の部分行列ヒートマップ（d>HEAT_MAX なら行・列を sampling）
  - onoff boxplot : on-mask vs off-mask の値分布（t ごと）
  - t-trend       : t ごとの |W| 平均（全体 / on / off）の推移

W を明示的に持たない lowrank / lora / lincomb も compute_W が少数サンプルで
materialize する（n_vis ≤ 8 → ≤ 32MB 程度）。
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from local_paths import resolve_path

from guided_diffusion import dist_util
from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260609_mathmlp import build_math_field, MathML_Hybrid, load_hybrid_state_dict

HEAT_MAX = 256        # heatmap 部分行列の最大辺
N_VIS = 4             # W を materialize するサンプル数（メモリ抑制）


def _to_bool(v):
    return str(v).lower() in ("true", "1", "yes", "y", "t")


def load_field_cfg(args):
    cfg = dict(
        model_type=args.model_type, rank=args.rank, K=args.K,
        use_mask_reg=_to_bool(args.use_mask_reg), soft=_to_bool(args.SoftReg),
        use_decay=_to_bool(args.use_decay),
        time_dim=args.time_dim, field_hidden=args.field_hidden,
        field_dropout=args.field_dropout,
        lowrank_penalty_subsample=args.lowrank_penalty_subsample,
        diffusion_steps=args.diffusion_steps,
    )
    if args.field_config and os.path.exists(args.field_config):
        with open(args.field_config) as f:
            loaded = json.load(f)
        for k in cfg:
            if k in loaded:
                cfg[k] = loaded[k]
        print(f"[viz] loaded field config: {cfg}")
    return cfg


def restore_hybrid(args, cfg, gene_list, device):
    field = build_math_field(
        model_type=cfg["model_type"], gene_list=gene_list,
        edge_tsv_path=args.edge_tsv_path, rank=cfg["rank"], K=cfg["K"],
        use_mask=cfg["use_mask_reg"], soft=cfg["soft"], use_decay=cfg["use_decay"],
        time_dim=cfg["time_dim"],
        hidden=cfg["field_hidden"], dropout=cfg["field_dropout"],
        lowrank_penalty_subsample=cfg["lowrank_penalty_subsample"], device=device,
    )
    ml = Cell_Unet(input_dim=len(gene_list)).to(device)
    hybrid = MathML_Hybrid(field, ml, timesteps=cfg["diffusion_steps"]).to(device)

    sd = dist_util.load_state_dict(args.model_path, map_location="cpu")
    # 可視化では strict=False（部分ロード許容）だが、全 key を必ずログ出力する
    load_hybrid_state_dict(hybrid, sd, strict=False, log=print)
    hybrid.eval()
    return field


def get_x_batch(args, gene_list, device):
    adata = sc.read_h5ad(args.data_dir)
    X = adata.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    idx = np.random.choice(X.shape[0], size=min(N_VIS, X.shape[0]), replace=False)
    return torch.tensor(X[idx], dtype=torch.float32, device=device)


def _subidx(d):
    if d <= HEAT_MAX:
        return np.arange(d)
    return np.sort(np.random.choice(d, size=HEAT_MAX, replace=False))


def plot_for_t(field, x, t, mask_np, outdir, tag):
    # lincomb の compute_W は厳密な W ではなく proxy（Σ_k a_k W_k）。タイトルに明記する。
    w_exact = getattr(field, "W_IS_EXACT", True)
    proxy_tag = "" if w_exact else "  [PROXY: Σ a_k W_k, not exact W]"
    W = field.compute_W(x, t).detach().cpu().numpy()      # (n_vis, d, d)
    Wmean = W.mean(axis=0)                                # (d, d)
    flat = W.reshape(-1)

    # ---- histogram (all / on / off) ----
    plt.figure(figsize=(7, 4))
    plt.hist(flat, bins=100, alpha=0.5, label="all", density=True)
    if mask_np is not None:
        on = W[:, mask_np == 1].reshape(-1)
        off = W[:, mask_np == 0].reshape(-1)
        if on.size:
            plt.hist(on, bins=100, alpha=0.5, label="on-mask", density=True)
        if off.size:
            plt.hist(off, bins=100, alpha=0.5, label="off-mask", density=True)
    plt.legend(); plt.title(f"W(x,t) histogram {tag}{proxy_tag}"); plt.xlabel("W value")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"hist_{tag}.png"), dpi=120); plt.close()

    # ---- heatmap (subsampled) ----
    ridx = _subidx(Wmean.shape[0]); cidx = _subidx(Wmean.shape[1])
    sub = Wmean[np.ix_(ridx, cidx)]
    plt.figure(figsize=(6, 5))
    vmax = np.abs(sub).max() + 1e-8
    plt.imshow(sub, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(); plt.title(f"mean W(x,t) heatmap {tag} ({sub.shape[0]}x{sub.shape[1]}){proxy_tag}")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"heat_{tag}.png"), dpi=120); plt.close()

    stats = {"all_absmean": float(np.abs(flat).mean())}
    if mask_np is not None:
        stats["on_absmean"] = float(np.abs(W[:, mask_np == 1]).mean()) if (mask_np == 1).any() else 0.0
        stats["off_absmean"] = float(np.abs(W[:, mask_np == 0]).mean()) if (mask_np == 0).any() else 0.0
    return stats


def plot_gamma(field, outdir):
    """減衰項 softplus(gamma) の分布を可視化。

    gamma は名前 (field.gamma) で参照するので K を変えても呼び出しはズレない
    （gamma は常に shape (d,)、W や coeff の成分数 K とは独立）。
    """
    if not getattr(field, "use_decay", False):
        print("[viz] use_decay=False; skip gamma plot")
        return
    import torch.nn.functional as F
    gamma_raw = field.gamma.detach().cpu()                  # (d,) 生パラメタ
    gamma_pos = F.softplus(gamma_raw).numpy()               # 非負化後（実効的な減衰率）
    plt.figure(figsize=(6, 4))
    plt.hist(gamma_pos, bins=60, alpha=0.7)
    plt.xlabel("softplus(gamma)  (effective decay rate, >=0)")
    plt.ylabel("count")
    plt.title(f"decay term softplus(gamma) distribution  (d={gamma_pos.shape[0]})")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "gamma_decay_hist.png"), dpi=120); plt.close()
    print(f"[viz] gamma softplus: mean={float(gamma_pos.mean()):.4f} "
          f"min={float(gamma_pos.min()):.4f} max={float(gamma_pos.max()):.4f}")


def main():
    args = create_argparser().parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.edge_tsv_path = resolve_path(args.edge_tsv_path)
    np.random.seed(1234); torch.manual_seed(1234)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = args.output_dir if args.output_dir else os.getcwd()
    outdir = os.path.join(base, f"{timestamp}_Wviz")
    os.makedirs(outdir, exist_ok=True)
    print(f"[viz] output dir: {outdir}")

    dist_util.setup_dist()
    device = dist_util.dev()

    cfg = load_field_cfg(args)
    adata = sc.read_h5ad(args.data_dir)
    gene_list = adata.var["gene_name"].tolist()

    field = restore_hybrid(args, cfg, gene_list, device)
    x = get_x_batch(args, gene_list, device)

    # 減衰項 softplus(gamma) の分布（gamma は名前参照なので K 非依存）
    plot_gamma(field, outdir)

    if not getattr(field, "W_IS_EXACT", True):
        print("[viz] ⚠ WARNING: this model's compute_W is a PROXY (Σ_k a_k W_k), "
              "NOT the exact W(x,t). Cross-model W comparison needs care.")

    mask_np = field.mask.detach().cpu().numpy() if field.mask is not None else None
    T = cfg["diffusion_steps"]
    t_values = sorted(set([0, T // 2, T - 1]))

    trend = {"t": [], "all": [], "on": [], "off": []}
    for t in t_values:
        tag = f"t{t}"
        stats = plot_for_t(field, x, t, mask_np, outdir, tag)
        trend["t"].append(t)
        trend["all"].append(stats.get("all_absmean", 0.0))
        trend["on"].append(stats.get("on_absmean", 0.0))
        trend["off"].append(stats.get("off_absmean", 0.0))
        print(f"[viz] {tag}: {stats}")

    # ---- t-trend ----
    plt.figure(figsize=(6, 4))
    plt.plot(trend["t"], trend["all"], "o-", label="all")
    if mask_np is not None:
        plt.plot(trend["t"], trend["on"], "s-", label="on-mask")
        plt.plot(trend["t"], trend["off"], "^-", label="off-mask")
    plt.xlabel("t"); plt.ylabel("mean |W|"); plt.legend()
    plt.title(f"|W(x,t)| vs t ({cfg['model_type']})")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "W_abs_vs_t.png"), dpi=120); plt.close()

    with open(os.path.join(outdir, "W_stats.json"), "w") as f:
        json.dump({"model_type": cfg["model_type"],
                   "w_is_exact": bool(getattr(field, "W_IS_EXACT", True)),
                   "trend": trend,
                   "info": field.get_model_info()}, f, indent=2)
    print(f"[viz] done. Output directory: {outdir}")


def create_argparser():
    defaults = dict(
        model_path="",
        field_config="",
        data_dir="/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad",
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        output_dir="",
        diffusion_steps=1000,
        # fallback field hyperparams
        model_type="lowrank", rank=16, K=8, use_mask_reg=True, SoftReg=True, use_decay=True,
        time_dim=64, field_hidden=256, field_dropout=0.0, lowrank_penalty_subsample=8,
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", default=v)
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser


if __name__ == "__main__":
    main()
