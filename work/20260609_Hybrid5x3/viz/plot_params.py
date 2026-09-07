#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_params.py  (20260609)  — パラメタ / 有効作用素 W の 2 階層可視化
====================================================================

可視化を **2 階層** に分ける:

1. raw parameter visualization（checkpoint の state_dict そのもの。学習が動いているか）
   - param_dist_W.png        : 静的 W 系（geneode W / lincomb expertW_k / matsum A_k / lora W0+Δ_k）の
                               on/off×k 分布。**モデル間比較には使わない**（同一モデルの step 変化を見る）。
   - param_dist_misc.png     : gamma(decay) / b / MLP / cellunet / other。**gamma は必ず misc 側**。
   - scale_vs_step.png       : normed_learned_scale の exp(log_scale) を step に対して。
   ※ gamma は param_dist_misc に含める（専用 gamma_decay 図は作らない）。

2. effective operator visualization（compute_W(x,t) が作る有効作用素。**モデル間比較はこちら**）
   - effective_W_metrics.csv : 全モデル共通の metric（off_on_ratio / sparsity / hoyer / effective_rank ...）
   - W_abs_vs_t.png          : mean|W(x,t)| vs t（all/on/off）
   - W_hist_by_step_t*.png   : 代表 t（0/T//2/T-1）で all/on/off の W 分布を checkpoint 別に重ねる
   - W_off_on_ratio_t_vs_step.png / W_off_mass_fraction_t_vs_step.png
   - W_effective_rank_t_vs_step.png / W_density_abs_gt_*_t_vs_step.png
   - W_meanabs_t_vs_step.png / W_offmask_t_vs_step.png（従来。主役は ratio / mass_fraction）
   - W_sparsity_vs_step.png  : hoyer / density / top1pct を step に対して
   - top_W_edges.csv         : |W| 上位エッジ表
   - branch-specific（model_introspection.py、--skip_branch_specific で抑制）

checkpoint 選択: model000000.pt(raw init) + EMA 5 点（0,N/4,N/2,3N/4,N 近傍）= 最大 6（_restore.find_viz_checkpoints）。
full W_heat（n×n はほぼノイズ）は **default off**（--plot_full_W_heat で出す）。
LinComb の compute_W は proxy（W_IS_EXACT=False）→ タイトル/CSV/ログに [PROXY] 明示。
plain baseline / compute_W 無し / checkpoint 不足でも落ちず、可能な部分だけ描く。
run_all_viz.py 経由（params のみ）でも動く（新 CLI は安全な default）。
"""

import argparse
import csv
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
import model_introspection as MI  # noqa: E402

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


def parse_eps(s):
    """'1e-4,1e-3,1e-2' → [(1e-4,'1e-4'), ...]。token は CSV/図の列名に使う。"""
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append((float(tok), tok))
        except ValueError:
            pass
    return out or [(1e-4, "1e-4"), (1e-3, "1e-3"), (1e-2, "1e-2")]


# =========================================================
# 1. raw parameter visualization
# =========================================================
_W_PREFIX = ("W", "expertW", "A", "Delta")   # ← gamma は入れない（gamma は misc 側）


def _is_w_label(lab):
    return lab.startswith(_W_PREFIX) or lab.endswith("_on") or lab.endswith("_off")


def plot_param_dist_grid(ckpts, cfg, output_dir):
    """行=param 群, 列=checkpoint(init+EMA) の格子で各セルにヒストグラム。

    W 系(on/off×k) は param_dist_W.png（同一モデルの step 変化を見る用途）、
    gamma を含むそれ以外は param_dist_misc.png、scalar(log_scale) は scale_vs_step.png。
    subplot title に label / step / kind / μ / σ を入れる。
    """
    if not ckpts:
        print("[params] no checkpoints for param-dist"); return
    per_ck, scalars_series, union = [], {}, []
    for ck in ckpts:
        sd = R.dist_util.load_state_dict(ck["path"], map_location="cpu")
        groups, scalars = R.extract_param_groups(sd, cfg)
        per_ck.append(groups)
        for lab in groups:
            if lab not in union:
                union.append(lab)
        for lab, v in scalars.items():
            scalars_series.setdefault(lab, []).append((ck["step"], v))

    w_labels = [l for l in union if _is_w_label(l)]
    misc_labels = [l for l in union if not _is_w_label(l)]   # gamma はここに入る

    def grid(labels, fname, title):
        if not labels:
            return
        nr, nc = len(labels), len(ckpts)
        fig, axes = plt.subplots(nr, nc, figsize=(max(2.8 * nc, 4), max(2.1 * nr, 2.5)),
                                 squeeze=False)
        for i, lab in enumerate(labels):
            for j, g in enumerate(per_ck):
                ax = axes[i][j]
                ck = ckpts[j]
                d = g.get(lab)
                if d is not None and d.size:
                    ax.hist(d, bins=60, color="steelblue", alpha=0.85)
                    ax.set_title(f"{lab}\n{ck['label']} [{ck['kind']}]\n"
                                 f"μ={d.mean():.2g} σ={d.std():.2g}", fontsize=6)
                else:
                    ax.text(0.5, 0.5, "-", ha="center", va="center")
                    ax.set_title(f"{lab}\n{ck['label']} [{ck['kind']}]", fontsize=6)
                ax.tick_params(labelsize=5)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(os.path.join(output_dir, fname), dpi=130); plt.close(fig)

    grid(w_labels, "param_dist_W.png", "raw W params (on/off × k) per checkpoint  [同一モデルの step 変化用]")
    grid(misc_labels, "param_dist_misc.png", "raw params (gamma/b/MLP/cellunet/other) per checkpoint")

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
# 2. effective operator visualization
# =========================================================
def plot_W_abs_and_heat(model, X_real, diffusion, output_dir, device,
                        plot_full_W_heat=False, n_vis=4):
    """単一モデルの W_abs_vs_t（all/on/off）。--plot_full_W_heat 指定時のみ full W_heat。"""
    if not R.has_ode_branch(model):
        print("[params] plain baseline: skip W viz"); return
    ode = model.ode_model
    T = diffusion.num_timesteps
    n_vis = min(n_vis, X_real.shape[0])
    x = torch.from_numpy(np.asarray(X_real[:n_vis])).float().to(device)
    mask = MI.model_mask(model)
    proxy = "" if getattr(ode, "W_IS_EXACT", True) else "  [PROXY]"

    trend = {"t": [], "all": [], "on": [], "off": []}
    for t in sorted(set([0, T // 2, T - 1])):
        W, _ = MI.effective_W(model, x, t)
        if W is None:
            print("[params] no W available; skip W_abs/heat"); return
        Wm = W.mean(axis=0)
        aW = np.abs(Wm)
        trend["t"].append(t); trend["all"].append(float(aW.mean()))
        trend["on"].append(float(aW[mask == 1].mean()) if mask is not None and (mask == 1).any() else np.nan)
        trend["off"].append(float(aW[mask == 0].mean()) if mask is not None and (mask == 0).any() else np.nan)

        if plot_full_W_heat:
            vmax = aW.max() + 1e-8
            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(Wm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest", aspect="equal")
            ax.set_axis_off()
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            ax.set_title(f"mean W heatmap t={t} ({Wm.shape[0]}x{Wm.shape[1]}){proxy}")
            fig.savefig(os.path.join(output_dir, f"W_heat_t{t}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)
    if not plot_full_W_heat:
        print("[params] full W_heat skipped (default; pass --plot_full_W_heat to emit n×n heatmaps)")

    plt.figure(figsize=(6, 4))
    plt.plot(trend["t"], trend["all"], "o-", label="all")
    if mask is not None:
        plt.plot(trend["t"], trend["on"], "s-", label="on-mask")
        plt.plot(trend["t"], trend["off"], "^-", label="off-mask")
    plt.xlabel("t"); plt.ylabel("mean |W|"); plt.legend(); plt.title(f"|W(x,t)| vs t{proxy}")
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "W_abs_vs_t.png"), dpi=150); plt.close()
    print("[params] W_abs_vs_t saved")


def compute_effective_W_metrics(ckpts, cfg, gene_list, diffusion, x_real, edge_tsv_path,
                                output_dir, device, model_name, eps_list,
                                n_t_grid=16, max_viz_cells=4, max_svd_dim=512, top_edges=100,
                                write_csv=True, csv_name="effective_W_metrics.csv"):
    """全 checkpoint × t-grid で有効作用素 W(x,t) の metric を計算し CSV / 中間テンソルを返す。

    返り値: dict(M, t_grid, rep_t, Wm_store, is_exact, branch, mode, ok)
      - M[metric] = 2D array [len(t_grid), len(ckpts)]（heatmap 用）
      - Wm_store[(j, t)] = Wm (d,d)（代表 t × checkpoint。hist_by_step / top_edges 用）
    """
    branch = cfg.get("ode_branch", cfg.get("model_type", "geneode"))
    mode = cfg.get("hybrid_norm_mode", "ratio_reg")
    T = diffusion.num_timesteps
    t_grid = list(np.unique(np.linspace(0, T - 1, n_t_grid).round().astype(int)))
    rep_t = sorted(set([0, T // 2, T - 1]))
    all_t = sorted(set(t_grid) | set(rep_t))
    n = min(max_viz_cells, x_real.shape[0])
    x = torch.from_numpy(np.asarray(x_real[:n])).float().to(device)

    metric_keys = (["mean_abs_all", "mean_abs_on", "mean_abs_off", "off_on_ratio", "off_mass_fraction"]
                   + [f"density_abs_gt_{lab}" for _, lab in eps_list]
                   + [f"fraction_abs_lt_{lab}" for _, lab in eps_list]
                   + ["hoyer_sparsity", "effective_rank", "top1pct_mass_fraction"])
    M = {k: np.full((len(t_grid), len(ckpts)), np.nan) for k in metric_keys}
    Wm_store = {}
    rows = []
    is_exact_global = True
    any_W = False

    for j, ck in enumerate(ckpts):
        try:
            model = R.build_model(cfg, ck["path"], gene_list, T, edge_tsv_path, device)
        except Exception as e:  # noqa: BLE001
            print(f"[params] build_model failed for {ck['label']} ({type(e).__name__}: {e}); skip")
            continue
        if not R.has_ode_branch(model):
            print(f"[params] {ck['label']}: plain baseline (no ode_model) → skip effective W")
            continue
        ode = model.ode_model
        if not (hasattr(ode, "compute_W") or getattr(ode, "W", None) is not None):
            print(f"[params] {ck['label']}: no compute_W/W → skip effective W"); continue
        mask = MI.model_mask(model)
        is_exact = bool(getattr(ode, "W_IS_EXACT", True))
        is_exact_global = is_exact_global and is_exact
        any_W = True
        if not is_exact:
            print(f"[params] {ck['label']}: LinComb compute_W is proxy, not exact W(x,t) [PROXY]")

        for t in all_t:
            W, _ = MI.effective_W(model, x, int(t))
            if W is None:
                continue
            Wm = W.mean(axis=0)
            m = MI.w_metrics(Wm, mask=mask, eps_list=eps_list, max_svd_dim=max_svd_dim)
            if t in t_grid:
                i = t_grid.index(t)
                for k in metric_keys:
                    M[k][i, j] = m.get(k, np.nan)
            if t in rep_t:
                Wm_store[(j, t)] = Wm
            row = {"model_name": model_name, "ode_branch": branch, "hybrid_norm_mode": mode,
                   "checkpoint_label": ck["label"], "checkpoint_step": ck["step"],
                   "checkpoint_kind": ck["kind"], "t": int(t), "n_cells_used": int(n),
                   "W_IS_EXACT": bool(is_exact)}
            row.update({k: m.get(k, np.nan) for k in metric_keys})
            rows.append(row)
        del model

    # --- CSV（write_csv=False の heatmap 用呼び出しでは書かない / csv_name で別名可）---
    if write_csv and rows:
        header = (["model_name", "ode_branch", "hybrid_norm_mode", "checkpoint_label",
                   "checkpoint_step", "checkpoint_kind", "t", "n_cells_used"]
                  + metric_keys + ["W_IS_EXACT"])
        with open(os.path.join(output_dir, csv_name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[params] {csv_name} saved ({len(rows)} rows, W_IS_EXACT={is_exact_global})")
    elif not rows:
        print("[params] no effective W rows (plain/no compute_W) → skip CSV")

    return dict(M=M, t_grid=t_grid, rep_t=rep_t, Wm_store=Wm_store,
                is_exact=is_exact_global, branch=branch, mode=mode, ok=any_W,
                gene_list=gene_list, top_edges=top_edges)


def plot_W_metric_heatmaps(res, ckpts, output_dir, eps_list):
    """t（縦）× checkpoint（横）の metric ヒートマップ群。"""
    M, t_grid = res["M"], res["t_grid"]
    proxy = "" if res["is_exact"] else "  [PROXY]"
    _kshort = {"model_init": "init", "ema": "ema", "model": "mdl"}
    xticks = [f"{_kshort.get(c['kind'], c['kind'])}:{c['step']}" for c in ckpts]
    mid = eps_list[min(1, len(eps_list) - 1)][1]   # 1e-3 相当
    plots = [
        ("off_on_ratio", "W_off_on_ratio_t_vs_step.png", "off/on mean|W| ratio"),
        ("off_mass_fraction", "W_off_mass_fraction_t_vs_step.png", "off-mask mass fraction Σ|W_off|/Σ|W|"),
        ("effective_rank", "W_effective_rank_t_vs_step.png", "effective rank of mean W"),
        (f"density_abs_gt_{mid}", f"W_density_abs_gt_{mid}_t_vs_step.png", f"density |W|>{mid}"),
        ("mean_abs_all", "W_meanabs_t_vs_step.png", "mean|W(x,t)|"),
        ("mean_abs_off", "W_offmask_t_vs_step.png", "off-mask mean|W(x,t)|"),
    ]
    for key, fname, ttl in plots:
        Mk = M.get(key)
        if Mk is None or np.all(np.isnan(Mk)):
            continue
        plt.figure(figsize=(max(7, 0.9 * len(ckpts) + 2), 6))
        plt.imshow(Mk, aspect="auto", cmap="viridis", origin="lower", interpolation="nearest")
        plt.colorbar()
        plt.xlabel("checkpoint (kind:step)"); plt.ylabel("diffusion t")
        plt.xticks(range(len(ckpts)), xticks, rotation=45, ha="right", fontsize=7)
        plt.yticks(range(len(t_grid)), [str(int(t)) for t in t_grid], fontsize=7)
        plt.title(f"{ttl}  (t × checkpoint){proxy}")
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches="tight")
        plt.close()
    print(f"[params] W metric t-vs-step heatmaps saved ({len(ckpts)} cols × {len(t_grid)} t)")


def plot_W_sparsity_vs_step(res, ckpts, output_dir):
    """sparsity 系（hoyer / density>1e-3 / top1pct mass）を step に対して（t-grid 平均）。"""
    M = res["M"]
    steps = [c["step"] for c in ckpts]
    keys = [("hoyer_sparsity", "Hoyer sparsity"),
            (next((k for k in M if k.startswith("density_abs_gt_")), "hoyer_sparsity"), "density |W|>eps"),
            ("top1pct_mass_fraction", "top1% mass fraction")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    drew = False
    for ax, (key, lab) in zip(axes, keys):
        Mk = M.get(key)
        if Mk is None or np.all(np.isnan(Mk)):
            ax.text(0.5, 0.5, "n/a", ha="center", va="center"); ax.set_title(lab); continue
        ys = np.nanmean(Mk, axis=0)   # t 平均
        ax.plot(steps, ys, "o-"); ax.set_xlabel("training step"); ax.set_ylabel(lab)
        ax.set_title(f"{lab} vs step (t-mean)"); ax.grid(True, alpha=0.3); drew = True
    fig.suptitle("effective W sparsity vs training step")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if drew:
        fig.savefig(os.path.join(output_dir, "W_sparsity_vs_step.png"), dpi=150)
    plt.close(fig)
    if drew:
        print("[params] W_sparsity_vs_step saved")


def plot_W_hist_by_step(res, ckpts, mask, output_dir, eps_list):
    """代表 t ごとに all/on/off の W 分布を checkpoint 別に重ねる（t ごとに別ファイル）。"""
    Wm_store, rep_t = res["Wm_store"], res["rep_t"]
    proxy = "" if res["is_exact"] else " [PROXY]"
    mid = eps_list[min(1, len(eps_list) - 1)]
    for t in rep_t:
        present = [(j, ck) for j, ck in enumerate(ckpts) if (j, t) in Wm_store]
        if not present:
            continue
        ncol = len(present)
        fig, axes = plt.subplots(1, ncol, figsize=(max(3.2 * ncol, 4), 3.6), squeeze=False)
        for col, (j, ck) in enumerate(present):
            ax = axes[0][col]
            Wm = Wm_store[(j, t)]
            flat = Wm.ravel()
            ax.hist(flat, bins=80, alpha=0.5, density=True, label="all", color="gray")
            on_dens = ratio = np.nan
            if mask is not None and Wm.shape == mask.shape:
                on = Wm[mask == 1]; off = Wm[mask == 0]
                if on.size:
                    ax.hist(on, bins=80, alpha=0.5, density=True, label="on")
                if off.size:
                    ax.hist(off, bins=80, alpha=0.5, density=True, label="off")
                on_dens = MI.density_gt(Wm, mid[0])
                mo = np.abs(off).mean() if off.size else np.nan
                mn = np.abs(on).mean() if on.size else np.nan
                ratio = (mo / mn) if (np.isfinite(mn) and mn > 0) else np.nan
            ax.set_title(f"{ck['label']} [{ck['kind']}]\nt={t} μ={flat.mean():.2g} σ={flat.std():.2g}\n"
                         f"dens>{mid[1]}={on_dens:.2g} off/on={ratio:.2g}", fontsize=6)
            ax.tick_params(labelsize=5); ax.legend(fontsize=5)
        fig.suptitle(f"W(x,t={t}) distribution by checkpoint{proxy}", fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(output_dir, f"W_hist_by_step_t{t}.png"), dpi=140); plt.close(fig)
    print(f"[params] W_hist_by_step saved (rep_t={rep_t})")


def write_top_W_edges(res, ckpts, mask, output_dir):
    """代表 t × checkpoint で |W| 上位エッジ表を CSV に書く（gene 名が無ければ index）。"""
    Wm_store, rep_t = res["Wm_store"], res["rep_t"]
    gene_list = res.get("gene_list")
    top_n = int(res.get("top_edges", 100))
    fields = ["checkpoint_label", "checkpoint_step", "checkpoint_kind", "t", "src_gene", "tgt_gene",
              "W_value", "abs_W", "is_on_mask", "rank", "W_IS_EXACT"]
    n_written = 0
    with open(os.path.join(output_dir, "top_W_edges.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for t in rep_t:
            for j, ck in enumerate(ckpts):
                Wm = Wm_store.get((j, t))
                if Wm is None:
                    continue
                aW = np.abs(Wm)
                flat_idx = np.argsort(aW, axis=None)[::-1][:top_n]
                for rank, fi in enumerate(flat_idx):
                    src, tgt = np.unravel_index(fi, Wm.shape)
                    sg = gene_list[src] if (gene_list is not None and src < len(gene_list)) else int(src)
                    tg = gene_list[tgt] if (gene_list is not None and tgt < len(gene_list)) else int(tgt)
                    on = (int(mask[src, tgt]) if (mask is not None and Wm.shape == mask.shape) else "")
                    w.writerow({"checkpoint_label": ck["label"], "checkpoint_step": ck["step"],
                                "checkpoint_kind": ck["kind"], "t": int(t), "src_gene": sg, "tgt_gene": tg,
                                "W_value": float(Wm[src, tgt]), "abs_W": float(aW[src, tgt]),
                                "is_on_mask": on, "rank": rank, "W_IS_EXACT": bool(res["is_exact"])})
                    n_written += 1
    print(f"[params] top_W_edges.csv saved ({n_written} rows)")


# =========================================================
def main():
    args = create_argparser().parse_args()
    args.data_dir = R.resolve_path(args.data_dir)
    args.edge_tsv_path = R.resolve_path(args.edge_tsv_path)
    if args.skip_W_t_step:   # 後方互換: 旧フラグは effective_W_metrics 抑制に寄せる
        args.skip_effective_W_metrics = True
    eps_list = parse_eps(args.sparsity_eps)
    np.random.seed(1234); torch.manual_seed(1234)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out = setup_output_directory(default_output_dir(args, ROLE), args.model_name)
    print(f"[params] output: {out}")

    adata_real = None
    if args.data_dir and os.path.exists(args.data_dir):
        adata_real = load_real_data(args.data_dir)

    if not (args.model_path and os.path.exists(args.model_path)):
        print(f"[params] no model_path → nothing to do. Output directory: {out}"); return

    cfg = R.normalize_config(R.load_config(args.config))
    cfg["model_name"] = cfg.get("model_name", args.model_name)
    diffusion = R.build_diffusion(cfg["diffusion_steps"])
    # checkpoint 選択: raw init(model000000) + EMA 5 点（0,N/4,N/2,3N/4,N 近傍）= 最大 6
    ckpts = R.find_viz_checkpoints(args.model_path, R.load_config(args.config),
                                   max_ema_points=args.max_ema_points)
    print(f"[params] checkpoints ({len(ckpts)}): "
          + ", ".join(f"{c['label']}[{c['kind']}]" for c in ckpts))

    # 1) raw parameter visualization（gamma は param_dist_misc に入る。専用 gamma_decay 図は出さない）
    if not args.skip_param_dist:
        plot_param_dist_grid(ckpts, cfg, out)

    # 2) effective operator visualization
    gene_list = R.load_gene_list(args.data_dir) if args.data_dir else None
    if gene_list is None or adata_real is None:
        print("[params] no data_dir/gene_list → skip effective W (need data). param-dist done")
        print(f"[params] done. Output directory: {out}"); return

    x_real = R.to_dense(adata_real.X).astype(np.float32)
    model = R.build_model(cfg, args.model_path, gene_list, diffusion.num_timesteps,
                          args.edge_tsv_path, device)
    if not R.has_ode_branch(model):
        print("[params] plain baseline: skip all effective-W viz")
        print(f"[params] done. Output directory: {out}"); return

    if not args.skip_W:
        plot_W_abs_and_heat(model, x_real, diffusion, out, device,
                            plot_full_W_heat=args.plot_full_W_heat, n_vis=args.max_viz_cells)

    if not args.skip_effective_W_metrics:
        mask = MI.model_mask(model)
        # (a) 従来粒度（ckpts=--max_ema_points, n_t_grid）: effective_W_metrics.csv /
        #     W_sparsity_vs_step / W_hist_by_step / top_W_edges。← heatmap はここでは描かない
        res = compute_effective_W_metrics(
            ckpts, cfg, gene_list, diffusion, x_real, args.edge_tsv_path, out, device,
            model_name=cfg["model_name"], eps_list=eps_list, n_t_grid=args.n_t_grid,
            max_viz_cells=args.max_viz_cells, max_svd_dim=args.max_svd_dim, top_edges=args.top_edges,
            write_csv=True, csv_name="effective_W_metrics.csv")
        if res["ok"]:
            plot_W_sparsity_vs_step(res, ckpts, out)
            plot_W_hist_by_step(res, ckpts, mask, out, eps_list)
            write_top_W_edges(res, ckpts, mask, out)
        else:
            print("[params] no effective W available across checkpoints → skip W metric figures")

        # (b) heatmap 専用の高粒度（model000000 + EMA --heatmap_max_ema_points, t=--heatmap_n_t_grid）
        heatmap_ckpts = R.find_viz_checkpoints(
            args.model_path, R.load_config(args.config),
            max_ema_points=args.heatmap_max_ema_points, log_targets=True)
        print(f"[params] heatmap checkpoint selection ({len(heatmap_ckpts)}):")
        for c in heatmap_ckpts:
            print(f"  {c['label']:<18} | step={c['step']:<8} | kind={c['kind']:<10} | path={c['path']}")
        heatmap_res = compute_effective_W_metrics(
            heatmap_ckpts, cfg, gene_list, diffusion, x_real, args.edge_tsv_path, out, device,
            model_name=cfg["model_name"], eps_list=eps_list, n_t_grid=args.heatmap_n_t_grid,
            max_viz_cells=args.max_viz_cells, max_svd_dim=args.max_svd_dim, top_edges=args.top_edges,
            write_csv=True, csv_name="effective_W_metrics_heatmap.csv")
        if heatmap_res["ok"]:
            plot_W_metric_heatmaps(heatmap_res, heatmap_ckpts, out, eps_list)
        else:
            print("[params] no effective W (heatmap grid) → skip heatmaps")

    # branch-specific（checkpoint step 横軸。find_viz_checkpoints の全 ckpt を使用）
    if not args.skip_branch_specific:
        mid_eps, mid_lab = eps_list[min(1, len(eps_list) - 1)]   # 代表 eps（既定 1e-3）
        MI.plot_branch_specific_across_steps(
            ckpts, cfg, gene_list, diffusion, x_real, args.edge_tsv_path, out, device,
            n_t_grid=min(args.n_t_grid, 8), max_cells=args.max_viz_cells,
            sparsity_eps=mid_eps, eps_label=mid_lab)

    print(f"[params] done. Output directory: {out}")


def create_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="")
    p.add_argument("--config", default="", help="exp/field/hybrid config json")
    p.add_argument("--data_dir", default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad")
    p.add_argument("--edge_tsv_path", default="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv")
    p.add_argument("--output_dir", default="", help="未指定なら --model_path から {base}/viz/params を導出")
    p.add_argument("--model_name", default="hybrid5x3")
    # checkpoint 選択
    p.add_argument("--max_ema_points", type=int, default=5, help="EMA checkpoint 点数（init を除く）")
    p.add_argument("--max_ckpts", type=int, default=8, help="(legacy fallback 用)")
    # effective W
    p.add_argument("--n_t_grid", type=int, default=16, help="t×checkpoint metric の t 点数")
    # effective W heatmap だけ高粒度（他可視化には影響しない）
    p.add_argument("--heatmap_max_ema_points", type=int, default=10,
                   help="t×checkpoint heatmap に使う EMA checkpoint 点数（init を除く）")
    p.add_argument("--heatmap_n_t_grid", type=int, default=20,
                   help="t×checkpoint heatmap に使う diffusion t grid 点数")
    p.add_argument("--max_viz_cells", type=int, default=N_VIS, help="compute_W に使う cell 数（少数）")
    p.add_argument("--top_edges", type=int, default=100, help="top_W_edges.csv の上位件数")
    p.add_argument("--sparsity_eps", default="1e-4,1e-3,1e-2", help="density/fraction の閾値（カンマ区切り）")
    p.add_argument("--max_svd_dim", type=int, default=512, help="effective_rank の SVD 行列サイズ上限")
    # full W_heat は default off（n×n はほぼノイズ）
    p.add_argument("--plot_full_W_heat", action="store_true", help="full 解像度 W_heat_t*.png を出す")
    # skip
    p.add_argument("--skip_W", action="store_true")
    p.add_argument("--skip_param_dist", action="store_true")
    p.add_argument("--skip_effective_W_metrics", action="store_true")
    p.add_argument("--skip_branch_specific", action="store_true")
    # 後方互換（旧 --skip_W_t_step は effective_W_metrics 抑制に寄せる）
    p.add_argument("--skip_W_t_step", action="store_true", help="(deprecated) → --skip_effective_W_metrics")
    return p


if __name__ == "__main__":
    main()
