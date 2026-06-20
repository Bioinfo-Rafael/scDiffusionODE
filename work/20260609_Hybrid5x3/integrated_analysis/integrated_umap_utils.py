#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
integrated_umap_utils.py  (20260609 / integrated_analysis)
==========================================================

`runs/` 配下の 20 構成（5 ODE branch × 4 variant）の generated サンプルを使い、
**2 系統**の UMAP 可視化を作る helper 群。

  A) per-model 個別比較: 各 config_label について real(既定 50000) + その model の gen(既定 3000)
     だけで個別 UMAP を計算 → 20 枚の PNG ＋ それを 1 枚にまとめた facets。
  B) integrated 統合比較: real(既定 全部 ~150k) + 全 model の gen(各 500) を 1 つの UMAP に。
     overlay 3 種（real_annotation / origin / branch_marker_variant_color）＋ 横並び summary。

設計方針:
- **既存 pipeline / training / sampling / viz は import も変更もしない**（torch 不要）。
  各 run の成果物ファイル（samples.npz / exp_config.json / checkpoint の存在）だけ読む。
- どの run（どの timestamp dir）を参照したかを必ず JSON/CSV に残す（再現性）。
- notebook（`tune_integrated_umap.ipynb`）と CLI（`run_integrated_umap.py`）が共有。

このファイルは `integrated_analysis/` 配下に閉じており、既存コードへ副作用を持たない。
"""

import os
import re
import sys
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# パス・定数
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
HYBRID_DIR = os.path.dirname(HERE)          # work/20260609_Hybrid5x3
DEFAULT_DATA_DIR = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"

# annotation 列の優先順位（最初に在るものを使う）。Superclass 先頭（カテゴリ少なく凡例が読める）。
DEFAULT_ANNOTATION_PRIORITY = ["Superclass", "Subclass", "ClassAnn", "celltype", "final_annotation"]

ALL_CONFIG_LABELS = [
    f"{b}__{v}"
    for b in ("geneode", "lowrank", "lincomb", "matsum", "lora")
    for v in ("none", "ratio_reg", "scale_model_x", "scale_model_ml_emb")
]
CKPT_SUBDIR = os.path.join("checkpoints", "hybrid5x3")
SAMPLE_KEY = "cell_gen"

# per-model の generated 色は赤で固定
PER_MODEL_GEN_COLOR = "#D62728"

# config_label = "<branch>__<variant>" の分解と、integrated の branch×variant 可視化用
BRANCH_ORDER = ["geneode", "lowrank", "lincomb", "matsum", "lora"]
VARIANT_ORDER = ["none", "ratio_reg", "scale_model_x", "scale_model_ml_emb"]
# branch -> marker（5 枝）
BRANCH_MARKERS = {"geneode": "o", "lowrank": "^", "lincomb": "s", "matsum": "D", "lora": "P"}


def split_config_label(label):
    """'lowrank__scale_model_ml_emb' -> ('lowrank', 'scale_model_ml_emb')。"""
    parts = str(label).split("__", 1)
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _find_repo_root(start=HERE):
    d = start
    for _ in range(8):
        if os.path.exists(os.path.join(d, "local_paths.py")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None


def resolve_path(p):
    """local_paths.resolve_path があれば使い、無ければ fallback。絶対に例外で落とさない。"""
    if not p:
        return p
    root = _find_repo_root()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from local_paths import resolve_path as _rp  # type: ignore
        return _rp(p)
    except Exception:
        pass
    try:
        if os.path.exists(p):
            return p
        s = str(p)
        REMOTE = "/home/suzuki/Projects/scDiffusion"
        if root and s.startswith(REMOTE):
            cand = os.path.join(root, os.path.relpath(s, REMOTE))
            if os.path.exists(cand):
                return cand
        if root and not os.path.isabs(s):
            cand = os.path.join(root, s)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    return p


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _rel(path, base=HYBRID_DIR):
    try:
        return os.path.relpath(path, base)
    except Exception:
        return path


# ---------------------------------------------------------------------------
# run 探索・選択
# ---------------------------------------------------------------------------
def _ckpt_step(fname):
    m = re.search(r"model0*(\d+)\.pt$", fname)
    if m:
        return int(m.group(1))
    m = re.search(r"ema_[0-9.]+_0*(\d+)\.pt$", fname)
    if m:
        return int(m.group(1))
    return -1


def _pick_checkpoint(train_dir):
    """model*.pt の step 最大を優先。無ければ ema_0.9999_*.pt の step 最大。"""
    ck_dir = os.path.join(train_dir, CKPT_SUBDIR)
    models = glob.glob(os.path.join(ck_dir, "model*.pt"))
    if models:
        best = max(models, key=lambda f: _ckpt_step(os.path.basename(f)))
        return best, _ckpt_step(os.path.basename(best)), "model"
    emas = glob.glob(os.path.join(ck_dir, "ema_0.9999_*.pt"))
    if emas:
        best = max(emas, key=lambda f: _ckpt_step(os.path.basename(f)))
        return best, _ckpt_step(os.path.basename(best)), "ema"
    return None, None, None


def _inspect_run(run_dir):
    """1 つの timestamp run_dir が採用条件（npz + exp_config + checkpoint）を満たすか。"""
    npzs = sorted(glob.glob(os.path.join(run_dir, "sample", "*", "SampledData", "*.npz")))
    if not npzs:
        return None, "missing samples.npz"
    sample_path = npzs[-1]
    cfgs = sorted(glob.glob(os.path.join(run_dir, "train", "*", "exp_config.json")))
    if not cfgs:
        return None, "missing exp_config.json"
    exp_config_path = cfgs[-1]
    train_dir = os.path.dirname(exp_config_path)
    ckpt_path, ckpt_step, ckpt_kind = _pick_checkpoint(train_dir)
    if ckpt_path is None:
        return None, "missing checkpoint (model*.pt / ema_0.9999_*.pt)"
    loss_cands = glob.glob(os.path.join(train_dir, CKPT_SUBDIR, "loss_details.csv"))
    pcmd = os.path.join(run_dir, "pipeline_command.txt")
    return {
        "run_dir": os.path.abspath(run_dir), "run_dir_rel": _rel(run_dir),
        "timestamp_dir": os.path.basename(run_dir.rstrip("/")),
        "sample_path": os.path.abspath(sample_path), "sample_path_rel": _rel(sample_path),
        "exp_config_path": os.path.abspath(exp_config_path), "exp_config_path_rel": _rel(exp_config_path),
        "checkpoint_path": os.path.abspath(ckpt_path), "checkpoint_path_rel": _rel(ckpt_path),
        "checkpoint_step": ckpt_step, "checkpoint_kind": ckpt_kind,
        "loss_path": os.path.abspath(loss_cands[0]) if loss_cands else None,
        "pipeline_command_path": os.path.abspath(pcmd) if os.path.exists(pcmd) else None,
    }, None


def _timestamp_dirs(config_dir, run_suffix):
    if not os.path.isdir(config_dir):
        return []
    subs = [d for d in glob.glob(os.path.join(config_dir, "*")) if os.path.isdir(d)]
    subs.sort(key=lambda p: os.path.basename(p), reverse=True)
    if run_suffix:
        preferred = [d for d in subs if run_suffix in os.path.basename(d)]
        if preferred:
            return preferred
    return subs


def filter_selected(selected, include=None, exclude=None):
    """selected（dict[label]->rec）を include/exclude で絞る。

    include が空でなければ include に在るものだけ、exclude に在るものは除く。
    両方空なら selected をそのまま返す（= 従来挙動・後方互換）。順序は保つ。
    """
    inc = set(include or [])
    exc = set(exclude or [])
    if not inc and not exc:
        return dict(selected)
    return {k: v for k, v in selected.items()
            if (not inc or k in inc) and (k not in exc)}


def select_runs(runs_root, run_suffix="", config_labels=None):
    """各 config label について採用 run を 1 つ選ぶ（新しい順で最初に条件を満たすもの）。"""
    labels = config_labels or ALL_CONFIG_LABELS
    selected, skipped = {}, {}
    for label in labels:
        config_dir = os.path.join(runs_root, label)
        ts_dirs = _timestamp_dirs(config_dir, run_suffix)
        if not ts_dirs:
            skipped[label] = {"reason": "no run directory", "config_dir_rel": _rel(config_dir)}
            continue
        rec, last_reason, tried = None, "no valid timestamp dir", []
        for d in ts_dirs:
            rec_try, reason = _inspect_run(d)
            tried.append({"timestamp_dir": os.path.basename(d.rstrip("/")), "reason": reason or "ok"})
            if rec_try is not None:
                rec = rec_try
                break
            last_reason = reason
        if rec is None:
            skipped[label] = {"reason": last_reason, "config_dir_rel": _rel(config_dir), "tried": tried}
        else:
            rec["config_label"] = label
            rec["passed_over"] = [t for t in tried if t["reason"] != "ok"]
            selected[label] = rec
    return selected, skipped


# ---------------------------------------------------------------------------
# データ読込
# ---------------------------------------------------------------------------
def _to_dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _sanitize(X):
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    return X


def choose_annotation_col(obs_columns, priority=None):
    priority = priority or DEFAULT_ANNOTATION_PRIORITY
    for c in priority:
        if c in obs_columns:
            return c
    return None


def load_real_full(data_dir, annotation_priority=None, log=print):
    """real h5ad を**全件**読み込み、annotation 列を選んで返す（subsample はしない）。

    返り値 (adata_real, annotation_col)。obs: origin='Real' / annotation / config_label='__real__'。
    var_names は g0.. に揃える（gen と結合可能に）。
    """
    import anndata as ad
    resolved = resolve_path(data_dir)
    if not os.path.exists(resolved):
        raise FileNotFoundError(
            f"real data h5ad が見つかりません: {data_dir} -> {resolved}\n"
            f"--data_dir で正しい h5ad を指定してください。")
    log(f"[real] loading {resolved}")
    adata = ad.read_h5ad(resolved)
    ann_col = choose_annotation_col(list(adata.obs.columns), annotation_priority)
    adata.X = _to_dense(adata.X)
    adata.obs = adata.obs.copy()
    adata.obs["origin"] = "Real"
    adata.obs["config_label"] = "__real__"
    adata.obs["annotation"] = (adata.obs[ann_col].astype(str).values if ann_col else "real")
    adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    log(f"[real] shape={adata.shape} annotation_col={ann_col}")
    return adata, ann_col


def subsample_real(adata_real, n, seed=0, log=print):
    """real を n 件に間引いた copy（n<=0 なら全件 copy）。"""
    if not n or n <= 0 or adata_real.n_obs <= n:
        return adata_real.copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(adata_real.n_obs, size=n, replace=False))
    log(f"[real] subsample {adata_real.n_obs} -> {n}")
    return adata_real[idx].copy()


def load_gen_one(rec, n, seed=0, n_vars_expected=None, log=print):
    """1 つの model の cell_gen を読み、n 件に間引いた AnnData を返す。失敗時 None。"""
    import anndata as ad
    label = rec.get("config_label", "?")
    try:
        z = np.load(rec["sample_path"], allow_pickle=True)
    except Exception as e:
        log(f"[gen] {label}: npz load 失敗 ({type(e).__name__}: {e})")
        return None
    if SAMPLE_KEY not in z:
        log(f"[gen] {label}: '{SAMPLE_KEY}' が npz に無い")
        return None
    X = _to_dense(z[SAMPLE_KEY])
    if X.ndim != 2 or X.shape[0] == 0:
        log(f"[gen] {label}: 空 or 非2D")
        return None
    if n_vars_expected is not None and X.shape[1] != n_vars_expected:
        log(f"[gen] {label}: 遺伝子次元不一致 ({X.shape[1]} != {n_vars_expected})")
        return None
    if n and n > 0 and X.shape[0] > n:
        rng = np.random.default_rng(seed)
        X = X[np.sort(rng.choice(X.shape[0], size=n, replace=False))]
    gen = ad.AnnData(_to_dense(X))
    gen.var_names = [f"g{i}" for i in range(gen.n_vars)]
    gen.obs["origin"] = "Generated"
    gen.obs["config_label"] = label
    gen.obs["annotation"] = "__generated__"
    return gen


def load_gen_all(selected, per_model, seed=0, n_vars_expected=None, log=print):
    """全 model の cell_gen を各 per_model 件で読み、1 つの AnnData に結合。"""
    import anndata as ad
    blocks, counts = [], {}
    for label, rec in selected.items():
        g = load_gen_one(rec, per_model, seed=seed, n_vars_expected=n_vars_expected, log=log)
        if g is None or g.n_obs == 0:
            counts[label] = 0
            continue
        blocks.append(g)
        counts[label] = g.n_obs
        log(f"[gen] {label}: {g.n_obs} cells")
    if not blocks:
        return None, counts
    gen = ad.concat(blocks, axis=0, join="inner", index_unique=None)
    return gen, counts


def build_combined(adata_real, adata_gen, log=print):
    """real + gen を 1 つの AnnData に。X dense・NaN/Inf 除去・var 揃え。"""
    import anndata as ad
    if adata_gen is None or adata_gen.n_obs == 0:
        combined = adata_real.copy()
    else:
        combined = ad.concat([adata_real, adata_gen], axis=0, join="inner", index_unique=None)
    combined.X = _sanitize(_to_dense(combined.X))
    for c in ("origin", "config_label", "annotation"):
        combined.obs[c] = combined.obs[c].astype(str)
    return combined


# ---------------------------------------------------------------------------
# UMAP 計算
# ---------------------------------------------------------------------------
def compute_umap(combined, n_pcs=50, n_neighbors=15, min_dist=0.5, seed=0, log=print):
    """PCA -> neighbors -> UMAP。obsm['X_umap'] を付けて返す。"""
    import scanpy as sc
    np.random.seed(seed)
    sc.settings.verbosity = 0
    n_pcs = int(min(n_pcs, max(2, min(combined.n_obs - 1, combined.n_vars - 1))))
    sc.pp.pca(combined, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(combined, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
    sc.tl.umap(combined, min_dist=min_dist, random_state=seed)
    return combined


# ---------------------------------------------------------------------------
# 配色・凡例
# ---------------------------------------------------------------------------
def _palette(n):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex
    if n <= 0:
        return []
    if n <= 10:
        return [to_hex(plt.get_cmap("tab10")(i)) for i in range(n)]
    if n <= 20:
        return [to_hex(plt.get_cmap("tab20")(i)) for i in range(n)]
    cmap = plt.get_cmap("gist_ncar")
    return [to_hex(cmap(x)) for x in np.linspace(0.02, 0.98, n)]


def make_real_color_map(categories):
    cats = sorted(set(map(str, categories)))
    cols = _palette(len(cats))
    return {c: cols[i] for i, c in enumerate(cats)}


def make_model_color_map(config_labels):
    present = set(config_labels)
    ordered = [l for l in ALL_CONFIG_LABELS if l in present]
    extra = sorted(present - set(ordered))
    labs = ordered + extra
    cols = _palette(max(len(labs), 1))
    return {l: cols[i] for i, l in enumerate(labs)}


def make_variant_color_map(config_labels):
    """variant（正則化方法）-> 色。VARIANT_ORDER 優先。"""
    variants = {split_config_label(l)[1] for l in config_labels}
    ordered = [v for v in VARIANT_ORDER if v in variants]
    extra = sorted(variants - set(ordered))
    vs = ordered + extra
    cols = _palette(max(len(vs), 1))
    return {v: cols[i] for i, v in enumerate(vs)}


def make_branch_marker_map(config_labels):
    """branch（元モデル）-> marker。未知 branch は 'o'。"""
    branches = {split_config_label(l)[0] for l in config_labels}
    ordered = [b for b in BRANCH_ORDER if b in branches] + sorted(branches - set(BRANCH_ORDER))
    return {b: BRANCH_MARKERS.get(b, "o") for b in ordered}


def _legend_handles(cmap, labels):
    from matplotlib.lines import Line2D
    return [Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                   markerfacecolor=cmap.get(str(l), "#000000"), markeredgecolor="none",
                   label=str(l)) for l in labels]


def _add_legend(ax, handles, title=None, fontsize=7, ncol=1):
    if not handles:
        return
    if len(handles) > 14 and ncol == 1:
        ncol = 2
    ax.legend(handles=handles, title=title, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=fontsize, title_fontsize=fontsize + 1, frameon=False,
              ncol=ncol, handletextpad=0.3, labelspacing=0.3, columnspacing=0.8)


# ---------------------------------------------------------------------------
# 描画プリミティブ（ax に描く。individual / summary が共有）
# ---------------------------------------------------------------------------
def _xy(adata):
    return adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1]


def _masks(adata):
    o = adata.obs["origin"].values
    return (o == "Real"), (o == "Generated")


def _draw_real_annotation(ax, combined, real_cmap, ps, annotation_col=None, legend=True):
    x, y = _xy(combined)
    real_m, gen_m = _masks(combined)
    ann = combined.obs["annotation"].values
    rcolors = np.array([real_cmap.get(str(a), "#999999") for a in ann])
    ax.scatter(x[real_m], y[real_m], s=ps["size_real"], c=rcolors[real_m],
               alpha=ps["alpha_real"], linewidths=0, rasterized=True)
    ax.scatter(x[gen_m], y[gen_m], s=ps["size_gen"], c="#202020",
               alpha=min(ps["alpha_gen"], 0.5), linewidths=0, rasterized=True)
    ax.set_title("Real (annotation) + Generated (dark gray)" +
                 (f"  [{annotation_col}]" if annotation_col else ""))
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        cats = sorted(set(str(a) for a in ann[real_m]))
        h = _legend_handles(real_cmap, cats) + _legend_handles({"Generated": "#202020"}, ["Generated"])
        _add_legend(ax, h, title=(annotation_col or "annotation"))


def _draw_origin(ax, combined, ps, legend=True):
    x, y = _xy(combined)
    real_m, gen_m = _masks(combined)
    ax.scatter(x[real_m], y[real_m], s=ps["size_real"], c="#BBBBBB", alpha=ps["alpha_real"],
               linewidths=0, rasterized=True, label="Real")
    ax.scatter(x[gen_m], y[gen_m], s=ps["size_gen"], c="#D62728", alpha=ps["alpha_gen"],
               linewidths=0, rasterized=True, label="Generated")
    ax.set_title("Real (gray) vs Generated (red)")
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        lg = ax.legend(loc="best", markerscale=2, framealpha=0.8)
        for hh in (lg.legend_handles if hasattr(lg, "legend_handles") else lg.legendHandles):
            hh.set_alpha(1.0)


def _draw_branch_marker_variant(ax, combined, variant_cmap, branch_marker_map, ps, legend=True):
    """generated を **branch=marker・variant=color** で描く。real は薄灰背景。

    config_label を split_config_label で (branch, variant) に分解し、
    branch ごとに marker、variant ごとに色を割り当てる。
    """
    from matplotlib.lines import Line2D
    x, y = _xy(combined)
    real_m, gen_m = _masks(combined)
    lab = combined.obs["config_label"].values
    branch = np.array([split_config_label(l)[0] for l in lab])
    variant = np.array([split_config_label(l)[1] for l in lab])
    ax.scatter(x[real_m], y[real_m], s=ps["size_real"], c="#DDDDDD",
               alpha=min(ps["alpha_real"], 0.3), linewidths=0, rasterized=True)
    branches = [b for b in BRANCH_ORDER if b in set(branch[gen_m])] + \
               sorted(set(branch[gen_m]) - set(BRANCH_ORDER))
    variants = [v for v in VARIANT_ORDER if v in set(variant[gen_m])] + \
               sorted(set(variant[gen_m]) - set(VARIANT_ORDER))
    for br in branches:
        for va in variants:
            m = gen_m & (branch == br) & (variant == va)
            if not m.any():
                continue
            ax.scatter(x[m], y[m], s=ps["size_gen"], marker=branch_marker_map.get(br, "o"),
                       c=variant_cmap.get(va, "#000000"), alpha=ps["alpha_gen"],
                       linewidths=0, rasterized=True)
    ax.set_title("Generated: branch=marker, variant=color (real = gray bg)")
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        bh = [Line2D([0], [0], marker=branch_marker_map.get(b, "o"), linestyle="none",
                     color="#444444", markersize=7, label=b) for b in branches]
        vh = [Line2D([0], [0], marker="o", linestyle="none", markeredgecolor="none",
                     markerfacecolor=variant_cmap.get(v, "#000000"), markersize=7, label=v)
              for v in variants]
        leg1 = ax.legend(handles=bh, title="branch (marker)", loc="upper left",
                         bbox_to_anchor=(1.01, 1.0), fontsize=7, title_fontsize=8, frameon=False)
        ax.add_artist(leg1)
        ax.legend(handles=vh, title="variant (color)", loc="upper left",
                  bbox_to_anchor=(1.01, 0.55), fontsize=7, title_fontsize=8, frameon=False)


# ---------------------------------------------------------------------------
# B) integrated overlay（individual + summary）
# ---------------------------------------------------------------------------
def plot_integrated_overlays(combined, real_cmap, variant_cmap, branch_marker_map,
                             out_dir, ps, annotation_col=None, log=print):
    """integrated overlay 3 種 + 横並び summary を保存（by_model は出さない）。"""
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    _draw_real_annotation(ax, combined, real_cmap, ps, annotation_col=annotation_col)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "integrated_real_annotation.png"),
                                    dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    _draw_origin(ax, combined, ps)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "integrated_origin.png"),
                                    dpi=200, bbox_inches="tight"); plt.close(fig)

    # branch=marker, variant=color
    fig, ax = plt.subplots(figsize=(11, 8))
    _draw_branch_marker_variant(ax, combined, variant_cmap, branch_marker_map, ps)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "integrated_branch_marker_variant_color.png"),
                                    dpi=200, bbox_inches="tight"); plt.close(fig)

    # 横並び summary（3 panel）
    fig, axes = plt.subplots(1, 3, figsize=(26, 8))
    _draw_real_annotation(axes[0], combined, real_cmap, ps, annotation_col=annotation_col)
    _draw_origin(axes[1], combined, ps)
    _draw_branch_marker_variant(axes[2], combined, variant_cmap, branch_marker_map, ps)
    fig.suptitle("Integrated comparison: real (all) + generated (per-model)", y=1.02)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "integrated_summary_side_by_side.png"),
                                    dpi=150, bbox_inches="tight"); plt.close(fig)
    log("[integrated] saved real_annotation / origin / branch_marker_variant_color / summary_side_by_side")


# ---------------------------------------------------------------------------
# A) per-model（個別 + facets）
# ---------------------------------------------------------------------------
def _draw_per_model_panel(ax, x, y, real_m, gen_m, ann, real_cmap, label, ps):
    """real を Superclass(annotation) 色、generated を赤で上に重ねる 1 panel。"""
    rcolors = np.array([real_cmap.get(str(a), "#999999") for a in ann])
    ax.scatter(x[real_m], y[real_m], s=ps["size_real"], c=rcolors[real_m],
               alpha=ps["alpha_real"], linewidths=0, rasterized=True)
    ax.scatter(x[gen_m], y[gen_m], s=ps["size_gen"], c=PER_MODEL_GEN_COLOR,
               alpha=ps["alpha_gen"], linewidths=0, rasterized=True)
    ax.set_title(f"{label}  (gen n={int(gen_m.sum())})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def run_per_model(selected, real_pm, real_cmap, out_dir, ps, params, log=print):
    """各 model について real_pm + その model の gen で個別 UMAP を計算 → 個別 PNG ＋ facets。

    real は Superclass(annotation) 色、generated は**赤で固定**。
    real_pm は事前に per_model_real_cells に間引いた共通 real。返り値: per-model 情報の list。
    """
    import matplotlib.pyplot as plt
    pm_dir = os.path.join(out_dir, "per_model")
    os.makedirs(pm_dir, exist_ok=True)
    panels = []   # (label, x, y, real_m, gen_m, ann)
    info = []
    n_vars = real_pm.n_vars
    for i, (label, rec) in enumerate(selected.items(), 1):
        gen = load_gen_one(rec, params["per_model_gen_cells"], seed=params["seed"],
                           n_vars_expected=n_vars, log=log)
        combined = build_combined(real_pm, gen, log=log)
        combined = compute_umap(combined, n_pcs=params["n_pcs"], n_neighbors=params["n_neighbors"],
                                min_dist=params["min_dist"], seed=params["seed"], log=log)
        x, y = _xy(combined)
        real_m, gen_m = _masks(combined)
        ann = combined.obs["annotation"].values

        # 個別 PNG（real=Superclass 色 + generated=赤）
        fig, ax = plt.subplots(figsize=(10, 8))
        _draw_per_model_panel(ax, x, y, real_m, gen_m, ann, real_cmap, label, ps)
        cats = sorted(set(str(a) for a in ann[real_m]))
        h = _legend_handles(real_cmap, cats) + _legend_handles({"Generated": PER_MODEL_GEN_COLOR}, ["Generated"])
        _add_legend(ax, h, title="Superclass", fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(pm_dir, f"{label}.png"),
                                        dpi=170, bbox_inches="tight"); plt.close(fig)
        log(f"[per_model] ({i}/{len(selected)}) {label}.png  (real={int(real_m.sum())}, gen={int(gen_m.sum())})")

        panels.append((label, x, y, real_m, gen_m, ann))
        info.append({"config_label": label, "real_cells": int(real_m.sum()),
                     "gen_cells": int(gen_m.sum())})

    # facets（個別と同じ embedding を使う）
    if panels:
        n = len(panels)
        ncol = 4 if n > 4 else n
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.4 * nrow), squeeze=False)
        for i, (label, x, y, real_m, gen_m, ann) in enumerate(panels):
            _draw_per_model_panel(axes[i // ncol][i % ncol], x, y, real_m, gen_m, ann, real_cmap, label, ps)
        for j in range(n, nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        # figure 全体に 1 つだけ凡例（Superclass + Generated）
        all_cats = sorted(set(str(a) for a in panels[0][5][panels[0][3]]))
        fh = _legend_handles(real_cmap, all_cats) + _legend_handles({"Generated": PER_MODEL_GEN_COLOR}, ["Generated"])
        fig.legend(handles=fh, title="Superclass / Generated(red)", loc="center left",
                   bbox_to_anchor=(1.0, 0.5), fontsize=8, title_fontsize=9, frameon=False)
        fig.suptitle("Per-model: real (Superclass colors) + each model's generated (red). Independent UMAP per panel.", y=1.003)
        fig.tight_layout(); fig.savefig(os.path.join(pm_dir, "all_models_facets.png"),
                                        dpi=140, bbox_inches="tight"); plt.close(fig)
        log(f"[per_model] all_models_facets.png ({n} panels)")
    return info


# ---------------------------------------------------------------------------
# metadata 保存
# ---------------------------------------------------------------------------
def save_selection_metadata(out_dir, selected, skipped, *, runs_root, run_suffix,
                            data_dir, data_dir_resolved, annotation_col, created_at):
    os.makedirs(out_dir, exist_ok=True)
    config = {
        "created_at": created_at,
        "runs_root": os.path.abspath(runs_root), "runs_root_rel": _rel(runs_root),
        "run_suffix": run_suffix, "data_dir": data_dir, "data_dir_resolved": data_dir_resolved,
        "annotation_col": annotation_col,
        "selection_rule": ("latest timestamp directory (preferring run_suffix) that has "
                           "sample/*/SampledData/*.npz, train/*/exp_config.json, and a checkpoint "
                           "(model*.pt max step, else ema_0.9999_*.pt max step)"),
        "n_selected": len(selected), "n_skipped": len(skipped),
        "models": selected, "skipped": skipped,
    }
    with open(os.path.join(out_dir, "selected_runs_config.json"), "w") as f:
        json.dump(config, f, indent=2, default=str)
    if selected:
        rows = []
        for label, rec in selected.items():
            row = {"config_label": label}
            row.update({k: rec.get(k) for k in (
                "timestamp_dir", "checkpoint_step", "checkpoint_kind", "run_dir", "run_dir_rel",
                "sample_path", "sample_path_rel", "exp_config_path", "checkpoint_path",
                "loss_path", "pipeline_command_path")})
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "selected_runs.csv"), index=False)
    else:
        pd.DataFrame(columns=["config_label"]).to_csv(os.path.join(out_dir, "selected_runs.csv"), index=False)
    if skipped:
        srows = [{"config_label": l, "reason": v.get("reason"), "config_dir_rel": v.get("config_dir_rel")}
                 for l, v in skipped.items()]
        pd.DataFrame(srows).to_csv(os.path.join(out_dir, "skipped_runs.csv"), index=False)
    else:
        pd.DataFrame(columns=["config_label", "reason"]).to_csv(
            os.path.join(out_dir, "skipped_runs.csv"), index=False)
    return config


def save_run_config(out_dir, params):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(params, f, indent=2, default=str)


def save_color_maps(out_dir, real_cmap, model_cmap, annotation_col=None,
                    variant_cmap=None, branch_marker_map=None):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"annotation": k, "annotation_col": annotation_col, "color": v}
                  for k, v in real_cmap.items()]).to_csv(
        os.path.join(out_dir, "color_map_real_annotation.csv"), index=False)
    pd.DataFrame([{"config_label": k, "color": v} for k, v in model_cmap.items()]).to_csv(
        os.path.join(out_dir, "model_color_map.csv"), index=False)
    if variant_cmap is not None:
        pd.DataFrame([{"variant": k, "color": v} for k, v in variant_cmap.items()]).to_csv(
            os.path.join(out_dir, "variant_color_map.csv"), index=False)
    if branch_marker_map is not None:
        pd.DataFrame([{"branch": k, "marker": v} for k, v in branch_marker_map.items()]).to_csv(
            os.path.join(out_dir, "branch_marker_map.csv"), index=False)


# ---------------------------------------------------------------------------
# 一括ドライバ（CLI / notebook 共通）
# ---------------------------------------------------------------------------
def default_params():
    return {
        "runs_root": "runs", "output_root": os.path.join("integrated_analysis", "outputs"),
        "run_suffix": "ALL100k", "data_dir": DEFAULT_DATA_DIR,
        "per_model_real_cells": 50000, "per_model_gen_cells": 3000,
        "integrated_real_cells": 0, "integrated_gen_per_model": 500,
        "integrated_include": [], "integrated_exclude": [],
        "seed": 0, "n_pcs": 50, "n_neighbors": 15, "min_dist": 0.5,
        "annotation_priority": list(DEFAULT_ANNOTATION_PRIORITY),
        # 見た目（per-model / integrated で別々の点サイズ）
        "pm_size_real": 3, "pm_size_gen": 14, "pm_alpha_real": 0.3, "pm_alpha_gen": 0.9,
        "int_size_real": 2, "int_size_gen": 10, "int_alpha_real": 0.25, "int_alpha_gen": 0.85,
    }


def discover_and_save(out_dir, params, log=print):
    """探索 → metadata 保存のみ（dry-run 用）。selected/skipped/config を返す。"""
    created_at = datetime.now().isoformat(timespec="seconds")
    os.makedirs(out_dir, exist_ok=True)
    selected, skipped = select_runs(params["runs_root"], params.get("run_suffix", ""))
    save_run_config(out_dir, {**params, "created_at": created_at, "dry_run": True})
    config = save_selection_metadata(
        out_dir, selected, skipped, runs_root=params["runs_root"],
        run_suffix=params.get("run_suffix", ""), data_dir=params["data_dir"],
        data_dir_resolved=resolve_path(params["data_dir"]), annotation_col=None, created_at=created_at)
    log(f"[discover] selected={len(selected)} skipped={len(skipped)} -> {_rel(out_dir)}")
    return selected, skipped, config


def run_full(params, do_per_model=True, do_integrated=True, log=print):
    """探索 → real 1 回ロード → per-model（A）+ integrated（B）→ 保存。out_dir を返す。"""
    created_at = datetime.now().isoformat(timespec="seconds")
    out_dir = params["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    save_run_config(out_dir, {**params, "created_at": created_at})

    selected, skipped = select_runs(params["runs_root"], params.get("run_suffix", ""))
    log(f"[run] selected={len(selected)} skipped={len(skipped)}")
    if not selected:
        raise RuntimeError(f"採用できる run が 0 件です（skipped={list(skipped)}）。")

    adata_real, ann_col = load_real_full(params["data_dir"],
                                         annotation_priority=params.get("annotation_priority"), log=log)
    save_selection_metadata(
        out_dir, selected, skipped, runs_root=params["runs_root"], run_suffix=params.get("run_suffix", ""),
        data_dir=params["data_dir"], data_dir_resolved=resolve_path(params["data_dir"]),
        annotation_col=ann_col, created_at=created_at)

    real_cmap = make_real_color_map(adata_real.obs["annotation"].unique())
    model_cmap = make_model_color_map(list(selected.keys()))
    variant_cmap = make_variant_color_map(list(selected.keys()))
    branch_marker_map = make_branch_marker_map(list(selected.keys()))
    save_color_maps(out_dir, real_cmap, model_cmap, annotation_col=ann_col,
                    variant_cmap=variant_cmap, branch_marker_map=branch_marker_map)

    pm_ps = {"size_real": params["pm_size_real"], "size_gen": params["pm_size_gen"],
             "alpha_real": params["pm_alpha_real"], "alpha_gen": params["pm_alpha_gen"]}
    int_ps = {"size_real": params["int_size_real"], "size_gen": params["int_size_gen"],
              "alpha_real": params["int_alpha_real"], "alpha_gen": params["int_alpha_gen"]}

    pm_info = None
    if do_per_model:
        log(f"[run] === A) per-model (real {params['per_model_real_cells']} + gen "
            f"{params['per_model_gen_cells']}/model) ===")
        real_pm = subsample_real(adata_real, params["per_model_real_cells"], seed=params["seed"], log=log)
        pm_info = run_per_model(selected, real_pm, real_cmap, out_dir, pm_ps, params, log=log)
        if pm_info is not None:
            pd.DataFrame(pm_info).to_csv(os.path.join(out_dir, "per_model_counts.csv"), index=False)
        del real_pm

    if do_integrated:
        # integrated に使う model を include/exclude で絞る（既定は空＝全 selected を使用）
        int_selected = filter_selected(selected, params.get("integrated_include"),
                                       params.get("integrated_exclude"))
        if not int_selected:
            raise RuntimeError("integrated 用に残る model が 0 件です。"
                               f"include={params.get('integrated_include')} exclude={params.get('integrated_exclude')}")
        log(f"[run] === B) integrated (real {'all' if params['integrated_real_cells'] <= 0 else params['integrated_real_cells']}"
            f" + gen {params['integrated_gen_per_model']}/model, models={len(int_selected)}/{len(selected)}) ===")
        if params.get("integrated_exclude"):
            log(f"[integrated] exclude={params['integrated_exclude']}")
        if params.get("integrated_include"):
            log(f"[integrated] include={params['integrated_include']}")
        real_int = subsample_real(adata_real, params["integrated_real_cells"], seed=params["seed"], log=log)
        gen_int, gen_counts = load_gen_all(int_selected, params["integrated_gen_per_model"],
                                           seed=params["seed"], n_vars_expected=adata_real.n_vars, log=log)
        combined = build_combined(real_int, gen_int, log=log)
        log(f"[integrated] combined shape={combined.shape} "
            f"(real={int((combined.obs['origin']=='Real').sum())}, "
            f"gen={int((combined.obs['origin']=='Generated').sum())})")
        combined = compute_umap(combined, n_pcs=params["n_pcs"], n_neighbors=params["n_neighbors"],
                                min_dist=params["min_dist"], seed=params["seed"], log=log)
        int_dir = os.path.join(out_dir, "integrated")
        plot_integrated_overlays(combined, real_cmap, variant_cmap, branch_marker_map,
                                 int_dir, int_ps, annotation_col=ann_col, log=log)
        pd.DataFrame([{"config_label": k, "gen_cells": v} for k, v in gen_counts.items()]).to_csv(
            os.path.join(out_dir, "integrated_gen_counts.csv"), index=False)

    log(f"[run] DONE -> {out_dir}")
    return out_dir, selected, skipped
