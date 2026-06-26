#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_lincomb_a_embedding.py  (20260609)
=======================================

学習済み **LinComb** モデルの係数 a(x,t) を K 次元の cell embedding として可視化する単独スクリプト。

LinCombField は  V(x,t) = Σ_k a_k(x,t) · softplus(W_k x + b_k) − decay(x)  で実装され、
a = coeff_net(x,t) を (B,K) で出す。本スクリプトはこの **raw signed a_k** を主解析として抽出し、
a-space の PCA/UMAP・係数統計を保存する（softmax は使わない。compute_W は LinComb では proxy なので使わない）。
補助指標として abs(a) / top_abs_k / abs_a_entropy なども保存する。

既存の学習コード・モデル本体・forward 挙動は一切変更しない。モデル復元は viz/_restore.py を使う。

使い方:
  cd work/20260609_Hybrid5x3/viz
  python plot_lincomb_a_embedding.py \
      --run_dir /path/to/runs/lincomb__none/20260617_180937 \
      --max_cells 50000 --t_values 0,499,999 \
      --color_cols Superclass,celltype,final_annotation

出力（--output_dir 未指定なら）:
  {run_dir}/viz/lincomb_a_embedding/{timestamp}_a_embedding/
"""

import argparse
import os
import sys
import re
import json
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                       # viz/（_restore）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # Hybrid5x3/（run_paths）
import _restore as R  # noqa: E402
import run_paths  # noqa: E402

# 後で run_all_viz.py に統合しやすいよう role 定数を置いておく（今回は単独実行）。
ROLE = "lincomb_a_embedding"


# ---------------------------------------------------------------------------
# checkpoint / config 自動検出
# ---------------------------------------------------------------------------
def _ckpt_step(path):
    """ema_0.9999_000123.pt / model000123.pt から step を取り出す（無ければ -1）。"""
    m = re.search(r"(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _first_nonempty(patterns):
    """patterns を順に recursive glob し、最初にマッチした list を返す（無ければ []）。"""
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits
    return []


def find_config(run_dir, explicit):
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--config が見つかりません: {explicit}")
        return os.path.abspath(explicit)
    cands = _first_nonempty([
        os.path.join(run_dir, "train", "**", "exp_config.json"),
        os.path.join(run_dir, "**", "exp_config.json"),
    ])
    if not cands:
        raise FileNotFoundError(
            f"exp_config.json が見つかりません（{run_dir}/train/**/ または {run_dir}/**/）。--config で明示してください。")
    # 複数あれば更新時刻が最新
    return os.path.abspath(max(cands, key=os.path.getmtime))


def find_checkpoint(run_dir, explicit):
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"--model_path が見つかりません: {explicit}")
        return os.path.abspath(explicit)
    cands = _first_nonempty([
        os.path.join(run_dir, "train", "**", "checkpoints", "**", "ema_*.pt"),
        os.path.join(run_dir, "train", "**", "checkpoints", "**", "model*.pt"),
        os.path.join(run_dir, "**", "ema_*.pt"),
        os.path.join(run_dir, "**", "model*.pt"),
    ])
    if not cands:
        raise FileNotFoundError(
            f"checkpoint が見つかりません（{run_dir} 配下の ema_*.pt / model*.pt）。--model_path で明示してください。")
    # 同一 tier 内で step 最大（同点は mtime 最新）
    return os.path.abspath(max(cands, key=lambda f: (_ckpt_step(f), os.path.getmtime(f))))


# ---------------------------------------------------------------------------
# 作図ヘルパ（matplotlib 直描き: Agg / close / category 多くても落ちない）
# ---------------------------------------------------------------------------
def _scatter(coords, values, path, title, categorical):
    """coords(N,2) を values で色付けして保存。categorical=True は離散、False は連続。"""
    if coords is None:
        return False
    x, y = coords[:, 0], coords[:, 1]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    if categorical:
        cats = pd.Categorical(pd.Series(values).astype(str))
        codes = cats.codes
        ncat = len(cats.categories)
        cmap = plt.get_cmap("tab20" if ncat <= 20 else "gist_ncar")
        for i, name in enumerate(cats.categories):
            mask = codes == i
            ax.scatter(x[mask], y[mask], s=4, linewidths=0,
                       color=cmap(i % cmap.N if ncat <= 20 else i / max(ncat - 1, 1)),
                       label=str(name))
        if ncat <= 30:   # 凡例が多すぎる時は付けない（落ちない・潰れない）
            ax.legend(markerscale=3, fontsize=6, loc="center left",
                      bbox_to_anchor=(1.0, 0.5), frameon=False)
    else:
        sm = ax.scatter(x, y, s=4, linewidths=0, c=np.asarray(values, dtype=float), cmap="viridis")
        fig.colorbar(sm, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def _lineplot_by_t(summary_df, value_col, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for k, g in summary_df.groupby("k"):
        gg = g.sort_values("t")
        ax.plot(gg["t"], gg[value_col], marker="o", markersize=3, label=f"a{int(k)}")
    ax.set_xlabel("diffusion timestep t"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if summary_df["k"].nunique() <= 24:
        ax.legend(fontsize=7, ncol=2, frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# a-space embedding（plot 用 copy にだけ scale/pca/umap）
# ---------------------------------------------------------------------------
def compute_embeddings(adata_a, K, seed):
    """plot 用 copy に scale→PCA→neighbors→UMAP。失敗しても (umap_coords, pca_coords, info) を返す。"""
    info = {"pca_ok": False, "umap_ok": False, "n_comps": 0, "reason": ""}
    umap_coords = pca_coords = None
    n_obs = adata_a.n_obs
    n_comps = min(K - 1, 20, max(n_obs - 1, 1))
    if K < 3 or n_comps < 2 or n_obs < 3:
        info["reason"] = f"K={K}, n_obs={n_obs} では PCA/UMAP を作れない（n_comps={n_comps}）。図は skip。"
        return umap_coords, pca_coords, info
    try:
        ap = adata_a.copy()
        sc.pp.scale(ap, max_value=10)                       # raw a を潰さないため plot copy にだけ適用
        sc.tl.pca(ap, n_comps=n_comps, random_state=seed)
        info["pca_ok"] = True; info["n_comps"] = int(n_comps)
        pca_coords = np.asarray(ap.obsm["X_pca"][:, :2])
        try:
            sc.pp.neighbors(ap, n_neighbors=min(15, n_obs - 1), n_pcs=min(n_comps, 20), random_state=seed)
            sc.tl.umap(ap, random_state=seed)
            umap_coords = np.asarray(ap.obsm["X_umap"])
            info["umap_ok"] = True
        except Exception as e:                              # UMAP だけ失敗 → PCA は残す
            info["reason"] = f"UMAP 失敗: {type(e).__name__}: {e}"
    except Exception as e:
        info["reason"] = f"PCA 失敗: {type(e).__name__}: {e}"
    return umap_coords, pca_coords, info


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    args = build_argparser().parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(f"--run_dir が存在しません: {run_dir}")

    config_path = find_config(run_dir, args.config)
    model_path = find_checkpoint(run_dir, args.model_path)
    data_dir = R.resolve_path(args.data_dir)
    edge_tsv_path = R.resolve_path(args.edge_tsv_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = args.output_dir if args.output_dir else os.path.join(run_dir, "viz", ROLE)
    out_dir = os.path.join(base_out, f"{ts}_a_embedding")
    os.makedirs(out_dir, exist_ok=True)

    # ---- モデル復元（既存 _restore を使用。学習コードは触らない） ----
    cfg = R.normalize_config(R.load_config(config_path))
    diffusion = R.build_diffusion(cfg["diffusion_steps"])
    gene_list = R.load_gene_list(data_dir)
    model = R.build_model(cfg, model_path, gene_list, diffusion.num_timesteps, edge_tsv_path, device)

    ode = getattr(model, "ode_model", None)
    if ode is None:
        raise SystemExit("[ERROR] このモデルには ode_model がありません（plain baseline?）。LinComb run を指定してください。")
    mtype = getattr(ode, "model_type", None)
    if mtype != "lincomb":
        raise SystemExit(
            f"[ERROR] このスクリプトは LinComb 専用です。ode_model.model_type='{mtype}'（ode_branch を確認）。"
            " config の ode_branch が 'lincomb' の run を指定してください。")
    if not hasattr(ode, "_coeffs"):
        raise SystemExit("[ERROR] ode_model に _coeffs() がありません。LinCombField ではない可能性があります。")
    K = int(getattr(ode, "K"))

    # ---- データ読み込み・サブサンプル ----
    adata = sc.read_h5ad(data_dir)
    n_full = adata.n_obs
    if args.max_cells and args.max_cells > 0 and args.max_cells < n_full:
        idx = np.sort(np.random.choice(n_full, args.max_cells, replace=False))
        adata = adata[idx].copy()                            # obs との対応を保つ
    else:
        idx = np.arange(n_full)
    used_index = np.asarray(idx)
    n = adata.n_obs

    X = R.to_dense(adata.X).astype(np.float32)
    if hasattr(R, "sanitize"):
        X = R.sanitize(X)                                    # NaN/Inf を 0 に

    # ---- t のパース・検証 ----
    T = diffusion.num_timesteps
    t_values = [int(s) for s in str(args.t_values).split(",") if str(s).strip() != ""]
    bad = [t for t in t_values if t < 0 or t >= T]
    if bad:
        raise SystemExit(f"[ERROR] t_values に範囲外があります {bad}（0..{T-1}）。")
    color_cols = [s.strip() for s in str(args.color_cols).split(",") if s.strip()]
    color_found = [c for c in color_cols if c in adata.obs.columns]

    print(f"[a-embed] config       : {config_path}")
    print(f"[a-embed] checkpoint   : {model_path}")
    print(f"[a-embed] data_dir     : {data_dir}")
    print(f"[a-embed] n_cells      : {n} (full={n_full})")
    print(f"[a-embed] K            : {K}")
    print(f"[a-embed] t_values     : {t_values}  use_noisy_xt={args.use_noisy_xt}")
    print(f"[a-embed] color_cols   : found={color_found} missing={[c for c in color_cols if c not in color_found]}")
    print(f"[a-embed] output_dir   : {out_dir}")

    created_files = []
    summary_rows = []   # per (t, k) 係数統計

    # ---- t ごとに a(x,t) を計算 → 保存・作図 ----
    for t in t_values:
        a_chunks = []
        for s in range(0, n, args.batch_size):
            xb = X[s:s + args.batch_size]
            x_tensor = torch.from_numpy(xb).to(device).float()
            B = x_tensor.shape[0]
            t_raw = torch.full((B,), int(t), dtype=torch.long, device=device)
            if args.use_noisy_xt:
                noise = torch.randn_like(x_tensor)
                x_in = diffusion.q_sample(x_tensor, t_raw, noise=noise)
            else:
                x_in = x_tensor                              # clean real expression x0
            model_t = diffusion._scale_timesteps(t_raw)
            a_b = ode._coeffs(x_in, model_t)                 # (B, K)
            a_chunks.append(a_b.detach().cpu().numpy())
        a = np.concatenate(a_chunks, axis=0).astype(np.float32)   # (n, K)

        # ---- 補助指標 ----
        abs_a = np.abs(a)
        top_abs_k = np.argmax(abs_a, axis=1).astype(int)
        top_abs_value = abs_a.max(axis=1)
        a_l2_norm = np.linalg.norm(a, axis=1)
        a_abs_sum = abs_a.sum(axis=1)
        denom = abs_a.sum(axis=1, keepdims=True)
        p = abs_a / np.where(denom > 0, denom, 1.0)
        ent_raw = -(p * np.log(p + 1e-12)).sum(axis=1)
        abs_a_entropy = np.clip(ent_raw / np.log(K), 0.0, 1.0) if K > 1 else np.zeros(n, dtype=np.float32)

        # ---- a-space を AnnData 化 ----
        adata_a = anndata.AnnData(a.copy())
        adata_a.obs = adata.obs.copy()
        adata_a.var_names = [f"a{k}" for k in range(K)]
        adata_a.obs["top_abs_k"] = pd.Categorical(top_abs_k.astype(str))
        adata_a.obs["top_abs_value"] = top_abs_value
        adata_a.obs["a_l2_norm"] = a_l2_norm
        adata_a.obs["a_abs_sum"] = a_abs_sum
        adata_a.obs["abs_a_entropy"] = abs_a_entropy
        for k in range(K):
            adata_a.obs[f"a{k}"] = a[:, k]
            adata_a.obs[f"abs_a{k}"] = abs_a[:, k]

        # ---- CSV 保存 ----
        df = pd.DataFrame({"cell_index": used_index})
        for c in adata.obs.columns:
            df[c] = adata.obs[c].values
        for k in range(K):
            df[f"a{k}"] = a[:, k]
        for k in range(K):
            df[f"abs_a{k}"] = abs_a[:, k]
        df["top_abs_k"] = top_abs_k
        df["top_abs_value"] = top_abs_value
        df["a_l2_norm"] = a_l2_norm
        df["a_abs_sum"] = a_abs_sum
        df["abs_a_entropy"] = abs_a_entropy
        csv_path = os.path.join(out_dir, f"lincomb_a_values_t{t}.csv")
        df.to_csv(csv_path, index=False)
        created_files.append(os.path.basename(csv_path))

        # ---- h5ad 保存 ----
        h5_path = os.path.join(out_dir, f"lincomb_a_space_t{t}.h5ad")
        try:
            adata_a.write(h5_path)
            created_files.append(os.path.basename(h5_path))
        except Exception as e:
            print(f"[a-embed][WARN] h5ad 保存失敗 t={t}: {type(e).__name__}: {e}")

        # ---- 係数統計（per k） ----
        for k in range(K):
            summary_rows.append({
                "t": int(t), "k": int(k),
                "mean_a": float(a[:, k].mean()), "std_a": float(a[:, k].std()),
                "mean_abs_a": float(abs_a[:, k].mean()), "std_abs_a": float(abs_a[:, k].std()),
                "fraction_top_abs_k": float((top_abs_k == k).mean()),
            })

        # ---- PCA / UMAP（plot 用 copy にだけ scale。失敗しても CSV/h5ad は残す） ----
        umap_coords, pca_coords, emb_info = compute_embeddings(adata_a, K, args.seed)
        if not emb_info["umap_ok"] or not emb_info["pca_ok"]:
            print(f"[a-embed][WARN] t={t} embedding: pca_ok={emb_info['pca_ok']} "
                  f"umap_ok={emb_info['umap_ok']} {emb_info['reason']}")

        # ---- 図（UMAP / PCA） ----
        def _save(coords, basis, col, values, categorical):
            if coords is None:
                return
            fname = f"lincomb_a_{basis}_{col}_t{t}.png"
            if _scatter(coords, values, os.path.join(out_dir, fname),
                        f"{basis.upper()} a-space  {col}  t={t}", categorical):
                created_files.append(fname)

        # 指定の基本図
        _save(umap_coords, "umap", "top_abs_k", top_abs_k, True)
        _save(umap_coords, "umap", "abs_a_entropy", abs_a_entropy, False)
        _save(umap_coords, "umap", "a_l2_norm", a_l2_norm, False)
        _save(pca_coords, "pca", "top_abs_k", top_abs_k, True)
        _save(pca_coords, "pca", "abs_a_entropy", abs_a_entropy, False)
        # annotation 列（存在するもの）
        for col in color_found:
            vals = adata.obs[col].values
            _save(umap_coords, "umap", col, vals, True)
            _save(pca_coords, "pca", col, vals, True)
        # 各 a_k / abs_a_k を連続値で（全 k）
        for k in range(K):
            _save(umap_coords, "umap", f"a{k}", a[:, k], False)
            _save(umap_coords, "umap", f"abs_a{k}", abs_a[:, k], False)

        print(f"[a-embed] t={t} done (cells={n}, K={K})")

    # ---- 係数 summary（全 t） ----
    summary_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(out_dir, "lincomb_a_summary_by_t.csv")
    summary_df.to_csv(sum_csv, index=False)
    created_files.append(os.path.basename(sum_csv))
    if not summary_df.empty:
        _lineplot_by_t(summary_df, "mean_abs_a", "mean |a_k| by t", "mean |a_k|",
                       os.path.join(out_dir, "lincomb_a_mean_abs_by_t.png"))
        _lineplot_by_t(summary_df, "fraction_top_abs_k", "top-expert fraction by t",
                       "fraction cells with top |a_k|",
                       os.path.join(out_dir, "lincomb_top_expert_fraction_by_t.png"))
        created_files += ["lincomb_a_mean_abs_by_t.png", "lincomb_top_expert_fraction_by_t.png"]

    # ---- model_info / summary.json ----
    model_info = {}
    for attr in ("model_type", "K", "use_decay", "W_IS_EXACT", "soft"):
        if hasattr(ode, attr):
            v = getattr(ode, attr)
            model_info[attr] = v if isinstance(v, (int, float, str, bool)) else str(v)

    summary = {
        "run_dir": run_dir,
        "model_path": model_path,
        "config": config_path,
        "data_dir": data_dir,
        "edge_tsv_path": edge_tsv_path,
        "n_cells_used": int(n),
        "n_cells_full": int(n_full),
        "K": int(K),
        "t_values": t_values,
        "use_noisy_xt": bool(args.use_noisy_xt),
        "output_dir": out_dir,
        "color_cols_found": color_found,
        "model_info": model_info,
        "created_at": ts,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    created_files.append("summary.json")

    # ---- 標準出力サマリ ----
    print("\n==================== DONE ====================")
    print(f"config      : {config_path}")
    print(f"checkpoint  : {model_path}")
    print(f"data_dir    : {data_dir}")
    print(f"n_cells     : {n}")
    print(f"K           : {K}")
    print(f"output_dir  : {out_dir}")
    print("main files  :")
    head = [f for f in ("summary.json", "lincomb_a_summary_by_t.csv",
                        "lincomb_a_mean_abs_by_t.png", "lincomb_top_expert_fraction_by_t.png")
            if f in created_files]
    per_t = [f"lincomb_a_values_t{t}.csv / lincomb_a_space_t{t}.h5ad / lincomb_a_umap_top_abs_k_t{t}.png ..."
             for t in t_values]
    for f in head:
        print(f"  - {f}")
    for line in per_t:
        print(f"  - {line}")
    print(f"  ({len(created_files)} files total under output_dir)")


def build_argparser():
    p = argparse.ArgumentParser(
        description="学習済み LinComb モデルの係数 a(x,t) を K 次元 cell embedding として可視化する（単独実行）。")
    p.add_argument("--run_dir", required=True,
                   help="対象 run dir。config/checkpoint をこの配下から自動検出する。")
    p.add_argument("--model_path", default="", help="checkpoint を明示（未指定なら run_dir から自動検出）")
    p.add_argument("--config", default="", help="exp_config.json を明示（未指定なら run_dir から自動検出）")
    p.add_argument("--data_dir", default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad")
    p.add_argument("--edge_tsv_path", default="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv")
    p.add_argument("--output_dir", default="", help="未指定なら {run_dir}/viz/lincomb_a_embedding/{ts}_a_embedding/")
    p.add_argument("--max_cells", type=int, default=50000, help="使う細胞数（0 以下なら全細胞）")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--t_values", default="0,499,999", help="a を評価する diffusion timestep（カンマ区切り）")
    p.add_argument("--color_cols", default="Superclass,celltype,final_annotation",
                   help="annotation 色付け列（存在するものだけ使う。カンマ区切り）")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--use_noisy_xt", action="store_true",
                   help="true なら q_sample で x_t を作って a(x_t,t)。false（既定）は clean x0 を a(x0,t) に入れる。")
    return p


if __name__ == "__main__":
    main()
