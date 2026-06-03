# ======================
# File: embed_velocity.py
# ======================
#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import scvelo as scv
import scipy.sparse as sp


# ==========================================
# 1. 初期設定とパスの追加
# ==========================================
def setup_environment():
    scdiffusion_path = "/home/suzuki/Projects/scDiffusion"
    if scdiffusion_path not in sys.path:
        sys.path.insert(0, scdiffusion_path)

    # いまのコードを踏襲
    scv.set_figure_params(transparent=False)


def resolve_run_dir() -> Path:
    """
    compute_velocity_metrics.py が書いた latest_velocity_compare_run.txt を読む。
    必要なら手で RUN_DIR を直指定してもよい。
    """
    RUN_DIR = None  # 例: "/absolute/path/to/20260330_123456_velocity_compare"

    if RUN_DIR is not None:
        return Path(RUN_DIR)

    latest_run_txt = Path(os.getcwd()) / "latest_velocity_compare_run.txt"
    if not latest_run_txt.exists():
        raise FileNotFoundError(
            f"{latest_run_txt} not found. "
            "先に compute_velocity_metrics.py を同じ作業ディレクトリで実行してください。"
        )

    run_dir = Path(latest_run_txt.read_text(encoding="utf-8").strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    return run_dir


# ==========================================
# 2. plotting
# ==========================================
def plot_velocity_panels(adata_subset, group_name: str, outdir: Path, vkey: str):
    print(f"\n=== Plotting {group_name} / {vkey} ===")
    outdir.mkdir(parents=True, exist_ok=True)

    color_col = "celltype"
    palette = adata_subset.uns.get(f"{color_col}_colors", None)
    if palette is not None:
        palette = list(palette)

    save_path = outdir / "1_velocity_stream.png"
    scv.pl.velocity_embedding_stream(
        adata_subset,
        basis="umap",
        vkey=vkey,
        color=color_col,
        palette=palette,
        show=False,
        legend_loc="right margin",
        size=5,
        alpha=1,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    save_path = outdir / "2_velocity_arrow.png"
    scv.pl.velocity_embedding(
        adata_subset,
        basis="umap",
        vkey=vkey,
        color=color_col,
        palette=palette,
        show=False,
        arrow_length=3,
        arrow_size=2,
        legend_loc="right margin",
        size=5,
        alpha=1,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    save_path = outdir / "3_velocity_grid.png"
    scv.pl.velocity_embedding_grid(
        adata_subset,
        basis="umap",
        vkey=vkey,
        color=color_col,
        palette=palette,
        show=False,
        legend_loc="right margin",
        size=5,
        alpha=1,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    if "distances" in adata_subset.obsp:
        if "neighbors" not in adata_subset.uns:
            adata_subset.uns["neighbors"] = {}
        adata_subset.uns["neighbors"]["distances"] = adata_subset.obsp["distances"]
        adata_subset.uns["neighbors"]["connectivities"] = adata_subset.obsp["connectivities"]

    try:
        scv.tl.paga(adata_subset, groups=color_col, vkey=vkey)
        save_path = outdir / "6_paga_velocity.png"

        scv.pl.paga(
            adata_subset,
            basis="umap",
            color=color_col,
            palette=palette,
            node_size_scale=0.6,
            min_edge_width=0.5,
            max_edge_width=2,
            threshold=0.05,
            alpha=0.8,
            show=False,
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"PAGA error for {group_name}/{vkey}: {e}")


# ==========================================
# 3. embedding pipeline
# ==========================================
def run_velocity_embedding_pipeline(adata_subset, vkey: str):
    """
    ここは velocity_by_Superclass.py の流れを踏襲:
      PCA(50) -> neighbors(15, n_pcs=40) -> UMAP
      velocity_graph(xkey='X', backend='loky', n_jobs=32)
      velocity_embedding(basis='umap')
    """
    print(f"\n=== Velocity embedding for {vkey} ===")

    sc.tl.pca(adata_subset, svd_solver="arpack", n_comps=50)
    sc.pp.neighbors(adata_subset, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_subset)

    scv.tl.velocity_graph(
        adata_subset,
        vkey=vkey,
        xkey="X",
        backend="loky",
        n_jobs=32,
    )

    scv.tl.velocity_embedding(
        adata_subset,
        basis="umap",
        vkey=vkey,
    )


# ==========================================
# 4. main
# ==========================================
def main():
    setup_environment()
    run_dir = resolve_run_dir()

    base_h5ad_path = run_dir / "adata_eryth_base.h5ad"
    velocity_npz_path = run_dir / "velocities_eryth.npz"
    out_h5ad_path = run_dir / "adata_eryth_embedded.h5ad"

    if not base_h5ad_path.exists():
        raise FileNotFoundError(base_h5ad_path)
    if not velocity_npz_path.exists():
        raise FileNotFoundError(velocity_npz_path)

    print(f"Loading adata from: {base_h5ad_path}")
    adata = sc.read_h5ad(base_h5ad_path)

    print(f"Loading velocities from: {velocity_npz_path}")
    velocity_data = np.load(velocity_npz_path, allow_pickle=True)
    V1 = np.asarray(velocity_data["V1"], dtype=np.float32)
    V2 = np.asarray(velocity_data["V2"], dtype=np.float32)
    obs_names = velocity_data["obs_names"].astype(str)
    var_names = velocity_data["var_names"].astype(str)

    if not np.array_equal(obs_names, adata.obs_names.astype(str).to_numpy()):
        raise ValueError("obs_names mismatch between npz and adata")
    if not np.array_equal(var_names, adata.var_names.astype(str).to_numpy()):
        raise ValueError("var_names mismatch between npz and adata")

    # いまのコードでは xkey='X' を使うので layers['X'] を用意
    if "X" not in adata.layers:
        if sp.issparse(adata.X):
            adata.layers["X"] = adata.X.copy()
        else:
            adata.layers["X"] = np.asarray(adata.X).copy()

    # 一時的に velocity を layer に積む
    adata.layers["V1"] = V1
    adata.layers["V2"] = V2

    # 埋め込みは各 vkey ごとに同じ条件で回す
    # ※ UMAP は両者で共通座標になるよう、同じ adata 上で同じ設定を使う
    #    ただし velocity_graph / velocity_embedding は vkey ごとに別
    run_velocity_embedding_pipeline(adata, vkey="V1")
    run_velocity_embedding_pipeline(adata, vkey="V2")

    # 参考図を保存
    figs_dir = run_dir / "velocity_plots_eryth"
    plot_velocity_panels(adata, "Erythropoietic", figs_dir / "V1", vkey="V1")
    plot_velocity_panels(adata, "Erythropoietic", figs_dir / "V2", vkey="V2")

    # notebook 側では npz から高次元 V1/V2 を読むので、h5ad には残さなくてよい
    for key in ["V1", "V2"]:
        if key in adata.layers:
            del adata.layers[key]

    # 重複を減らすため layers["X"] も不要なら削る
    if "X" in adata.layers:
        del adata.layers["X"]

    adata.write(out_h5ad_path)

    summary = {
        "base_h5ad_path": str(base_h5ad_path),
        "velocity_npz_path": str(velocity_npz_path),
        "out_h5ad_path": str(out_h5ad_path),
        "saved_obsm_keys": list(adata.obsm.keys()),
        "saved_uns_keys": list(adata.uns.keys()),
    }
    with open(run_dir / "embed_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(out_h5ad_path)
    print(figs_dir / "V1")
    print(figs_dir / "V2")


if __name__ == "__main__":
    main()