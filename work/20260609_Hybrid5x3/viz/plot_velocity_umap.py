#!/usr/bin/env python3
"""
plot_velocity_umap.py  (20260609)  — 旧 velocity_by_superclass.py をリネーム

Velocity UMAP Visualization Script (Superclass Split)。
work/20260215_embryonic/20260224_084326_Lamda5/velocity_by_Superclass.py をほぼそのまま流用。
変更点は最小限:
  - モデル復元のみ新モデル対応（_restore.build_model 経由、5 枝 + plain）。元の GeneODE 直構築から差し替え。
  - 図は 1_velocity_stream.png / 2_velocity_arrow.png の 2 枚のみ（grid / paga は削除）。
  - パスは引数で受ける（pipeline / run_all_viz から呼べるように）。/home/suzuki... は local_paths で fallback。
scvelo の使い方（velocity_graph xkey="X" / loky / n_jobs=32 → velocity_embedding → stream/arrow）は元コードのまま。
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import anndata
import scvelo as scv
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                       # viz/（_restore）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # Hybrid5x3/（run_paths）
import _restore as R  # noqa: E402
import run_paths  # noqa: E402

ROLE = "velocity"


def _apply_scvelo_compat_shims():
    """ローカル env（numpy>=1.24 / pandas>=2.0）で scvelo 0.2.5 を動かすための互換 shim。

    互換 env（remote: numpy<1.24 / pandas<2.0）では条件に当たらず **no-op**。
      - numpy>=1.24: ragged な np.array が ValueError → object 配列にフォールバック
        （velocity_graph → compute_cosines → parallelize の `np.array(res)` 対策）
      - pandas>=2.0: cat.categories の setter が無い → scvelo set_legend 用に追加
    """
    try:
        if tuple(int(x) for x in np.__version__.split(".")[:2]) >= (1, 24):
            _orig = np.array
            if getattr(np.array, "__name__", "") != "_safe_array":
                def _safe_array(o, *a, **k):
                    try:
                        return _orig(o, *a, **k)
                    except ValueError:
                        return _orig(o, *a, dtype=object, **k)
                np.array = _safe_array
    except Exception:
        pass
    try:
        if int(pd.__version__.split(".")[0]) >= 2:
            from pandas.core.arrays.categorical import Categorical as _Cat
            if _Cat.categories.fset is None:
                def _cat_set(self, new):
                    self._categories = pd.Index(new)
                _Cat.categories = property(_Cat.categories.fget, _cat_set)
    except Exception:
        pass


_apply_scvelo_compat_shims()


# ==========================================
# 2. パイプライン・作図関数の定義
# ==========================================
def _plot_stream_arrow(adata_subset, outdir, color_col, palette, stream_name, arrow_name):
    """stream / arrow の 2 枚を 1 つの palette で描く（色違いで使い回すヘルパ）。"""
    # ① Stream embedding
    scv.pl.velocity_embedding_stream(
        adata_subset, basis="umap", vkey="velocity_ode",
        color=color_col, palette=palette, show=False, legend_loc="right margin",
        size=5, alpha=1,
    )
    plt.savefig(os.path.join(outdir, stream_name), dpi=300, bbox_inches="tight")
    plt.close()

    # ② Arrow embedding (単一細胞レベル)
    scv.pl.velocity_embedding(
        adata_subset, basis="umap", vkey="velocity_ode",
        color=color_col, palette=palette, show=False, arrow_length=3, arrow_size=2, legend_loc="right margin",
        size=5, alpha=1,
    )
    plt.savefig(os.path.join(outdir, arrow_name), dpi=300, bbox_inches="tight")
    plt.close()


def run_velocity_pipeline(adata_subset, group_name, base_outdir, color_col="celltype", n_jobs=32):
    print(f"\n=== Processing Group: {group_name} ===")
    outdir = os.path.join(base_outdir, str(group_name).replace(" ", "_").replace("/", "_"))
    os.makedirs(outdir, exist_ok=True)

    # --- 1. 埋め込みの再計算 ---
    sc.tl.pca(adata_subset, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata_subset, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_subset)

    # --- 2. Velocity Graphの再計算 ---
    # ご指定通り、lokyとn_jobs=32を使用
    scv.tl.velocity_graph(
        adata_subset,
        vkey="velocity_ode",
        xkey="X",
        backend="loky",
        n_jobs=n_jobs
    )

    # --- 3. Velocity Embedding の計算 ---
    scv.tl.velocity_embedding(adata_subset, basis="umap", vkey="velocity_ode")

    # --- 4. 作図（2 通りの配色を両方出す → 1 Superclass あたり 4 枚）---
    # (A) apply_lineage_palette を入れずに描写（既存 uns 色 or scvelo デフォルト）
    palette = adata_subset.uns.get(f"{color_col}_colors", None)
    if palette is not None:
        palette = list(palette)
    _plot_stream_arrow(adata_subset, outdir, color_col, palette,
                       "1_velocity_stream.png", "2_velocity_arrow.png")

    # (B) apply_lineage_palette を入れて描写（系列グラデーション色）
    try:
        apply_lineage_palette(adata_subset, group_name, celltype_col=color_col)
        palette_lin = adata_subset.uns.get(f"{color_col}_colors", None)
        if palette_lin is not None:
            palette_lin = list(palette_lin)
        _plot_stream_arrow(adata_subset, outdir, color_col, palette_lin,
                           "3_velocity_stream_lineage.png", "4_velocity_arrow_lineage.png")
    except Exception as e:
        print(f"  [lineage palette] skipped ({type(e).__name__}: {e})")

    print(f"Finished plotting for {group_name}")

#
# 配色割り当てのための系列定義
#


# ----------------------------
# 1) Superclassごとの系列定義（PDFの並びをそのまま）
# ※ 【】で囲われた “adataに無い” 可能性のあるやつは入れてもOK（後でpresentだけ拾う）
# ----------------------------
LINEAGES = {
    "Fibroblast": [
        ("Pia", [
            "Primary meninx",
            "Primary meninx-hindbrain",
            "Committed pia precursor",
            "Committed pia precursor-anterior",
            "Pia",
            "Pia-forebrain",
        ], "Greens"),
        ("Arachnoid", [
            "Primary meninx",
            "Committed arachnoid precursor",
            "Committed arachnoid precursor-forebrain",
            "Committed arachnoid precursor-pia junction",
            "Arachnoid",
            "Arachnoid barrier",
            "Arachnoid-dura border",
            "Arachnoid-forebrain",
            "Arachnoid-hindbrain",
        ], "Purples"),
        ("Dura", [
            "Skull/dura progenitors",
            "Committed dura precursor-early",
            "Committed dura precursor",
            "Committed dura precursor-forebrain",
            "Inner dura",
            "Outer dura / periosteal dura",
        ], "Reds"),
        ("Skull/OsteoChondro", [
            "Cranial mesenchyme / skull-side progenitors",
            "Osteoprogenitor",
            "Osteogenic",
            "Osteoblast/osteocyte",
            "Chondrogenic",
            "Chondrocyte",
        ], "Blues"),
    ],

    "Vascular endothelial": [
        ("Vascular", [
            "Angioblast / endothelial progenitor",
            "Primary vascular plexus",
            "Endothelial",
            "Tip cell",
            "Tip cell - arterial",
            "Arterial capillary",
            "Arteriole",
            "Artery",
            "Large artery",
            "Venous capillary",
            "Venous lymphatic-like",
            "Venous lymphatic-like - non-leptomeningeal",
            "Fatty acid metabolic - non-leptomeningeal",
        ], "viridis"),
    ],

    "Vascular perivascular": [
        ("Mural", [
            "Perivascular mesenchymal progenitor",
            "Pericyte",
            "Smooth muscle cell",
            "Smooth muscle cell, mature",
            "ChP Pericyte",
            "ChP Perivascular",
        ], "cividis"),
    ],

    "Erythropoietic": [
        ("Erythroid", [
            "HSC",
            "Early precursor/BFU-E",
            "Precursor/CFU-E",
            "Erythroblast",
            "Cycling erythroblast",
            "Reticulocyte/RBC",
            "ChP Reticulocyte/RBC",
        ], "magma"),
    ],

    "Immune": [
        ("Mono→Macro", [
            "HSC",
            "CMP/GMP",
            "Monocytes",
            "S100A12-high monocytes",
            "Macrophages",
            "Macrophages Monocyte-like",
            "Macrophages-innate immunity",
            "Macrophages-iron metabolism",
            "Dendritic cells",
            "ChP Macrophage",
            "ChP Macrophages",
        ], "Oranges"),
        ("Microglia", [
            "Yolk sac macrophages",
            "Microglia",
            "ChP Microglia",
        ], "Greys"),
        ("Granulocyte", [
            "HSC",
            "CMP",
            "GMP",
            "Promyelocytes",
            "Metamyelocytes",
            "mature granulocytes",
        ], "YlGn"),
        ("Lymphoid", [
            "HSC",
            "CLP",
            "ILC precursors",
            "CD56-bright NK cells",
            "B lineage",
        ], "PuBu"),
        ("Mega/Platelet", [
            "HSC",
            "CMP",
            "MEP",
            "MEMP-MEP",
            "Megakaryocytes",
            "Platelets",
        ], "RdPu"),
        ("Mast", [
            "HSC",
            "myeloid progenitor",
            "Mast cells",
        ], "pink"),
    ],

    "Neural crest": [
        ("NC", [
            "Pre-EMT neural crest epithelia",
            "Pre-EMT neural crest",
            "Neural crest progenitor",
            "Schwann cell precursor",
            "Schwann cell",
            "Peripheral neuroblast",
            "Peripheral neuron",
            "LTMR mechanoreceptor",
        ], "Spectral"),
    ],

    "Neural": [
        ("Neuro", [
            "Neuroepithelial cell",
            "Forebrain radial glia",
            "Ventral forebrain radial glia",
            "Anteromedial cerebral pole radial glia",
            "Midbrain hem radial glia",
            "Isthmus radial glia",
            "Spinal cord radial glia",
            "Forebrain progenitor",
            "Midbrain hem progenitor",
            "Midbrain progenitor",
            "Midbrain-hindbrain progenitor",
            "Midbrain intermediate progenitor",
            "Forebrain neuroblast",
            "Midbrain neuroblast",
            "Forebrain neuron",
            "Pontine nuclei lineage migrating neuron",
            "Midbrain interneuron",
            "Hypothalamus interneuron",
        ], "plasma"),
        ("Oligo", [
            "Oligodendrocyte progenitor",
            "Committed oligodendrocyte progenitor",
            "Pre-oligodendrocyte",
            "Oligodendrocyte",
        ], "Blues"),
    ],

    "Epithelial": [
        ("ChP_Epi", [
            "Roof plate / ependymal-like epithelium",
            "ChP Epithelial progenitor",
            "ChP Epithelia",
            "ChP Differentiated epithelia",
            "ChP Ciliogenic progenitor",
        ], "BuPu"),
    ],
}

def _gradient_hex(cmap_name, n, lo=0.25, hi=0.9):
    cmap = plt.get_cmap(cmap_name)
    xs = np.linspace(lo, hi, max(n, 1))
    return [mcolors.to_hex(cmap(x)) for x in xs]

def apply_lineage_palette(adata_subset, superclass_name, celltype_col="celltype"):
    # 1) カテゴリを掃除して legend を短くする
    adata_subset.obs[celltype_col] = adata_subset.obs[celltype_col].astype("category")
    adata_subset.obs[celltype_col] = adata_subset.obs[celltype_col].cat.remove_unused_categories()
    present = list(adata_subset.obs[celltype_col].cat.categories)

    # 2) 系列→色 の辞書を作る
    mapping = {}
    ordered = []

    for lineage_name, seq, cmap in LINEAGES.get(superclass_name, []):
        seq_present = [x for x in seq if x in present and x not in ordered]
        if not seq_present:
            continue
        cols = _gradient_hex(cmap, len(seq_present))
        for ct, col in zip(seq_present, cols):
            mapping[ct] = col
        ordered.extend(seq_present)

    # 3) 余り（系列に入ってないやつ）
    leftovers = [ct for ct in present if ct not in ordered]
    tab = plt.get_cmap("tab20").colors
    for i, ct in enumerate(sorted(leftovers)):
        mapping[ct] = mcolors.to_hex(tab[i % len(tab)])

    # 4) legend の順序：系列順 → 余り
    new_order = ordered + [ct for ct in present if ct in leftovers]
    adata_subset.obs[celltype_col] = adata_subset.obs[celltype_col].cat.reorder_categories(
        new_order, ordered=True
    )

    # 5) scanpy/scvelo が参照する uns に色配列を入れる（カテゴリ順に合わせる）
    colors = [mapping[ct] for ct in adata_subset.obs[celltype_col].cat.categories]
    adata_subset.uns[f"{celltype_col}_colors"] = np.array(colors, dtype=str)


# ==========================================
# real vs gen UMAP（旧 eval_model_io から移設。viz/velocity 直下に保存・凡例なし）
# ==========================================
def create_combined_anndata(adata_real, sample_path):
    npz = np.load(sample_path, allow_pickle=True)
    if "cell_gen" not in npz:
        print("[velocity] 'cell_gen' not in npz; skip real-vs-gen umap")
        return None, None
    cell_gen = npz["cell_gen"]
    adata_gen = anndata.AnnData(np.asarray(cell_gen, dtype=np.float32))
    if adata_gen.n_vars == adata_real.n_vars:
        adata_gen.var_names = adata_real.var_names
    adata_real.obs["cell_origin"] = "Real"
    adata_gen.obs["cell_origin"] = "Generated"
    adata_gen.obs["final_annotation"] = "Generated"
    combined = anndata.concat([adata_real, adata_gen], axis=0)
    combined.obs["cell_origin"] = combined.obs["cell_origin"].astype("category")
    combined.obs["final_annotation"] = combined.obs["final_annotation"].astype("category")
    return combined, adata_gen


def plot_umap_analysis(combined, output_dir, max_cells=0):
    """real vs gen の UMAP（2 panel）。**凡例なし**。output_dir 直下に umap_analysis.png。
    max_cells<=0 は制限なし（real+gen 全 cell）。"""
    if max_cells and max_cells > 0 and combined.n_obs > max_cells:
        idx = np.random.choice(combined.n_obs, max_cells, replace=False)
        combined = combined[idx].copy()
    combined.X = R.sanitize(R.to_dense(combined.X))   # z-score 済み前提、NaN/Inf 除去のみ
    R.embed_umap(combined)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    sc.pl.umap(combined, color="cell_origin", ax=axes[0], show=False, legend_loc=None,
               palette={"Real": "lightgray", "Generated": "red"},
               title="Real (gray) vs Generated (red)", size=20, alpha=0.6)
    sc.pl.umap(combined, color="final_annotation", ax=axes[1], show=False, legend_loc=None,
               title="annotation (+Generated)", size=20)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "umap_analysis.png"), dpi=200, bbox_inches="tight"); plt.close()
    print("[velocity] umap_analysis saved (no legend)")


# ==========================================
# 3. メイン処理
# ==========================================
@torch.no_grad()
def main():
    args = create_argparser().parse_args()
    args.data_dir = R.resolve_path(args.data_dir)
    args.edge_tsv_path = R.resolve_path(args.edge_tsv_path)
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    # scVeloの図のデフォルト設定
    scv.set_figure_params(transparent=False)

    # --output_dir 未指定なら --model_path から {base}/viz/velocity を導出（単体実行用）
    out_root = args.output_dir
    if not out_root:
        b = run_paths.infer_run_base(args.model_path) if args.model_path else ""
        out_root = os.path.join(b, "viz", ROLE) if b else os.getcwd()
    base_output_dir = os.path.join(out_root, "velocity_by_superclass")
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Output will be saved to: {base_output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 推論用モデルの構築とロード（新モデル: _restore.build_model 経由）
    print("Loading model and calculating velocity_ode...")
    cfg = R.normalize_config(R.load_config(args.config))
    diffusion = R.build_diffusion(cfg["diffusion_steps"])
    gene_list = R.load_gene_list(args.data_dir)
    model = R.build_model(cfg, args.model_path, gene_list, diffusion.num_timesteps,
                          args.edge_tsv_path, device)
    if not R.has_ode_branch(model):
        print("[velocity] plain baseline: no ODE branch, skip"); return

    # データのロード
    print(f"Loading real data from {args.data_dir}...")
    adata = sc.read_h5ad(args.data_dir)
    # max_cells<=0 は「制限なし（全 cell）」。正の値のときだけサブサンプル。
    if args.max_cells and args.max_cells > 0 and adata.n_obs > args.max_cells:
        idx = np.random.choice(adata.n_obs, args.max_cells, replace=False)
        adata = adata[idx].copy()
    print(f"[velocity] using {adata.n_obs} cells (max_cells={args.max_cells or 'unlimited'})")

    # ODEモデルを用いた Velocity の計算
    X = R.to_dense(adata.X).astype(np.float32)
    adata.X = X
    V = R.compute_velocity(model, X, args.velocity_t, device)
    V[~np.isfinite(V)] = 0.0

    if "X" not in adata.layers:
        adata.layers["X"] = adata.X.copy()
    adata.layers["velocity_ode"] = V
    print("Base velocity calculation completed.")

    # real vs gen UMAP（旧 eval_io から移設。viz/velocity 直下＝out_root に凡例なしで保存）
    if args.sample_path and os.path.exists(args.sample_path):
        adata_real = adata.copy()
        label_col = R.auto_label_col(adata_real)
        adata_real.obs["final_annotation"] = (adata_real.obs[label_col].astype(str) if label_col else "real")
        combined, _ = create_combined_anndata(adata_real, args.sample_path)
        if combined is not None:
            try:
                plot_umap_analysis(combined, out_root, max_cells=args.max_cells)
            except Exception as e:
                print(f"[velocity] real-vs-gen umap failed ({type(e).__name__}: {e})")

    color_col = R.auto_label_col(adata) or "celltype"

    # ----------------------------------------
    # Superclassごとの分割と処理
    # ----------------------------------------
    if args.group_col in adata.obs.columns:
        superclasses = adata.obs[args.group_col].unique()
        print(f"\nFound {len(superclasses)} {args.group_col}: {superclasses}")
        for sc_name in superclasses:
            # データの切り出し
            adata_sub = adata[adata.obs[args.group_col] == sc_name].copy()

            # 細胞数が極端に少ない場合はエラーになるためスキップ
            if adata_sub.n_obs < args.min_cells:
                print(f"\nSkipping {sc_name} because it has only {adata_sub.n_obs} cells.")
                continue

            try:
                run_velocity_pipeline(adata_sub, str(sc_name), base_output_dir, color_col, args.n_jobs)
            except Exception as e:
                print(f"[velocity] {sc_name} failed ({type(e).__name__}: {e}); continue")
    else:
        print(f"\n'{args.group_col}' column not found in adata.obs. Running on whole dataset.")
        try:
            run_velocity_pipeline(adata, "all", base_output_dir, color_col, args.n_jobs)
        except Exception as e:
            print(f"[velocity] whole failed ({type(e).__name__}: {e})")

    print(f"\n=== All processing complete. Files saved in {base_output_dir} ===")


def create_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--sample_path", default="", help="生成サンプル npz（あれば real vs gen UMAP を viz/velocity 直下に描く）")
    p.add_argument("--data_dir", default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad")
    p.add_argument("--edge_tsv_path", default="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv")
    p.add_argument("--output_dir", default="")
    p.add_argument("--group_col", default="Superclass")
    p.add_argument("--velocity_t", type=float, default=0.0)
    p.add_argument("--max_cells", type=int, default=0,
                   help="velocity/UMAP に使う cell 上限。0 以下=制限なし（全 cell）。重い場合のみ正の値で間引く")
    p.add_argument("--min_cells", type=int, default=15)
    p.add_argument("--n_jobs", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    return p


if __name__ == "__main__":
    main()
