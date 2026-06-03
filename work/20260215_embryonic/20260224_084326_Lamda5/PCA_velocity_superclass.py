#!/usr/bin/env python3
"""
Velocity UMAP Visualization Script (Whole & Superclass Split)
次元削減をPCAで行う
"""

import os
import sys
import numpy as np
import scanpy as sc
import scvelo as scv
import anndata
import torch
import scipy.sparse as sp
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
import matplotlib.colors as mcolors
# ==========================================
# 1. 初期設定とパスの追加
# ==========================================
def setup_environment():
    scdiffusion_path = "/home/suzuki/Projects/scDiffusion"
    if scdiffusion_path not in sys.path:
        sys.path.insert(0, scdiffusion_path)
    
    # scVeloの図のデフォルト設定
    # scv.set_figure_params()
    scv.set_figure_params(transparent=False)

    # 保存用ディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(os.getcwd(), f"{timestamp}_PCA_velocity")
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Output will be saved to: {base_output_dir}")
    
    return base_output_dir

# ==========================================
# 2. パイプライン・作図関数の定義
# ==========================================
def compute_velocity_umap_from_layer(adata, layer_key="velocity_ode", out_key="velocity_umap", n_components=2):
    """
    指定したlayerの高次元velocity行列をPCAで2次元化し、obsmに保存する関数。
    scveloのplot関数がデフォルトで参照できるよう、キーは "velocity_umap" 等とする。
    """
    if layer_key not in adata.layers:
        raise ValueError(f"Layer '{layer_key}' not found in adata.layers.")
        
    V = adata.layers[layer_key]
    if sp.issparse(V):
        V = V.toarray()
    
    # 欠損値や無限大を0に置換
    V = np.asarray(V, dtype=np.float32)
    V[~np.isfinite(V)] = 0.0
    
    # PCAで2次元に圧縮
    pca = PCA(n_components=n_components)
    v_pca = pca.fit_transform(V)
    
    print(f"  [PCA Velocity] Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # scveloのデフォルト描画が参照するキーに保存
    adata.obsm[out_key] = v_pca
    # adata.obsm[out_key] = np.random.randn(adata.n_obs, n_components)
    # print(f"  [Random Velocity] Assigned random values to adata.obsm['{out_key}']")

def run_velocity_pipeline(adata_subset, group_name, base_outdir):
    print(f"\n=== Processing Group: {group_name} ===")
    outdir = os.path.join(base_outdir, group_name.replace(" ", "_"))
    os.makedirs(outdir, exist_ok=True)
    
    # --- 1. 細胞のUMAP埋め込みの計算 ---
    sc.tl.pca(adata_subset, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata_subset, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_subset) # adata_subset.obsm["X_umap"] が作成される
    
    # --- 2. Velocity Graph と Velocity Embedding の計算はスキップ ---
    # scv.tl.velocity_graph と scv.tl.velocity_embedding は呼ばない
    
    # --- 3. Velocity のPCA 2次元化 ---
    # 高次元velocityをPCAで2Dに落とし、obsm["velocity_umap"] に保存
    compute_velocity_umap_from_layer(
        adata_subset, 
        layer_key="velocity_ode", 
        out_key="velocity_umap"
    )
    
    # --- 4. 作図 ---
    color_col = "celltype"

    # ★ここで系列色を適用（paletteが未設定でも強制的に作る）
    apply_lineage_palette(adata_subset, group_name, celltype_col=color_col)

    palette = adata_subset.uns.get(f"{color_col}_colors", None)
    if palette is not None:
        palette = list(palette) 



    # プロット関数からは vkey="velocity_ode" を削除。
    # basis="umap" を指定することで、scveloは自動的に adata_subset.obsm["velocity_umap"] を探しに行きます。
    
    # ① Stream embedding
    save_path = os.path.join(outdir, "1_velocity_stream.png")
    scv.pl.velocity_embedding_stream(
        adata_subset, basis="umap", 
        color=color_col, palette=palette, show=False, legend_loc="right margin",
        size=5, alpha=1,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # ② Arrow embedding (単一細胞レベル)
    save_path = os.path.join(outdir, "2_velocity_arrow.png")
    scv.pl.velocity_embedding(
        adata_subset, basis="umap", 
        color=color_col, palette=palette, show=False, arrow_length=3, arrow_size=2, legend_loc="right margin",
        size=5, alpha=1,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # ③ Grid embedding
    save_path = os.path.join(outdir, "3_velocity_grid.png")
    scv.pl.velocity_embedding_grid(
        adata_subset, basis="umap", 
        color=color_col, palette=palette, show=False, legend_loc="right margin",
        size=5, alpha=1,
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # ⑥ PAGA (Velocity Graph非依存の通常PAGAとして実行)
    if 'distances' in adata_subset.obsp:
        if 'neighbors' not in adata_subset.uns:
            adata_subset.uns['neighbors'] = {}
        adata_subset.uns['neighbors']['distances'] = adata_subset.obsp['distances']
        adata_subset.uns['neighbors']['connectivities'] = adata_subset.obsp['connectivities']
        
    try:
        # velocity_graphを計算していないため、vkey指定による方向付きPAGAはエラーになります。
        # 代わりに、vkey引数を削除して非velocityな通常のPAGAとしてトポロジーを計算します。
        scv.tl.paga(adata_subset, groups=color_col)
        save_path = os.path.join(outdir, "6_paga_velocity.png")

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
            show=False
        )

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"PAGA error for {group_name}: {e}")

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
# 3. メイン処理
# ==========================================
def main():
    base_output_dir = setup_environment()
    
    # モジュールのインポート (パス追加後にインポート)
    try:
        from guided_diffusion.cell_model import Cell_Unet
        from ODE.ode_analysis1106 import GeneODE, ODE_ML_Hybrid
    except ImportError as e:
        print(f"Error importing scDiffusion modules: {e}")
        print("Please ensure scdiffusion_path is correctly set.")
        return

    # データのロード
    adata_path = '/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad'
    print(f"Loading real data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    # 推論用モデルのパス（必要に応じて書き換えてください）
    # model_path = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260224_084326_Lamda5/20260224_084328_train/checkpoints/pbmc68k_soft_20251127_hvg1024/ema_0.9999_100000.pt"
    model_path = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260224_084326_Lamda5/20260224_084328_train/checkpoints/pbmc68k_soft_20251127_hvg1024/ema_0.9999_000000.pt"
    edge_tsv_path = "/home/suzuki/Projects/scDiffusion/external_data/Mouse_tf_target_edges.tsv"
    print(f"Loading model from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_list = list(adata.var_names)

    print("Loading model and calculating velocity_ode...")
    # モデルの構築とロード
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    ode = GeneODE(gene_list=gene_list, edge_tsv_path=edge_tsv_path, soft=True, device=device)
    ml_model = Cell_Unet(input_dim=len(gene_list))
    hybrid_model = ODE_ML_Hybrid(ode_model=ode, ml_model=ml_model, timesteps=1000).to(device)
    hybrid_model.load_state_dict(state_dict, strict=False)
    hybrid_model.eval()

    # ODEモデルを用いた Velocity の計算
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    n = X.shape[0]
    batch_size = 1024
    V = np.zeros_like(X, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(X[start:end]).to(device)
            vb = hybrid_model.ode_model(xb).detach().cpu().numpy()
            V[start:end] = np.asarray(vb, dtype=np.float32)

    V[~np.isfinite(V)] = 0.0

    if "X" not in adata.layers:
        adata.layers["X"] = adata.X.copy()
    adata.layers["velocity_ode"] = V
    print("Base velocity calculation completed.")

    # ----------------------------------------
    # パイプラインの実行 (全体とSuperclass分割)
    # ----------------------------------------
    
    # 1. 全体データ (Whole) の処理
    # adata_whole = adata.copy()
    # run_velocity_pipeline(adata_whole, "Whole", base_output_dir)

    # 2. Superclassごとの分割と処理
    if "Superclass" in adata.obs.columns:
        superclasses = adata.obs["Superclass"].unique()
        print(f"\nFound {len(superclasses)} Superclasses: {superclasses}")
        for sc_name in superclasses:
            # データの切り出し
            adata_sub = adata[adata.obs['Superclass'] == sc_name].copy()
            
            # 細胞数が極端に少ない場合はエラーになるためスキップ
            if adata_sub.n_obs < 15:
                print(f"\nSkipping {sc_name} because it has only {adata_sub.n_obs} cells.")
                continue
                
            run_velocity_pipeline(adata_sub, str(sc_name), base_output_dir)
    else:
        print("\n'Superclass' column not found in adata.obs. Skipping split processing.")

    print(f"\n=== All processing complete. Files saved in {base_output_dir} ===")

if __name__ == "__main__":
    main()