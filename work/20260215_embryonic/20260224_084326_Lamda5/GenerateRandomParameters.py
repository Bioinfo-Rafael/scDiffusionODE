#!/usr/bin/env python3
"""
Velocity UMAP Visualization Script (Randomized W, 10 iterations: PCA & scvelo)
Wx＋ｂについてランダムサンプリングしたパラメタで作図。実験４
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
    
    scv.set_figure_params(transparent=False)

    # 保存用トップディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(os.getcwd(), f"{timestamp}_random_velocity")
    
    # 指定されたディレクトリ構造を事前に作成
    plot_types = ['velocity_embedding_stream', 'velocity_embedding', 'velocity_embedding_grid']
    methods = ['PCA', 'scvelo.tl.velocity_embedding']
    
    for ptype in plot_types:
        for method in methods:
            os.makedirs(os.path.join(base_output_dir, ptype, method), exist_ok=True)
            
    print(f"Output will be saved to: {base_output_dir}")
    
    return base_output_dir

# ==========================================
# 2. パイプライン・作図関数の定義
# ==========================================
def compute_velocity_umap_from_layer(adata, layer_key="velocity_ode", out_key="velocity_umap", n_components=2):
    if layer_key not in adata.layers:
        raise ValueError(f"Layer '{layer_key}' not found in adata.layers.")
        
    V = adata.layers[layer_key]
    if sp.issparse(V):
        V = V.toarray()
    
    V = np.asarray(V, dtype=np.float32)
    V[~np.isfinite(V)] = 0.0
    
    pca = PCA(n_components=n_components)
    v_pca = pca.fit_transform(V)
    
    adata.obsm[out_key] = v_pca

def run_velocity_pipeline(adata_subset, group_name, base_outdir, method, iteration_idx):
    """
    iteration_idx: 1~5 の数値
    method: 'PCA' または 'scvelo.tl.velocity_embedding'
    """
    print(f"  --> Plotting {group_name} (Method: {method}, Iteration: {iteration_idx}/5)")
    
    # --- 1. UMAP ---
    sc.tl.pca(adata_subset, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata_subset, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_subset)
    
    # --- 2. 手法に応じたVelocity埋め込みの計算 ---
    if method == 'scvelo.tl.velocity_embedding':
        scv.tl.velocity_graph(
            adata_subset, 
            vkey="velocity_ode", 
            xkey="X",
            backend="loky",
            n_jobs=32
        )
        scv.tl.velocity_embedding(adata_subset, basis="umap", vkey="velocity_ode")
    else: # PCA
        compute_velocity_umap_from_layer(
            adata_subset, 
            layer_key="velocity_ode", 
            out_key="velocity_umap"
        )
    
    # --- 3. 配色の適用 ---
    color_col = "celltype"
    apply_lineage_palette(adata_subset, group_name, celltype_col=color_col)
    
    palette = adata_subset.uns.get(f"{color_col}_colors", None)
    if palette is not None:
        palette = list(palette) 

    # ファイル名に使用する安全なグループ名
    group_safe = group_name.replace(" ", "_")
    filename = f"{group_safe}_iter{iteration_idx}.png"

    # 共通の描画引数 (PCAとscveloで vkey の有無が変わる点のみ調整)
    kwargs = {
        "color": color_col, 
        "palette": palette, 
        "show": False, 
        "legend_loc": "right margin",
        "size": 5, 
        "alpha": 1
    }
    if method == 'scvelo.tl.velocity_embedding':
        kwargs["vkey"] = "velocity_ode"

    # --- 4. 作図 ---
    # ① Stream embedding
    save_path = os.path.join(base_outdir, "velocity_embedding_stream", method, filename)
    scv.pl.velocity_embedding_stream(adata_subset, basis="umap", **kwargs)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # ② Arrow embedding (単一細胞レベル)
    save_path = os.path.join(base_outdir, "velocity_embedding", method, filename)
    scv.pl.velocity_embedding(adata_subset, basis="umap", arrow_length=3, arrow_size=2, **kwargs)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # ③ Grid embedding
    save_path = os.path.join(base_outdir, "velocity_embedding_grid", method, filename)
    scv.pl.velocity_embedding_grid(adata_subset, basis="umap", **kwargs)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    # ※PAGAは今回保存先が指定されていないためコメントアウトしてスキップします
    # ...


# ----------------------------
# 配色割り当てのための系列定義 (変更なし)
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
    adata_subset.obs[celltype_col] = adata_subset.obs[celltype_col].astype("category")
    adata_subset.obs[celltype_col] = adata_subset.obs[celltype_col].cat.remove_unused_categories()
    present = list(adata_subset.obs[celltype_col].cat.categories)

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

    leftovers = [ct for ct in present if ct not in ordered]
    tab = plt.get_cmap("tab20").colors
    for i, ct in enumerate(sorted(leftovers)):
        mapping[ct] = mcolors.to_hex(tab[i % len(tab)])

    new_order = ordered + [ct for ct in present if ct in leftovers]
    adata_subset.obs[celltype_col] = adata_subset.obs[celltype_col].cat.reorder_categories(
        new_order, ordered=True
    )

    colors = [mapping[ct] for ct in adata_subset.obs[celltype_col].cat.categories]
    adata_subset.uns[f"{celltype_col}_colors"] = np.array(colors, dtype=str)

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    base_output_dir = setup_environment()
    
    try:
        from guided_diffusion.cell_model import Cell_Unet
        from ODE.ode_analysis1106 import GeneODE, ODE_ML_Hybrid
    except ImportError as e:
        print(f"Error importing scDiffusion modules: {e}")
        return

    # データのロード
    adata_path = '/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad'
    print(f"Loading real data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    model_path = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260224_084326_Lamda5/20260224_084328_train/checkpoints/pbmc68k_soft_20251127_hvg1024/ema_0.9999_000000.pt"
    edge_tsv_path = "/home/suzuki/Projects/scDiffusion/external_data/Mouse_tf_target_edges.tsv"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_list = list(adata.var_names)

    print("Loading model structure (parameters will be overwritten by random sampling)...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    ode = GeneODE(gene_list=gene_list, edge_tsv_path=edge_tsv_path, soft=True, device=device)
    ml_model = Cell_Unet(input_dim=len(gene_list))
    hybrid_model = ODE_ML_Hybrid(ode_model=ode, ml_model=ml_model, timesteps=1000).to(device)
    
    # モデル構造の形式だけ一致させるために一度ロードします
    hybrid_model.load_state_dict(state_dict, strict=False)
    hybrid_model.eval()

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    batch_size = 1024

    # ----------------------------------------
    # 計10回のループ処理
    # ----------------------------------------
    for total_iter in range(10):
        # 最初の5回はPCA、残りの5回はscvelo
        if total_iter < 5:
            method = "PCA"
            iter_idx = total_iter + 1
        else:
            method = "scvelo.tl.velocity_embedding"
            iter_idx = (total_iter - 5) + 1
            
        print(f"\n==================================================")
        print(f"Global Iteration {total_iter+1}/10 | Method: {method} | Iter: {iter_idx}/5")
        print(f"==================================================")
        
        # ★ W(モデルのパラメータ)を標準正規分布(randn)でサンプリングして上書き
        with torch.no_grad():
            for param in hybrid_model.ode_model.parameters():
                param.data = torch.randn_like(param.data)

        # ランダム化されたモデルを用いて Velocity の計算
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
            
        # 毎回のランダム計算結果でvelocity_odeを更新
        adata.layers["velocity_ode"] = V
        print(f"-> Base velocity calculated using Randomized W.")

        # Superclassごとの分割と作図処理
        if "Superclass" in adata.obs.columns:
            superclasses = adata.obs["Superclass"].unique()
            for sc_name in superclasses:
                if sc_name != "Erythropoietic":
                    continue
                adata_sub = adata[adata.obs['Superclass'] == sc_name].copy()
                
                if adata_sub.n_obs < 15:
                    continue
                
                run_velocity_pipeline(adata_sub, str(sc_name), base_output_dir, method, iter_idx)
        else:
            print("\n'Superclass' column not found in adata.obs. Skipping split processing.")

    print(f"\n=== All 10 iterations complete. Files saved in {base_output_dir} ===")

if __name__ == "__main__":
    main()