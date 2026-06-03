#!/usr/bin/env python3
"""
Visualization script for training and sampling results
Replicates the logic from script_diffusion_umap_pbmc68k_soft20251106.ipynb
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scanpy as sc
import anndata
import torch
from datetime import datetime
import re

# -----------------------------------------------------------------------------
# 1. Setup & Utilities
# -----------------------------------------------------------------------------

def setup_output_directory(model_name, total_steps, base_output_dir=None):
    """Create unique output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if base_output_dir:
        # Use base_output_dir from train script
        viz_output_dir = os.path.join(base_output_dir, f"{timestamp}_viz")
        
        # Handle existing directory with suffix
        counter = 1
        original_viz_output_dir = viz_output_dir
        while os.path.exists(viz_output_dir):
            viz_output_dir = f"{original_viz_output_dir}_{counter}"
            counter += 1
        
        os.makedirs(viz_output_dir, exist_ok=True)
        return viz_output_dir
    else:
        # Fallback to original behavior
        dir_name = f"{model_name}_{total_steps}steps_{timestamp}"
        output_dir = f"output/analysis/{dir_name}"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

def save_metadata(output_dir, model_path, sample_path, loss_path):
    metadata_file = os.path.join(output_dir, "analysis_metadata.txt")
    with open(metadata_file, 'w') as f:
        f.write(f"Analysis Metadata\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Sample Path: {sample_path}\n")
        f.write(f"Output Directory: {output_dir}\n")
    return metadata_file

def save_parser_args(output_dir, args):
    """Persist parser arguments used for this run."""
    args_file = os.path.join(output_dir, "parser_args.txt")
    with open(args_file, 'w') as f:
        f.write("Parser Arguments\n")
        f.write(f"Saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for key, value in sorted(vars(args).items()):
            if isinstance(value, anndata.AnnData):
                desc = f"AnnData(shape={value.shape}, obs_keys=list({list(value.obs_keys())}))"
            else:
                desc = str(value)
            f.write(f"{key}: {desc}\n")
    return args_file

# -----------------------------------------------------------------------------
# 2. Parameter Analysis (Sorted & Robust)
# -----------------------------------------------------------------------------

def extract_weights(path):
    """Extract weights handling both 'ode_model' and 'ODE' keys"""
    if not os.path.exists(path):
        return None
    
    try:
        ckpt = torch.load(path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
    except Exception as e:
        print(f"Error loading checkpoint {path}: {e}")
        return None

    weights = {}
    # Check both naming conventions
    prefixes = ["ODE", "ode_model"]
    
    for param_name in ["W", "mask", "b", "gamma"]:
        found = False
        for prefix in prefixes:
            key = f"{prefix}.{param_name}"
            if key in state:
                weights[f"ODE.{param_name}"] = state[key]
                found = True
                break
        if not found:
            weights[f"ODE.{param_name}"] = None

    # ML parameters
    ml_keys = [k for k in state.keys() if ("ml_model" in k or "ML" in k) and "weight" in k]
    ml_params = []
    for k in ml_keys:
        w = state[k].flatten().detach().cpu().numpy()
        ml_params.append(w)
    
    weights["ML.params"] = np.concatenate(ml_params) if len(ml_params) > 0 else np.array([])
    return weights

def find_checkpoint_models(model_path):
    """Find models and sort by step number (Ascending)"""
    base_dir = os.path.dirname(model_path)
    model_paths = {}
    
    if os.path.exists(base_dir):
        files = [f for f in os.listdir(base_dir) if f.startswith('model') and f.endswith('.pt')]
        
        def get_step_number(filename):
            nums = re.findall(r'\d+', filename)
            if nums:
                return int(nums[-1])
            return float('inf')

        # Sort files based on step number
        files.sort(key=get_step_number)
        
        for filename in files:
            full_path = os.path.join(base_dir, filename)
            model_paths[filename] = full_path
    
    # Fallback if no list found
    if not model_paths and os.path.exists(model_path):
        filename = os.path.basename(model_path)
        model_paths[filename] = model_path
    
    return model_paths

def plot_parameter_distributions(model_paths, output_dir):
    if not model_paths:
        return None
    
    weights_dict = {}
    for label, path in model_paths.items():
        weights = extract_weights(path)
        if weights is not None:
            weights_dict[label] = weights
    
    if not weights_dict:
        return None
    
    n_cols = len(weights_dict)
    fig, axes = plt.subplots(5, n_cols, figsize=(4 * n_cols, 15))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
        
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for col_idx, (label, wdict) in enumerate(weights_dict.items()):
        W = wdict.get("ODE.W")
        mask = wdict.get("ODE.mask")
        b = wdict.get("ODE.b")
        gamma = wdict.get("ODE.gamma")
        ml_params = wdict.get("ML.params")

        def plot_hist(data, ax, title, color):
            if data is None or len(data) == 0:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            else:
                ax.hist(data, bins=60, color=color, alpha=0.7)
                stats = f"mean:{np.mean(data):.2e}\nstd:{np.std(data):.2e}"
                ax.text(0.95, 0.95, stats, transform=ax.transAxes, ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.5))
            ax.set_title(title, fontsize=9)

        # Plotting
        w_on = W[mask.bool()].flatten().detach().cpu().numpy() if W is not None and mask is not None else []
        plot_hist(w_on, axes[0, col_idx], f"{label}\nW(Known)", "C0")

        w_off = W[(1 - mask).bool()].flatten().detach().cpu().numpy() if W is not None and mask is not None else []
        plot_hist(w_off, axes[1, col_idx], "W(Unknown)", "C1")

        b_vals = b.flatten().detach().cpu().numpy() if b is not None else []
        plot_hist(b_vals, axes[2, col_idx], "Bias b", "C2")

        g_vals = torch.nn.functional.softplus(gamma).flatten().detach().cpu().numpy() if gamma is not None else [] #⭐️新規変更点20260413 gamma
        plot_hist(g_vals, axes[3, col_idx], "Gamma", "C3")

        plot_hist(ml_params, axes[4, col_idx], "CellUnet", "C4")

    # Labels
    row_labels = ["W(Known)", "W(Unknown)", "Bias", "Gamma", "CellUnet"]
    for row in range(5):
        axes[row, 0].set_ylabel(row_labels[row], fontweight='bold')

    plt.suptitle("Parameter Distributions (Sorted by Steps)", fontsize=16)
    save_path = os.path.join(output_dir, "parameter_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

# -----------------------------------------------------------------------------
# 3. Data Processing & UMAP (Replicating Notebook Logic)
# -----------------------------------------------------------------------------

# def load_and_preprocess_real_data():
#     """Load and preprocess real data, ensuring labels exist."""
#     adata_path = 'data_preparation/pbmc68k.h5ad'
#     if not os.path.exists(adata_path):
#         # Fallback for different execution contexts
#         adata_path = '/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad'
    
#     print(f"Loading real data from {adata_path}...")
#     adata = sc.read_h5ad(adata_path)
    
#     # --- Preprocessing (Same as Notebook) ---
#     print("Preprocessing real data...")
#     sc.pp.filter_cells(adata, min_genes=10)
#     sc.pp.filter_genes(adata, min_cells=3)
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     sc.pp.highly_variable_genes(adata, n_top_genes=1024, subset=True, inplace=True)
#     sc.pp.scale(adata, zero_center=True)
    
#     # --- Label Handling (The Fix) ---
#     # ノートブックと同様、ラベルがなければ計算する
#     label_col = None
    
#     # 1. 既存の有力なラベルを探す
#     for cand in ['Bulk_labels', 'bulk_labels', 'Louvain', 'louvain', 'celltype']:
#         if cand in adata.obs.columns:
#             label_col = cand
#             print(f"✅ Found existing label column: '{label_col}'")
#             break
            
#     # 2. なければ Louvain クラスタリングを実行
#     if label_col is None:
#         print("⚠️ No labels found. Running Louvain clustering (replicating notebook logic)...")
#         sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
#         sc.tl.louvain(adata, key_added='louvain')
#         label_col = 'louvain'
#         print("✅ Labels generated.")

#     # 3. 統一的な列名 'final_annotation' にコピー
#     adata.obs['final_annotation'] = adata.obs[label_col].astype(str)
    
#     return adata

def create_combined_anndata(adata_real, sample_path):
    """Combine real and generated data."""
    print(f"Loading generated samples from {sample_path}...")
    npz_data = np.load(sample_path, allow_pickle=True)
    
    if 'cell_gen' not in npz_data:
        print("Error: 'cell_gen' key not found in npz file.")
        return None, None
        
    cell_gen = npz_data['cell_gen']
    
    # Create AnnData for Generated
    adata_gen = anndata.AnnData(cell_gen)
    adata_gen.var_names = adata_real.var_names # Align genes
    
    # --- Set Metadata ---
    # 1. Cell Origin
    adata_real.obs['cell_origin'] = 'Real'
    adata_gen.obs['cell_origin'] = 'Generated'
    
    # 2. Cell Type (Final Annotation)
    # Real data already has 'final_annotation' from load function
    # Generated data gets 'Generated' label
    adata_gen.obs['final_annotation'] = 'Generated'
    
    # --- Concatenate ---
    print("Concatenating datasets...")
    adata_combined = anndata.concat([adata_real, adata_gen], axis=0)
    
    # Convert to category for plotting
    adata_combined.obs['cell_origin'] = adata_combined.obs['cell_origin'].astype('category')
    adata_combined.obs['final_annotation'] = adata_combined.obs['final_annotation'].astype('category')
    
    return adata_combined, adata_gen

def plot_umap_analysis(adata_combined, output_dir):
    """Run PCA/Neighbors/UMAP and Plot"""
    print("Running UMAP pipeline on combined data...")
    
    # ノートブックのパラメータに準拠
    sc.pp.scale(adata_combined, max_value=10)
    sc.tl.pca(adata_combined, svd_solver='arpack', n_comps=50)
    sc.pp.neighbors(adata_combined, n_neighbors=15, n_pcs=40)
    sc.tl.umap(adata_combined)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: Origin (Real vs Generated)
    # Generated (Red) on top of Real (Gray)
    sc.pl.umap(adata_combined, color="cell_origin", ax=axes[0], show=False, 
               palette={"Real": "lightgray", "Generated": "red"},
               title="Real (Gray) vs Generated (Red)",
               size=20, alpha=0.6)
    
    # Plot 2: Cell Types
    sc.pl.umap(adata_combined, color="final_annotation", ax=axes[1], show=False, 
               title="Cell Types (with Generated cluster)",
               size=20)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, "umap_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def plot_gene_correlation(adata_real, adata_gen, output_dir):
    """Simple gene correlation plot"""
    # Calculate means (handle sparse matrices if necessary)
    real_mean = np.mean(adata_real.X, axis=0)
    if hasattr(real_mean, 'A1'): real_mean = real_mean.A1
        
    gen_mean = np.mean(adata_gen.X, axis=0)
    if hasattr(gen_mean, 'A1'): gen_mean = gen_mean.A1
        
    plt.figure(figsize=(7, 7))
    plt.scatter(real_mean, gen_mean, s=10, alpha=0.5, c='purple')
    
    # 45-degree line
    lims = [
        np.min([plt.xlim(), plt.ylim()]),  
        np.max([plt.xlim(), plt.ylim()])
    ]
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
    
    corr = np.corrcoef(real_mean, gen_mean)[0, 1]
    plt.title(f"Gene Expression Correlation (R={corr:.3f})")
    plt.xlabel("Real Data Mean")
    plt.ylabel("Generated Data Mean")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(output_dir, "gene_correlation.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def plot_loss_curves(loss_csv_path, output_dir):
    if not os.path.exists(loss_csv_path):
        return None

    try:
        df = pd.read_csv(loss_csv_path)
    except Exception as e:
        print(f"Error reading loss CSV: {e}")
        return None

    fig, ax1 = plt.subplots(figsize=(12, 8))

    # 左軸：メイン損失
    if 'original_loss' in df.columns:
        ax1.plot(
            df['step'], df['original_loss'],
            label='Reconst Error',
            linestyle='-',
            color='blue',
            linewidth=2,
            alpha=0.8,
        )

    if 'total_loss' in df.columns:
        ax1.plot(
            df['step'], df['total_loss'],
            label='Total Loss',
            linestyle='-',
            color='red',
            linewidth=2,
            alpha=0.8,
        )

    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss Value')
    ax1.set_title('Training Loss Curves')
    ax1.grid(True, alpha=0.3)

    # 右軸：正則化（緑）
    ax2 = None
    if 'reg_weighted' in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(
            df['step'], df['reg_weighted'],
            label='Weighted Reg (λ*Reg)',
            linestyle='-',
            color='green',
            linewidth=1.8,
            alpha=0.8,
        )
        ax2.set_ylabel('Regularization Loss', color='green')
        ax2.tick_params(axis='y', colors='green')
        ax2.spines['right'].set_color('green')

    # legend を “ax1 と ax2” から集めて結合（重複除去つき）
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = (ax2.get_legend_handles_labels() if ax2 is not None else ([], []))

    # 同名ラベルがあったら潰す（後勝ちで1個になる）
    uniq = dict(zip(l1 + l2, h1 + h2))
    ax1.legend(uniq.values(), uniq.keys(), loc='upper right')

    os.makedirs(output_dir, exist_ok=True)
    loss_plot_path = os.path.join(output_dir, "loss_curves_compare.png")
    fig.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return loss_plot_path


def plot_loss_curves_simple(loss_csv_path, output_dir):
    if not os.path.exists(loss_csv_path):
        return None

    try:
        df = pd.read_csv(loss_csv_path)
    except Exception as e:
        print(f"Error reading loss CSV: {e}")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # Reconst Error（青）
    if 'original_loss' in df.columns:
        ax.plot(
            df['step'],
            df['original_loss'],
            label='Reconst Error',
            color='blue',
            linewidth=2,
            alpha=0.8
        )

    # Total Loss（赤）
    if 'total_loss' in df.columns:
        ax.plot(
            df['step'],
            df['total_loss'],
            label='Total Loss',
            color='red',
            linewidth=2,
            alpha=0.8
        )

    # Weighted Reg（緑）
    if 'reg_weighted' in df.columns:
        ax.plot(
            df['step'],
            df['reg_weighted'],
            label='Weighted Reg (λ*Reg)',
            color='green',
            linewidth=2,
            alpha=0.8
        )

    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss Value')
    ax.set_title('Training Loss Curves')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "loss_curves_simple.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return save_path

# Velocity UMAP Plotting (Using ODE Model)
def plot_velocity_umap(adata_real, model_path, output_dir, edge_tsv_path):
    """
    ベストモデルの重みをロードし、Velocity UMAPを計算して作図する関数。
    既存ファイルを汚さないよう、必要なインポートや処理を内部に閉じ込めています。
    """
    import scvelo as scv
    import torch
    import scipy.sparse as sp
    import sys
    
    # scDiffusionのモジュールを読み込めるようパスを追加
    if "/home/suzuki/Projects/scDiffusion" not in sys.path:
        sys.path.insert(0, "/home/suzuki/Projects/scDiffusion")
    
    from guided_diffusion.cell_model import Cell_Unet
    from ODE.ode_analysis1106 import GeneODE, ODE_ML_Hybrid

    print("Generating Velocity UMAP from the trained model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gene_list = list(adata_real.var_names)
    
    # 1. モデルの構築とロード (ode_softは一旦Trueとしています)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    ode = GeneODE(gene_list=gene_list, edge_tsv_path=edge_tsv_path, soft=True, device=device)
    ml_model = Cell_Unet(input_dim=len(gene_list))
    hybrid_model = ODE_ML_Hybrid(ode_model=ode, ml_model=ml_model, timesteps=1000).to(device)
    hybrid_model.load_state_dict(state_dict, strict=False)
    hybrid_model.eval()

    # 2. ODEモデルを用いた Velocity の計算
    X = adata_real.X
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
            # ODEモデルの出力 (dx/dt) を取得
            vb = hybrid_model.ode_model(xb).detach().cpu().numpy()
            V[start:end] = np.asarray(vb, dtype=np.float32)


    # # ---- scVelo安定化のため CSR + float64 に統一 ----
    
    # V[~np.isfinite(V)] = 0.0
    # import scipy.sparse as sp
    # adata_real.X = sp.csr_matrix(
    #     np.asarray(adata_real.X, dtype=np.float64)
    # )

    # adata_real.layers["X"] = adata_real.X.copy()

    # adata_real.layers["velocity_ode"] = sp.csr_matrix(
    #     V.astype(np.float64)
    # )            
    V[~np.isfinite(V)] = 0.0
    
    if "X" not in adata_real.layers:
        adata_real.layers["X"] = adata_real.X.copy()
    adata_real.layers["velocity_ode"] = V

    # 3. UMAP座標の確認・生成とVelocity描画
    if "X_pca" not in adata_real.obsm:
        sc.tl.pca(adata_real, svd_solver='arpack', n_comps=50)
    if "neighbors" not in adata_real.uns:
        scv.pp.neighbors(adata_real, n_neighbors=15, n_pcs=40)
    if "X_umap" not in adata_real.obsm:
        sc.tl.umap(adata_real)

    # グラフの作成
    scv.tl.velocity_graph(adata_real, 
                          vkey="velocity_ode", 
                          xkey="X",# これが無いと.layers["spliced"]を参照する
                          backend="loky",
                          n_jobs=32 )
    scv.tl.velocity_embedding(adata_real, basis="umap", vkey="velocity_ode")

    # 作図と保存
    save_path = os.path.join(output_dir, "velocity_umap.png")
    scv.pl.velocity_embedding_stream(
        adata_real,
        basis="umap",
        vkey="velocity_ode",
        color="final_annotation",
        legend_loc="right margin",
        show=False
    )
    import matplotlib.pyplot as plt
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved Velocity UMAP to {save_path}")


# -----------------------------------------------------------------------------
# 4. Main Execution
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--sample_path', required=True)
    parser.add_argument('--loss_path', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--total_steps', required=True)
    parser.add_argument('--skip_umap', action='store_true')
    parser.add_argument('--data_dir', type=str, default="", help="Path to preprocessed real data")
    parser.add_argument('--output_dir', type=str, default="", help="Output directory passed from train script")
    args = parser.parse_args()
    
    print("=== Starting Visualization Analysis ===")
    output_dir = setup_output_directory(args.model_name, args.total_steps, args.output_dir)
    save_metadata(output_dir, args.model_path, args.sample_path, args.loss_path)
    save_parser_args(output_dir, args)
    print(f"Output Directory: {output_dir}")

    # 1. Loss
    plot_loss_curves(args.loss_path, output_dir)
    plot_loss_curves_simple(args.loss_path, output_dir)
    # 2. Parameter Analysis (Sorted)
    print("Analyzing parameters...")
    model_paths = find_checkpoint_models(args.model_path)
    plot_parameter_distributions(model_paths, output_dir)

    # 3. Data Processing (Real + Generated)
    #    Load preprocessed real data
    if args.data_dir:
        print(f"Loading preprocessed data from: {args.data_dir}")
        adata_real = sc.read_h5ad(args.data_dir)
        adata_real.obs['final_annotation'] = adata_real.obs["celltype"].astype(str)
        adata_combined, adata_gen = create_combined_anndata(adata_real, args.sample_path)
        
        if adata_combined is None:
            print("Error combining data. Exiting.")
            return

        # 4. Gene Correlation
        print("Plotting gene correlation...")
        plot_gene_correlation(adata_real, adata_gen, output_dir)

        # 5. UMAP
        if not args.skip_umap:
            print("Generating UMAP...")
            # plot_umap_analysis(adata_combined, output_dir)

            # # --- ここから追加 ---
            # # 6. Velocity UMAP
            # # trainスクリプトと同一のTSVパスを指定します
            # edge_tsv = "/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv" #間違ってたfound at 20260416
            # plot_velocity_umap(adata_real, args.model_path, output_dir, edge_tsv_path=edge_tsv)
            # # --- ここまで追加 ---
            
    else:
        print("No data_dir provided. Skipping gene correlation and UMAP analysis.")

    print(f"=== Analysis Complete. Saved to {output_dir} ===")

if __name__ == "__main__":
    main()