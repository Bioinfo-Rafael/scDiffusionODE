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

def setup_output_directory(model_name, total_steps):
    """Create unique output directory"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

        g_vals = gamma.flatten().detach().cpu().numpy() if gamma is not None else []
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

def load_and_preprocess_real_data():
    """Load and preprocess real data, ensuring labels exist."""
    adata_path = 'data_preparation/pbmc68k.h5ad'
    if not os.path.exists(adata_path):
        # Fallback for different execution contexts
        adata_path = '/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad'
    
    print(f"Loading real data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # --- Preprocessing (Same as Notebook) ---
    print("Preprocessing real data...")
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1024, subset=True, inplace=True)
    sc.pp.scale(adata, zero_center=True)
    
    # --- Label Handling (The Fix) ---
    # ノートブックと同様、ラベルがなければ計算する
    label_col = None
    
    # 1. 既存の有力なラベルを探す
    for cand in ['Bulk_labels', 'bulk_labels', 'Louvain', 'louvain', 'celltype']:
        if cand in adata.obs.columns:
            label_col = cand
            print(f"✅ Found existing label column: '{label_col}'")
            break
            
    # 2. なければ Louvain クラスタリングを実行
    if label_col is None:
        print("⚠️ No labels found. Running Louvain clustering (replicating notebook logic)...")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.louvain(adata, key_added='louvain')
        label_col = 'louvain'
        print("✅ Labels generated.")

    # 3. 統一的な列名 'final_annotation' にコピー
    adata.obs['final_annotation'] = adata.obs[label_col].astype(str)
    
    return adata

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
    """Plot loss curves from CSV file"""
    if not os.path.exists(loss_csv_path):
        return None
    
    try:
        df = pd.read_csv(loss_csv_path)
    except Exception as e:
        print(f"Error reading loss CSV: {e}")
        return None
    
    plt.figure(figsize=(12, 8))
    if 'original_loss' in df.columns:
        plt.plot(df['step'], df['original_loss'], label='Original Loss', color='blue', linewidth=2, alpha=0.7)
    if 'total_loss' in df.columns:
        plt.plot(df['step'], df['total_loss'], label='Total Loss', color='red', linewidth=2, linestyle='--')
    
    if 'reg_weighted' in df.columns:
        ax2 = plt.gca().twinx()
        ax2.plot(df['step'], df['reg_weighted'], label='Weighted Reg (λ*Reg)', color='green', linewidth=1.5, alpha=0.6)
        ax2.set_ylabel('Regularization Loss', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss Value')
    plt.title('Training Loss Curves')
    plt.grid(True, alpha=0.3)
    
    lines, labels = plt.gca().get_legend_handles_labels()
    if 'reg_weighted' in df.columns:
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='upper right')
    else:
        plt.legend(loc='upper right')
    
    loss_plot_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return loss_plot_path

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
    parser.add_argument('--data_preprocessed', type=anndata.AnnData, default=None, help="Preprocessed real data AnnData object")
    args = parser.parse_args()
    
    print("=== Starting Visualization Analysis ===")
    output_dir = setup_output_directory(args.model_name, args.total_steps)
    save_metadata(output_dir, args.model_path, args.sample_path, args.loss_path)
    save_parser_args(output_dir, args)
    print(f"Output Directory: {output_dir}")

    # 1. Loss
    plot_loss_curves(args.loss_path, output_dir)

    # 2. Parameter Analysis (Sorted)
    print("Analyzing parameters...")
    model_paths = find_checkpoint_models(args.model_path)
    plot_parameter_distributions(model_paths, output_dir)

    # 3. Data Processing (Real + Generated)
    #    Replicating Notebook logic: Load -> Preprocess -> Check/Calc Labels -> Concat
    adata_real = args.data_preprocessed #load_and_preprocess_real_data()
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
        plot_umap_analysis(adata_combined, output_dir)

    print(f"=== Analysis Complete. Saved to {output_dir} ===")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# """
# Visualization script for training and sampling results
# Generates loss plots and UMAP analysis plots
# """

# import argparse
# import os
# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import scanpy as sc
# import anndata
# import torch
# from datetime import datetime

# def setup_output_directory(model_name, total_steps):
#     """Create unique output directory"""
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     dir_name = f"{model_name}_{total_steps}steps_{timestamp}"
#     output_dir = f"/home/suzuki/Projects/scDiffusion/exp_script/{dir_name}"
#     os.makedirs(output_dir, exist_ok=True)
#     return output_dir

# def save_metadata(output_dir, model_path, sample_path, loss_path):
#     """Save metadata about used files"""
#     metadata_file = os.path.join(output_dir, "analysis_metadata.txt")
#     with open(metadata_file, 'w') as f:
#         f.write(f"Analysis Metadata\n")
#         f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
#         f.write(f"Model Path: {model_path}\n")
#         f.write(f"Sample Path: {sample_path}\n")
#         f.write(f"Loss Details Path: {loss_path}\n")
#         f.write(f"Output Directory: {output_dir}\n")
#     return metadata_file

# def extract_weights(path):
#     """Extract weights from model checkpoint for parameter analysis"""
#     if not os.path.exists(path):
#         print(f"Model file not found: {path}")
#         return None
        
#     ckpt = torch.load(path, map_location="cpu")
#     state = ckpt.get("model_state_dict", ckpt)
#     weights = {}

#     # --- GeneODE 部分 ---
#     for key in ["ode_model.W", "ode_model.mask", "ode_model.b", "ode_model.gamma"]:
#         weights[key] = state.get(key)

#     # --- CellUnet 部分 (ML部分) ---
#     ml_keys = [k for k in state.keys() if "ml_model" in k and "weight" in k]
#     ml_params = []
#     for k in ml_keys:
#         w = state[k].flatten().detach().cpu().numpy()
#         ml_params.append(w)
#     weights["ml_model.params"] = np.concatenate(ml_params) if len(ml_params) > 0 else np.array([])

#     return weights

# def find_checkpoint_models(model_path):
#     """Find available checkpoint models from training"""
#     base_dir = os.path.dirname(model_path)
#     model_paths = {}
    
#     # Look for checkpoints in the same directory
#     if os.path.exists(base_dir):
#         for filename in os.listdir(base_dir):
#             if filename.startswith('model') and filename.endswith('.pt'):
#                 full_path = os.path.join(base_dir, filename)
#                 model_paths[filename] = full_path
    
#     # If no checkpoints found, use the provided model path
#     if not model_paths and os.path.exists(model_path):
#         filename = os.path.basename(model_path)
#         model_paths[filename] = model_path
    
#     return model_paths

# def plot_parameter_distributions(model_paths, output_dir):
#     """Plot parameter distributions over training steps (GeneODE + CellUnet)"""
#     if not model_paths:
#         print("No model files found for parameter analysis")
#         return None
    
#     # Extract weights from all models
#     weights_dict = {}
#     for label, path in model_paths.items():
#         weights = extract_weights(path)
#         if weights is not None:
#             weights_dict[label] = weights
    
#     if not weights_dict:
#         print("Could not extract weights from any model files")
#         return None
    
#     # Create histogram plots
#     fig, axes = plt.subplots(5, len(weights_dict), figsize=(5 * len(weights_dict), 15))
#     if len(weights_dict) == 1:
#         axes = axes.reshape(-1, 1)
#     plt.subplots_adjust(hspace=0.4)

#     for col_idx, (label, wdict) in enumerate(weights_dict.items()):
#         W = wdict["ode_model.W"]
#         mask = wdict["ode_model.mask"]
#         b = wdict["ode_model.b"]
#         gamma = wdict["ode_model.gamma"]
#         ml_params = wdict["ml_model.params"]
#         def plot_hist(data, ax, title, color):
#             if data is None or len(data) == 0:
#                 ax.text(0.5, 0.5, "missing", ha="center", va="center")
#                 ax.set_title(title)
#             else:
#                 ax.hist(data, bins=60, color=color, alpha=0.7)
#                 ax.set_title(title)

#         # 1️⃣ W(mask=1)
#         w_on = W[mask.bool()].flatten().detach().cpu().numpy() if W is not None and mask is not None else []
#         plot_hist(w_on, axes[0, col_idx], f"{label} | W(mask=1)", "C0")

#         # 2️⃣ W(mask=0)
#         w_off = W[(1 - mask).bool()].flatten().detach().cpu().numpy() if W is not None and mask is not None else []
#         plot_hist(w_off, axes[1, col_idx], f"{label} | W(mask=0)", "C1")

#         # 3️⃣ b
#         b_vals = b.flatten().detach().cpu().numpy() if b is not None else []
#         plot_hist(b_vals, axes[2, col_idx], f"{label} | b (all)", "C2")

#         # 4️⃣ gamma
#         g_vals = gamma.flatten().detach().cpu().numpy() if gamma is not None else []
#         plot_hist(g_vals, axes[3, col_idx], f"{label} | gamma (all)", "C3")

#         # 5️⃣ CellUnet
#         plot_hist(ml_params, axes[4, col_idx], f"{label} | CellUnet params", "C4")

#     # Add row labels
#     row_labels = ["W(mask=1)", "W(mask=0)", "b", "gamma", "CellUnet"]
#     for row in range(5):
#         axes[row, 0].set_ylabel(row_labels[row])

#     plt.suptitle("Parameter Distributions over Training Steps (GeneODE + CellUnet)", fontsize=18)
    
#     # Save plot
#     param_plot_path = os.path.join(output_dir, "parameter_distributions.png")
#     plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return param_plot_path

# def plot_loss_curves(loss_csv_path, output_dir):
#     """Plot loss curves from CSV file"""
#     if not os.path.exists(loss_csv_path):
#         print(f"Loss file not found: {loss_csv_path}")
#         return None
    
#     # Load loss data
#     df = pd.read_csv(loss_csv_path)
    
#     # Create plot
#     plt.figure(figsize=(12, 8))
    
#     # Plot original loss
#     plt.plot(df['step'], df['original_loss'], label='Original Loss', color='blue', linewidth=2)
    
#     # Plot total loss (always show if available)
#     if 'total_loss' in df.columns:
#         plt.plot(df['step'], df['total_loss'], label='Total Loss', color='red', linewidth=2)
    
#     # Plot regularization term (always show if column exists, even if zero)
#     if 'reg_value' in df.columns:
#         plt.plot(df['step'], df['reg_value'], label='Regularization Term', color='green', linewidth=2)
    
#     # Plot regularization weighted term (always show if column exists, even if zero)
#     if 'reg_weighted' in df.columns:
#         plt.plot(df['step'], df['reg_weighted'], label='Weighted Regularization (λ*Reg)', color='orange', linewidth=2)
    
#     plt.xlabel('Training Steps')
#     plt.ylabel('Loss Value')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Save plot
#     loss_plot_path = os.path.join(output_dir, "loss_curves.png")
#     plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return loss_plot_path

# def load_and_preprocess_data():
#     """Load and preprocess original PBMC68k data"""
#     adata_path = '/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad'
#     adata = sc.read_h5ad(adata_path)
    
#     # Apply same preprocessing as training
#     sc.pp.filter_cells(adata, min_genes=10)
#     sc.pp.filter_genes(adata, min_cells=3)
#     sc.pp.normalize_total(adata, target_sum=1e4)
#     sc.pp.log1p(adata)
#     sc.pp.highly_variable_genes(adata, n_top_genes=1024, subset=True, inplace=True)
#     sc.pp.scale(adata, zero_center=True)
    
#     return adata

# def create_combined_data(adata_real, sample_path):
#     """Combine real and generated data"""
#     # Load generated samples
#     npz_data = np.load(sample_path, allow_pickle=True)
#     cell_gen = npz_data['cell_gen']
    
#     # Check for NaN values
#     if np.any(np.isnan(cell_gen)):
#         print("Warning: Generated data contains NaN values")
#         return None, None, None
    
#     # Create AnnData for generated samples
#     adata_gen = anndata.AnnData(cell_gen)
#     adata_gen.var_names = adata_real.var_names
#     adata_gen.obs['cell_origin'] = 'Generated'
    
#     # Add cell_origin to real data
#     adata_real.obs['cell_origin'] = 'Real'
    
#     # Combine datasets
#     adata_combined = anndata.concat([adata_real, adata_gen], axis=0)
    
#     # # Add celltype information (use existing or create dummy)
#     # if 'celltype' not in adata_combined.obs.columns:
#     #     # Create dummy celltypes
#     #     n_real = adata_real.n_obs
#     #     n_gen = adata_gen.n_obs
#     #     celltypes = ['celltype_' + str(i % 8) for i in range(n_real + n_gen)]
#     #     adata_combined.obs['celltype'] = celltypes
    
#     # return adata_combined, adata_real, adata_gen
#     # Add celltype information
#     # ノートブックに合わせて 'Bulk_labels' を使用する
#     label_col = 'Bulk_labels'
    
#     # Realデータにラベルがあるか確認
#     if label_col in adata_real.obs.columns:
#         # Generatedデータには "Generated" というラベルを割り当てる（比較のため）
#         adata_gen.obs[label_col] = 'Generated'
        
#         # 結合しなおす（obsを正しく反映させるため）
#         adata_combined = anndata.concat([adata_real, adata_gen], axis=0)
        
#         # 念のためカテゴリ型に変換（プロット時のエラー防止）
#         adata_combined.obs[label_col] = adata_combined.obs[label_col].astype('category')
#     else:
#         print(f"Warning: {label_col} not found in data. Using dummy labels.")
#         # ここで初めてダミー処理（バックアップ策）
#         n_total = adata_combined.n_obs
#         adata_combined.obs[label_col] = ['celltype_' + str(i % 8) for i in range(n_total)]

#     return adata_combined, adata_real, adata_gen

# def plot_gene_correlation(adata_real, adata_gen, output_dir):
#     """Plot gene expression correlation between real and generated data"""
#     # Calculate mean gene expression
#     real_gene_means = np.mean(adata_real.X, axis=0)
#     gen_gene_means = np.mean(adata_gen.X, axis=0)
    
#     # Flatten if needed
#     if hasattr(real_gene_means, 'A1'):
#         real_gene_means = real_gene_means.A1
#     if hasattr(gen_gene_means, 'A1'):
#         gen_gene_means = gen_gene_means.A1
    
#     # Create scatter plot
#     plt.figure(figsize=(8, 8))
#     plt.scatter(real_gene_means, gen_gene_means, alpha=0.5, s=15)
    
#     # Add diagonal line
#     min_val = min(np.min(real_gene_means), np.min(gen_gene_means))
#     max_val = max(np.max(real_gene_means), np.max(gen_gene_means))
#     plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
#     plt.xlabel('Real Data Mean Expression')
#     plt.ylabel('Generated Data Mean Expression')
#     plt.grid(True, alpha=0.3)
    
#     # Save plot
#     corr_plot_path = os.path.join(output_dir, "gene_correlation.png")
#     plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return corr_plot_path

# def preprocess_for_umap(adata_combined):
#     """Preprocess combined data for UMAP analysis"""
#     print("Starting UMAP preprocessing...")
    
#     # Basic preprocessing for UMAP
#     print("Scaling data...")
#     sc.pp.scale(adata_combined, max_value=10)
    
#     print("Running PCA...")
#     sc.tl.pca(adata_combined, svd_solver='arpack', n_comps=50)
    
#     print("Computing neighbors...")
#     sc.pp.neighbors(adata_combined, n_neighbors=15, n_pcs=40)
    
#     print("Computing UMAP...")
#     sc.tl.umap(adata_combined)
    
#     print("UMAP computation completed!")
#     print(f"UMAP coordinates shape: {adata_combined.obsm['X_umap'].shape}")

# def plot_umap_analysis(adata_combined, output_dir):
#     """Create UMAP plots"""
#     # Preprocess data for UMAP (only when UMAP is actually needed)
#     preprocess_for_umap(adata_combined)
    
#     # Create subplot
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
#     # Get subsets
#     mask_real = adata_combined.obs["cell_origin"] == "Real"
#     mask_gen = adata_combined.obs["cell_origin"] == "Generated"
    
#     adata_real = adata_combined[mask_real].copy()
#     adata_gen = adata_combined[mask_gen].copy()
    
#     # Plot Real data (gray)
#     if adata_real.n_obs > 0:
#         sc.pl.umap(
#             adata_real,
#             color=None,
#             ax=axes[0],
#             frameon=False,
#             show=False,
#             size=35
#         )
    
#     # Plot Generated data on top (red)
#     if adata_gen.n_obs > 0:
#         # Shuffle for better visualization
#         idx = np.arange(adata_gen.n_obs)
#         np.random.shuffle(idx)
#         adata_gen = adata_gen[idx].copy()
        
#         sc.pl.umap(
#             adata_gen,
#             color="cell_origin",
#             ax=axes[0],
#             frameon=False,
#             show=False,
#             size=35,
#             palette={"Generated": "red"},
#             legend_loc="right margin"
#         )
    
#     # Plot by celltype
#     plot_col = "Bulk_labels" if "Bulk_labels" in adata_combined.obs.columns else "celltype"
#     sc.pl.umap(
#         adata_combined,
#         color=plot_col,
#         ax=axes[1],
#         frameon=False,
#         show=False,
#         size=35
#     )
    
#     plt.tight_layout()
    
#     # Save plot
#     umap_plot_path = os.path.join(output_dir, "umap_analysis.png")
#     plt.savefig(umap_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return umap_plot_path

# def main():
#     parser = argparse.ArgumentParser(description='Generate analysis plots')
#     parser.add_argument('--model_path', required=True, help='Path to trained model')
#     parser.add_argument('--sample_path', required=True, help='Path to generated samples')
#     parser.add_argument('--loss_path', required=True, help='Path to loss details CSV')
#     parser.add_argument('--model_name', required=True, help='Model name')
#     parser.add_argument('--total_steps', required=True, help='Total training steps')
#     parser.add_argument('--skip_umap', action='store_true', default=False, help='Skip UMAP analysis (default: False)')
    
#     args = parser.parse_args()
    
#     print("Starting visualization analysis...")
    
#     # Create output directory
#     output_dir = setup_output_directory(args.model_name, args.total_steps)
#     print(f"Output directory: {output_dir}")
    
#     # Save metadata
#     metadata_file = save_metadata(output_dir, args.model_path, args.sample_path, args.loss_path)
#     print(f"Metadata saved to: {metadata_file}")
    
#     # 1. Plot loss curves
#     print("Generating loss curves...")
#     loss_plot = plot_loss_curves(args.loss_path, output_dir)
#     if loss_plot:
#         print(f"Loss plot saved to: {loss_plot}")
    
#     # 2. Load and preprocess data
#     print("Loading and preprocessing data...")
#     adata_real = load_and_preprocess_data()
    
#     # 3. Create combined dataset
#     print("Combining real and generated data...")
#     adata_combined, adata_real_subset, adata_gen = create_combined_data(adata_real, args.sample_path)
    
#     if adata_combined is None:
#         print("Error: Could not create combined dataset (NaN in generated data)")
#         return
    
#     # 4. Parameter distribution analysis
#     print("Generating parameter distribution analysis...")
#     model_paths = find_checkpoint_models(args.model_path)
#     param_plot = plot_parameter_distributions(model_paths, output_dir)
#     if param_plot:
#         print(f"Parameter distribution plot saved to: {param_plot}")
    
#     # 5. Gene correlation plot
#     print("Generating gene correlation plot...")
#     corr_plot = plot_gene_correlation(adata_real_subset, adata_gen, output_dir)
#     print(f"Gene correlation plot saved to: {corr_plot}")
    
#     # 6. UMAP analysis (conditional)
#     umap_plot = None
#     if not args.skip_umap:
#         print("Generating UMAP analysis...")
#         umap_plot = plot_umap_analysis(adata_combined, output_dir)
#         print(f"UMAP analysis plot saved to: {umap_plot}")
#     else:
#         print("Skipping UMAP analysis (--skip_umap flag set)")
    
#     print("\n" + "="*60)
#     print("VISUALIZATION ANALYSIS COMPLETED")
#     print("="*60)
#     print(f"Output directory: {output_dir}")
#     print("Generated files:")
#     print(f"  - {metadata_file}")
#     if loss_plot:
#         print(f"  - {loss_plot}")
#     if param_plot:
#         print(f"  - {param_plot}")
#     print(f"  - {corr_plot}")
#     if umap_plot:
#         print(f"  - {umap_plot}")
#     print("="*60)

# if __name__ == "__main__":
#     main()