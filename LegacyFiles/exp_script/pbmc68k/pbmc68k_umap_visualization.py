#!/usr/bin/env python3
"""
UMAP Visualization Script for PBMC68k Dataset
This script decodes the generated latent embedding into gene expression matrix 
and creates UMAP visualizations comparing real vs generated cells
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sys
import warnings
warnings.filterwarnings('ignore')

# Add path to VAE module
sys.path.append('/home/suzuki/Projects/scDiffusion/VAE')
from VAE_model import VAE

# Set up scanpy settings
sc.settings.verbosity = 3
sc.settings.set_figure_params(dpi=80, facecolor='white')

print("Environment set up successfully!")

def load_VAE():
    """Load VAE model for PBMC68k dataset"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    autoencoder = VAE(
        num_genes=32738,  # PBMC68k dataset has 32738 genes
        device=device,
        seed=0,
        loss_ae='mse',
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    
    model_path = '/home/suzuki/Projects/scDiffusion/output/checkpoint/AE/pbmc68k/model_seed=0_step=9999.pt'
    
    try:
        if device == 'cpu':
            state_dict = torch.load(model_path, map_location='cpu')
        else:
            state_dict = torch.load(model_path)
        
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
        print(f"VAE model loaded successfully!")
        return autoencoder
    except Exception as e:
        print(f"Error loading VAE model: {e}")
        return None

# Load the VAE model
print("Loading VAE model...")
autoencoder = load_VAE()

# Load pancreas dataset
print("Loading pancreas dataset...")
adata = sc.read_h5ad('/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad')
print(f"Original data shape: {adata.shape}")
print(f"Available obs columns: {list(adata.obs.columns)}")

# Find the correct cell type column
cell_type_col = None
for col in ['celltype', 'cell_type', 'Celltype', 'Cell_type']:
    if col in adata.obs.columns:
        cell_type_col = col
        break

if cell_type_col:
    print(f"Using cell type column: {cell_type_col}")
    print(f"Cell types: {list(adata.obs[cell_type_col].unique())}")
    celltype = adata.obs[cell_type_col]
else:
    print("Warning: No cell type column found")
    celltype = None

# Get gene expression data (already log-normalized)
if hasattr(adata.X, 'toarray'):
    cell_data = adata.X.toarray()
else:
    cell_data = adata.X

print(f"Cell data shape: {cell_data.shape}")
print(f"Gene names: {len(adata.var_names)} genes")

# Store gene names for later use
gene_names = adata.var_names.tolist()

# Load unconditional generated samples
print("Loading unconditional generated samples...")
unconditional_path = '/home/suzuki/Projects/scDiffusion/output/simulated_samples/pbmc68k/classifier_guided/unconditional_samples.npz'

cell_gen = None
try:
    npzfile = np.load(unconditional_path, allow_pickle=True)
    print(f"Available keys in file: {list(npzfile.keys())}")

    # Use the correct key
    if 'cell_gen' in npzfile.keys():
        cell_gen_latent = npzfile['cell_gen'][:3000]  # Use 3000 samples
        print("Using 'cell_gen' key")
    elif 'arr_0' in npzfile.keys():
        cell_gen_latent = npzfile['arr_0'][:3000]
        print("Using 'arr_0' key")
    else:
        key = list(npzfile.keys())[0]
        cell_gen_latent = npzfile[key][:3000]
        print(f"Using key: {key}")

    print(f"Loaded latent samples shape: {cell_gen_latent.shape}")

    # Decode to gene expression space
    if autoencoder is not None:
        with torch.no_grad():
            device = next(autoencoder.parameters()).device
            cell_gen_tensor = torch.tensor(cell_gen_latent, dtype=torch.float32).to(device)
            cell_gen = autoencoder(cell_gen_tensor, return_decoded=True).cpu().detach().numpy()

        print(f"Decoded unconditional samples shape: {cell_gen.shape}")
    else:
        print("Failed to load VAE model")
        
except FileNotFoundError:
    print(f"Unconditional samples file not found at: {unconditional_path}")
except Exception as e:
    print(f"Error loading unconditional samples: {e}")

# Load conditional generated samples for PBMC68k
print("Loading conditional generated samples...")
conditional_base_path = '/home/suzuki/Projects/scDiffusion/output/simulated_samples/pbmc68k/classifier_guided/type_{}_{}.npz'
cell_types = ['CD8+ T cells', 'B cells', 'CD4+ T cells', 'NK cells', 'CD34+', 'Megakaryocytes', 'Monocytes', 'Dendritic cells', 'Erythroid cells', 'CD16+ Monocytes', 'Plasmacytoid dendritic cells']

conditional_samples = []
gen_class = []
cell_gen_conditional = None

for i in range(11):
    cond_path = conditional_base_path.format(i, i)
    try:
        npzfile = np.load(cond_path, allow_pickle=True)
        print(f"Type {i} ({cell_types[i]}) - Available keys: {list(npzfile.keys())}")
        
        # Use the correct key
        if 'cell_gen' in npzfile.keys():
            cond_samples = npzfile['cell_gen'][:500]  # 500 samples per type
        elif 'arr_0' in npzfile.keys():
            cond_samples = npzfile['arr_0'][:500]
        else:
            key = list(npzfile.keys())[0]
            cond_samples = npzfile[key][:500]
        
        conditional_samples.append(cond_samples)
        gen_class.extend([f'gen {cell_types[i]}'] * len(cond_samples))
        print(f"Loaded {len(cond_samples)} samples for {cell_types[i]}")
        
    except FileNotFoundError:
        print(f"Warning: File not found for type {i} ({cell_types[i]})")
        continue

if conditional_samples and autoencoder is not None:
    # Combine all conditional samples
    all_conditional_latent = np.vstack(conditional_samples)
    print(f"Total conditional latent samples: {all_conditional_latent.shape}")
    
    # Decode conditional samples
    with torch.no_grad():
        device = next(autoencoder.parameters()).device
        cond_tensor = torch.tensor(all_conditional_latent, dtype=torch.float32).to(device)
        cell_gen_conditional = autoencoder(cond_tensor, return_decoded=True).cpu().detach().numpy()
    
    print(f"Decoded conditional samples shape: {cell_gen_conditional.shape}")
else:
    print("No conditional samples found or VAE model not available")
    cell_gen_conditional = None
    gen_class = []

# Gene expression correlation analysis
if cell_gen is not None:
    print("Analyzing gene expression correlation...")
    # Calculate mean expression per gene
    real_gene_means = cell_data.mean(axis=0)
    gen_gene_means = cell_gen.mean(axis=0)
    
    # Calculate correlation
    corr_coef = np.corrcoef(real_gene_means, gen_gene_means)[0, 1]
    print(f"Unconditional generation correlation: {corr_coef:.4f}")

# Conditional correlation if available
if cell_gen_conditional is not None:
    cond_gene_means = cell_gen_conditional.mean(axis=0)
    corr_coef_cond = np.corrcoef(real_gene_means, cond_gene_means)[0, 1]
    print(f"Conditional generation correlation: {corr_coef_cond:.4f}")

# Prepare data for UMAP visualization
print("Preparing data for UMAP...")

# Choose which generated data to use (prefer conditional if available)
if cell_gen_conditional is not None:
    cell_gen_final = cell_gen_conditional
    gen_class_final = gen_class
    print("Using conditional generated data for UMAP")
elif cell_gen is not None:
    cell_gen_final = cell_gen
    gen_class_final = [f"gen_Cell" for i in range(cell_gen.shape[0])]
    print("Using unconditional generated data for UMAP")
else:
    print("No generated data available!")
    cell_gen_final = None

if cell_gen_final is not None:
    # Combine real and generated data
    all_data = np.concatenate((cell_data, cell_gen_final), axis=0)
    adata_combined = ad.AnnData(all_data, dtype=np.float32)
    
    print(f"Combined data shape: {adata_combined.shape}")
    
    # Create cell labels (real vs generated)
    cell_names = [f"Real_Cell" for i in range(cell_data.shape[0])] + \
                 [f"Generated_Cell" for i in range(cell_gen_final.shape[0])]
    adata_combined.obs['cell_origin'] = cell_names
    
    # Add cell type information if available
    if celltype is not None and cell_gen_conditional is not None:
        # For real cells, use actual cell types
        real_cell_types = celltype.values if hasattr(celltype, 'values') else celltype
        # For generated cells, use generated cell types
        all_cell_types = list(real_cell_types) + gen_class_final
        adata_combined.obs['celltype'] = all_cell_types
        print(f"Added cell type information: {len(set(all_cell_types))} unique types")
    else:
        print("Cell type information not available or incomplete")
    
    print("Data preparation completed!")
    
    # UMAP preprocessing and computation
    print("Preprocessing data for UMAP...")
    
    # The data is already log-normalized, so we proceed with feature selection
    sc.pp.highly_variable_genes(adata_combined, min_mean=0.0125, max_mean=3, min_disp=0.5)
    print(f"Found {sum(adata_combined.var.highly_variable)} highly variable genes")
    
    # Store raw data and filter
    adata_combined.raw = adata_combined
    adata_combined = adata_combined[:, adata_combined.var.highly_variable]
    
    print(f"Data after filtering: {adata_combined.shape}")
    
    # Scale data
    sc.pp.scale(adata_combined, max_value=10)
    
    # PCA
    sc.tl.pca(adata_combined, svd_solver='arpack')
    print("PCA completed!")
    
    # Neighbors and UMAP
    print("Computing neighbors...")
    sc.pp.neighbors(adata_combined, n_neighbors=15, n_pcs=40)
    
    print("Computing UMAP...")
    sc.tl.umap(adata_combined)
    
    print("UMAP computation completed!")
    
    # Create visualizations
    print("Creating UMAP visualizations...")
    
    # Plot UMAP colored by cell origin (real vs generated)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Real vs Generated
    sc.pl.umap(adata_combined, color="cell_origin", size=30, 
               title='Pancreas: Real vs Generated Cells', 
               ax=axes[0], show=False, frameon=False)
    
    # Plot 2: Cell types (if available)
    if 'celltype' in adata_combined.obs.columns:
        sc.pl.umap(adata_combined, color="celltype", size=30, 
                   title='Pancreas: Cell Types', 
                   ax=axes[1], show=False, frameon=False, 
                   legend_loc='right margin', legend_fontsize=8)
    else:
        # If no cell types, show another view
        sc.pl.umap(adata_combined, color="cell_origin", size=30, 
                   title='Pancreas: Real vs Generated (Alternative View)', 
                   ax=axes[1], show=False, frameon=False)
    
    plt.tight_layout()
    plt.savefig('/home/suzuki/Projects/scDiffusion/pancreas_umap_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("UMAP visualization saved to: /home/suzuki/Projects/scDiffusion/pancreas_umap_comparison.png")

else:
    print("Cannot create UMAP without generated data")

# Summary
print("\n=== PANCREAS DATASET UMAP ANALYSIS SUMMARY ===")
print(f"Real data shape: {cell_data.shape}")
if cell_gen is not None:
    print(f"Unconditional generated data shape: {cell_gen.shape}")
if cell_gen_conditional is not None:
    print(f"Conditional generated data shape: {cell_gen_conditional.shape}")
    print(f"Generated cell types: {set([ct.replace('gen ', '') for ct in gen_class])}")

if celltype is not None:
    print(f"Real cell types: {list(celltype.unique())}")

if 'corr_coef' in locals():
    print(f"Unconditional correlation: {corr_coef:.4f}")
if 'corr_coef_cond' in locals():
    print(f"Conditional correlation: {corr_coef_cond:.4f}")

print("\nUMAP visualizations have been created showing:")
print("1. Real vs Generated cells comparison")
print("2. Cell type distribution")
if cell_gen_conditional is not None:
    print("3. Individual cell type comparisons")

print("\nAnalysis completed!")
