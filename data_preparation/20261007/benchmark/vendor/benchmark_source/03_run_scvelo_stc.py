import argparse
import numpy as np
import scanpy as sc
import scvelo as scv
import gc
from pathlib import Path
DATA_DIR = Path("/root/autodl-tmp/dataset") 

global_parser = argparse.ArgumentParser(description="compute velocity")
global_parser.add_argument('--dataset', type=str, help="the name of the dataset")
global_parser.add_argument('--clusterkey', type=str, help="the key of cluster")
global_parser.add_argument('--X_embed', type=str, help="the key of X_embed")
global_parser.add_argument('--K_FOLD', type=int, required=True, help="the number of K_FOLD")
global_parser.add_argument('--seed', type=int, help="random seed")
global_args = global_parser.parse_args()

DATASET = global_args.dataset
CLUSTER_KEY = global_args.clusterkey
VIS_EMB = global_args.X_embed
K_FOLD = global_args.K_FOLD
SEED = global_args.seed

SAVE_DATA = True
if SAVE_DATA:
    (DATA_DIR / DATASET / "processed").mkdir(parents=True, exist_ok=True)

def run_scvelo_st(fold):
    adata = sc.read_h5ad(DATA_DIR / DATASET / "processed" / f"adata_preprocessed_{fold}.h5ad")
    scv.tl.velocity(adata, mode="stochastic", vkey='scvelo_stc_velocity')
    scv.tl.velocity_graph(adata, vkey='scvelo_stc_velocity')
    scv.tl.velocity_pseudotime(adata,vkey='scvelo_stc_velocity')
    adata.obs['scvelo_stc_time'] = adata.obs['scvelo_stc_velocity_pseudotime']
    if SAVE_DATA:
        adata.write_h5ad(DATA_DIR / DATASET / "processed" / f"adata_run_scvelo_stc_{fold}.h5ad")
    del adata
    gc.collect()

if K_FOLD == 0:
    fold = 'full'
    run_scvelo_stc(fold)
else:
    for fold in range(K_FOLD):
        run_scvelo_stc(fold)