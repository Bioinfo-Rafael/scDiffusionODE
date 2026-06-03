#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a large batch of cell samples from a trained hybrid diffusion model (Soft HVG1024 version 20251105).

【Soft制約 HVG1024版 - 20251105】
- ODEモデル + Cell_Unetのハイブリッドモデル (ODE-ML Hybrid) を使用
- cell_train_soft20251105_hvg1024.py で学習したモデルを使用
- HVG処理により1024遺伝子に削減
- 学習時と同一の前処理 (_preprocessed_zscore_hvg1024_20251105.h5ad) を利用



CUDA_VISIBLE_DEVICES="" python /home/suzuki/Projects/scDiffusion/cell_sample_20251105_soft_hvg1024.py \
    --model_path /home/suzuki/Projects/scDiffusion/output/diffusion_checkpoint_soft_hvg1024_20251105/pbmc68k_soft_20251105_hvg1024/model001000.pt \
    --data_dir /home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k_preprocessed_zscore_hvg1024_20251105.h5ad
を実行

"""

import argparse
import numpy as np
import torch
import torch.distributed as dist
import random
import scanpy as sc

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_analysis import GeneODE, ODE_ML_Hybrid


def save_data(all_cells, save_path):
    np.savez(save_path, cell_gen=all_cells)
    return


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(1234)
    args = create_argparser().parse_args()

    # ============ Setup ============
    dist_util.setup_dist()
    logger.configure(dir='output/checkpoint/sample_logs')

    logger.log("Creating diffusion process...")
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # ============ Load HVG gene list ============
    logger.log("Loading preprocessed h5ad (must match training)...")
    adata = sc.read_h5ad(args.data_dir)
    gene_list = adata.var["gene_name"].tolist()
    logger.log(f"Loaded {len(gene_list)} HVG genes from preprocessed file.")

    device = dist_util.dev()
    logger.log(f"Using device: {device}")

    # ============ Recreate Hybrid Model ============
    logger.log("Recreating ODE-ML Hybrid model (Soft constraint enabled)...")
    ode = GeneODE(
        gene_list=gene_list,
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        soft=True,
        device=device,
    )
    ml_model = Cell_Unet(input_dim=len(gene_list)).to(device)
    hybrid = ODE_ML_Hybrid(
        ode_model=ode,
        ml_model=ml_model,
        timesteps=diffusion.num_timesteps
    ).to(device)

    # ============ Load checkpoint ============
    logger.log(f"Loading checkpoint from: {args.model_path}")
    sd = dist_util.load_state_dict(args.model_path, map_location="cpu")

    # Handle nested state dicts (e.g., 'ema', 'model')
    if any(k.startswith("ema") for k in sd.keys()):
        logger.log("Detected EMA weights, loading from 'ema' key")
        sd = sd.get('ema', sd)
    if "model" in sd:
        sd = sd["model"]

    missing, unexpected = hybrid.load_state_dict(sd, strict=False)
    logger.log(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    hybrid.eval()

    # ============ Sampling ============
    logger.log("Starting sampling process...")
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    all_cells = []
    generated = 0
    total = args.num_samples
    batch = args.batch_size

    while generated < total:
        n = min(batch, total - generated)
        sample, _ = sample_fn(
            hybrid,
            (n, len(gene_list)),
            clip_denoised=args.clip_denoised,
        )
        all_cells.append(sample.detach().cpu().numpy())
        generated += n
        logger.log(f"Generated {generated}/{total} samples")

    arr = np.concatenate(all_cells, axis=0)
    save_data(arr, args.sample_dir)
    logger.log(f"Saved {arr.shape[0]} samples to {args.sample_dir}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    logger.log("Sampling complete!")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=500,   # 生成サンプル数
        batch_size=50,     # バッチサイズ
        use_ddim=False,
        model_path="/home/suzuki/Projects/scDiffusion/output/diffusion_checkpoint_soft_hvg1024_20251105/pbmc68k_soft_20251105_hvg1024/model001000.pt",
        data_dir="/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k_preprocessed_zscore_hvg1024_20251105.h5ad",
        sample_dir="/home/suzuki/Projects/scDiffusion/output/simulated_samples/pbmc68k_soft_20251105_hvg1024_500samples.npz",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()