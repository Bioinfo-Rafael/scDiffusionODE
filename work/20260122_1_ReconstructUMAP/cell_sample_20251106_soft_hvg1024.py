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
import os
import sys
import glob
from datetime import datetime

# Add scDiffusion directory to path for importing ODE and guided_diffusion modules
sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_analysis1106 import GeneODE, ODE_ML_Hybrid


def save_data(all_cells, save_path):
    # ファイルが既に存在する場合、末尾に数字を付けて上書きを防ぐ
    if os.path.exists(save_path):
        base_path = save_path.replace('.npz', '')
        # 既存の番号付きファイルを検索
        pattern = f"{base_path}_*.npz"
        existing_files = glob.glob(pattern)
        
        # 既存の番号を取得
        numbers = []
        for f in existing_files:
            try:
                # ファイル名から番号を抽出
                num_str = f.replace(base_path + '_', '').replace('.npz', '')
                if num_str.isdigit():
                    numbers.append(int(num_str))
            except:
                continue
        
        # 次の番号を決定
        next_num = max(numbers) + 1 if numbers else 1
        save_path = f"{base_path}_{next_num:03d}.npz"
        print(f"File already exists. Saving as: {save_path}")
    
    np.savez(save_path, cell_gen=all_cells)
    return save_path


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    setup_seed(1234)
    args = create_argparser().parse_args()

    # Create base output directory from train script
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.getcwd()
    
    # Create timestamped sample subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_output_dir = os.path.join(base_output_dir, f"{timestamp}_sample")
    
    # Handle existing directory with suffix
    counter = 1
    original_sample_output_dir = sample_output_dir
    while os.path.exists(sample_output_dir):
        sample_output_dir = f"{original_sample_output_dir}_{counter}"
        counter += 1
    
    os.makedirs(sample_output_dir, exist_ok=True)
    logger.log(f"Sample output directory created: {sample_output_dir}")

    # Create subdirectories for logs and sampled data
    log_dir = os.path.join(sample_output_dir, "log")
    sampled_data_dir = os.path.join(sample_output_dir, "SampledData")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sampled_data_dir, exist_ok=True)

    # ============ Setup ============
    dist_util.setup_dist()
    logger.configure(dir=log_dir)

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
    
    # Save to SampledData directory
    base_filename = os.path.basename(args.sample_name) if args.sample_name else "samples.npz"
    sample_file_path = os.path.join(sampled_data_dir, base_filename)
    actual_save_path = save_data(arr, sample_file_path)
    logger.log(f"Saved {arr.shape[0]} samples to {actual_save_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    logger.log("Sampling complete!")
    
    # 最終出力: 保存されたファイルパス
    print(f"\n{'='*60}")
    print(f"SAMPLING COMPLETED SUCCESSFULLY")
    print(f"Saved file: {actual_save_path}")
    print(f"Number of samples: {arr.shape[0]}")
    print(f"Number of genes: {arr.shape[1]}")
    print(f"{'='*60}")
    
    # シェルスクリプト用の環境変数形式で出力
    print(f"\n# Shell script variables:")
    print(f"SAMPLE_FILE_PATH='{actual_save_path}'")
    print(f"NUM_SAMPLES={arr.shape[0]}")
    print(f"NUM_GENES={arr.shape[1]}")
    print(f"MODEL_PATH_USED='{args.model_path}'")
    print(f"DATA_DIR_USED='{args.data_dir}'")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=500,   # 生成サンプル数
        batch_size=50,     # バッチサイズ
        use_ddim=False,
        model_path="/home/suzuki/Projects/scDiffusion/output/diffusion_checkpoint_soft_hvg1024_20251105/pbmc68k_soft_20251105_hvg1024/model001000.pt",
        data_dir="/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k_preprocessed_zscore_hvg1024_20251105.h5ad",
        sample_name="pbmc68k.npz",
        output_dir="",  # Output directory passed from train script
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()