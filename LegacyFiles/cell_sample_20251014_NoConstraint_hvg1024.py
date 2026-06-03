"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce sa        num_samples=1000,  # サンプル数を1000に変更ples for FID evaluation.

【NoConstraint HVG1024版】
- ODEモデルを使用しない、Cell_Unetのみのバージョン
- cell_train_20251014_hvg1024_NoConstraint.pyで学習したモデルを使用
"""
import argparse

import numpy as np
import torch as th
import torch.distributed as dist
import random
import scanpy as sc

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    diffusion_defaults,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.cell_model import Cell_Unet


def save_data(all_cells, traj, data_dir):
    cell_gen = all_cells
    np.savez(data_dir, cell_gen=cell_gen)
    return

def main():
    setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='output/checkpoint/sample_logs')

    logger.log("creating model and diffusion...")
    # Create diffusion only (same as in cell_train_20251014_hvg1024_NoConstraint.py)
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # Load gene list from data - with same preprocessing as cell_train_20251014_NoConstraint_hvg1024.py
    logger.log("loading gene list from h5ad file...")
    adata = sc.read_h5ad(args.data_dir)
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Apply normalization and log transformation
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Apply HVG selection (same as in training) - HVG 1024 version 20251014
    logger.log("Applying HVG selection (1024 genes)...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1024,
        subset=True,
        inplace=True
    )
    
    gene_list = adata.var_names.tolist()
    logger.log(f"Found {len(gene_list)} genes after HVG selection")
    

    # Create Cell_Unet model (NoConstraint version - no ODE)
    logger.log("creating Cell_Unet model (NoConstraint)...")
    model = Cell_Unet(
        input_dim=len(gene_list),
    )
    model.to(dist_util.dev())
    
    # Load the trained model state
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.eval()

    logger.log("sampling...")
    all_cells = []
    while len(all_cells) * args.batch_size < args.num_samples:
        model_kwargs = {}
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, traj = sample_fn(
            model,
            (args.batch_size, len(gene_list)),  # Use actual gene count
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            start_time=diffusion.betas.shape[0],
        )

        # Safe distributed handling
        world_size = 1
        if dist.is_available() and dist.is_initialized():
            try:
                world_size = dist.get_world_size()
            except ValueError:
                world_size = 1
        if world_size > 1:
            gathered_samples = [th.zeros_like(sample) for _ in range(world_size)]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_cells.extend([s.cpu().numpy() for s in gathered_samples])
        else:
            # Single process mode
            all_cells.append(sample.cpu().numpy())
        logger.log(f"created {len(all_cells) * args.batch_size} samples")

    arr = np.concatenate(all_cells, axis=0)
    save_data(arr, traj, args.sample_dir)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=1000,  # サンプル数を1000に設定 - HVG1024版
        batch_size=100,    # バッチサイズを調整
        use_ddim=False,
        model_path="output/checkpoint/backbone_NoConstraint_hvg1024/pbmc68k_NoConstraint_20251014_hvg1024/model002000.pt",  # 2000ステップのモデルパス
        sample_dir="output/simulated_samples/pbmc68k_NoConstraint_20251014_hvg1024_1000samples.npz",              # NoConstraint HVG1024版の出力ファイル（1000サンプル版）
        data_dir="data_preparation/pbmc68k.h5ad"
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def setup_seed(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()