"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    # create_gaussian_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.cell_model import Cell_Unet
import torch

import numpy as np
import random
import scanpy as sc



def main():
    
    setup_seed(1234)
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='../output/logs/'+args.model_name)  # log file

    logger.log("creating model and diffusion...")
    _ , diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # # diffusion だけ作成
    # diffusion = create_gaussian_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys())
    # )


    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Load and preprocess data
    logger.log("loading and preprocessing h5ad file...")
    adata = sc.read_h5ad(args.data_dir)
    logger.log(f"Original data shape: {adata.shape}")
    
    # Apply filtering
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    logger.log(f"Data shape after filtering: {adata.shape}")
    
    # Apply normalization and log transformation
    logger.log("Applying normalization and log transformation...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    logger.log("Normalization and log transformation completed")
    
    
    # ####################################################################
    # --- HVG 選択をここで挿入 ---
    logger.log("Selecting highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=1024,         # HVG 1024 遺伝子を残す - 20251014 version
        # flavor="seurat_v3",       # 好みのフレーバーを選ぶ（’seurat_v3’, ‘seurat’, ‘pearson_residuals’ 等）
        subset=True,               # True にすれば即座に adata を HVG のみにサブセット化
        inplace=True
    )
    logger.log(f"Number of HVG kept: {adata.n_vars}")
    ####################################################################


    # Apply z-score normalization
    logger.log("Applying z-score normalization...")
    sc.pp.scale(adata, zero_center=True)
    logger.log("Z-score normalization completed")
    
    # Save preprocessed data
    preprocessed_path = args.data_dir.replace('.h5ad', '_preprocessed_zscore.h5ad')
    adata.write(preprocessed_path)
    logger.log(f"Preprocessed data saved to: {preprocessed_path}")
    
    # Get gene count for model
    gene_list = adata.var_names.tolist()
    logger.log(f"Found {len(gene_list)} genes")

    logger.log("creating data loader...")
    data = load_data(
        data_dir=preprocessed_path,
        batch_size=args.batch_size,
        # vae_path=args.vae_path,
        train_vae=True,
        preprocess=False,
    )

    model = Cell_Unet(
        input_dim=len(gene_list),
        # hidden_dim=[2000],
        # dropout=dropout
    )

    model.to(dist_util.dev())


    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        model_name=args.model_name,
        save_dir=args.save_dir
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="data_preparation/pbmc68k.h5ad",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=500000,
        batch_size=128,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # vae_path = 'output/Autoencoder_checkpoint/pbmc68k_AE/model_seed=0_step=0.pt',
        model_name="pbmc68k_NoConstraint_20251014_hvg1024",
        save_dir='output/diffusion_checkpoint_NoConstraint_hvg1024'
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    main()