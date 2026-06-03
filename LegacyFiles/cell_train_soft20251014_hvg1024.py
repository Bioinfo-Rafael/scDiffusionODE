"""
Train a diffusion model on images.
# HVG 1024 version (20251014)

ODEモデル + CellUnetを組み合わせたハイブリッドモデルを学習する。
ただしCellUnetはinput_dimがadataの遺伝子の個数全て。

【Soft制約版 HVG 1024】
- GeneODEでsoft=Trueを設定
- ODE正則化項(ode_reg_lambda=0.1)を有効化
- HVG処理により1024遺伝子に削減 (20251014)
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
from ODE.ode_analysis import GeneODE, ODE_ML_Hybrid

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

    # Data loader will be created after preprocessing


    """
    HVG 1024 version (20251014) - 前処理統一
    """
    
    # Load and preprocess data - same as cell_train_20251014_NoConstraint_hvg1024.py
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
    
    ####################################################################
    # --- HVG 選択をここで挿入 ---
    logger.log("Selecting highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        # n_top_genes=1024,         # n_top_genesを指定しない - すべてのHVGを使用
        # flavor="seurat_v3",       # 使用するフレーバーは必要に応じて指定（例：'seurat_v3'）
        subset=True,               # HVG のみにサブセット化
        inplace=True
    )
    logger.log(f"Number of HVG kept: {adata.n_vars}")
    ####################################################################
    
    # Apply z-score normalization
    logger.log("Applying z-score normalization...")
    sc.pp.scale(adata, zero_center=True)
    logger.log("Z-score normalization completed")
    
    # Save preprocessed data
    preprocessed_path = args.data_dir.replace('.h5ad', '_preprocessed_zscore_hvg_all.h5ad')
    adata.write(preprocessed_path)
    logger.log(f"Preprocessed data saved to: {preprocessed_path}")
    
    # Get gene list for model
    gene_list = adata.var_names.tolist()
    logger.log(f"Found {len(gene_list)} genes after HVG processing")
    
    # Create data loader with preprocessed data
    logger.log("creating data loader...")
    data = load_data(
        data_dir=preprocessed_path,
        batch_size=args.batch_size,
        # vae_path=args.vae_path,
        train_vae=True,
        preprocess=False,
    )
    
    # Create GeneODE instance with soft=True
    logger.log("creating GeneODE instance with soft=True...")
    logger.log(f"Gene list length: {len(gene_list)}")
    logger.log(f"Device: {dist_util.dev()}")
    
    try:
        ode = GeneODE(
            gene_list=gene_list,
            edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
            soft=True,  # ★ Soft制約を有効化
            device=dist_util.dev()
        )
        logger.log("GeneODE created successfully!")
    except Exception as e:
        logger.log(f"ERROR creating GeneODE: {e}")
        logger.log(f"Error type: {type(e)}")
        # Try without soft parameter to see if it works
        logger.log("Trying without soft parameter...")
        ode = GeneODE(
            gene_list=gene_list,
            edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
            device=dist_util.dev()
        )
        logger.log("GeneODE created without soft parameter - manual setting required")
        ode.soft = True  # Set manually after creation
    
    # Get timesteps from diffusion
    timesteps = diffusion.num_timesteps
    logger.log(f"Using {timesteps} timesteps for hybrid model")
    

    model = Cell_Unet(
        input_dim=len(gene_list),
        # hidden_dim,
        # dropout=dropout
    )
    model.to(dist_util.dev())



    # Create hybrid model
    logger.log("creating ODE-ML hybrid model...")
    hybrid_model = ODE_ML_Hybrid(
        ode_model=ode,
        ml_model=model,
        timesteps=timesteps
    )
    hybrid_model.to(dist_util.dev())
    


    logger.log("training...")
    TrainLoop(
        #model=model,
        model = hybrid_model,
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
        save_dir=args.save_dir,
        ode_reg_lambda=args.ode_reg_lambda,  # ★ ODE正則化項を0.1に設定
        ode_reg_norm=args.ode_reg_norm  # L1正則化を使用
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="data_preparation/pbmc68k.h5ad",  # HVG 1024 version (20251014)
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=2000,  # 短縮して2000ステップに設定
        batch_size=128,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        # vae_path = 'output/Autoencoder_checkpoint/pbmc68k_AE/model_seed=0_step=0.pt',
        model_name="pbmc68k_soft_20251014_hvg1024",  # HVG 1024 version (20251014)
        save_dir='output/diffusion_checkpoint_soft_hvg1024',  # HVG 1024 version (20251014)
        ode_reg_lambda=0.1,   # デフォルト値
        ode_reg_norm='l1'     # 既存の正則化タイプもついでに引数化可能
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