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
import os
import sys
from datetime import datetime

# Add scDiffusion directory to path for importing ODE and guided_diffusion modules
sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')

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
from ODE.ode_analysis20260413 import GeneODE, ODE_ML_HybridLearnedScale
def main():
    setup_seed(1234)
    torch.autograd.set_detect_anomaly(True)
    args = create_argparser().parse_args()

    # Create base output directory from train script
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.getcwd()
    
    # Create timestamped train subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_output_dir = os.path.join(base_output_dir, f"{timestamp}_train")
    
    # Handle existing directory with suffix
    counter = 1
    original_train_output_dir = train_output_dir
    while os.path.exists(train_output_dir):
        train_output_dir = f"{original_train_output_dir}_{counter}"
        counter += 1
    
    os.makedirs(train_output_dir, exist_ok=True)
    logger.log(f"Training output directory created: {train_output_dir}")

    # Create subdirectories for logs and checkpoints
    log_dir = os.path.join(train_output_dir, "log")
    checkpoint_dir = os.path.join(train_output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=log_dir)  # log file

    logger.log("creating model and diffusion...")
    _ , diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    # # diffusion だけ作成
    # diffusion = create_gaussian_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys())
    # )


    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Load preprocessed data
    logger.log("loading preprocessed h5ad file...")
    adata = sc.read_h5ad(args.data_dir)
    logger.log(f"Data loaded. Shape: {adata.shape}")
    
    # Check data structure
    logger.log(f"adata.var columns: {list(adata.var.columns)}")
    logger.log(f"'gene_name' column exists in adata.var: {'gene_name' in adata.var.columns}")
    logger.log(f"adata.obs columns: {list(adata.obs.columns)}")
    logger.log(f"'celltype' column exists in adata.obs: {'celltype' in adata.obs.columns}")
    logger.log(f"adata.n_vars (number of genes): {adata.n_vars}")
    logger.log(f"adata.n_obs (number of cells): {adata.n_obs}")
    
    # Get gene list for model
    gene_list = list(adata.var["gene_name"].unique())
    logger.log(f"Found {len(gene_list)} genes")
    
    # Create data loader with preprocessed data
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        # vae_path=args.vae_path,
        train_vae=True,
        preprocess=False,
    )
    
    # Create GeneODE instance with configurable soft regularization
    logger.log(f"{args.SoftReg}")
    if isinstance(args.SoftReg, str):
        v = args.SoftReg.lower()
        if v in ("true", "1", "yes", "y", "t"):
            args.SoftReg = True
        elif v in ("false", "0", "no", "n", "f"):
            args.SoftReg = False
        else:
            raise ValueError(
                f"Invalid value for --SoftReg: {args.SoftReg}. "
                "Use true/false."
            )


    logger.log(f"creating GeneODE instance with soft={args.SoftReg}...")
    logger.log(f"Gene list length: {len(gene_list)}")
    logger.log(f"Device: {dist_util.dev()}")
    



    try:
        ode = GeneODE(
            gene_list=gene_list,
            edge_tsv_path=args.edge_tsv_path,
            soft=args.SoftReg,  # ★ Soft制約の有効/無効を引数で制御
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
            edge_tsv_path=args.edge_tsv_path,
            device=dist_util.dev()
            
        )
        logger.log("GeneODE created without soft parameter - manual setting required")
        ode.soft = args.SoftReg  # Set manually after creation
    
    # Get timesteps from diffusion
    timesteps = diffusion.num_timesteps
    logger.log(f"Using {timesteps} timesteps for hybrid model")
    

    model = Cell_Unet(
        input_dim=len(gene_list),
        # hidden_num=[1024, 512, 256, 256], #→SmallerUnetでつかう
        # dropout=dropout
    )
    model.to(dist_util.dev())



    # Create hybrid model
    logger.log("creating ODE-ML hybrid model...")
    hybrid_model = ODE_ML_HybridLearnedScale(
        ode_model=ode,
        ml_model=model,
        timesteps=timesteps,
        scale_hidden_dim=32,
        scale_min=0.5,
        scale_max=8.0,
        init_scale=3.0,
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
        save_dir=checkpoint_dir,  # Save to {timestamp}_train/checkpoints
        ode_reg_lambda=args.ode_reg_lambda,  # ★ ODE正則化項
        ode_reg_norm=args.ode_reg_norm,  # L1正則化を使用
        save_loss_details=args.save_loss_details  # 損失詳細記録
    ).run_loop()
    
    # Find the latest checkpoint (best model)
    import glob
    model_files = glob.glob(os.path.join(checkpoint_dir, "*", "ema_*.pt"))
    if not model_files:
        model_files = glob.glob(os.path.join(checkpoint_dir, "*", "model*.pt"))
    
    if model_files:
        # Get the latest model file
        latest_model = max(model_files, key=os.path.getmtime)
        logger.log(f"Latest model found: {latest_model}")
        
        # Output for train script
        model_dir = os.path.dirname(latest_model)
        total_steps = args.lr_anneal_steps  # or extract from filename
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"TRAINED_MODEL_PATH='{latest_model}'")
        print(f"MODEL_DIR='{model_dir}'")
        print(f"CHECKPOINT_DIR='{checkpoint_dir}'")
        print(f"TOTAL_STEPS={total_steps}")
        print(f"MODEL_NAME='{args.model_name}'")
        print(f"{'='*60}")
    else:
        logger.log("Warning: No model files found in checkpoint directory")
        print(f"CHECKPOINT_DIR='{checkpoint_dir}'")


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
        model_name="pbmc68k_soft_20251127_hvg1024",  # HVG 1024 version (20251127)
        save_dir='output/diffusion_checkpoint_soft_hvg1024_20251127',  # HVG 1024 version (20251127)
        output_dir="",  # Output directory passed from train script
        ode_reg_lambda=0.0005,   # デフォルト値
        ode_reg_norm='l1',     # 既存の正則化タイプもついでに引数化可能
        save_loss_details=True,  # 損失の詳細記録を保存
        SoftReg=True,  # GeneODEのsoft制約を有効化（デフォルトTrue）
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv"  # Gene regulatory network path
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