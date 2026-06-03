"""
Train a diffusion model on images.

ODEモデル + CellUnetを組み合わせたハイブリッドモデルを学習する。

1.adataをフィルタリング→highly_variable_genesを計算
2.高変動遺伝子を選択
3.選択した遺伝子のみに基づいてモデルを学習

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

    """
    追加
    """
    
    # Load and filter adata
    logger.log("loading gene list from h5ad file...")
    adata = sc.read_h5ad(args.data_dir)
    logger.log(f"Original: {adata.n_obs} cells, {adata.n_vars} genes")

    # Cell and gene filtering
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)
    logger.log(f"After filter: {adata.n_obs} cells, {adata.n_vars} genes")

    # Highly variable genes selection
    sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=args.n_top_genes, inplace=True)
    hvgs = adata.var['highly_variable']
    adata = adata[:, hvgs].copy()
    logger.log(f"Selected {adata.n_vars} highly variable genes")

    # Save processed .h5ad file
    out_path = args.data_dir.replace('.h5ad', '') + '_processed.h5ad'
    adata.write(out_path)
    logger.log(f"Processed AnnData saved to {out_path}")

    gene_list = adata.var_names.tolist()
    logger.log(f"Proceeding with {len(gene_list)} genes")




    logger.log("creating data loader...")
    data = load_data(
        data_dir=out_path,
        batch_size=args.batch_size,
        # vae_path=args.vae_path,
        train_vae=True,
        preprocess=False
    )



    # Create GeneODE instance
    logger.log("creating GeneODE instance...")
    ode = GeneODE(
        gene_list=gene_list,
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        device=dist_util.dev()
    )
    
    # Get timesteps from diffusion
    timesteps = diffusion.num_timesteps
    logger.log(f"Using {timesteps} timesteps for hybrid model")
    

    model = Cell_Unet(
        input_dim=args.n_top_genes,
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
        save_dir=args.save_dir
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/data1/lep/Workspace/guided-diffusion/data/tabula_muris/all.h5ad",
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
        # vae_path = 'output/Autoencoder_checkpoint/muris_AE/model_seed=0_step=0.pt',
        model_name="muris_diffusion",
        save_dir='output/diffusion_checkpoint',
        n_top_genes=5000,  # Number of highly variable genes to select:  NEW ★★★→adata.highly_variable_genesにつかう
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
