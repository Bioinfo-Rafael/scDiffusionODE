#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train script with selectable ODE model family.

Selectable ODE families
-----------------------
1) mixture_softplus
   V(x,t) = sum_k alpha_k(x,t) f_k(x),   f_k(x)=softplus(xW_k+b_k)
   alpha is MLP + softmax, optionally Top-K sparse.

2) matrix_dict
   W(x,t) = sum_k alpha_k(x,t) A_k
   alpha can be MLP or linear, optionally Top-K sparse.

3) lowrank_residual
   W(x,t) = W0 + sum_k alpha_k(x,t) Delta_k,   Delta_k = U_k V_k^T
   alpha can be MLP or linear, optionally Top-K sparse.
"""

import argparse
import os
import sys
from datetime import datetime
import glob

sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_loader import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch
import numpy as np
import random
import scanpy as sc

from ODE.ode_analysis20260420 import create_ode_model_from_args, ODE_ML_Hybrid, validate_ode_args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def _normalize_bool_flag(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.lower()
        if s in ("true", "1", "yes", "y", "t"):
            return True
        if s in ("false", "0", "no", "n", "f"):
            return False
    raise ValueError(f"Invalid boolean value: {v}")



def main():
    setup_seed(1234)
    torch.autograd.set_detect_anomaly(True)
    args = create_argparser().parse_args()
    args.SoftReg = _normalize_bool_flag(args.SoftReg)
    args.use_time_embedding = _normalize_bool_flag(args.use_time_embedding)
    args.learn_W0 = _normalize_bool_flag(args.learn_W0)
    validate_ode_args(args)

    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        base_output_dir = os.getcwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_output_dir = os.path.join(base_output_dir, f"{timestamp}_train")
    counter = 1
    original_train_output_dir = train_output_dir
    while os.path.exists(train_output_dir):
        train_output_dir = f"{original_train_output_dir}_{counter}"
        counter += 1

    os.makedirs(train_output_dir, exist_ok=True)
    logger.log(f"Training output directory created: {train_output_dir}")

    log_dir = os.path.join(train_output_dir, "log")
    checkpoint_dir = os.path.join(train_output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=log_dir)

    logger.log("creating model and diffusion...")
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("loading preprocessed h5ad file...")
    adata = sc.read_h5ad(args.data_dir)
    logger.log(f"Data loaded. Shape: {adata.shape}")
    gene_list = list(adata.var["gene_name"].unique())
    logger.log(f"Found {len(gene_list)} genes")

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_vae=True,
        preprocess=False,
    )

    logger.log(
        f"creating ODE model: type={args.ode_model_type}, alpha_mode={args.alpha_mode}, "
        f"alpha_topk={args.alpha_topk}, num_bases={args.num_bases}, num_experts={args.num_experts}, rank={args.rank}"
    )
    ode = create_ode_model_from_args(
        args=args,
        gene_list=gene_list,
        device=dist_util.dev(),
    )
    logger.log("ODE model created successfully!")

    timesteps = diffusion.num_timesteps
    logger.log(f"Using {timesteps} timesteps for hybrid model")

    hybrid_model = ODE_ML_Hybrid(
        ode_model=ode,
        ml_model=None,
        timesteps=timesteps,
        use_hybrid=False,
    ).to(dist_util.dev())

    logger.log("training...")
    TrainLoop(
        model=hybrid_model,
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
        save_dir=checkpoint_dir,
        ode_reg_lambda=args.ode_reg_lambda,
        ode_reg_norm=args.ode_reg_norm,
        save_loss_details=args.save_loss_details,
    ).run_loop()

    model_files = glob.glob(os.path.join(checkpoint_dir, "*", "ema_*.pt"))
    if not model_files:
        model_files = glob.glob(os.path.join(checkpoint_dir, "*", "model*.pt"))

    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        logger.log(f"Latest model found: {latest_model}")
        model_dir = os.path.dirname(latest_model)
        total_steps = args.lr_anneal_steps

        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"TRAINED_MODEL_PATH='{latest_model}'")
        print(f"MODEL_DIR='{model_dir}'")
        print(f"CHECKPOINT_DIR='{checkpoint_dir}'")
        print(f"TOTAL_STEPS={total_steps}")
        print(f"MODEL_NAME='{args.model_name}'")
        print(f"ODE_MODEL_TYPE='{args.ode_model_type}'")
        print(f"ALPHA_MODE='{args.alpha_mode}'")
        print(f"ALPHA_TOPK={args.alpha_topk}")
        print(f"{'='*60}")
    else:
        logger.log("Warning: No model files found in checkpoint directory")
        print(f"CHECKPOINT_DIR='{checkpoint_dir}'")



def create_argparser():
    defaults = dict(
        data_dir="data_preparation/pbmc68k.h5ad",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=2000,
        batch_size=128,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=200000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_name="pbmc68k_dynamic_ode_20260420",
        save_dir='output/diffusion_checkpoint_dynamic_ode_20260420',
        output_dir="",
        ode_reg_lambda=0.0005,
        ode_reg_norm='l1',
        save_loss_details=True,
        SoftReg=True,
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        use_time_embedding=True,
        time_hidden_dim=64,

        # ODE family selector
        ode_model_type="lowrank_residual",   # mixture_softplus / matrix_dict / lowrank_residual

        # alpha selector
        alpha_mode="mlp",                    # mlp / linear
        alpha_topk=0,                         # 0 means disabled
        alpha_hidden_dim=128,
        alpha_dropout=0.0,
        alpha_temperature=1.0,

        # family-specific sizes
        num_experts=8,
        num_bases=8,
        rank=16,
        learn_W0=True,
        w_ema=0.95,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
