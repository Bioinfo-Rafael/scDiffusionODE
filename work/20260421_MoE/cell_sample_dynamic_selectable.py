#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample script with selectable ODE model family.

Important:
- Use exactly the same ODE-related arguments as in training.
- In particular: ode_model_type / alpha_mode / alpha_topk / num_bases / num_experts / rank / learn_W0
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

sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ODE.ode_analysis20260420 import create_ode_model_from_args, ODE_ML_Hybrid, validate_ode_args



def save_data(all_cells, save_path):
    if os.path.exists(save_path):
        base_path = save_path.replace('.npz', '')
        pattern = f"{base_path}_*.npz"
        existing_files = glob.glob(pattern)
        numbers = []
        for f in existing_files:
            try:
                num_str = f.replace(base_path + '_', '').replace('.npz', '')
                if num_str.isdigit():
                    numbers.append(int(num_str))
            except Exception:
                continue
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
    sample_output_dir = os.path.join(base_output_dir, f"{timestamp}_sample")
    counter = 1
    original_sample_output_dir = sample_output_dir
    while os.path.exists(sample_output_dir):
        sample_output_dir = f"{original_sample_output_dir}_{counter}"
        counter += 1

    os.makedirs(sample_output_dir, exist_ok=True)
    logger.log(f"Sample output directory created: {sample_output_dir}")

    log_dir = os.path.join(sample_output_dir, "log")
    sampled_data_dir = os.path.join(sample_output_dir, "SampledData")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(sampled_data_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=log_dir)

    logger.log("Creating diffusion process...")
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    logger.log("Loading preprocessed h5ad (must match training)...")
    adata = sc.read_h5ad(args.data_dir)
    gene_list = adata.var["gene_name"].tolist()
    logger.log(f"Loaded {len(gene_list)} HVG genes from preprocessed file.")

    device = dist_util.dev()
    logger.log(f"Using device: {device}")

    logger.log(
        f"Recreating ODE model: type={args.ode_model_type}, alpha_mode={args.alpha_mode}, "
        f"alpha_topk={args.alpha_topk}, num_bases={args.num_bases}, num_experts={args.num_experts}, rank={args.rank}"
    )
    ode = create_ode_model_from_args(
        args=args,
        gene_list=gene_list,
        device=device,
    )

    hybrid = ODE_ML_Hybrid(
        ode_model=ode,
        ml_model=None,
        timesteps=diffusion.num_timesteps,
        use_hybrid=False,
    ).to(device)

    logger.log(f"Loading checkpoint from: {args.model_path}")
    sd = dist_util.load_state_dict(args.model_path, map_location="cpu")

    if any(k.startswith("ema") for k in sd.keys()):
        logger.log("Detected EMA weights, loading from 'ema' key")
        sd = sd.get('ema', sd)
    if "model" in sd:
        sd = sd["model"]

    missing, unexpected = hybrid.load_state_dict(sd, strict=False)
    logger.log(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    hybrid.eval()

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
    base_filename = os.path.basename(args.sample_name) if args.sample_name else "samples.npz"
    sample_file_path = os.path.join(sampled_data_dir, base_filename)
    actual_save_path = save_data(arr, sample_file_path)
    logger.log(f"Saved {arr.shape[0]} samples to {actual_save_path}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    logger.log("Sampling complete!")

    print(f"\n{'='*60}")
    print("SAMPLING COMPLETED SUCCESSFULLY")
    print(f"Saved file: {actual_save_path}")
    print(f"Number of samples: {arr.shape[0]}")
    print(f"Number of genes: {arr.shape[1]}")
    print(f"ODE_MODEL_TYPE='{args.ode_model_type}'")
    print(f"ALPHA_MODE='{args.alpha_mode}'")
    print(f"ALPHA_TOPK={args.alpha_topk}")
    print(f"{'='*60}")
    print(f"\n# Shell script variables:")
    print(f"SAMPLE_FILE_PATH='{actual_save_path}'")
    print(f"NUM_SAMPLES={arr.shape[0]}")
    print(f"NUM_GENES={arr.shape[1]}")
    print(f"MODEL_PATH_USED='{args.model_path}'")
    print(f"DATA_DIR_USED='{args.data_dir}'")



def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=500,
        batch_size=50,
        use_ddim=False,
        model_path="/home/suzuki/Projects/scDiffusion/output/model.pt",
        data_dir="/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k_preprocessed.h5ad",
        sample_name="pbmc68k.npz",
        output_dir="",
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        SoftReg=True,
        use_time_embedding=True,
        time_hidden_dim=64,

        # ODE family selector
        ode_model_type="lowrank_residual",

        # alpha selector
        alpha_mode="mlp",
        alpha_topk=0,
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
