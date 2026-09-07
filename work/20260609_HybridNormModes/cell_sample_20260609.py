#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate cell samples from a trained ODE-ML HybridNorm model (20260609).

cell_sample_20260421.py をベースに ODE_ML_HybridNorm（3 mode）で再構築する。
normed_learned_scale の checkpoint を正しく復元するには学習時と同じ hybrid_norm_mode が必要なため、
--hybrid_config（学習時の hybrid_config.json）から復元する。無指定時は CLI 値を使う。
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
import json
from datetime import datetime

sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from local_paths import resolve_path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "20260609_Hybrid5x3"))
import run_paths  # 単体実行時 --model_path から {base}/sample を導出

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260421_regODEMLratio import GeneODE
from ODE.ode_20260609_hybridnorm import ODE_ML_HybridNorm


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


def _to_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y", "t")


def resolve_hybrid_cfg(args):
    cfg = dict(
        hybrid_norm_mode=args.hybrid_norm_mode,
        hybrid_scale_init=args.hybrid_scale_init,
        hybrid_scale_eps=args.hybrid_scale_eps,
        reverse_coef=_to_bool(args.reverse_coef),
        SoftReg=_to_bool(args.SoftReg),
    )
    if args.hybrid_config and os.path.exists(args.hybrid_config):
        with open(args.hybrid_config) as f:
            loaded = json.load(f)
        for k in cfg:
            if k in loaded:
                cfg[k] = loaded[k]
        logger.log(f"Loaded hybrid config from {args.hybrid_config}: {cfg}")
    else:
        logger.log(f"No hybrid_config; using argparse values: {cfg}")
    return cfg


def main():
    setup_seed(1234)
    args = create_argparser().parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.edge_tsv_path = resolve_path(args.edge_tsv_path)

    # --output_dir 指定時はそれ。未指定の単体実行時は --model_path から {base}/sample を導出。
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        _b = run_paths.infer_run_base(args.model_path) if args.model_path else ""
        base_output_dir = os.path.join(_b, "sample") if _b else os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_output_dir = os.path.join(base_output_dir, f"{timestamp}_sample")
    counter = 1
    original = sample_output_dir
    while os.path.exists(sample_output_dir):
        sample_output_dir = f"{original}_{counter}"
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
    logger.log(f"Loaded {len(gene_list)} genes.")

    device = dist_util.dev()
    cfg = resolve_hybrid_cfg(args)

    logger.log("Recreating ODE_ML_HybridNorm model...")
    ode = GeneODE(
        gene_list=gene_list,
        edge_tsv_path=args.edge_tsv_path,
        soft=cfg["SoftReg"],
        device=device,
    )
    ml_model = Cell_Unet(input_dim=len(gene_list)).to(device)
    hybrid = ODE_ML_HybridNorm(
        ode_model=ode,
        ml_model=ml_model,
        timesteps=diffusion.num_timesteps,
        hybrid_norm_mode=cfg["hybrid_norm_mode"],
        hybrid_scale_init=cfg["hybrid_scale_init"],
        hybrid_scale_eps=cfg["hybrid_scale_eps"],
        reverse_coef=cfg["reverse_coef"],
    ).to(device)
    logger.log(f"hybrid info: {hybrid.get_model_info()}")

    logger.log(f"Loading checkpoint from: {args.model_path}")
    sd = dist_util.load_state_dict(args.model_path, map_location="cpu")
    if any(k.startswith("ema") for k in sd.keys()):
        logger.log("Detected EMA weights, loading from 'ema' key")
        sd = sd.get('ema', sd)
    if "model" in sd:
        sd = sd["model"]

    missing, unexpected = hybrid.load_state_dict(sd, strict=False)
    logger.log(f"Missing keys ({len(missing)}): {list(missing)}")
    logger.log(f"Unexpected keys ({len(unexpected)}): {list(unexpected)}")
    hybrid.eval()

    logger.log("Starting sampling process...")
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    all_cells = []
    generated = 0
    total = args.num_samples
    batch = args.batch_size
    while generated < total:
        n = min(batch, total - generated)
        sample, _ = sample_fn(hybrid, (n, len(gene_list)), clip_denoised=args.clip_denoised)
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
    print(f"SAMPLING COMPLETED SUCCESSFULLY")
    print(f"Saved file: {actual_save_path}")
    print(f"{'='*60}")
    print(f"\n# Shell script variables:")
    print(f"SAMPLE_FILE_PATH='{actual_save_path}'")
    print(f"NUM_SAMPLES={arr.shape[0]}")
    print(f"NUM_GENES={arr.shape[1]}")
    print(f"MODEL_PATH_USED='{args.model_path}'")


def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=500,
        batch_size=50,
        use_ddim=False,
        model_path="",
        data_dir="/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad",
        sample_name="samples.npz",
        output_dir="",
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        hybrid_config="",                # 学習時の hybrid_config.json（推奨）
        # ---- fallback（hybrid_config が無いとき）----
        hybrid_norm_mode="ratio_reg",
        hybrid_scale_init=1.0,
        hybrid_scale_eps=1e-8,
        SoftReg=True,
        reverse_coef=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
