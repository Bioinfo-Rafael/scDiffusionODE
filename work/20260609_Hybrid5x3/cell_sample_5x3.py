#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified sampling for the 5×3 + baseline experiments (20260609).

exp_config.json から ode_branch / hybrid_norm_mode / rank / K 等を復元して build_denoiser で
同型再構築し、checkpoint をロードしてサンプリングする。
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import run_paths  # 単体実行時 --model_path から {base}/sample を導出

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from ODE.ode_20260609_hybrid5x3 import build_denoiser


def save_data(all_cells, save_path):
    if os.path.exists(save_path):
        base_path = save_path.replace('.npz', '')
        existing = glob.glob(f"{base_path}_*.npz")
        nums = []
        for f in existing:
            s = f.replace(base_path + '_', '').replace('.npz', '')
            if s.isdigit():
                nums.append(int(s))
        save_path = f"{base_path}_{(max(nums)+1) if nums else 1:03d}.npz"
        print(f"File exists. Saving as: {save_path}")
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


def resolve_cfg(args):
    cfg = dict(
        ode_branch=args.ode_branch, hybrid_norm_mode=args.hybrid_norm_mode,
        rank=args.rank, K=args.K, SoftReg=_to_bool(args.SoftReg),
        ode_reg_lambda=args.ode_reg_lambda, time_dim=args.time_dim,
        field_hidden=args.field_hidden, field_dropout=args.field_dropout,
        lowrank_penalty_subsample=args.lowrank_penalty_subsample,
        use_decay=_to_bool(args.use_decay), ratio_reg_weight=args.ratio_reg_weight,
        ratio_reg_target=args.ratio_reg_target, hybrid_scale_init=args.hybrid_scale_init,
        hybrid_scale_eps=args.hybrid_scale_eps,
        scale_model_type=args.scale_model_type, scale_input_source=args.scale_input_source,
        ode_input_source=args.ode_input_source, scale_hidden=args.scale_hidden,
        scale_eps=args.scale_eps,
    )
    if args.exp_config and os.path.exists(args.exp_config):
        with open(args.exp_config) as f:
            loaded = json.load(f)
        for k in cfg:
            if k in loaded:
                cfg[k] = loaded[k]
        logger.log(f"Loaded exp_config: {cfg}")
    else:
        logger.log(f"No exp_config; using argparse: {cfg}")
    return cfg


def main():
    setup_seed(1234)
    args = create_argparser().parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.edge_tsv_path = resolve_path(args.edge_tsv_path)

    # --output_dir 指定時はそれ。未指定の単体実行時は --model_path から {base}/sample を導出。
    if args.output_dir:
        base = args.output_dir
    else:
        _b = run_paths.infer_run_base(args.model_path) if args.model_path else ""
        base = os.path.join(_b, "sample") if _b else os.getcwd()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base, f"{ts}_sample")
    c = 1
    orig = out_dir
    while os.path.exists(out_dir):
        out_dir = f"{orig}_{c}"; c += 1
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "log")
    data_out = os.path.join(out_dir, "SampledData")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_out, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=log_dir)

    logger.log("creating diffusion...")
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    adata = sc.read_h5ad(args.data_dir)
    gene_list = list(adata.var["gene_name"].unique())
    logger.log(f"{len(gene_list)} genes")
    device = dist_util.dev()

    cfg = resolve_cfg(args)
    model = build_denoiser(
        ode_branch=cfg["ode_branch"], gene_list=gene_list, edge_tsv_path=args.edge_tsv_path,
        timesteps=diffusion.num_timesteps, hybrid_norm_mode=cfg["hybrid_norm_mode"],
        rank=cfg["rank"], K=cfg["K"], soft=cfg["SoftReg"], ode_reg_lambda=cfg["ode_reg_lambda"],
        time_dim=cfg["time_dim"], field_hidden=cfg["field_hidden"], field_dropout=cfg["field_dropout"],
        lowrank_penalty_subsample=cfg["lowrank_penalty_subsample"], use_decay=cfg["use_decay"],
        ratio_reg_weight=cfg["ratio_reg_weight"], ratio_reg_target=cfg["ratio_reg_target"],
        hybrid_scale_init=cfg["hybrid_scale_init"], hybrid_scale_eps=cfg["hybrid_scale_eps"],
        scale_model_type=cfg["scale_model_type"], scale_input_source=cfg["scale_input_source"],
        ode_input_source=cfg["ode_input_source"], scale_hidden=cfg["scale_hidden"],
        scale_eps=cfg["scale_eps"],
        device=device,
    )
    model.to(device)

    logger.log(f"loading checkpoint: {args.model_path}")
    sd = dist_util.load_state_dict(args.model_path, map_location="cpu")
    if any(k.startswith("ema") for k in sd.keys()):
        sd = sd.get('ema', sd)
    if "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.log(f"Missing keys ({len(missing)}): {list(missing)}")
    logger.log(f"Unexpected keys ({len(unexpected)}): {list(unexpected)}")
    model.eval()

    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop
    all_cells, generated = [], 0
    while generated < args.num_samples:
        n = min(args.batch_size, args.num_samples - generated)
        sample, _ = sample_fn(model, (n, len(gene_list)), clip_denoised=args.clip_denoised)
        all_cells.append(sample.detach().cpu().numpy())
        generated += n
        logger.log(f"Generated {generated}/{args.num_samples}")

    arr = np.concatenate(all_cells, axis=0)
    name = os.path.basename(args.sample_name) if args.sample_name else "samples.npz"
    saved = save_data(arr, os.path.join(data_out, name))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    print(f"\nSAMPLE_FILE_PATH='{saved}'")
    print(f"NUM_SAMPLES={arr.shape[0]}")
    print(f"NUM_GENES={arr.shape[1]}")


def create_argparser():
    defaults = dict(
        clip_denoised=False, num_samples=500, batch_size=50, use_ddim=False,
        model_path="", data_dir="/home/suzuki/Projects/scDiffusion/data_preparation/pbmc68k.h5ad",
        sample_name="samples.npz", output_dir="",
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        exp_config="",
        # fallback (exp_config が無いとき)
        ode_branch="geneode", hybrid_norm_mode="ratio_reg", rank=16, K=8, SoftReg=True,
        ode_reg_lambda=5.0, time_dim=64, field_hidden=256, field_dropout=0.0,
        lowrank_penalty_subsample=8, use_decay=True, ratio_reg_weight=1.0, ratio_reg_target=1.0,
        hybrid_scale_init=1.0, hybrid_scale_eps=1e-8,
        scale_model_type="none", scale_input_source="ml_emb", ode_input_source="none",
        scale_hidden=128, scale_eps=1e-8,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
