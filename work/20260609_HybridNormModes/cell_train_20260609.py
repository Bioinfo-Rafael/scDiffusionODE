"""
Train ODE-ML hybrid with selectable output-integration mode (20260609).

cell_train_20260421.py をベースに、ODE_ML_Hybrid を ODE_ML_HybridNorm（3 mode 対応）へ差替。
3 mode（排他）を --hybrid_norm_mode で選ぶ:
  ratio_reg | normed_learned_scale | none

既存ファイル（guided_diffusion/train_util.py, ODE/ode_20260421_regODEMLratio.py,
work/20260421_RegODEMLratio/*）は一切変更しない。GeneODE は ode_20260421 から再利用する。
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add scDiffusion directory to path for importing ODE and guided_diffusion modules
sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from local_paths import resolve_path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "20260609_Hybrid5x3"))  # run_paths を共有
import run_paths  # 出力構造 {work}/runs/{model}/{date}/{train,...}

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
from guided_diffusion.cell_model import Cell_Unet
import torch
import numpy as np
import random
import scanpy as sc
from ODE.ode_20260421_regODEMLratio import GeneODE
from ODE.ode_20260609_hybridnorm import ODE_ML_HybridNorm


def main():
    setup_seed(1234)
    torch.autograd.set_detect_anomaly(True)
    args = create_argparser().parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.edge_tsv_path = resolve_path(args.edge_tsv_path)

    # --output_dir 指定時はそれを使う（pipeline 経由）。未指定の単体実行時は
    # {work}/runs/{model}/{date}/train を既定にする（記述的 model 名）。
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        _model = f"hybridnorm__{args.hybrid_norm_mode}"
        base_output_dir = run_paths.stage_dir(run_paths.run_base(_HERE, _model), "train")

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

    # ---- SoftReg flag ----
    soft = _to_bool(args.SoftReg)

    timesteps = diffusion.num_timesteps
    logger.log(f"Using {timesteps} timesteps for hybrid model")

    # ---- GeneODE (既存クラスを再利用) ----
    logger.log(f"creating GeneODE instance with soft={soft}...")
    ode = GeneODE(
        gene_list=gene_list,
        edge_tsv_path=args.edge_tsv_path,
        soft=soft,
        device=dist_util.dev(),
    )
    ode.ratio_reg_weight = args.ratio_reg_weight
    ode.ratio_reg_target = args.ratio_reg_target

    # ---- ML model ----
    ml_model = Cell_Unet(input_dim=len(gene_list))
    ml_model.to(dist_util.dev())

    # ---- hybrid (3 mode) ----
    logger.log(f"creating ODE_ML_HybridNorm: mode={args.hybrid_norm_mode}, "
               f"scale_init={args.hybrid_scale_init}, scale_eps={args.hybrid_scale_eps}")
    hybrid_model = ODE_ML_HybridNorm(
        ode_model=ode,
        ml_model=ml_model,
        timesteps=timesteps,
        hybrid_norm_mode=args.hybrid_norm_mode,
        hybrid_scale_init=args.hybrid_scale_init,
        hybrid_scale_eps=args.hybrid_scale_eps,
        reverse_coef=_to_bool(args.reverse_coef),
    )
    hybrid_model.to(dist_util.dev())
    logger.log(f"hybrid info: {hybrid_model.get_model_info()}")

    # ---- dump config json for sampling restore ----
    config = {
        "hybrid_norm_mode": args.hybrid_norm_mode,
        "hybrid_scale_init": args.hybrid_scale_init,
        "hybrid_scale_eps": args.hybrid_scale_eps,
        "reverse_coef": _to_bool(args.reverse_coef),
        "SoftReg": soft,
        "ode_reg_lambda": args.ode_reg_lambda,
        "ode_reg_norm": args.ode_reg_norm,
        "ratio_reg_weight": args.ratio_reg_weight,
        "ratio_reg_target": args.ratio_reg_target,
        "edge_tsv_path": args.edge_tsv_path,
        "data_dir": args.data_dir,
        "diffusion_steps": args.diffusion_steps,
        "n_genes": len(gene_list),
    }
    config_path = os.path.join(train_output_dir, "hybrid_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.log(f"hybrid config saved: {config_path}")

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

    import glob
    model_files = glob.glob(os.path.join(checkpoint_dir, "*", "ema_*.pt"))
    if not model_files:
        model_files = glob.glob(os.path.join(checkpoint_dir, "*", "model*.pt"))

    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        logger.log(f"Latest model found: {latest_model}")
        model_dir = os.path.dirname(latest_model)
        total_steps = args.lr_anneal_steps
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"TRAINED_MODEL_PATH='{latest_model}'")
        print(f"MODEL_DIR='{model_dir}'")
        print(f"CHECKPOINT_DIR='{checkpoint_dir}'")
        print(f"HYBRID_CONFIG='{config_path}'")
        print(f"TOTAL_STEPS={total_steps}")
        print(f"MODEL_NAME='{args.model_name}'")
        print(f"{'='*60}")
    else:
        logger.log("Warning: No model files found in checkpoint directory")
        print(f"CHECKPOINT_DIR='{checkpoint_dir}'")
        print(f"HYBRID_CONFIG='{config_path}'")


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
        model_name="hybridnorm_20260609",
        save_dir='output/hybridnorm_20260609',
        output_dir="",
        ode_reg_lambda=0.0005,
        ode_reg_norm='l1',
        save_loss_details=True,
        SoftReg=True,
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        ratio_reg_weight=1.0,
        ratio_reg_target=1.0,
        # ---- hybrid integration mode ----
        hybrid_norm_mode="ratio_reg",    # ratio_reg | normed_learned_scale | none
        hybrid_scale_init=1.0,           # normed_learned_scale の scale 初期値
        hybrid_scale_eps=1e-8,           # 正規化 & scale eps
        reverse_coef=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def _to_bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise ValueError(f"Invalid boolean value: {v}. Use true/false.")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
