"""
Train a math-structure + MLP hybrid denoising model (20260609).

数理構造 (W(x,t)) を MLP でパラメタ化した field を、Cell_Unet と t-scheduler で
blend する hybrid denoising model を学習する。4 系統を --model_type で切替:

  lowrank : W(x,t) = U(x,t) V_low(x,t)^T
  lincomb : V = Σ_k a_k(x,t) softplus(W_k x + b_k)
  matsum  : W(x,t) = Σ_k a_k(x,t) A_k
  lora    : W(x,t) = W_0 + Σ_k a_k(x,t) U_k V_k^T

cell_train_20260421.py をベースに、GeneODE を build_math_field に差し替えただけ。
TrainLoop / diffusion / data loader は無変更。正則化は既存 hook
(model.ode_model.off_mask_penalty) をそのまま利用する。
"""

import argparse
import os
import sys
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
import json
import scanpy as sc
from ODE.ode_20260609_mathmlp import build_math_field, MathML_Hybrid


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
        _model = f"mathmlp__{args.model_type}"
        base_output_dir = run_paths.stage_dir(run_paths.run_base(_HERE, _model), "train")

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
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # Load preprocessed data
    logger.log("loading preprocessed h5ad file...")
    adata = sc.read_h5ad(args.data_dir)
    logger.log(f"Data loaded. Shape: {adata.shape}")

    # Get gene list for model
    gene_list = list(adata.var["gene_name"].unique())
    logger.log(f"Found {len(gene_list)} genes")

    # Create data loader with preprocessed data (gene space, no VAE)
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_vae=True,
        preprocess=False,
    )

    # ---- parse boolean-ish flags ----
    use_mask_reg = _to_bool(args.use_mask_reg)
    soft = _to_bool(args.SoftReg)
    use_decay = _to_bool(args.use_decay)

    timesteps = diffusion.num_timesteps
    logger.log(f"Using {timesteps} timesteps for hybrid model")

    # ---- build math field via factory ----
    logger.log(
        f"creating math field: model_type={args.model_type}, rank={args.rank}, "
        f"K={args.K}, use_mask_reg={use_mask_reg}, soft={soft}"
    )
    field = build_math_field(
        model_type=args.model_type,
        gene_list=gene_list,
        edge_tsv_path=args.edge_tsv_path,
        rank=args.rank,
        K=args.K,
        use_mask=use_mask_reg,
        soft=soft,
        time_dim=args.time_dim,
        hidden=args.field_hidden,
        dropout=args.field_dropout,
        lowrank_penalty_subsample=args.lowrank_penalty_subsample,
        use_decay=use_decay,
        device=dist_util.dev(),
    )
    field.ratio_reg_weight = args.ratio_reg_weight
    field.ratio_reg_target = args.ratio_reg_target
    # LowRank の subsample W キャッシュは正則化が有効なときだけ作る（無駄構築の防止）
    field.enable_offmask_cache = bool(soft and args.ode_reg_lambda > 0)
    logger.log(f"field info: {field.get_model_info()} "
               f"(enable_offmask_cache={field.enable_offmask_cache})")

    # ---- ML model (Cell_Unet) ----
    ml_model = Cell_Unet(input_dim=len(gene_list))
    ml_model.to(dist_util.dev())

    # ---- hybrid ----
    logger.log("creating math-ML hybrid model...")
    hybrid_model = MathML_Hybrid(
        field=field,
        ml_model=ml_model,
        timesteps=timesteps,
    )
    hybrid_model.to(dist_util.dev())

    # ---- dump config json for sampling/visualization restore ----
    config = {
        "model_type": args.model_type,
        "rank": args.rank,
        "K": args.K,
        "use_mask_reg": use_mask_reg,
        "soft": soft,
        "use_decay": use_decay,
        "time_dim": args.time_dim,
        "field_hidden": args.field_hidden,
        "field_dropout": args.field_dropout,
        "lowrank_penalty_subsample": args.lowrank_penalty_subsample,
        "ode_reg_lambda": args.ode_reg_lambda,
        "ode_reg_norm": args.ode_reg_norm,
        "ratio_reg_weight": args.ratio_reg_weight,
        "ratio_reg_target": args.ratio_reg_target,
        "edge_tsv_path": args.edge_tsv_path,
        "data_dir": args.data_dir,
        "diffusion_steps": args.diffusion_steps,
        "n_genes": len(gene_list),
    }
    config_path = os.path.join(train_output_dir, "field_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.log(f"field config saved: {config_path}")

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

    # Find the latest checkpoint (best model)
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
        print(f"FIELD_CONFIG='{config_path}'")
        print(f"TOTAL_STEPS={total_steps}")
        print(f"MODEL_NAME='{args.model_name}'")
        print(f"{'='*60}")
    else:
        logger.log("Warning: No model files found in checkpoint directory")
        print(f"CHECKPOINT_DIR='{checkpoint_dir}'")
        print(f"FIELD_CONFIG='{config_path}'")


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
        model_name="mathmlp_20260609",
        save_dir='output/mathmlp_20260609',
        output_dir="",
        ode_reg_lambda=0.0005,
        ode_reg_norm='l1',
        save_loss_details=True,
        SoftReg=True,
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        ratio_reg_weight=1.0,
        ratio_reg_target=1.0,
        # ---- math field specific ----
        model_type="lowrank",            # lowrank | lincomb | matsum | lora
        rank=16,                         # low-rank / LoRA の r
        K=8,                             # lincomb / matsum / lora の成分数
        use_mask_reg=True,               # off-mask 罰則 (True) / W 全体罰則 (False)
        use_decay=True,                  # GeneODE 風の減衰項 -softplus(gamma)*x を加える
        time_dim=64,
        field_hidden=256,
        field_dropout=0.0,
        lowrank_penalty_subsample=8,
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
