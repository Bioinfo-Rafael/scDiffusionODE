"""
Unified cell_train for the 5 ODE-branch × 3 hybrid-mode experiment matrix (20260609).

引数で ODE 枝（geneode/lowrank/lincomb/matsum/lora/plain）と hybrid mode を選び、
build_denoiser で 1 つの denoising model を作って TrainLoop に渡す統合スクリプト。

既存ファイルは無編集。GeneODE / build_math_field / Cell_Unet を import 再利用する
ODE/ode_20260609_hybrid5x3.py を経由してモデルを構築する。
"""

import argparse
import os
import sys
import json
import shlex
from datetime import datetime

sys.path.insert(0, '/home/suzuki/Projects/scDiffusion')
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from local_paths import resolve_path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
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
import torch
import numpy as np
import random
import scanpy as sc
from ODE.ode_20260609_hybrid5x3 import build_denoiser


def main():
    args = create_argparser().parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.edge_tsv_path = resolve_path(args.edge_tsv_path)
    setup_seed(args.seed)
    torch.autograd.set_detect_anomaly(True)

    # --output_dir 指定時はそれを使う（pipeline/matrix 経由）。未指定の単体実行時は
    # {work}/runs/{model}/{date}/train を既定にする（番号でない記述的 model 名）。
    if args.output_dir:
        base_output_dir = args.output_dir
    else:
        _model = f"{args.ode_branch}__{args.hybrid_norm_mode}"
        base_output_dir = run_paths.stage_dir(
            run_paths.run_base(_HERE, _model, suffix=args.name), "train")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_output_dir = os.path.join(base_output_dir, f"{timestamp}_train")
    counter = 1
    original = train_output_dir
    while os.path.exists(train_output_dir):
        train_output_dir = f"{original}_{counter}"
        counter += 1
    os.makedirs(train_output_dir, exist_ok=True)

    log_dir = os.path.join(train_output_dir, "log")
    checkpoint_dir = os.path.join(train_output_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=log_dir)
    logger.log(f"Training output directory: {train_output_dir}")

    logger.log("creating diffusion...")
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("loading preprocessed h5ad...")
    adata = sc.read_h5ad(args.data_dir)
    gene_list = list(adata.var["gene_name"].unique())
    logger.log(f"Found {len(gene_list)} genes")

    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_vae=True,       # VAE を通さず遺伝子空間そのまま（全実験共通）
        preprocess=False,
    )

    soft = _to_bool(args.SoftReg)
    timesteps = diffusion.num_timesteps

    logger.log(
        f"building denoiser: ode_branch={args.ode_branch}, hybrid_norm_mode={args.hybrid_norm_mode}, "
        f"rank={args.rank}, K={args.K}, soft={soft}, ode_reg_lambda={args.ode_reg_lambda}"
    )
    model = build_denoiser(
        ode_branch=args.ode_branch,
        gene_list=gene_list,
        edge_tsv_path=args.edge_tsv_path,
        timesteps=timesteps,
        hybrid_norm_mode=args.hybrid_norm_mode,
        rank=args.rank,
        K=args.K,
        soft=soft,
        ode_reg_lambda=args.ode_reg_lambda,
        time_dim=args.time_dim,
        field_hidden=args.field_hidden,
        field_dropout=args.field_dropout,
        lowrank_penalty_subsample=args.lowrank_penalty_subsample,
        use_decay=_to_bool(args.use_decay),
        ratio_reg_weight=args.ratio_reg_weight,
        ratio_reg_target=args.ratio_reg_target,
        hybrid_scale_init=args.hybrid_scale_init,
        hybrid_scale_eps=args.hybrid_scale_eps,
        scale_model_type=args.scale_model_type,
        scale_input_source=args.scale_input_source,
        ode_input_source=args.ode_input_source,
        scale_hidden=args.scale_hidden,
        scale_eps=args.scale_eps,
        reverse_coef=_to_bool(args.reverse_coef),
        regime_gate_mode=args.regime_gate_mode,
        regime_gate_type=args.regime_gate_type,
        t_s=_optional_float(args.t_s),
        gate_tau=args.gate_tau,
        device=dist_util.dev(),
    )
    model.to(dist_util.dev())
    if hasattr(model, "get_model_info"):
        logger.log(f"model info: {model.get_model_info()}")
    else:
        logger.log("model: plain Cell_Unet (baseline, no ODE / no hook)")

    # ---- dump exp config json for sampling restore ----
    config = {
        "ode_branch": args.ode_branch,
        "hybrid_norm_mode": args.hybrid_norm_mode,
        "rank": args.rank,
        "K": args.K,
        "SoftReg": soft,
        "ode_reg_lambda": args.ode_reg_lambda,
        "ode_reg_norm": args.ode_reg_norm,
        "ratio_reg_weight": args.ratio_reg_weight,
        "ratio_reg_target": args.ratio_reg_target,
        "hybrid_scale_init": args.hybrid_scale_init,
        "hybrid_scale_eps": args.hybrid_scale_eps,
        "scale_model_type": args.scale_model_type,
        "scale_input_source": args.scale_input_source,
        "ode_input_source": args.ode_input_source,
        "scale_hidden": args.scale_hidden,
        "scale_eps": args.scale_eps,
        "reverse_coef": _to_bool(args.reverse_coef),
        "regime_gate_mode": args.regime_gate_mode,
        "regime_gate_type": args.regime_gate_type,
        "t_s": _optional_float(args.t_s),
        "gate_tau": args.gate_tau,
        "time_dim": args.time_dim,
        "field_hidden": args.field_hidden,
        "field_dropout": args.field_dropout,
        "lowrank_penalty_subsample": args.lowrank_penalty_subsample,
        "use_decay": _to_bool(args.use_decay),
        "edge_tsv_path": args.edge_tsv_path,
        "data_dir": args.data_dir,
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
        "n_genes": len(gene_list),
        # ---- 実行条件の完全記録（再現用。sample 復元は上のキーだけ参照するので追加は無害）----
        "run_name": args.name,
        "batch_size": args.batch_size,
        "microbatch": args.microbatch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "ema_rate": args.ema_rate,
        "lr_anneal_steps": args.lr_anneal_steps,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "schedule_sampler": args.schedule_sampler,
        "use_fp16": args.use_fp16,
        "model_name": args.model_name,
        "train_output_dir": train_output_dir,
        "command": " ".join(shlex.quote(s) for s in sys.argv),  # 叩いたコマンドそのまま
        "all_args": {k: v for k, v in sorted(vars(args).items())},  # 全引数スナップショット
    }
    config_path = os.path.join(train_output_dir, "exp_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.log(f"exp config saved: {config_path}")

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
        save_dir=checkpoint_dir,
        ode_reg_lambda=args.ode_reg_lambda,
        ode_reg_norm=args.ode_reg_norm,
        save_loss_details=args.save_loss_details,
    ).run_loop()

    import glob
    model_files = glob.glob(os.path.join(checkpoint_dir, "*", "ema_*.pt"))
    if not model_files:
        model_files = glob.glob(os.path.join(checkpoint_dir, "*", "model*.pt"))
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED: ode_branch={args.ode_branch} mode={args.hybrid_norm_mode}")
    if model_files:
        latest_model = max(model_files, key=os.path.getmtime)
        print(f"TRAINED_MODEL_PATH='{latest_model}'")
        print(f"MODEL_DIR='{os.path.dirname(latest_model)}'")
    print(f"CHECKPOINT_DIR='{checkpoint_dir}'")
    print(f"EXP_CONFIG='{config_path}'")
    print(f"TOTAL_STEPS={args.lr_anneal_steps}")
    print(f"MODEL_NAME='{args.model_name}'")
    print(f"{'='*60}")


def create_argparser():
    defaults = dict(
        data_dir="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=30000,
        batch_size=128,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=1000,
        save_interval=5000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        model_name="hybrid5x3",
        save_dir='output/hybrid5x3',
        output_dir="",
        name="",   # run 識別名（exp_config.json に run_name で記録、output_dir 未指定時は日付 dir 末尾に付与）
        seed=1234,
        # ---- regularization (existing hook) ----
        SoftReg=True,
        ode_reg_lambda=5.0,
        ode_reg_norm='l1',
        save_loss_details=True,
        ratio_reg_weight=1.0,
        ratio_reg_target=1.0,
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        # ---- branch / mode selection ----
        ode_branch="geneode",            # geneode|lowrank|lincomb|matsum|lora|plain
        hybrid_norm_mode="ratio_reg",    # ratio_reg|none|scale_model|normed_learned_scale(deprecated)
        rank=16,
        K=8,
        time_dim=64,
        field_hidden=256,
        field_dropout=0.0,
        lowrank_penalty_subsample=8,
        use_decay=True,
        hybrid_scale_init=1.0,
        hybrid_scale_eps=1e-8,
        # ---- scale_model mode（hybrid_norm_mode="scale_model" のとき simple を指定）----
        scale_model_type="none",         # none|simple
        scale_input_source="ml_emb",     # ml_emb|x
        ode_input_source="none",         # none|x_ml_emb
        scale_hidden=128,
        scale_eps=1e-8,
        reverse_coef=False,
        regime_gate_mode="none",
        regime_gate_type="sigmoid",
        t_s="",
        gate_tau=20.0,
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


def _optional_float(v):
    if v is None or str(v).strip().lower() in ("", "none"):
        return None
    return float(v)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()
