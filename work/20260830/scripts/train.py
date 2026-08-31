#!/usr/bin/env python3
"""Train one 20260830 condition using the local TrainLoop subclass."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import latest_raw_checkpoint, load_experiment_config, read_json, write_json  # noqa: E402


def _device(torch, dist_util, requested):
    choice = str(requested).lower()
    if choice == "auto":
        dist_util.setup_dist()
        return dist_util.dev()
    device = torch.device(choice)
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dist_util.dev = lambda: device
    return device


def _genes(adata):
    if "gene_name" not in adata.var.columns:
        raise KeyError("AnnData .var must contain gene_name")
    genes = [str(value) for value in adata.var["gene_name"].tolist()]
    if len(genes) != adata.n_vars or len(set(genes)) != len(genes):
        raise ValueError("gene_name must be a unique one-to-one mapping")
    return genes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--resume", nargs="?", const="auto", default="")
    args = parser.parse_args(argv)

    import scanpy as sc
    import torch
    from guided_diffusion import dist_util, logger
    from guided_diffusion.cell_datasets_loader import load_data
    from guided_diffusion.resample import create_named_schedule_sampler
    from guided_diffusion.script_util import create_gaussian_diffusion
    from models import build_model_from_config
    from training import TrainLoop20260830

    config = load_experiment_config(args.config)
    target = Path(args.run_dir).resolve()
    expected_root = (SUITE_ROOT / "runs").resolve()
    target.relative_to(expected_root)
    if target == expected_root:
        raise ValueError("run-dir must be an experiment/batch directory below runs/")
    target.mkdir(parents=True, exist_ok=True)
    stored = target / "exp_config.json"
    if stored.exists() and read_json(stored) != config:
        raise RuntimeError("run directory already contains a different config")
    write_json(stored, config)

    seed = int(config["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    device = _device(torch, dist_util, config.get("device", "auto"))
    logger.configure(dir=str(target / "logs" / "train"))
    diffusion = create_gaussian_diffusion(
        steps=int(config["diffusion_steps"]),
        learn_sigma=bool(config["learn_sigma"]),
        noise_schedule=str(config["noise_schedule"]),
        use_kl=bool(config["use_kl"]),
        predict_xstart=bool(config["predict_xstart"]),
        rescale_timesteps=bool(config["rescale_timesteps"]),
        rescale_learned_sigmas=bool(config["rescale_learned_sigmas"]),
        timestep_respacing=config["timestep_respacing"],
    )
    adata = sc.read_h5ad(config["data_dir"], backed="r")
    try:
        genes = _genes(adata)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()
    model = build_model_from_config(config, genes, diffusion.num_timesteps, device)
    write_json(target / "model_info.json", {
        "wrapper": type(model).__name__,
        "denoising_output": "ml_model_only",
        "ode": model.ode_model.get_model_info(),
    })
    data = load_data(
        data_dir=config["data_dir"],
        batch_size=int(config["batch_size"]),
        train_vae=True,
        preprocess=False,
    )
    resume = ""
    if args.resume:
        resume_path = latest_raw_checkpoint(target) if args.resume == "auto" else Path(args.resume).resolve()
        if resume_path is None or not Path(resume_path).is_file():
            raise FileNotFoundError("resume checkpoint not found")
        Path(resume_path).resolve().relative_to((target / "checkpoints").resolve())
        resume = str(resume_path)
    segment_root = target / "checkpoints"
    existing = [int(path.name.split("_")[1]) for path in segment_root.glob("segment_*")]
    segment = segment_root / f"segment_{max(existing, default=-1) + 1:03d}"
    segment.mkdir(parents=True)
    sampler = create_named_schedule_sampler(config["schedule_sampler"], diffusion)
    TrainLoop20260830(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=int(config["batch_size"]),
        microbatch=int(config["microbatch"]),
        lr=float(config["lr"]),
        ema_rate=str(config["ema_rate"]),
        log_interval=int(config["log_interval"]),
        save_interval=int(config["save_interval"]),
        resume_checkpoint=resume,
        use_fp16=bool(config["use_fp16"]),
        fp16_scale_growth=float(config["fp16_scale_growth"]),
        schedule_sampler=sampler,
        weight_decay=float(config["weight_decay"]),
        lr_anneal_steps=int(config["total_steps"]),
        model_name="model",
        save_dir=str(segment),
        ode_reg_lambda=float(config["ode_reg_lambda"]),
        ode_reg_norm=str(config["ode_reg_norm"]),
        save_loss_details=bool(config["save_loss_details"]),
        cell_ode_reg_lambda_20260830=float(config["cell_ode_reg_lambda_20260830"]),
    ).run_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
