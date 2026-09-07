#!/usr/bin/env python3
"""Restore and sample a 2026-07-07 experiment run."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
REPO_ROOT = WORK_ROOT.parent.parent
for path in (str(REPO_ROOT), str(WORK_ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from guided_diffusion import dist_util, logger  # noqa: E402
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults  # noqa: E402
from ODE.ode_20260609_mathmlp import load_hybrid_state_dict  # noqa: E402
from scripts.common import (  # noqa: E402
    artifact_dirs, build_model_from_config, command_string, read_json,
    update_manifest, write_json,
)


def _seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = read_json(run_dir / "exp_config.json")
    manifest = read_json(run_dir / "manifest.json")
    checkpoint = args.model_path or manifest.get("checkpoint_path", "")
    if not checkpoint:
        raise FileNotFoundError("checkpoint_path is missing from manifest; pass --model-path")
    paths = artifact_dirs(run_dir)
    output_dir = paths["samples"]
    if args.dry_run:
        print(json.dumps({
            "command": command_string(), "run_dir": str(run_dir),
            "checkpoint": checkpoint, "output_dir": str(output_dir),
        }, indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    (run_dir / "commands" / "sample.txt").write_text(command_string() + "\n", encoding="utf-8")
    dist_util.setup_dist()
    logger.configure(dir=str(paths["logs"] / "sample"))
    seed = int(config.get("seed", 1234))
    _seed(seed)

    diffusion_args = model_and_diffusion_defaults()
    for key in diffusion_args:
        if key in config:
            diffusion_args[key] = config[key]
    _, diffusion = create_model_and_diffusion(**diffusion_args)
    adata = sc.read_h5ad(config["data_dir"], backed="r")
    gene_list = list(adata.var["gene_name"].unique())
    model = build_model_from_config(config, gene_list, diffusion.num_timesteps, dist_util.dev())
    state = dist_util.load_state_dict(checkpoint, map_location="cpu")
    load_hybrid_state_dict(model, state, strict=True, log=logger.log)
    model.to(dist_util.dev()).eval()

    num_samples = int(args.num_samples or config.get("num_samples", 500))
    batch_size = int(args.batch_size or config.get("sample_batch_size", 50))
    use_ddim = bool(config.get("use_ddim", False))
    sample_fn = diffusion.ddim_sample_loop if use_ddim else diffusion.p_sample_loop
    generated, chunks = 0, []
    while generated < num_samples:
        n = min(batch_size, num_samples - generated)
        sample_kwargs = {
            "clip_denoised": bool(config.get("clip_denoised", False)),
        }
        if not use_ddim:
            # gaussian_diffusion.py keeps a legacy start_time=1000 default.
            # Pass the actual schedule length so non-1000 smoke/config runs work.
            sample_kwargs["start_time"] = diffusion.num_timesteps
        sample, _ = sample_fn(model, (n, len(gene_list)), **sample_kwargs)
        chunks.append(sample.detach().cpu().numpy())
        generated += n
        logger.log(f"Generated {generated}/{num_samples}")
    array = np.concatenate(chunks, axis=0)
    sample_path = output_dir / "samples.npz"
    if sample_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_path = output_dir / f"samples_{stamp}.npz"
    np.savez(sample_path, cell_gen=array)
    write_json(output_dir / "sample_manifest.json", {
        "run_dir": str(run_dir), "checkpoint_path": checkpoint,
        "sample_path": str(sample_path), "num_samples": int(array.shape[0]),
        "num_genes": int(array.shape[1]), "seed": seed,
        "regime_gate_mode": config.get("regime_gate_mode", "none"),
        "t_s": config.get("t_s"), "gate_tau": config.get("gate_tau"),
    })
    stages = manifest.get("stages", {})
    stages["sample"] = {
        "status": "completed", "finished_at": datetime.now().isoformat(),
        "sample_path": str(sample_path),
    }
    update_manifest(run_dir, stages=stages, sample_path=str(sample_path))
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    print(f"SAMPLE_FILE_PATH='{sample_path}'")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main()
