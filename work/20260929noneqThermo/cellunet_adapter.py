"""Frozen START_X CellUNet output interpreted literally as V_t(x)."""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch

SUITE = "work.20260916_x0predict_hybrid_additive"


def restore_stage1(checkpoint=None, campaign=None):
    checkpoints = importlib.import_module(f"{SUITE}.training.checkpoints")
    common = importlib.import_module(f"{SUITE}.common")
    if campaign is not None:
        payload, record = checkpoints.canonical_stage1(Path(campaign).expanduser().resolve())
        checkpoint = record["checkpoint"]
        builder = importlib.import_module(f"{SUITE}.models").build_model
    else:
        payload, _ = checkpoints.load_stage1(checkpoint)
        builder = importlib.import_module("work.20260915_x0predict.models").build_model
    meta = payload["metadata"]
    config = meta["effective_config"]
    if (config["objective"] != "stage1" or not config["predict_xstart"]
            or meta["checkpoint_kind"] != "ema" or meta["step"] != 30000
            or config["total_steps"] != 30000 or config.get("ts_layer") is not None):
        raise ValueError("requires final 30k START_X Stage1 EMA using h5ad.X")
    model = builder(config, meta["gene_names"], state=payload["state_dict"])
    model.requires_grad_(False).eval()
    return model, meta, Path(checkpoint).expanduser().resolve(), common.file_hash


class CellUNetVelocity:
    def __init__(self, model, genes, batch_size=128):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.model = model.requires_grad_(False).eval()
        self.genes = list(genes)
        self.batch_size = batch_size

    def __call__(self, X, genes, inputs, device):
        if list(genes) != self.genes:
            raise ValueError("h5ad gene order differs from Stage1 checkpoint")
        if X.ndim != 2 or X.shape[1] != len(self.genes) or not np.isfinite(X).all():
            raise ValueError("invalid h5ad.X shape or nonfinite values")
        timestep = inputs["timestep"]
        if isinstance(timestep, bool) or not isinstance(timestep, int) or timestep < 0:
            raise ValueError("timestep must be a nonnegative integer")
        model = self.model.to(device).eval()
        parts = []
        with torch.inference_mode():
            for start in range(0, len(X), self.batch_size):
                x = torch.as_tensor(X[start:start + self.batch_size], dtype=torch.float32, device=device)
                # The native sampler passes [batch, 1] unscaled raw timesteps.
                t = torch.full((len(x), 1), timestep, dtype=torch.long, device=device)
                velocity = model(x, t)
                if velocity.shape != x.shape or not torch.isfinite(velocity).all():
                    raise ValueError("invalid CellUNet output")
                parts.append(velocity.cpu().numpy())
        return np.concatenate(parts), {
            "definition": "V_t(x) = CellUNet(x, t); raw START_X prediction treated as velocity",
            "timestep": timestep, "training_timestep_range": [0, 999],
            "outside_training_timestep_range": timestep > 999,
            "input": "unnoised h5ad.X", "subtract_x": False,
            "sampling_used": False, "timestep_rescaled": False,
        }
