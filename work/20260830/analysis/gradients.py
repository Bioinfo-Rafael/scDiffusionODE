"""Post-hoc gradient interaction analysis; no optimizer is constructed or stepped."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd
import torch

from .metrics import diffusion_training_target


_MODEL_STEP = re.compile(r"model(?P<training_step>[0-9]+)\.pt$")


def checkpoint_training_step(path) -> int:
    name = Path(path).name
    match = _MODEL_STEP.search(name)
    if match:
        return int(match.group("training_step"))
    numbers = re.findall(r"([0-9]+)", name)
    if not numbers:
        raise ValueError(f"cannot parse training step from checkpoint: {path}")
    return int(numbers[-1])


def select_analysis_checkpoints(
    checkpoint_paths: Iterable[Path],
    fractions: Sequence[float] = (0.10, 0.33, 0.66, 1.0),
) -> list[dict]:
    """Select nearest available checkpoints to early/middle/late/final targets."""

    candidates = sorted(
        ((checkpoint_training_step(path), Path(path)) for path in checkpoint_paths),
        key=lambda item: item[0],
    )
    if not candidates:
        return []
    maximum = candidates[-1][0]
    labels = ("early_10pct", "middle_33pct", "late_66pct", "final_100pct")
    selected: dict[Path, dict] = {}
    for fraction, label in zip(fractions, labels):
        target = maximum * float(fraction)
        training_step, path = min(
            candidates, key=lambda item: (abs(item[0] - target), item[0])
        )
        if path not in selected:
            selected[path] = {
                "checkpoint_path": str(path.resolve()),
                "checkpoint_training_step": int(training_step),
                "checkpoint_stage": label,
                "target_training_fraction": float(fraction),
            }
        else:
            selected[path]["checkpoint_stage"] += "+" + label
    return sorted(selected.values(), key=lambda row: row["checkpoint_training_step"])


def parameter_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def gradient_pair_stats(
    first: Sequence[torch.Tensor | None],
    second: Sequence[torch.Tensor | None],
    eps: float = 1e-30,
) -> dict[str, float]:
    if len(first) != len(second):
        raise ValueError("gradient sequences must have the same parameter scope")
    norm_first2 = 0.0
    norm_second2 = 0.0
    dot = 0.0
    for left, right in zip(first, second):
        if left is not None:
            norm_first2 += float(left.detach().double().square().sum())
        if right is not None:
            norm_second2 += float(right.detach().double().square().sum())
        if left is not None and right is not None:
            dot += float((left.detach().double() * right.detach().double()).sum())
    norm_first = norm_first2 ** 0.5
    norm_second = norm_second2 ** 0.5
    denominator = norm_first * norm_second
    return {
        "first_gradient_norm": norm_first,
        "second_gradient_norm": norm_second,
        "gradient_norm_ratio_second_over_first": norm_second / (norm_first + eps),
        "gradient_cosine": dot / denominator if denominator > eps else 0.0,
    }


def gradient_metrics_for_model(
    model,
    diffusion,
    x_start: torch.Tensor,
    diffusion_timestep: int,
    *,
    cell_ode_lambda: float,
    ode_reg_lambda: float,
    seed: int,
    device,
    metadata: Mapping,
) -> dict:
    """Compare two Cell gradient sources and two ODE gradient sources.

    This function uses ``torch.autograd.grad`` only.  It never calls
    ``backward()``, constructs an optimizer, or performs an optimizer step.
    """

    if x_start.ndim != 2 or x_start.shape[0] == 0:
        raise ValueError("gradient x_start must be a non-empty (cells, genes) tensor")
    if not 0 <= int(diffusion_timestep) < diffusion.num_timesteps:
        raise ValueError("diffusion_timestep out of range")
    before = parameter_fingerprint(model)
    was_training = model.training
    model.train()
    try:
        torch.manual_seed(int(seed) + int(diffusion_timestep) * 1_000_003)
        clean = x_start.to(device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + int(diffusion_timestep) * 1_000_003)
        noise = torch.randn(clean.shape, generator=generator, dtype=torch.float64).to(device)
        t = torch.full(
            (clean.shape[0],), int(diffusion_timestep), dtype=torch.long, device=device
        )
        x_t = diffusion.q_sample(clean, t, noise=noise)
        _target, target_name = diffusion_training_target(diffusion, clean, x_t, t, noise)
        # Use the shared implementation so the diffusion component remains
        # exact even if target/loss configuration changes in a checkpoint.
        diffusion_terms = diffusion.training_losses(
            model, clean, t, model_kwargs={}, noise=noise
        )
        diffusion_loss = diffusion_terms["loss"].mean()
        consistency_raw = model.consistency_penalty_20260830().mean()
        consistency_weighted = consistency_raw * float(cell_ode_lambda)
        prior_raw = model.ode_model.off_mask_penalty("l1")
        prior_weighted = prior_raw * float(ode_reg_lambda)

        cell_parameters = [
            parameter for parameter in model.ml_model.parameters()
            if parameter.requires_grad
        ]
        ode_parameters = [
            parameter for parameter in model.ode_model.parameters()
            if parameter.requires_grad
        ]
        cell_diff = torch.autograd.grad(
            diffusion_loss, cell_parameters, retain_graph=True, allow_unused=True
        )
        cell_cons = torch.autograd.grad(
            consistency_weighted, cell_parameters, retain_graph=True, allow_unused=True
        )
        ode_prior = torch.autograd.grad(
            prior_weighted, ode_parameters, retain_graph=True, allow_unused=True
        )
        ode_cons = torch.autograd.grad(
            consistency_weighted, ode_parameters, allow_unused=True
        )
        cell_stats = gradient_pair_stats(cell_diff, cell_cons)
        ode_stats = gradient_pair_stats(ode_prior, ode_cons)
    finally:
        model.train(was_training)
    after = parameter_fingerprint(model)
    if before != after:
        raise RuntimeError("gradient analysis mutated model state")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("autograd.grad unexpectedly populated parameter .grad")
    return {
        **dict(metadata),
        "diffusion_timestep": int(diffusion_timestep),
        "diffusion_target": target_name,
        "gradient_analyzed_cells": int(x_start.shape[0]),
        "diffusion_loss_for_gradient": float(diffusion_loss.detach()),
        "cell_ode_consistency_raw_for_gradient_20260830": float(consistency_raw.detach()),
        "cell_ode_consistency_weighted_for_gradient_20260830": float(
            consistency_weighted.detach()
        ),
        "ode_regularization_raw_for_gradient": float(prior_raw.detach()),
        "ode_regularization_weighted_for_gradient": float(prior_weighted.detach()),
        "cell_diffusion_gradient_norm": cell_stats["first_gradient_norm"],
        "cell_consistency_gradient_norm": cell_stats["second_gradient_norm"],
        "cell_gradient_norm_ratio": cell_stats[
            "gradient_norm_ratio_second_over_first"
        ],
        "cell_gradient_cosine": cell_stats["gradient_cosine"],
        "ode_prior_gradient_norm": ode_stats["first_gradient_norm"],
        "ode_consistency_gradient_norm": ode_stats["second_gradient_norm"],
        "ode_gradient_norm_ratio": ode_stats[
            "gradient_norm_ratio_second_over_first"
        ],
        "ode_gradient_cosine": ode_stats["gradient_cosine"],
        "optimizer_constructed": False,
        "optimizer_step_performed": False,
        "parameter_fingerprint_before": before,
        "parameter_fingerprint_after": after,
    }


def analyze_gradients(
    model,
    diffusion,
    x_start,
    diffusion_timesteps,
    *,
    cell_ode_lambda,
    ode_reg_lambda,
    seed,
    device,
    metadata,
) -> pd.DataFrame:
    rows = [
        gradient_metrics_for_model(
            model,
            diffusion,
            x_start,
            diffusion_timestep,
            cell_ode_lambda=cell_ode_lambda,
            ode_reg_lambda=ode_reg_lambda,
            seed=seed,
            device=device,
            metadata=metadata,
        )
        for diffusion_timestep in diffusion_timesteps
    ]
    return pd.DataFrame(rows)


__all__ = [
    "analyze_gradients",
    "checkpoint_training_step",
    "gradient_metrics_for_model",
    "gradient_pair_stats",
    "parameter_fingerprint",
    "select_analysis_checkpoints",
]
