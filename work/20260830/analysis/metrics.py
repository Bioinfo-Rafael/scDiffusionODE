"""Numerically safe Cell-target and Cell-ODE metrics by diffusion timestep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from guided_diffusion import gaussian_diffusion as gd


DEFAULT_EPS = 1e-12


@dataclass
class RunningPearson:
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_x2: float = 0.0
    sum_y2: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x64 = x.detach().double()
        y64 = y.detach().double()
        self.n += x64.numel()
        self.sum_x += float(x64.sum())
        self.sum_y += float(y64.sum())
        self.sum_x2 += float(x64.square().sum())
        self.sum_y2 += float(y64.square().sum())
        self.sum_xy += float((x64 * y64).sum())

    def value(self, eps: float = DEFAULT_EPS) -> float:
        if self.n == 0:
            return 0.0
        numerator = self.sum_xy - self.sum_x * self.sum_y / self.n
        var_x = max(self.sum_x2 - self.sum_x * self.sum_x / self.n, 0.0)
        var_y = max(self.sum_y2 - self.sum_y * self.sum_y / self.n, 0.0)
        denominator = (var_x * var_y) ** 0.5
        return float(numerator / denominator) if denominator > eps else 0.0


def safe_row_pearson(x: torch.Tensor, y: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("Pearson inputs must have the same (cells, genes) shape")
    x64, y64 = x.double(), y.double()
    x_centered = x64 - x64.mean(dim=1, keepdim=True)
    y_centered = y64 - y64.mean(dim=1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=1)
    denominator = x_centered.square().sum(dim=1).sqrt() * y_centered.square().sum(dim=1).sqrt()
    return torch.where(denominator > eps, numerator / denominator, torch.zeros_like(numerator))


def safe_row_cosine(x: torch.Tensor, y: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("cosine inputs must have the same (cells, genes) shape")
    x64, y64 = x.double(), y.double()
    numerator = (x64 * y64).sum(dim=1)
    denominator = x64.square().sum(dim=1).sqrt() * y64.square().sum(dim=1).sqrt()
    return numerator / (denominator + eps)


def safe_row_nmse(x: torch.Tensor, y: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("NMSE inputs must have the same (cells, genes) shape")
    x64, y64 = x.double(), y.double()
    numerator = 2.0 * (x64 - y64).square().sum(dim=1)
    denominator = x64.square().sum(dim=1) + y64.square().sum(dim=1)
    return numerator / (denominator + eps)


def distribution_stats(prefix: str, values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        array = np.asarray([0.0], dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_q25": float(np.quantile(array, 0.25)),
        f"{prefix}_q75": float(np.quantile(array, 0.75)),
    }


def diffusion_training_target(diffusion, x_start, x_t, t, noise):
    """Use the exact target selection in GaussianDiffusion.training_losses."""

    if diffusion.model_mean_type == gd.ModelMeanType.EPSILON:
        return noise, "epsilon"
    if diffusion.model_mean_type == gd.ModelMeanType.START_X:
        return x_start, "x_start"
    if diffusion.model_mean_type == gd.ModelMeanType.PREVIOUS_X:
        return diffusion.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0], "previous_x"
    raise NotImplementedError(f"unsupported model_mean_type: {diffusion.model_mean_type}")


def _metadata(metadata: Mapping, diffusion_timestep: int, cells: int, target_name: str) -> dict:
    return {
        **dict(metadata),
        "diffusion_timestep": int(diffusion_timestep),
        "analyzed_cells": int(cells),
        "diffusion_target": target_name,
    }


def evaluate_diffusion_timesteps(
    model,
    diffusion,
    x_start: torch.Tensor,
    diffusion_timesteps: Iterable[int],
    *,
    batch_size: int,
    seed: int,
    device,
    metadata: Mapping,
    eps: float = DEFAULT_EPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate normal metrics under no_grad without calling wrapper.forward."""

    if x_start.ndim != 2 or x_start.shape[0] == 0:
        raise ValueError("x_start must be a non-empty (cells, genes) tensor")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    values = sorted({int(value) for value in diffusion_timesteps})
    if not values or values[0] < 0 or values[-1] >= diffusion.num_timesteps:
        raise ValueError("diffusion timesteps are empty or out of range")

    was_training = model.training
    model.eval()
    diffusion_rows, cell_ode_rows = [], []
    try:
        with torch.no_grad():
            for diffusion_timestep in values:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(seed) + diffusion_timestep * 1_000_003)
                target_corr = RunningPearson()
                ode_corr = RunningPearson()
                target_pearson, target_mse, target_cosine = [], [], []
                ode_pearson, ode_cosine, ode_mse, ode_nmse = [], [], [], []
                cell_norms, ode_norms, norm_ratios = [], [], []
                target_name = ""
                for start in range(0, x_start.shape[0], batch_size):
                    stop = min(start + batch_size, x_start.shape[0])
                    clean = x_start[start:stop].to(device)
                    noise = torch.randn(
                        clean.shape, generator=generator, dtype=torch.float64
                    ).to(device)
                    t = torch.full(
                        (clean.shape[0],), diffusion_timestep,
                        dtype=torch.long, device=device,
                    )
                    x_t = diffusion.q_sample(clean, t, noise=noise)
                    scaled_t = diffusion._scale_timesteps(t).unsqueeze(1)
                    cell = model.ml_model(x_t, scaled_t)
                    ode = model.ode_model(x_t, scaled_t)
                    target, target_name = diffusion_training_target(
                        diffusion, clean, x_t, t, noise
                    )
                    if cell.shape != ode.shape or cell.shape != target.shape:
                        raise RuntimeError(
                            f"metric shape mismatch: cell={tuple(cell.shape)}, "
                            f"ode={tuple(ode.shape)}, target={tuple(target.shape)}"
                        )
                    target_corr.update(cell, target)
                    ode_corr.update(cell, ode)
                    target_pearson.extend(safe_row_pearson(cell, target, eps).cpu().numpy())
                    target_mse.extend((cell.double() - target.double()).square().mean(1).cpu().numpy())
                    target_cosine.extend(safe_row_cosine(cell, target, eps).cpu().numpy())
                    ode_pearson.extend(safe_row_pearson(cell, ode, eps).cpu().numpy())
                    ode_cosine.extend(safe_row_cosine(cell, ode, eps).cpu().numpy())
                    ode_mse.extend((cell.double() - ode.double()).square().mean(1).cpu().numpy())
                    ode_nmse.extend(safe_row_nmse(cell, ode, eps).cpu().numpy())
                    cell_norm = cell.double().norm(dim=1)
                    ode_norm = ode.double().norm(dim=1)
                    cell_norms.extend(cell_norm.cpu().numpy())
                    ode_norms.extend(ode_norm.cpu().numpy())
                    norm_ratios.extend((ode_norm / (cell_norm + eps)).cpu().numpy())

                base = _metadata(metadata, diffusion_timestep, x_start.shape[0], target_name)
                diffusion_rows.append({
                    **base,
                    "cell_target_pearson_global": target_corr.value(eps),
                    "cell_target_mse_global": float(np.mean(target_mse)),
                    **distribution_stats("cell_target_pearson", target_pearson),
                    **distribution_stats("cell_target_mse", target_mse),
                    **distribution_stats("cell_target_cosine", target_cosine),
                })
                cell_ode_rows.append({
                    **base,
                    "cell_ode_pearson_global": ode_corr.value(eps),
                    "cell_ode_mse_global": float(np.mean(ode_mse)),
                    **distribution_stats("cell_ode_pearson", ode_pearson),
                    **distribution_stats("cell_ode_cosine", ode_cosine),
                    **distribution_stats("cell_ode_mse", ode_mse),
                    **distribution_stats("cell_ode_nmse", ode_nmse),
                    **distribution_stats("cell_norm", cell_norms),
                    **distribution_stats("ode_norm", ode_norms),
                    **distribution_stats("ode_cell_norm_ratio", norm_ratios),
                })
    finally:
        model.train(was_training)
    return pd.DataFrame(diffusion_rows), pd.DataFrame(cell_ode_rows)


__all__ = [
    "RunningPearson",
    "diffusion_training_target",
    "distribution_stats",
    "evaluate_diffusion_timesteps",
    "safe_row_cosine",
    "safe_row_nmse",
    "safe_row_pearson",
]
