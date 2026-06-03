#!/usr/bin/env python3
"""
Offline contribution analysis for a trained scDiffusion hybrid model.

This script reconstructs the same diffusion / GeneODE / Cell_Unet / ODE_ML_Hybrid
configuration used in training, then performs offline forward noising:

    x_t = q_sample(x_0, t, noise=eps)

For selected timesteps t, it computes the weighted branch outputs

    ode_contrib = r_t * ode_model(x_t)
    ml_contrib  = (1-r_t) * ml_model(x_t, scaled_t, y)

and compares them against the true Gaussian noise eps that was injected.

Saved outputs:
  - timestep_metrics.csv
  - overview_metrics.png
  - ode_norm_ratio.png
  - ml_norm_ratio.png
  - ode_alignment.png
  - ml_alignment.png
  - analysis_config.json

Notes
-----
- The model call follows guided-diffusion training semantics:
      model(x_t, diffusion._scale_timesteps(t), **model_kwargs)
  so the ML branch sees the same timestep representation as during training.
- The hybrid scheduler is evaluated with the *same timestep tensor actually passed
  into the hybrid model* (effective r_t), plus the raw integer timestep is saved
  separately for plotting / interpretation.
- By default this script averages over different cells and one fresh epsilon draw
  per cell. Increase --num_noise_draws only if you want lower Monte Carlo noise.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

import torch.nn.functional as F

def _ensure_repo_on_path(repo_root: str) -> None:
    repo_root = os.path.abspath(repo_root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


@dataclass
class RunningMoments:
    sums: MutableMapping[str, float] = field(default_factory=dict)
    sums_sq: MutableMapping[str, float] = field(default_factory=dict)
    count: int = 0

    def update(self, values: Mapping[str, torch.Tensor]) -> None:
        first = True
        for key, tensor in values.items():
            arr = tensor.detach().float().reshape(-1).cpu()
            if arr.numel() == 0:
                continue
            s = float(arr.sum().item())
            ss = float((arr * arr).sum().item())
            self.sums[key] = self.sums.get(key, 0.0) + s
            self.sums_sq[key] = self.sums_sq.get(key, 0.0) + ss
            if first:
                self.count += int(arr.numel())
                first = False

    def finalize(self, prefix: str = "") -> Dict[str, float]:
        out: Dict[str, float] = {}
        n = max(self.count, 1)
        for key, s in self.sums.items():
            mean = s / n
            ss = self.sums_sq[key]
            var = max(ss / n - mean * mean, 0.0)
            std = math.sqrt(var)
            out[f"{prefix}{key}_mean"] = mean
            out[f"{prefix}{key}_std"] = std
        out[f"{prefix}n"] = self.count
        return out


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_tensor(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x
    if hasattr(x, "toarray"):
        x = x.toarray()
    return torch.as_tensor(np.asarray(x))


def flatten_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).clamp_min(eps)


def flatten_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x.reshape(x.shape[0], -1) * y.reshape(y.shape[0], -1)).sum(dim=1)


def detach_cond(cond: Mapping[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in cond.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu()
        elif isinstance(value, np.ndarray):
            out[key] = torch.from_numpy(value).cpu()
        else:
            out[key] = value
    return out


def move_cond_to_device(cond: Mapping[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in cond.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def collect_eval_subset(data_iter: Iterable[Tuple[torch.Tensor, Mapping[str, object]]], max_cells: int) -> List[Tuple[torch.Tensor, Dict[str, object]]]:
    subset: List[Tuple[torch.Tensor, Dict[str, object]]] = []
    collected = 0
    while collected < max_cells:
        batch, cond = next(data_iter)
        batch = to_tensor(batch).float().cpu()
        cond = detach_cond(cond)
        remaining = max_cells - collected
        if batch.shape[0] > remaining:
            batch = batch[:remaining]
            trimmed_cond: Dict[str, object] = {}
            for key, value in cond.items():
                if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= remaining:
                    trimmed_cond[key] = value[:remaining]
                else:
                    trimmed_cond[key] = value
            cond = trimmed_cond
        subset.append((batch, cond))
        collected += batch.shape[0]
    return subset


def build_timestep_list(num_timesteps: int, num_t_points: int, explicit: Optional[str]) -> List[int]:
    if explicit:
        values = sorted({int(v.strip()) for v in explicit.split(",") if v.strip()})
        for t in values:
            if t < 0 or t >= num_timesteps:
                raise ValueError(f"t={t} is outside [0, {num_timesteps - 1}].")
        return values
    if num_t_points <= 0 or num_t_points >= num_timesteps:
        return list(range(num_timesteps))
    values = np.linspace(0, num_timesteps - 1, num_t_points)
    values = np.unique(np.round(values).astype(int))
    return values.tolist()


def prepare_output_dir(root: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, f"offline_hybrid_analysis_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_metric(
    df: pd.DataFrame,
    x_col: str,
    mean_col: str,
    std_col: Optional[str],
    y_label: str,
    title: str,
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    x = df[x_col].to_numpy()
    y = df[mean_col].to_numpy()
    ax.plot(x, y)
    if std_col is not None and std_col in df.columns:
        s = df[std_col].to_numpy()
        ax.fill_between(x, y - s, y + s, alpha=0.2)
    ax.set_xlabel("forward diffusion timestep t")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_overview(df: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax = axes[0, 0]
    ax.plot(df["t"], df["ode_norm_ratio_mean"], label="ODE")
    ax.plot(df["t"], df["ml_norm_ratio_mean"], label="ML")
    ax.set_title("Weighted output norm / true noise norm")
    ax.set_ylabel("norm ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(df["t"], df["ode_align_mean"], label="ODE")
    ax.plot(df["t"], df["ml_align_mean"], label="ML")
    ax.set_title("Alignment with true noise")
    ax.set_ylabel(r"<contrib, eps> / ||eps||^2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(df["t"], df["effective_r_mean"], label="effective r_t")
    ax.set_title("Effective ODE mixing weight")
    ax.set_xlabel("forward diffusion timestep t")
    ax.set_ylabel("r_t")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["t"], df["hybrid_mse_norm_mean"], label="hybrid")
    ax.plot(df["t"], df["ode_mse_norm_mean"], label="ODE only")
    ax.plot(df["t"], df["ml_mse_norm_mean"], label="ML only")
    ax.set_title("Normalized MSE to true noise")
    ax.set_xlabel("forward diffusion timestep t")
    ax.set_ylabel("MSE / ||eps||^2")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


class OfflineHybridAnalyzer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(
            args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        _ensure_repo_on_path(args.scdiffusion_root)

        from guided_diffusion.cell_datasets_loader import load_data
        from guided_diffusion.script_util import (
            add_dict_to_argparser,
            args_to_dict,
            create_model_and_diffusion,
            model_and_diffusion_defaults,
        )
        from guided_diffusion.cell_model import Cell_Unet
        from ODE.ode_analysis20260413 import GeneODE, ODE_ML_HybridLearnedScale

        self._load_data_fn = load_data
        self._args_to_dict = args_to_dict
        self._create_model_and_diffusion = create_model_and_diffusion
        self._model_and_diffusion_defaults = model_and_diffusion_defaults
        self._Cell_Unet = Cell_Unet
        self._GeneODE = GeneODE
        self._ODE_ML_Hybrid = ODE_ML_HybridLearnedScale

    def build_model_and_diffusion(self):
        args = self.args
        _, diffusion = self._create_model_and_diffusion(
            **self._args_to_dict(args, self._model_and_diffusion_defaults().keys())
        )

        adata = sc.read_h5ad(args.data_dir)
        gene_list = list(adata.var["gene_name"].unique())

        ode = self._GeneODE(
            gene_list=gene_list,
            edge_tsv_path=args.edge_tsv_path,
            soft=args.SoftReg,
            device=self.device,
        )
        ml_model = self._Cell_Unet(input_dim=len(gene_list))
        # build_model_and_diffusion() を修正
        hybrid_model = self._ODE_ML_Hybrid(
            ode_model=ode,
            ml_model=ml_model,
            timesteps=diffusion.num_timesteps,
            scale_hidden_dim=args.scale_hidden_dim,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            init_scale=args.init_scale,
        )
        hybrid_model.to(self.device)

        state = torch.load(args.model_path, map_location=self.device)
        hybrid_model.load_state_dict(state, strict=True)
        hybrid_model.eval()

        return adata, diffusion, hybrid_model

    def load_eval_subset(self) -> List[Tuple[torch.Tensor, Dict[str, object]]]:
        """
        データをランダムサンプリングして計算量を削減する
        """
        args = self.args
        adata = sc.read_h5ad(args.data_dir)

        n = min(args.max_cells, adata.n_obs)
        idx = np.random.choice(adata.n_obs, size=n, replace=False)

        subset: List[Tuple[torch.Tensor, Dict[str, object]]] = []
        for start in range(0, n, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            batch = to_tensor(adata.X[batch_idx]).float().cpu()
            subset.append((batch, {}))

        return subset

    @torch.no_grad()
    def analyze(self) -> pd.DataFrame:
        args = self.args
        _, diffusion, model = self.build_model_and_diffusion()
        subset = self.load_eval_subset()
        timesteps = build_timestep_list(diffusion.num_timesteps, args.num_t_points, args.explicit_timesteps)

        rows: List[Dict[str, float]] = []
        for t in timesteps:
            moments = RunningMoments()
            for x0_cpu, cond_cpu in subset:
                x0 = x0_cpu.to(self.device).float()
                cond = move_cond_to_device(cond_cpu, self.device)
                batch_size = x0.shape[0]

                for _ in range(args.num_noise_draws):
                    t_raw = torch.full((batch_size,), int(t), device=self.device, dtype=torch.long)
                    noise = torch.randn_like(x0)
                    x_t = diffusion.q_sample(x0, t_raw, noise=noise)
                    model_t = diffusion._scale_timesteps(t_raw)

                    ode_raw = model.ode_model(x_t)
                    ml_raw = model.ml_model(x_t, model_t, **cond)

                    # raw 指標用
                    ode_raw_norm = flatten_norm(ode_raw)
                    ml_raw_norm = flatten_norm(ml_raw)

                    ode_unit = F.normalize(ode_raw, p=2, dim=-1, eps=1e-8)
                    ml_unit  = F.normalize(ml_raw,  p=2, dim=-1, eps=1e-8)

                    effective_r = model._scheduler(model_t, device=x_t.device, dtype=x_t.dtype)
                    while effective_r.dim() < x_t.dim():
                        effective_r = effective_r.unsqueeze(-1)

                    scale = model.scale_net(
                        x=x_t,
                        t=model_t,
                        T=model.T,
                        ode_out=ode_raw,
                        ml_out=ml_raw,
                    )  # (batch, 1)

                    while scale.dim() < x_t.dim():
                        scale = scale.unsqueeze(-1)

                    ode_contrib = scale * effective_r * ode_unit
                    ml_contrib  = scale * (1.0 - effective_r) * ml_unit
                    hybrid_pred = ode_contrib + ml_contrib

                    eps_norm = flatten_norm(noise)
                    eps_sq = flatten_dot(noise, noise).clamp_min(1e-12)
                    ode_norm = flatten_norm(ode_contrib)
                    ml_norm = flatten_norm(ml_contrib)

                    scale_flat = scale.reshape(batch_size, -1)[:, 0]
                    r_flat = effective_r.reshape(batch_size, -1)[:, 0]

                    values = {
                        "effective_r": r_flat,

                        "scale": scale_flat,
                        "scale_times_r": scale_flat * r_flat,
                        "scale_times_1mr": scale_flat * (1.0 - r_flat),

                        "ode_norm_ratio": ode_norm / eps_norm,
                        "ml_norm_ratio": ml_norm / eps_norm,
                        "ode_align": flatten_dot(ode_contrib, noise) / eps_sq,
                        "ml_align": flatten_dot(ml_contrib, noise) / eps_sq,
                        "hybrid_mse_norm": flatten_dot(hybrid_pred - noise, hybrid_pred - noise) / eps_sq,
                        "ode_mse_norm": flatten_dot(ode_contrib - noise, ode_contrib - noise) / eps_sq,
                        "ml_mse_norm": flatten_dot(ml_contrib - noise, ml_contrib - noise) / eps_sq,

                        "ode_raw_norm_ratio": ode_raw_norm / eps_norm,
                        "ml_raw_norm_ratio": ml_raw_norm / eps_norm,
                        "ode_raw_align": flatten_dot(ode_raw, noise) / eps_sq,
                        "ml_raw_align": flatten_dot(ml_raw, noise) / eps_sq,
                    }
                    moments.update(values)

            row = {
                "t": int(t),
                "reverse_step": int(diffusion.num_timesteps - 1 - t),
            }
            row.update(moments.finalize())
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
        return df


def build_parser() -> argparse.ArgumentParser:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--scdiffusion_root",
        type=str,
        default="/home/suzuki/Projects/scDiffusion",
    )
    known, _ = bootstrap.parse_known_args()
    _ensure_repo_on_path(known.scdiffusion_root)

    from guided_diffusion.script_util import add_dict_to_argparser, model_and_diffusion_defaults

    defaults = dict(
        data_dir="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad",
        batch_size=128,
        SoftReg=True,
        edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv" , #"/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
        scdiffusion_root="/home/suzuki/Projects/scDiffusion",
        model_path="/home/suzuki/Projects/scDiffusion/work/20260420_Normalized_LearnScale/20260421_061005_linear/20260421_061007_train/checkpoints/pbmc68k_soft_20251127_hvg1024/ema_0.9999_030000.pt",
        output_root="./offline_hybrid_analysis",
        max_cells=5000,
        num_t_points=64,
        explicit_timesteps="",
        num_noise_draws=20,
        seed=1234,
        device="",
        scale_hidden_dim=32,
        scale_min=0.5,
        scale_max=8.0,
        init_scale=3.0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser(
        description="Offline contribution analysis for hybrid model"
    )
    add_dict_to_argparser(parser, defaults)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.model_path:
        raise ValueError("--model_path is required.")

    set_seed(args.seed)
    out_dir = prepare_output_dir(args.output_root)

    analyzer = OfflineHybridAnalyzer(args)
    df = analyzer.analyze()

    csv_path = os.path.join(out_dir, "timestep_metrics.csv")
    df.to_csv(csv_path, index=False)

    plot_metric(
        df,
        x_col="t",
        mean_col="ode_norm_ratio_mean",
        std_col="ode_norm_ratio_std",
        y_label=r"|| r_t * ODE(x_t) || / || eps ||",
        title="ODE weighted output norm ratio",
        save_path=os.path.join(out_dir, "ode_norm_ratio.png"),
    )
    plot_metric(
        df,
        x_col="t",
        mean_col="ml_norm_ratio_mean",
        std_col="ml_norm_ratio_std",
        y_label=r"|| (1-r_t) * ML(x_t) || / || eps ||",
        title="ML weighted output norm ratio",
        save_path=os.path.join(out_dir, "ml_norm_ratio.png"),
    )
    plot_metric(
        df,
        x_col="t",
        mean_col="ode_align_mean",
        std_col="ode_align_std",
        y_label=r"< r_t * ODE(x_t), eps > / ||eps||^2",
        title="ODE alignment with true noise",
        save_path=os.path.join(out_dir, "ode_alignment.png"),
    )
    plot_metric(
        df,
        x_col="t",
        mean_col="ml_align_mean",
        std_col="ml_align_std",
        y_label=r"< (1-r_t) * ML(x_t), eps > / ||eps||^2",
        title="ML alignment with true noise",
        save_path=os.path.join(out_dir, "ml_alignment.png"),
    )
    plot_overview(df, os.path.join(out_dir, "overview_metrics.png"))


    plot_metric(
        df,
        x_col="t",
        mean_col="ode_raw_norm_ratio_mean",
        std_col="ode_raw_norm_ratio_std",
        y_label=r"|| ODE(x_t) || / || eps ||",
        title="ODE raw output norm ratio",
        save_path=os.path.join(out_dir, "ode_norm_ratio_unweighted.png"),
    )

    plot_metric(
        df,
        x_col="t",
        mean_col="ml_raw_norm_ratio_mean",
        std_col="ml_raw_norm_ratio_std",
        y_label=r"|| ML(x_t) || / || eps ||",
        title="ML raw output norm ratio",
        save_path=os.path.join(out_dir, "ml_norm_ratio_unweighted.png"),
    )

    plot_metric(
        df,
        x_col="t",
        mean_col="ode_raw_align_mean",
        std_col="ode_raw_align_std",
        y_label=r"< ODE(x_t), eps > / ||eps||^2",
        title="ODE raw alignment with true noise",
        save_path=os.path.join(out_dir, "ode_alignment_unweighted.png"),
    )

    plot_metric(
        df,
        x_col="t",
        mean_col="ml_raw_align_mean",
        std_col="ml_raw_align_std",
        y_label=r"< ML(x_t), eps > / ||eps||^2",
        title="ML raw alignment with true noise",
        save_path=os.path.join(out_dir, "ml_alignment_unweighted.png"),
    )

    plot_metric(
        df,
        x_col="t",
        mean_col="scale_mean",
        std_col="scale_std",
        y_label="scale(x_t, t)",
        title="SmallScaleNet output",
        save_path=os.path.join(out_dir, "scale_value.png"),
    )

    plot_metric(
        df,
        x_col="t",
        mean_col="scale_times_r_mean",
        std_col="scale_times_r_std",
        y_label="scale(x_t, t) * r_t",
        title="SmallScaleNet × ODE mixing weight",
        save_path=os.path.join(out_dir, "scale_times_r.png"),
    )

    plot_metric(
        df,
        x_col="t",
        mean_col="scale_times_1mr_mean",
        std_col="scale_times_1mr_std",
        y_label="scale(x_t, t) * (1-r_t)",
        title="SmallScaleNet × ML mixing weight",
        save_path=os.path.join(out_dir, "scale_times_1mr.png"),
    )

    config_path = os.path.join(out_dir, "analysis_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Offline hybrid contribution analysis completed.")
    print(f"Output directory : {out_dir}")
    print(f"Metrics CSV      : {csv_path}")
    print("Saved figures    :")
    print(f"  - {os.path.join(out_dir, 'ode_norm_ratio.png')}")
    print(f"  - {os.path.join(out_dir, 'ml_norm_ratio.png')}")
    print(f"  - {os.path.join(out_dir, 'ode_alignment.png')}")
    print(f"  - {os.path.join(out_dir, 'ml_alignment.png')}")
    print(f"  - {os.path.join(out_dir, 'overview_metrics.png')}")
    print("=" * 72)


if __name__ == "__main__":
    main()

# python analyze_hybrid_offline.py \
#   --scdiffusion_root /home/suzuki/Projects/scDiffusion \
#   --model_path /path/to/ema_0.9999_002000.pt \
#   --data_dir data_preparation/pbmc68k.h5ad \
#   --edge_tsv_path /home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv \
#   --output_root ./analysis_out \
#   --max_cells 512 \
#   --num_t_points 64 \
#   --num_noise_draws 1 \
#   --batch_size 128