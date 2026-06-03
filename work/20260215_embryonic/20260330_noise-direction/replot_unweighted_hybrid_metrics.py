#!/usr/bin/env python3
"""
Recompute unweighted ODE / ML branch metrics and generate 4 plots only.

Input:
  - Existing offline_hybrid_analysis output directory that contains:
      - timestep_metrics.csv
      - analysis_config.json
  - Original trained model / dataset paths are read from analysis_config.json

What this script changes:
  - It plots metrics for raw branch outputs:
      ODE(x_t)
      ML(x_t)
    instead of weighted outputs:
      r_t * ODE(x_t)
      (1-r_t) * ML(x_t)

Outputs:
  - unweighted_branch_analysis_<timestamp>/
      - timestep_metrics_unweighted.csv
      - ode_norm_ratio_unweighted.png
      - ml_norm_ratio_unweighted.png
      - ode_alignment_unweighted.png
      - ml_alignment_unweighted.png
      - run_config.json
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
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch


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
    out_dir = os.path.join(root, f"unweighted_branch_analysis_{ts}")
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


class OfflineHybridUnweightedAnalyzer:
    def __init__(self, cfg: dict, device_override: str = ""):
        self.cfg = dict(cfg)
        self.device = torch.device(
            device_override if device_override else (cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
        )

        scdiffusion_root = self.cfg["scdiffusion_root"]
        _ensure_repo_on_path(scdiffusion_root)

        from guided_diffusion.script_util import (
            args_to_dict,
            create_model_and_diffusion,
            model_and_diffusion_defaults,
        )
        from guided_diffusion.cell_model import Cell_Unet
        from ODE.ode_analysis1106 import GeneODE, ODE_ML_Hybrid

        self._args_to_dict = args_to_dict
        self._create_model_and_diffusion = create_model_and_diffusion
        self._model_and_diffusion_defaults = model_and_diffusion_defaults
        self._Cell_Unet = Cell_Unet
        self._GeneODE = GeneODE
        self._ODE_ML_Hybrid = ODE_ML_Hybrid

    def build_model_and_diffusion(self):
        cfg = self.cfg
        diffusion_kwargs = self._args_to_dict(
            argparse.Namespace(**cfg),
            self._model_and_diffusion_defaults().keys()
        )
        _, diffusion = self._create_model_and_diffusion(**diffusion_kwargs)

        adata = sc.read_h5ad(cfg["data_dir"])
        gene_list = list(adata.var["gene_name"].unique())

        ode = self._GeneODE(
            gene_list=gene_list,
            edge_tsv_path=cfg["edge_tsv_path"],
            soft=cfg["SoftReg"],
            device=self.device,
        )
        ml_model = self._Cell_Unet(input_dim=len(gene_list))
        hybrid_model = self._ODE_ML_Hybrid(
            ode_model=ode,
            ml_model=ml_model,
            timesteps=diffusion.num_timesteps,
        )
        hybrid_model.to(self.device)

        state = torch.load(cfg["model_path"], map_location=self.device)
        hybrid_model.load_state_dict(state, strict=True)
        hybrid_model.eval()

        return adata, diffusion, hybrid_model

    def load_eval_subset(self) -> List[Tuple[torch.Tensor, Dict[str, object]]]:
        cfg = self.cfg
        adata = sc.read_h5ad(cfg["data_dir"])

        max_cells = int(cfg["max_cells"])
        batch_size = int(cfg["batch_size"])

        n = min(max_cells, adata.n_obs)
        idx = np.random.choice(adata.n_obs, size=n, replace=False)

        subset: List[Tuple[torch.Tensor, Dict[str, object]]] = []
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            batch = to_tensor(adata.X[batch_idx]).float().cpu()
            subset.append((batch, {}))
        return subset

    @torch.no_grad()
    def analyze(self) -> pd.DataFrame:
        cfg = self.cfg
        _, diffusion, model = self.build_model_and_diffusion()
        subset = self.load_eval_subset()

        explicit_timesteps = cfg.get("explicit_timesteps", "")
        explicit_timesteps = explicit_timesteps if explicit_timesteps else None
        timesteps = build_timestep_list(
            diffusion.num_timesteps,
            int(cfg["num_t_points"]),
            explicit_timesteps,
        )

        rows: List[Dict[str, float]] = []
        for t in timesteps:
            moments = RunningMoments()
            for x0_cpu, cond_cpu in subset:
                x0 = x0_cpu.to(self.device).float()
                cond = move_cond_to_device(cond_cpu, self.device)
                batch_size = x0.shape[0]

                for _ in range(int(cfg["num_noise_draws"])):
                    t_raw = torch.full((batch_size,), int(t), device=self.device, dtype=torch.long)
                    noise = torch.randn_like(x0)
                    x_t = diffusion.q_sample(x0, t_raw, noise=noise)
                    model_t = diffusion._scale_timesteps(t_raw)

                    ode_raw = model.ode_model(x_t)
                    ml_raw = model.ml_model(x_t, model_t, **cond)
                    effective_r = model._scheduler(model_t, device=x_t.device, dtype=x_t.dtype)
                    while effective_r.dim() < x_t.dim():
                        effective_r = effective_r.unsqueeze(-1)

                    eps_norm = flatten_norm(noise)
                    eps_sq = flatten_dot(noise, noise).clamp_min(1e-12)

                    ode_raw_norm = flatten_norm(ode_raw)
                    ml_raw_norm = flatten_norm(ml_raw)

                    values = {
                        "effective_r": effective_r.reshape(batch_size, -1)[:, 0],
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

        return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def load_previous_config(previous_out_dir: str) -> dict:
    cfg_path = os.path.join(previous_out_dir, "analysis_config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"analysis_config.json not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    csv_path = os.path.join(previous_out_dir, "timestep_metrics.csv")
    if os.path.exists(csv_path):
        old_df = pd.read_csv(csv_path)
        if "t" in old_df.columns and len(old_df) > 0:
            cfg["num_t_points"] = int(len(old_df))
            cfg["explicit_timesteps"] = ",".join(str(int(t)) for t in old_df["t"].tolist())

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute unweighted ODE/ML branch metrics and plot 4 figures")
    parser.add_argument(
        "--previous_out_dir",
        type=str,
        required=True,
        help="Existing offline_hybrid_analysis output directory containing timestep_metrics.csv and analysis_config.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260330_noise-direction/offline_hybrid_analysis",
        help="Root directory where a new output folder will be created",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Optional device override, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override",
    )
    parser.add_argument(
        "--max_cells",
        type=int,
        default=None,
        help="Optional max_cells override",
    )
    parser.add_argument(
        "--num_noise_draws",
        type=int,
        default=None,
        help="Optional num_noise_draws override",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional batch_size override",
    )
    args = parser.parse_args()

    cfg = load_previous_config(args.previous_out_dir)

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.max_cells is not None:
        cfg["max_cells"] = args.max_cells
    if args.num_noise_draws is not None:
        cfg["num_noise_draws"] = args.num_noise_draws
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device:
        cfg["device"] = args.device

    set_seed(int(cfg["seed"]))
    out_dir = prepare_output_dir(args.output_root)

    analyzer = OfflineHybridUnweightedAnalyzer(cfg, device_override=args.device)
    df = analyzer.analyze()

    csv_path = os.path.join(out_dir, "timestep_metrics_unweighted.csv")
    df.to_csv(csv_path, index=False)

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

    run_cfg_path = os.path.join(out_dir, "run_config.json")
    with open(run_cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "previous_out_dir": args.previous_out_dir,
                "resolved_config": cfg,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("Unweighted branch analysis completed.")
    print(f"Input directory  : {args.previous_out_dir}")
    print(f"Output directory : {out_dir}")
    print(f"Metrics CSV      : {csv_path}")
    print("Saved figures    :")
    print(f"  - {os.path.join(out_dir, 'ode_norm_ratio_unweighted.png')}")
    print(f"  - {os.path.join(out_dir, 'ml_norm_ratio_unweighted.png')}")
    print(f"  - {os.path.join(out_dir, 'ode_alignment_unweighted.png')}")
    print(f"  - {os.path.join(out_dir, 'ml_alignment_unweighted.png')}")
    print("=" * 72)


if __name__ == "__main__":
    main()
