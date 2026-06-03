#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.stats import pearsonr, spearmanr

EPS = 1e-12
DEFAULT_SCDIFFUSION_PATH = "/home/suzuki/Projects/scDiffusion"
DEFAULT_ADATA_PATH = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"
DEFAULT_MODEL_PATH1 = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260224_084326_Lamda5/20260224_084328_train/checkpoints/pbmc68k_soft_20251127_hvg1024/ema_0.9999_100000.pt"
DEFAULT_MODEL_PATH2 = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260224_084326_Lamda5/20260224_084328_train/checkpoints/pbmc68k_soft_20251127_hvg1024/ema_0.9999_000000.pt"
DEFAULT_EDGE_TSV_PATH = "/home/suzuki/Projects/scDiffusion/external_data/Mouse_tf_target_edges.tsv"
DEFAULT_TARGET_SUPERCLASS = "Erythropoietic"
DEFAULT_TIMESTEPS = 1000
DEFAULT_BATCH_SIZE = 1024


@dataclass
class RunAssets:
    gene_names: list[str]
    X: np.ndarray | None
    obs_names: np.ndarray | None
    V1: np.ndarray | None
    V2: np.ndarray | None
    adata_path: str | None
    model_path1: str
    model_path2: str
    edge_tsv_path: str
    target_superclass: str | None


# -----------------------------------------------------------------------------
# basic utilities
# -----------------------------------------------------------------------------
def setup_environment(scdiffusion_path: str) -> None:
    if scdiffusion_path not in sys.path:
        sys.path.insert(0, scdiffusion_path)


def make_output_dir(prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path.cwd() / f"{timestamp}_{prefix}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {outdir}")
    return outdir


def ensure_dense_float32(X: Any) -> np.ndarray:
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0
    return X


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan
    if np.std(x) < EPS or np.std(y) < EPS:
        return np.nan
    return float(pearsonr(x, y)[0])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 3:
        return np.nan
    if np.std(x) < EPS or np.std(y) < EPS:
        return np.nan
    return float(spearmanr(x, y)[0])


def safe_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return np.nan, np.nan
    if np.std(x) < EPS:
        return np.nan, np.nan
    a, b = np.polyfit(x, y, deg=1)
    return float(a), float(b)


def finite_var(arr: np.ndarray) -> float:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return np.nan
    return float(np.var(vals))


# -----------------------------------------------------------------------------
# checkpoint / model helpers
# -----------------------------------------------------------------------------
def checkpoint_to_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        return checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    return checkpoint


def load_model(model_path: str, gene_list: list[str], edge_tsv_path: str, device: torch.device, timesteps: int):
    from guided_diffusion.cell_model import Cell_Unet
    from ODE.ode_analysis1106 import GeneODE, ODE_ML_Hybrid

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint_to_state_dict(checkpoint)

    ode = GeneODE(
        gene_list=gene_list,
        edge_tsv_path=edge_tsv_path,
        soft=True,
        device=device,
    )
    ml_model = Cell_Unet(input_dim=len(gene_list))
    hybrid_model = ODE_ML_Hybrid(
        ode_model=ode,
        ml_model=ml_model,
        timesteps=timesteps,
    ).to(device)
    hybrid_model.load_state_dict(state_dict, strict=False)
    hybrid_model.eval()
    return hybrid_model


def compute_velocity_matrix(hybrid_model, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    n = X.shape[0]
    V = np.zeros_like(X, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.from_numpy(X[start:end]).to(device)
            vb = hybrid_model.ode_model(xb).detach().cpu().numpy()
            V[start:end] = np.asarray(vb, dtype=np.float32)
    V[~np.isfinite(V)] = 0.0
    return V


def compute_alpha_and_velocity_batch(ode_model, xb: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xb = xb.float()
    W_eff = ode_model.W if ode_model.soft else (ode_model.W * ode_model.mask)
    Wx = xb @ W_eff + ode_model.b
    alpha = torch.sigmoid(Wx * ode_model.scale)
    velocity = alpha - ode_model.gamma * xb
    return (
        alpha.detach().cpu().numpy().astype(np.float64, copy=False),
        velocity.detach().cpu().numpy().astype(np.float64, copy=False),
        xb.detach().cpu().numpy().astype(np.float64, copy=False),
    )


def accumulate_ode_diagnostics(hybrid_model, X: np.ndarray, device: torch.device, batch_size: int) -> pd.DataFrame:
    n_cells, n_genes = X.shape
    ode = hybrid_model.ode_model
    gamma = ode.gamma.detach().cpu().numpy().astype(np.float64)

    sum_alpha = np.zeros(n_genes, dtype=np.float64)
    sum_alpha_sq = np.zeros(n_genes, dtype=np.float64)
    sum_abs_alpha = np.zeros(n_genes, dtype=np.float64)
    sum_abs_gamma_x = np.zeros(n_genes, dtype=np.float64)
    frac_alpha_low = np.zeros(n_genes, dtype=np.float64)
    frac_alpha_high = np.zeros(n_genes, dtype=np.float64)

    with torch.no_grad():
        for start in range(0, n_cells, batch_size):
            end = min(start + batch_size, n_cells)
            xb = torch.from_numpy(X[start:end]).to(device)
            alpha, _, x_np = compute_alpha_and_velocity_batch(ode, xb)
            sum_alpha += alpha.sum(axis=0)
            sum_alpha_sq += (alpha ** 2).sum(axis=0)
            sum_abs_alpha += np.abs(alpha).sum(axis=0)
            sum_abs_gamma_x += np.abs(gamma[None, :] * x_np).sum(axis=0)
            frac_alpha_low += (alpha < 0.05).sum(axis=0)
            frac_alpha_high += (alpha > 0.95).sum(axis=0)

    mean_alpha = sum_alpha / max(n_cells, 1)
    var_alpha = np.maximum(sum_alpha_sq / max(n_cells, 1) - mean_alpha ** 2, 0.0)
    std_alpha = np.sqrt(var_alpha)

    df = pd.DataFrame(
        {
            "gamma": gamma,
            "mean_alpha": mean_alpha,
            "std_alpha": std_alpha,
            "mean_abs_alpha": sum_abs_alpha / max(n_cells, 1),
            "mean_abs_gamma_x": sum_abs_gamma_x / max(n_cells, 1),
            "gamma_term_over_alpha": (sum_abs_gamma_x / max(n_cells, 1)) / ((sum_abs_alpha / max(n_cells, 1)) + EPS),
            "frac_alpha_lt_005": frac_alpha_low / max(n_cells, 1),
            "frac_alpha_gt_095": frac_alpha_high / max(n_cells, 1),
        }
    )
    return df


# -----------------------------------------------------------------------------
# data loading / reuse previous run
# -----------------------------------------------------------------------------
def fill_from_previous_run(previous_run_dir: Path, args: argparse.Namespace) -> argparse.Namespace:
    cfg_path = previous_run_dir / "run_config.json"
    if not cfg_path.exists():
        return args

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    for key, cfg_key in [
        ("adata_path", "adata_path"),
        ("model_path1", "model_path1"),
        ("model_path2", "model_path2"),
        ("edge_tsv_path", "edge_tsv_path"),
        ("target_superclass", "target_superclass"),
        ("timesteps", "timesteps"),
        ("batch_size", "batch_size"),
    ]:
        if getattr(args, key, None) in (None, ""):
            if cfg_key in cfg:
                setattr(args, key, cfg[cfg_key])
    return args


def load_assets(args: argparse.Namespace) -> RunAssets:
    previous_run_dir = Path(args.previous_run_dir).resolve() if args.previous_run_dir else None
    if previous_run_dir is not None:
        args = fill_from_previous_run(previous_run_dir, args)

    velocities_path = None
    adata_sub_path = None
    if previous_run_dir is not None:
        cand_vel = previous_run_dir / "velocities_eryth.npz"
        cand_adata = previous_run_dir / "adata_eryth_base.h5ad"
        velocities_path = cand_vel if cand_vel.exists() else None
        adata_sub_path = cand_adata if cand_adata.exists() else None

    V1 = None
    V2 = None
    gene_names: list[str] | None = None
    obs_names = None
    if velocities_path is not None:
        print(f"[INFO] Reusing saved velocities: {velocities_path}")
        npz = np.load(velocities_path, allow_pickle=True)
        V1 = np.asarray(npz["V1"], dtype=np.float32)
        V2 = np.asarray(npz["V2"], dtype=np.float32)
        gene_names = [str(x) for x in npz["var_names"].tolist()]
        obs_names = np.asarray(npz["obs_names"]).astype(str) if "obs_names" in npz else None

    X = None
    adata_path_used = None
    target_superclass = args.target_superclass

    if adata_sub_path is not None:
        print(f"[INFO] Reusing saved subset adata: {adata_sub_path}")
        adata_sub = sc.read_h5ad(adata_sub_path)
        X = ensure_dense_float32(adata_sub.X)
        if gene_names is None:
            gene_names = list(adata_sub.var_names.astype(str))
        if obs_names is None:
            obs_names = np.asarray(adata_sub.obs_names.astype(str))
        adata_path_used = str(adata_sub_path)
    elif args.adata_path:
        print(f"[INFO] Loading adata: {args.adata_path}")
        adata = sc.read_h5ad(args.adata_path)
        adata_path_used = args.adata_path
        if target_superclass:
            if "Superclass" not in adata.obs.columns:
                raise KeyError("'Superclass' column not found in adata.obs")
            adata = adata[adata.obs["Superclass"] == target_superclass].copy()
            if adata.n_obs == 0:
                raise ValueError(f"No cells found for Superclass == {target_superclass}")
            print(f"[INFO] Subset: {target_superclass}, n_obs={adata.n_obs}, n_vars={adata.n_vars}")
        X = ensure_dense_float32(adata.X)
        if gene_names is None:
            gene_names = list(adata.var_names.astype(str))
        if obs_names is None:
            obs_names = np.asarray(adata.obs_names.astype(str))
    else:
        raise ValueError("No adata source available. Provide --adata-path or --previous-run-dir containing adata_eryth_base.h5ad.")

    if gene_names is None:
        raise ValueError("gene_names could not be resolved")
    if args.model_path1 is None or args.model_path2 is None:
        raise ValueError("model_path1 / model_path2 could not be resolved")
    if args.edge_tsv_path is None:
        raise ValueError("edge_tsv_path could not be resolved")

    return RunAssets(
        gene_names=gene_names,
        X=X,
        obs_names=obs_names,
        V1=V1,
        V2=V2,
        adata_path=adata_path_used,
        model_path1=args.model_path1,
        model_path2=args.model_path2,
        edge_tsv_path=args.edge_tsv_path,
        target_superclass=target_superclass,
    )


# -----------------------------------------------------------------------------
# analysis
# -----------------------------------------------------------------------------
def compute_gene_metrics(gene_names: list[str], V1: np.ndarray, V2: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, gene in enumerate(gene_names):
        v1 = np.asarray(V1[:, j], dtype=np.float64)
        v2 = np.asarray(V2[:, j], dtype=np.float64)
        valid_mask = np.isfinite(v1) & np.isfinite(v2)
        vv1 = v1[valid_mask]
        vv2 = v2[valid_mask]
        n_valid = int(valid_mask.sum())

        pearson_r = safe_pearson(vv1, vv2)
        spearman_r = safe_spearman(vv1, vv2)
        slope, intercept = safe_linear_fit(vv1, vv2)

        rows.append(
            {
                "gene": gene,
                "n_valid_cells": n_valid,
                "pearson_r": pearson_r,
                "spearman_r": spearman_r,
                "slope_v2_on_v1": slope,
                "intercept_v2_on_v1": intercept,
                "mean_v1": float(np.mean(vv1)) if n_valid > 0 else np.nan,
                "mean_v2": float(np.mean(vv2)) if n_valid > 0 else np.nan,
                "std_v1": float(np.std(vv1)) if n_valid > 0 else np.nan,
                "std_v2": float(np.std(vv2)) if n_valid > 0 else np.nan,
                "mean_abs_v1": float(np.mean(np.abs(vv1))) if n_valid > 0 else np.nan,
                "mean_abs_v2": float(np.mean(np.abs(vv2))) if n_valid > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def merge_metrics(
    gene_metrics: pd.DataFrame,
    diag1: pd.DataFrame,
    diag2: pd.DataFrame,
) -> pd.DataFrame:
    diag1 = diag1.add_prefix("m1_")
    diag2 = diag2.add_prefix("m2_")
    df = pd.concat([gene_metrics.reset_index(drop=True), diag1.reset_index(drop=True), diag2.reset_index(drop=True)], axis=1)

    g1 = df["m1_gamma"].to_numpy(dtype=np.float64)
    g2 = df["m2_gamma"].to_numpy(dtype=np.float64)
    slope = df["slope_v2_on_v1"].to_numpy(dtype=np.float64)

    gamma_ratio = np.full_like(g1, np.nan, dtype=np.float64)
    valid_ratio = np.isfinite(g1) & np.isfinite(g2) & (np.abs(g1) > EPS)
    gamma_ratio[valid_ratio] = g2[valid_ratio] / g1[valid_ratio]

    df["gamma_ratio_g2_over_g1"] = gamma_ratio
    df["delta_slope_minus_gamma_ratio"] = slope - gamma_ratio
    df["abs_delta_slope_minus_gamma_ratio"] = np.abs(df["delta_slope_minus_gamma_ratio"])
    df["sign_match_slope_gamma_ratio"] = np.sign(df["slope_v2_on_v1"]) == np.sign(df["gamma_ratio_g2_over_g1"])
    df["alpha_saturation_score_m1"] = np.maximum(df["m1_frac_alpha_lt_005"], df["m1_frac_alpha_gt_095"])
    df["alpha_saturation_score_m2"] = np.maximum(df["m2_frac_alpha_lt_005"], df["m2_frac_alpha_gt_095"])
    df["gamma_dominance_mean"] = 0.5 * (df["m1_gamma_term_over_alpha"] + df["m2_gamma_term_over_alpha"])
    return df


# -----------------------------------------------------------------------------
# plots / reports
# -----------------------------------------------------------------------------
def _finite_xy(df: pd.DataFrame, xcol: str, ycol: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[xcol].to_numpy(dtype=np.float64)
    y = df[ycol].to_numpy(dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask], mask


def plot_slope_vs_gamma_ratio(df: pd.DataFrame, outpath: Path) -> None:
    x, y, mask = _finite_xy(df, "gamma_ratio_g2_over_g1", "slope_v2_on_v1")
    plt.figure(figsize=(8, 8))
    if x.size > 0:
        plt.scatter(x, y, s=18, alpha=0.65)
        corr = safe_pearson(x, y)
        lo = np.nanpercentile(np.concatenate([x, y]), 1)
        hi = np.nanpercentile(np.concatenate([x, y]), 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = min(np.nanmin(x), np.nanmin(y))
            hi = max(np.nanmax(x), np.nanmax(y))
        pad = 0.05 * max(hi - lo, 1e-6)
        lo, hi = lo - pad, hi + pad
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.5, label="y = x")
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
        plt.legend(loc="best")
        plt.title(f"Slope vs gamma2/gamma1\nN={x.size}, Pearson={corr:.4f}" if np.isfinite(corr) else f"Slope vs gamma2/gamma1\nN={x.size}")
    else:
        plt.text(0.5, 0.5, "No finite points", ha="center", va="center")
        plt.title("Slope vs gamma2/gamma1")
    plt.xlabel("gamma2 / gamma1")
    plt.ylabel("slope of regression V2 ~ a * V1 + b")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_delta_hist(df: pd.DataFrame, outpath: Path) -> dict[str, float]:
    delta = df["delta_slope_minus_gamma_ratio"].to_numpy(dtype=np.float64)
    delta = delta[np.isfinite(delta)]
    mean = float(np.mean(delta)) if delta.size else np.nan
    var = finite_var(delta)
    median = float(np.median(delta)) if delta.size else np.nan

    plt.figure(figsize=(9, 6))
    if delta.size > 0:
        plt.hist(delta, bins=60, alpha=0.75)
        plt.axvline(0.0, linestyle="--", linewidth=1.5)
        plt.title("Histogram of slope - gamma2/gamma1")
        caption = f"mean={mean:.6g}   var={var:.6g}   median={median:.6g}   N={delta.size}"
        plt.figtext(0.5, 0.01, caption, ha="center", fontsize=10)
    else:
        plt.text(0.5, 0.5, "No finite delta values", ha="center", va="center")
        plt.title("Histogram of slope - gamma2/gamma1")
    plt.xlabel("slope - gamma2/gamma1")
    plt.ylabel("gene count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    return {"mean": mean, "var": var, "median": median, "n": int(delta.size)}


def plot_gamma_dominance_vs_abs_delta(df: pd.DataFrame, outpath: Path) -> None:
    x, y, _ = _finite_xy(df, "gamma_dominance_mean", "abs_delta_slope_minus_gamma_ratio")
    plt.figure(figsize=(8, 6))
    if x.size > 0:
        plt.scatter(x, y, s=18, alpha=0.6)
        corr = safe_pearson(x, y)
        title = f"Gamma dominance vs |slope - gamma_ratio|\nN={x.size}"
        if np.isfinite(corr):
            title += f", Pearson={corr:.4f}"
        plt.title(title)
    else:
        plt.text(0.5, 0.5, "No finite points", ha="center", va="center")
        plt.title("Gamma dominance vs |slope - gamma_ratio|")
    plt.xlabel("0.5 * [(|gamma*x|/|alpha|)_m1 + (|gamma*x|/|alpha|)_m2]")
    plt.ylabel("|slope - gamma2/gamma1|")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def write_summary(df: pd.DataFrame, delta_stats: dict[str, float], outpath: Path, args: argparse.Namespace, assets: RunAssets) -> None:
    finite_mask = np.isfinite(df["gamma_ratio_g2_over_g1"].to_numpy(dtype=np.float64)) & np.isfinite(df["slope_v2_on_v1"].to_numpy(dtype=np.float64))
    x = df.loc[finite_mask, "gamma_ratio_g2_over_g1"].to_numpy(dtype=np.float64)
    y = df.loc[finite_mask, "slope_v2_on_v1"].to_numpy(dtype=np.float64)

    summary = {
        "n_genes_total": int(df.shape[0]),
        "n_genes_finite_slope_and_gamma_ratio": int(finite_mask.sum()),
        "pearson_slope_vs_gamma_ratio": safe_pearson(x, y),
        "spearman_slope_vs_gamma_ratio": safe_spearman(x, y),
        "delta_mean": delta_stats["mean"],
        "delta_variance": delta_stats["var"],
        "delta_median": delta_stats["median"],
        "top10_most_negative_pearson_genes": df.sort_values("pearson_r", ascending=True).head(10)["gene"].tolist(),
        "top10_smallest_abs_delta_genes": df.sort_values("abs_delta_slope_minus_gamma_ratio", ascending=True).head(10)["gene"].tolist(),
        "top10_largest_gamma_dominance_genes": df.sort_values("gamma_dominance_mean", ascending=False).head(10)["gene"].tolist(),
        "config": {
            "adata_path": assets.adata_path,
            "model_path1": assets.model_path1,
            "model_path2": assets.model_path2,
            "edge_tsv_path": assets.edge_tsv_path,
            "target_superclass": assets.target_superclass,
            "batch_size": args.batch_size,
            "timesteps": args.timesteps,
            "previous_run_dir": args.previous_run_dir,
        },
    }
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_readme(outdir: Path) -> None:
    text = """This directory contains a gene-wise comparison between
1) slope of the regression V2 ~ a * V1 + b
2) gamma2 / gamma1 extracted from the ODE checkpoints

Files:
- gene_level_gamma_slope_metrics.csv : main table
- slope_vs_gamma_ratio.png          : scatter plot with y=x
- delta_slope_minus_gamma_ratio_hist.png : histogram of slope - gamma2/gamma1
- gamma_dominance_vs_abs_delta.png  : optional support plot
- summary.json                      : compact numeric summary

Interpretation tip:
If out ≈ c - gamma * x gene-wise, then slope(V1->V2) should be close to gamma2/gamma1.
If alpha is saturated and/or gamma*x dominates alpha, this approximation tends to hold better.
"""
    with open(outdir / "README.txt", "w", encoding="utf-8") as f:
        f.write(text)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare slope(V1,V2) with gamma2/gamma1 in the ODE velocity model.")
    parser.add_argument("--scdiffusion-path", default=DEFAULT_SCDIFFUSION_PATH)
    parser.add_argument("--adata-path", default=DEFAULT_ADATA_PATH)
    parser.add_argument("--model-path1", default=DEFAULT_MODEL_PATH1)
    parser.add_argument("--model-path2", default=DEFAULT_MODEL_PATH2)
    parser.add_argument("--edge-tsv-path", default=DEFAULT_EDGE_TSV_PATH)
    parser.add_argument("--target-superclass", default=DEFAULT_TARGET_SUPERCLASS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--device", default=None, help="cpu / cuda. default: auto")
    parser.add_argument(
        "--previous-run-dir",
        default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260330_Analyze-20260224Lamda5/20260330_072038_velocity_compare",
        help="Optional previous velocity_compare directory. If available, velocities_eryth.npz and adata_eryth_base.h5ad are reused.",
    )
    parser.add_argument("--force-recompute-velocities", action="store_true", help="Ignore reused velocities and recompute V1/V2 from checkpoints.")
    parser.add_argument("--save-velocities", action="store_true", help="Save reused/recomputed velocities into the new output dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_environment(args.scdiffusion_path)
    outdir = make_output_dir("gamma_slope_velocity_analysis")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    assets = load_assets(args)

    model1 = load_model(assets.model_path1, assets.gene_names, assets.edge_tsv_path, device, args.timesteps)
    model2 = load_model(assets.model_path2, assets.gene_names, assets.edge_tsv_path, device, args.timesteps)

    if assets.V1 is None or assets.V2 is None or args.force_recompute_velocities:
        if assets.X is None:
            raise ValueError("X is required to compute velocities")
        print("[INFO] Computing V1 from model_path1 ...")
        V1 = compute_velocity_matrix(model1, assets.X, device, args.batch_size)
        print("[INFO] Computing V2 from model_path2 ...")
        V2 = compute_velocity_matrix(model2, assets.X, device, args.batch_size)
    else:
        V1 = assets.V1
        V2 = assets.V2
        print("[INFO] Using reused V1/V2 from previous run directory.")

    if args.save_velocities:
        np.savez_compressed(
            outdir / "velocities_reused_or_computed.npz",
            V1=V1.astype(np.float32),
            V2=V2.astype(np.float32),
            var_names=np.asarray(assets.gene_names, dtype=object),
            obs_names=assets.obs_names if assets.obs_names is not None else np.asarray([], dtype=object),
        )

    print("[INFO] Computing gene-wise slope / correlation metrics ...")
    gene_metrics = compute_gene_metrics(assets.gene_names, V1, V2)

    if assets.X is None:
        raise ValueError("X is required for gamma/alpha diagnostics")

    print("[INFO] Computing ODE diagnostics for model 1 ...")
    diag1 = accumulate_ode_diagnostics(model1, assets.X, device, args.batch_size)
    print("[INFO] Computing ODE diagnostics for model 2 ...")
    diag2 = accumulate_ode_diagnostics(model2, assets.X, device, args.batch_size)

    df = merge_metrics(gene_metrics, diag1, diag2)
    df = df.sort_values(["pearson_r", "abs_delta_slope_minus_gamma_ratio"], ascending=[True, True]).reset_index(drop=True)

    csv_path = outdir / "gene_level_gamma_slope_metrics.csv"
    parquet_path = outdir / "gene_level_gamma_slope_metrics.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    df.sort_values("pearson_r", ascending=True).head(100).to_csv(outdir / "top100_most_negative_pearson_genes.csv", index=False)
    df.sort_values("abs_delta_slope_minus_gamma_ratio", ascending=True).head(100).to_csv(outdir / "top100_smallest_abs_delta_genes.csv", index=False)
    df.sort_values("gamma_dominance_mean", ascending=False).head(100).to_csv(outdir / "top100_gamma_dominant_genes.csv", index=False)

    print("[INFO] Plotting ...")
    plot_slope_vs_gamma_ratio(df, outdir / "slope_vs_gamma_ratio.png")
    delta_stats = plot_delta_hist(df, outdir / "delta_slope_minus_gamma_ratio_hist.png")
    plot_gamma_dominance_vs_abs_delta(df, outdir / "gamma_dominance_vs_abs_delta.png")
    write_summary(df, delta_stats, outdir / "summary.json", args, assets)
    write_readme(outdir)

    run_record = {
        "timestamp": datetime.now().isoformat(),
        "cwd": str(Path.cwd()),
        "command": " ".join(sys.argv),
        "argv": sys.argv,
        "args": vars(args),
    }

    with open(outdir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(run_record, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(csv_path)
    print(parquet_path)
    print(outdir / "slope_vs_gamma_ratio.png")
    print(outdir / "delta_slope_minus_gamma_ratio_hist.png")
    print(outdir / "summary.json")


if __name__ == "__main__":
    main()
