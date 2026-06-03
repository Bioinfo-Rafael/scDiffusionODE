# ================================
# File: compute_velocity_metrics.py
# ================================
#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from scipy.stats import pearsonr, spearmanr


# ==========================================
# 1. 初期設定とパスの追加
# ==========================================
def setup_environment():
    scdiffusion_path = "/home/suzuki/Projects/scDiffusion"
    if scdiffusion_path not in sys.path:
        sys.path.insert(0, scdiffusion_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = os.path.join(os.getcwd(), f"{timestamp}_velocity_compare")
    os.makedirs(base_output_dir, exist_ok=True)
    print(f"Output will be saved to: {base_output_dir}")

    # 次スクリプト / notebook から参照しやすいように保存
    latest_run_txt = os.path.join(os.getcwd(), "latest_velocity_compare_run.txt")
    with open(latest_run_txt, "w", encoding="utf-8") as f:
        f.write(base_output_dir)

    return Path(base_output_dir)


# ==========================================
# 2. 固定設定
# ==========================================
ADATA_PATH = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"

# ここは今の環境のまま置いています。必要ならあとで差し替え。
MODEL_PATH1 = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260224_084326_Lamda5/20260224_084328_train/checkpoints/pbmc68k_soft_20251127_hvg1024/ema_0.9999_100000.pt"
MODEL_PATH2 = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/20260224_084326_Lamda5/20260224_084328_train/checkpoints/pbmc68k_soft_20251127_hvg1024/ema_0.9999_000000.pt"

EDGE_TSV_PATH = "/home/suzuki/Projects/scDiffusion/external_data/Mouse_tf_target_edges.tsv"

TARGET_SUPERCLASS = "Erythropoietic"
BATCH_SIZE = 1024
TIMESTEPS = 1000

EPS = 1e-12


# ==========================================
# 3. utility
# ==========================================
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


def load_model(model_path: str, gene_list: list[str], edge_tsv_path: str, device: torch.device):
    # setup_environment() 実行後に import する
    from guided_diffusion.cell_model import Cell_Unet
    from ODE.ode_analysis1106 import GeneODE, ODE_ML_Hybrid

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

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
        timesteps=TIMESTEPS,
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

        mean_abs_v1 = float(np.mean(np.abs(vv1))) if n_valid > 0 else np.nan
        mean_abs_v2 = float(np.mean(np.abs(vv2))) if n_valid > 0 else np.nan
        std_v1 = float(np.std(vv1)) if n_valid > 0 else np.nan
        std_v2 = float(np.std(vv2)) if n_valid > 0 else np.nan
        mean_v1 = float(np.mean(vv1)) if n_valid > 0 else np.nan
        mean_v2 = float(np.mean(vv2)) if n_valid > 0 else np.nan
        frac_zero_v1 = float(np.mean(np.isclose(vv1, 0.0, atol=1e-8))) if n_valid > 0 else np.nan
        frac_zero_v2 = float(np.mean(np.isclose(vv2, 0.0, atol=1e-8))) if n_valid > 0 else np.nan

        rows.append(
            {
                "gene": gene,
                "n_valid_cells": n_valid,
                "pearson_r": pearson_r,
                "pearson_r2": np.nan if pd.isna(pearson_r) else float(pearson_r ** 2),
                "spearman_r": spearman_r,
                "spearman_r2": np.nan if pd.isna(spearman_r) else float(spearman_r ** 2),
                "slope": slope,
                "intercept": intercept,
                "mean_v1": mean_v1,
                "mean_v2": mean_v2,
                "mean_abs_v1": mean_abs_v1,
                "mean_abs_v2": mean_abs_v2,
                "std_v1": std_v1,
                "std_v2": std_v2,
                "frac_zero_v1": frac_zero_v1,
                "frac_zero_v2": frac_zero_v2,
            }
        )

    df = pd.DataFrame(rows)
    df["min_mean_abs_v"] = np.minimum(df["mean_abs_v1"], df["mean_abs_v2"])
    df["min_std_v"] = np.minimum(df["std_v1"], df["std_v2"])
    df["max_frac_zero_v"] = np.maximum(df["frac_zero_v1"], df["frac_zero_v2"])
    return df.sort_values("pearson_r", ascending=True).reset_index(drop=True)


# ==========================================
# 4. main
# ==========================================
def main():
    outdir = setup_environment()

    # いまのコードの device 設定を踏襲
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    print(f"Loading adata from: {ADATA_PATH}")
    adata = sc.read_h5ad(ADATA_PATH)

    if "Superclass" not in adata.obs.columns:
        raise KeyError("'Superclass' column not found in adata.obs")

    adata_sub = adata[adata.obs["Superclass"] == TARGET_SUPERCLASS].copy()
    if adata_sub.n_obs == 0:
        raise ValueError(f"No cells found for Superclass == {TARGET_SUPERCLASS}")

    print(f"Subset: {TARGET_SUPERCLASS}, n_obs={adata_sub.n_obs}, n_vars={adata_sub.n_vars}")

    # いまのコードどおり adata.var_names をそのまま gene_list に使う
    gene_list = list(adata_sub.var_names)

    # いまのコードどおり adata.X を使う
    X = adata_sub.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = 0.0

    print("Loading model 1 and computing V1 ...")
    model1 = load_model(MODEL_PATH1, gene_list=gene_list, edge_tsv_path=EDGE_TSV_PATH, device=device)
    V1 = compute_velocity_matrix(model1, X=X, device=device, batch_size=BATCH_SIZE)

    print("Loading model 2 and computing V2 ...")
    model2 = load_model(MODEL_PATH2, gene_list=gene_list, edge_tsv_path=EDGE_TSV_PATH, device=device)
    V2 = compute_velocity_matrix(model2, X=X, device=device, batch_size=BATCH_SIZE)

    print("Computing gene-wise metrics ...")
    metrics_df = compute_gene_metrics(gene_names=gene_list, V1=V1, V2=V2)

    metrics_df.to_parquet(outdir / "gene_metrics_eryth.parquet", index=False)
    metrics_df.to_csv(outdir / "gene_metrics_eryth.csv", index=False)

    # notebook / embed script で順序一致確認のため obs_names / var_names も保存
    np.savez_compressed(
        outdir / "velocities_eryth.npz",
        V1=V1.astype(np.float32),
        V2=V2.astype(np.float32),
        obs_names=np.asarray(adata_sub.obs_names.astype(str)),
        var_names=np.asarray(adata_sub.var_names.astype(str)),
        target_superclass=np.asarray([TARGET_SUPERCLASS]),
    )

    # 最小限の subset も保存しておくと後段が安全
    adata_sub.write(outdir / "adata_eryth_base.h5ad")

    config = {
        "adata_path": ADATA_PATH,
        "model_path1": MODEL_PATH1,
        "model_path2": MODEL_PATH2,
        "edge_tsv_path": EDGE_TSV_PATH,
        "target_superclass": TARGET_SUPERCLASS,
        "device_used": str(device),
        "batch_size": BATCH_SIZE,
        "timesteps": TIMESTEPS,
        "n_obs_subset": int(adata_sub.n_obs),
        "n_vars_subset": int(adata_sub.n_vars),
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(outdir / "gene_metrics_eryth.parquet")
    print(outdir / "velocities_eryth.npz")
    print(outdir / "adata_eryth_base.h5ad")


if __name__ == "__main__":
    main()