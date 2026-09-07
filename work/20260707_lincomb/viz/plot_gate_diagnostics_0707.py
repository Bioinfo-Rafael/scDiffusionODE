#!/usr/bin/env python3
"""Gate/coefficient diagnostics for 2026-07-07 LinComb experiments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
REPO_ROOT = WORK_ROOT.parent.parent
LEGACY_VIZ = REPO_ROOT / "work" / "20260609_Hybrid5x3" / "viz"
for path in (str(REPO_ROOT), str(WORK_ROOT), str(WORK_ROOT / "scripts"), str(LEGACY_VIZ)):
    if path not in sys.path:
        sys.path.insert(0, path)

import _restore as restore  # noqa: E402
from scripts.common import artifact_dirs, command_string, read_json, update_manifest, write_json  # noqa: E402


def _subset_source(adata, layer, max_cells, seed):
    n = int(adata.n_obs)
    if max_cells > 0 and n > max_cells:
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(n, max_cells, replace=False))
    else:
        indices = np.arange(n)
    view = adata[indices].copy()
    source = view.X if layer is None else view.layers[layer]
    if hasattr(source, "toarray"):
        source = source.toarray()
    x = np.asarray(source, dtype=np.float32)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return view, indices, x


@torch.no_grad()
def _compute_at_t(ode, x, t_value, device, batch_size, active_threshold):
    coefficients, probabilities, outputs = [], [], []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start:start + batch_size]).to(device)
        tb = torch.full((xb.shape[0], 1), float(t_value), device=device)
        gate = ode.get_gate_values(xb, tb)
        coefficients.append(gate["coefficients"].cpu().numpy())
        probabilities.append(gate["probabilities"].cpu().numpy())
        outputs.append(ode(xb, tb).cpu().numpy())
    a = np.concatenate(coefficients).astype(np.float32)
    p = np.concatenate(probabilities).astype(np.float32)
    out = np.concatenate(outputs).astype(np.float32)
    abs_a = np.abs(a)
    denom = np.maximum(abs_a.sum(axis=1, keepdims=True), 1e-12)
    abs_share = abs_a / denom
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    metrics = {
        "mean_abs_a": abs_a.mean(axis=1),
        "l1_norm": abs_a.sum(axis=1),
        "l2_norm": np.linalg.norm(a, axis=1),
        "active_expert_count": (abs_share >= active_threshold).sum(axis=1),
        "top1_abs_share": abs_share.max(axis=1),
        "output_norm": np.linalg.norm(out, axis=1),
        "entropy": entropy,
        "top1_probability_share": p.max(axis=1),
        "top_expert": p.argmax(axis=1),
    }
    return a, p, out, metrics


def _save_gate_curve(model, diffusion, config, output_dir):
    t = np.arange(diffusion.num_timesteps, dtype=np.float32)
    alpha = np.asarray(diffusion.alphas_cumprod, dtype=np.float64)
    if hasattr(model, "_regime_ode_weight") and model.regime_gate_mode != "none":
        with torch.no_grad():
            w = model._regime_ode_weight(
                torch.from_numpy(t), torch.device("cpu"), torch.float32
            ).squeeze(-1).numpy()
        policy = "regime"
    elif hasattr(model, "_scheduler"):
        r = 1.0 - t / float(diffusion.num_timesteps - 1)
        w = 1.0 - r if getattr(model, "reverse_coef", False) else r
        policy = "reverse" if getattr(model, "reverse_coef", False) else "current"
    else:
        return None
    table = pd.DataFrame({"t": t.astype(int), "alpha_bar": alpha, "w_ode": w})
    curve_path = output_dir / "gate_curve.csv"
    table.to_csv(curve_path, index=False)
    # Compatibility with the descriptive filename in the experiment plan.
    table.to_csv(output_dir / "regime_gate_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, w, label="w_ode")
    if config.get("t_s") is not None:
        ax.axvline(float(config["t_s"]), color="tab:red", linestyle="--", label="T_s")
    ax.set(xlabel="diffusion timestep", ylabel="ODE weight", ylim=(-0.02, 1.02))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gate_curve.png", dpi=180)
    plt.close(fig)
    ts = config.get("t_s")
    summary = {
        "policy": policy,
        "mean_w_ode": float(np.mean(w)),
        "mean_w_ode_by_t": [float(value) for value in w],
        "w_ode_at_t0": float(w[0]),
        "w_ode_at_ts": (float(w[int(ts)]) if ts is not None else None),
        "w_ode_at_tmax": float(w[-1]),
    }
    return summary


def _plot_superclass(long_table, output_dir):
    if "Superclass" not in long_table.columns or long_table.empty:
        return
    counts = long_table.groupby("Superclass", observed=True).size().sort_values(ascending=False)
    groups = list(counts.head(12).index)
    subset = long_table[long_table["Superclass"].isin(groups)]
    data = [subset.loc[subset["Superclass"] == group, "a"].values for group in groups]
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 0.8), 5))
    ax.boxplot(data, labels=[str(g) for g in groups], showfliers=False)
    ax.set_ylabel("coefficient a (all experts)")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output_dir / "Superclass_a_distribution.png", dpi=180)
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = read_json(run_dir / "exp_config.json")
    manifest = read_json(run_dir / "manifest.json")
    checkpoint = args.model_path or manifest["checkpoint_path"]
    paths = artifact_dirs(run_dir)
    output_dir = paths["viz"] / "gate"
    if args.dry_run:
        print(json.dumps({
            "command": command_string(), "checkpoint": checkpoint,
            "output_dir": str(output_dir),
        }, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "commands" / "gate_diagnostics.txt").write_text(command_string() + "\n", encoding="utf-8")

    cfg = restore.normalize_config(config)
    diffusion = restore.build_diffusion(cfg["diffusion_steps"])
    gene_list = restore.load_gene_list(config["data_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = restore.build_model(
        cfg, checkpoint, gene_list, diffusion.num_timesteps,
        config["edge_tsv_path"], device,
    )
    ode = getattr(model, "ode_model", None)
    if ode is None or not hasattr(ode, "get_gate_values"):
        raise ValueError("gate diagnostics require ConfigurableLinCombField")

    adata = sc.read_h5ad(config["data_dir"])
    layer = config.get("ts_layer")
    adata_sub, indices, x = _subset_source(
        adata, layer, int(args.max_cells or config.get("diagnostic_max_cells", 50000)),
        int(config.get("seed", 1234)),
    )
    requested_t = args.t_values or str(config.get("diagnostic_t_values", "0,499,999"))
    t_values = sorted(set(int(v) for v in requested_t.split(",") if v.strip()))
    if config.get("t_s") is not None:
        t_values = sorted(set(t_values + [int(config["t_s"])]))
    bad = [value for value in t_values if not 0 <= value < diffusion.num_timesteps]
    if bad:
        raise ValueError(f"diagnostic timesteps out of range: {bad}")

    active_threshold = float(config.get("active_share_threshold", 0.05))
    summaries, superclass_rows, usage_rows = [], [], []
    for t_value in t_values:
        a, p, output, metrics = _compute_at_t(
            ode, x, t_value, device, args.batch_size, active_threshold,
        )
        cell = pd.DataFrame({"cell_index": indices, "t": t_value, **metrics})
        for column in ("Superclass", "celltype", "final_annotation"):
            if column in adata_sub.obs:
                cell[column] = adata_sub.obs[column].astype(str).to_numpy()
        for k in range(a.shape[1]):
            cell[f"a{k}"] = a[:, k]
            cell[f"p{k}"] = p[:, k]
        cell.to_csv(output_dir / f"cell_gate_metrics_t{t_value}.csv", index=False)

        for name, values in metrics.items():
            if name == "top_expert":
                continue
            summaries.append({
                "t": t_value, "metric": name,
                "mean": float(np.mean(values)), "std": float(np.std(values)),
                "median": float(np.median(values)),
            })
        summaries.append({
            "t": t_value, "metric": "coefficient_norm",
            "mean": float(np.linalg.norm(a)),
            "std": 0.0, "median": float(np.linalg.norm(a) / np.sqrt(len(a))),
        })
        for k, norm in enumerate(np.linalg.norm(ode.expert_W.detach().cpu().numpy(), axis=(1, 2))):
            summaries.append({
                "t": t_value, "metric": f"expert_W_norm_k{k}",
                "mean": float(norm), "std": 0.0, "median": float(norm),
            })
        labels = (
            adata_sub.obs["Superclass"].astype(str).to_numpy()
            if "Superclass" in adata_sub.obs else np.repeat("all", len(a))
        )
        for k in range(a.shape[1]):
            superclass_rows.append(pd.DataFrame({
                "t": t_value, "Superclass": labels,
                "expert": k, "a": a[:, k], "probability": p[:, k],
            }))
            usage_rows.append({
                "t": t_value, "expert": k,
                "mean_probability_per_expert": float(p[:, k].mean()),
                "expert_usage_count": int((p.argmax(axis=1) == k).sum()),
                "expert_usage_fraction": float((p.argmax(axis=1) == k).mean()),
            })

    pd.DataFrame(summaries).to_csv(output_dir / "coefficient_diagnostics_summary.csv", index=False)
    long_table = pd.concat(superclass_rows, ignore_index=True)
    try:
        long_table.to_parquet(output_dir / "Superclass_a_distribution.parquet", index=False)
    except Exception:
        long_table.to_csv(output_dir / "Superclass_a_distribution.csv", index=False)
    grouped = long_table.groupby(["t", "Superclass", "expert"], observed=True)["a"].agg(
        ["mean", "std", "median", "count", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)]
    ).reset_index()
    grouped.columns = ["t", "Superclass", "expert", "mean", "std", "median", "count", "q25", "q75"]
    grouped.to_csv(output_dir / "Superclass_a_summary.csv", index=False)
    pd.DataFrame(usage_rows).to_csv(output_dir / "expert_usage_histogram.csv", index=False)
    _plot_superclass(long_table, output_dir)
    curve_summary = _save_gate_curve(model, diffusion, config, output_dir)

    summary = {
        "lambda_max": config.get("lambda_max"),
        "t_s": config.get("t_s"), "gate_tau": config.get("gate_tau"),
        "regime_gate_mode": config.get("regime_gate_mode", "none"),
        "regime_gate_type": config.get("regime_gate_type", "sigmoid"),
        "gate_mode": config.get("gate_mode", "raw"),
        "n_cells": int(len(x)), "t_values": t_values,
        "active_share_threshold": active_threshold,
        **(curve_summary or {}),
    }
    write_json(output_dir / "summary.json", summary)
    update_manifest(run_dir, gate_diagnostics_path=str(output_dir), gate_diagnostics=summary)
    print(f"GATE_DIAGNOSTICS='{output_dir}'")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--t-values", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main()
