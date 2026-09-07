#!/usr/bin/env python3
"""Compute LinComb velocity projections once and render UMAP plus PAGA views."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
WORK_ROOT = HERE.parent
REPO_ROOT = WORK_ROOT.parent.parent
LEGACY_VIZ = REPO_ROOT / "work" / "20260609_Hybrid5x3" / "viz"
for path in (str(REPO_ROOT), str(WORK_ROOT), str(WORK_ROOT / "scripts"), str(LEGACY_VIZ)):
    if path not in sys.path:
        sys.path.insert(0, path)

import _restore as restore  # noqa: E402
import plot_lincomb_component_velocity_umap as component  # noqa: E402
import plot_lincomb_component_velocity_paga as paga  # noqa: E402
import plot_velocity_umap as total_velocity  # noqa: E402
from plot_velocity_umap import _apply_scvelo_compat_shims  # noqa: E402
from scripts.common import artifact_dirs, command_string, read_json, update_manifest, write_json  # noqa: E402


def _lineage_palette_in_base_order(base_state, lineage_state):
    color_map = dict(zip(lineage_state["categories"], lineage_state["colors"]))
    base_map = dict(zip(base_state["categories"], base_state["colors"]))
    return [color_map.get(category, base_map.get(category, "#808080")) for category in base_state["categories"]]


def _render_group(adata, group_name, outdir, ode, K, args, device, color_col):
    outdir.mkdir(parents=True, exist_ok=True)
    x = component.prepare_group_embedding(adata)
    base_state, lineage_state = component._prepare_palette_states(adata, group_name, color_col)
    # Lock category order once; palette changes must not invalidate PAGA adjacency.
    component._apply_palette_state(adata, color_col, base_state)
    lineage_palette = _lineage_palette_in_base_order(base_state, lineage_state)
    umap_figs, urows, ucols = component._new_grid_figures(K)
    paga_figs, prows, pcols = paga._new_paga_figures(K)
    stats, errors = {}, {}
    try:
        for panel_index, spec in enumerate(component._velocity_specs(K)):
            title, vkey, k = spec["title"], spec["vkey"], spec["k"]
            if k is None:
                velocity, sanity = component.compute_lincomb_total_velocity(
                    ode, x, args.velocity_t, device, args.batch_size,
                )
            else:
                velocity = component.compute_lincomb_component_velocity(
                    ode, x, k, args.velocity_t, device, args.batch_size,
                    args.component_mode,
                )
                sanity = None
            adata.layers[vkey] = velocity
            stats[title] = component.velocity_norm_stats(velocity)
            try:
                # This is the sole graph/embedding computation for this vkey.
                component._compute_velocity_projection(adata, vkey, args.n_jobs)
                row, col = divmod(panel_index, ucols)
                base_palette = list(base_state["colors"])
                for kind, key, palette_values in (
                    ("stream", "stream", base_palette),
                    ("arrow", "arrow", base_palette),
                    ("stream", "stream_lineage", lineage_palette),
                    ("arrow", "arrow_lineage", lineage_palette),
                ):
                    component._render_panel(
                        kind, adata, vkey, color_col, palette_values, title,
                        umap_figs[key][1][row, col],
                    )
                if k is None:
                    standalone = (
                        ("stream", base_palette, "1_velocity_stream.png"),
                        ("arrow", base_palette, "2_velocity_arrow.png"),
                        ("stream", lineage_palette, "3_velocity_stream_lineage.png"),
                        ("arrow", lineage_palette, "4_velocity_arrow_lineage.png"),
                    )
                    for kind, palette_values, filename in standalone:
                        fig, ax = plt.subplots(figsize=(7, 6))
                        component._render_panel(
                            kind, adata, vkey, color_col, palette_values, title, ax
                        )
                        fig.tight_layout()
                        fig.savefig(outdir / filename, dpi=240, bbox_inches="tight")
                        plt.close(fig)

                paga._clear_paga_state(adata, color_col)
                paga._compute_paga_projection(adata, vkey, color_col)
                prow, pcol = divmod(panel_index, pcols)
                paga._plot_paga_to_axis(
                    adata, vkey, color_col, base_palette, title,
                    paga_figs["paga"][1][prow, pcol], args,
                )
                paga._plot_paga_to_axis(
                    adata, vkey, color_col, lineage_palette, title,
                    paga_figs["paga_lineage"][1][prow, pcol], args,
                )
            except Exception as exc:
                errors[title] = f"{type(exc).__name__}: {exc}"
                component._mark_failed_panel(
                    umap_figs, panel_index, ucols, title, type(exc).__name__
                )
                prow, pcol = divmod(panel_index, pcols)
                paga._mark_failed_axis(paga_figs["paga"][1][prow, pcol], title, type(exc).__name__)
                paga._mark_failed_axis(paga_figs["paga_lineage"][1][prow, pcol], title, type(exc).__name__)

        component._save_grid_figures(
            umap_figs, K + 1, urows, ucols, str(outdir),
            base_legend_state=base_state, lineage_legend_state=lineage_state,
            color_col=color_col,
        )
        umap_figs = {}
        paga._save_paga_figures(
            paga_figs, K + 1, prows, pcols, str(outdir),
            base_state, {"categories": base_state["categories"], "colors": lineage_palette},
            color_col,
        )
        paga_figs = {}
    finally:
        component._close_grid_figures(umap_figs)
        component._close_grid_figures(paga_figs)

    cache_path = outdir / "velocity_projection_cache.h5ad"
    try:
        adata.write(cache_path)
    except Exception as exc:
        errors["cache_write"] = f"{type(exc).__name__}: {exc}"
    payload = {
        "group": str(group_name), "n_cells": int(adata.n_obs),
        "velocity_t": float(args.velocity_t), "component_mode": args.component_mode,
        "velocity_projection_computations": int(K + 1),
        "cache_path": str(cache_path), "stats": stats, "errors": errors,
    }
    write_json(outdir / "summary.json", payload)
    return payload


def main():
    args = build_parser().parse_args()
    run_dir = Path(args.run_dir).resolve()
    config = read_json(run_dir / "exp_config.json")
    manifest = read_json(run_dir / "manifest.json")
    checkpoint = args.model_path or manifest["checkpoint_path"]
    paths = artifact_dirs(run_dir)
    output_dir = paths["viz"] / "velocity"
    if args.dry_run:
        print(json.dumps({
            "command": command_string(), "checkpoint": checkpoint,
            "output_dir": str(output_dir),
            "policy": "one velocity_graph + one velocity_embedding per group/vkey",
        }, indent=2))
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "commands" / "velocity.txt").write_text(command_string() + "\n", encoding="utf-8")
    _apply_scvelo_compat_shims()

    cfg = restore.normalize_config(config)
    diffusion = restore.build_diffusion(cfg["diffusion_steps"])
    gene_list = restore.load_gene_list(config["data_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = restore.build_model(
        cfg, checkpoint, gene_list, diffusion.num_timesteps,
        config["edge_tsv_path"], device,
    )
    ode, K = component.validate_lincomb_model(model)
    adata = sc.read_h5ad(config["data_dir"])
    layer = config.get("ts_layer")
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"input layer '{layer}' is missing")
        adata.X = adata.layers[layer].copy()
    if args.max_cells > 0 and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(int(config.get("seed", 1234)))
        selected = np.sort(rng.choice(adata.n_obs, args.max_cells, replace=False))
        adata = adata[selected].copy()
    color_col = restore.auto_label_col(adata)
    if color_col is None:
        color_col = "_velocity_color"
        adata.obs[color_col] = pd.Categorical(["all"] * adata.n_obs)
    if args.group_col in adata.obs:
        group_values = adata.obs[args.group_col]
        groups = list(pd.unique(group_values))
    else:
        group_values, groups = None, ["all"]

    results, skipped = {}, {}
    for group in groups:
        name = "NA" if pd.isna(group) else str(group)
        if group_values is None:
            subset = adata.copy()
        else:
            mask = group_values.isna() if pd.isna(group) else group_values == group
            subset = adata[np.asarray(mask)].copy()
        if subset.n_obs < args.min_cells:
            skipped[name] = int(subset.n_obs)
            continue
        results[name] = _render_group(
            subset, name, output_dir / "velocity_by_superclass" / component._safe_group_name(name),
            ode, K, args, device, color_col,
        )
    summary = {
        "run_dir": str(run_dir), "checkpoint": checkpoint,
        "K": K, "groups": results, "skipped": skipped,
        "shared_projection_policy": "one graph/embedding computation per group/vkey",
    }
    sample_path = manifest.get("sample_path", "")
    if sample_path and os.path.exists(sample_path):
        try:
            adata_real = sc.read_h5ad(config["data_dir"])
            if layer is not None:
                adata_real.X = adata_real.layers[layer].copy()
            combined, _ = total_velocity.create_combined_anndata(adata_real, sample_path)
            if combined is not None:
                total_velocity.plot_umap_analysis(combined, str(output_dir), max_cells=args.max_cells)
                summary["real_vs_generated_umap"] = str(output_dir / "umap_analysis.png")
        except Exception as exc:
            summary["real_vs_generated_umap_error"] = f"{type(exc).__name__}: {exc}"
    write_json(output_dir / "summary.json", summary)
    update_manifest(run_dir, velocity_viz_path=str(output_dir), velocity_summary=summary)
    print(f"VELOCITY_VIZ='{output_dir}'")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--group-col", default="Superclass")
    parser.add_argument("--velocity-t", type=float, default=0.0)
    parser.add_argument("--component-mode", choices=("expert", "contribution"), default="expert")
    parser.add_argument("--max-cells", type=int, default=0)
    parser.add_argument("--min-cells", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=32)
    parser.add_argument("--paga-size", type=float, default=50)
    parser.add_argument("--paga-alpha", type=float, default=0.1)
    parser.add_argument("--paga-min-edge-width", type=float, default=2)
    parser.add_argument("--paga-node-size-scale", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main()
