#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinComb の total/expert velocity を Superclass 別 PAGA grid で比較する。

`plot_lincomb_component_velocity_umap.py` の計算・復元・palette helper を再利用し、
PAGA 固有の処理だけをこの単独スクリプトに実装する。既存の UMAP
stream/arrow スクリプトとその出力は変更しない。
"""

import argparse
import atexit
import gc
import math
import os
import shlex
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import _restore as R  # noqa: E402
from plot_lincomb_a_embedding import find_checkpoint, find_config  # noqa: E402
from plot_lincomb_component_velocity_umap import (  # noqa: E402
    _Tee,
    _add_shared_legend,
    _apply_palette_state,
    _apply_scvelo_compat_shims,
    _close_grid_figures,
    _compute_velocity_projection,
    _finalize_norm_accumulator,
    filter_adata_to_groups,
    _prepare_palette_states,
    _safe_group_name,
    _strip_axis_legends,
    _update_norm_accumulator,
    _velocity_specs,
    _write_json,
    compute_lincomb_component_velocity,
    compute_lincomb_total_velocity,
    prepare_group_embedding,
    validate_lincomb_model,
    velocity_norm_stats,
)


ROLE = "velocity_lincomb_component_paga"


def _new_paga_figures(K):
    n_panels = K + 1
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    figures = {
        "paga": plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
        ),
        "paga_lineage": plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
        ),
    }
    return figures, nrows, ncols


def _clear_paga_state(adata_subset, color_col=None):
    if "paga" in adata_subset.uns:
        del adata_subset.uns["paga"]
    if color_col is not None:
        size_key = f"{color_col}_sizes"
        if size_key in adata_subset.uns:
            del adata_subset.uns[size_key]


def _compute_paga_projection(adata_subset, vkey, color_col):
    """PAGA を計算し、vkey API fallback を使ったかを返す。"""
    try:
        scv.tl.paga(adata_subset, groups=color_col, vkey=vkey)
        return False
    except TypeError as exc:
        message = str(exc)
        unsupported_vkey = (
            "unexpected keyword argument 'vkey'" in message
            or 'unexpected keyword argument "vkey"' in message
            or ("vkey" in message and "unexpected keyword" in message)
        )
        if not unsupported_vkey:
            raise
        print(
            f"  [PAGA compatibility] scv.tl.paga has no vkey argument; "
            f"retry without vkey ({type(exc).__name__}: {exc})"
        )
        scv.tl.paga(adata_subset, groups=color_col)
        return True


def _plot_paga_to_axis(
    adata_subset,
    vkey,
    color_col,
    palette,
    title,
    ax,
    args,
):
    scv.pl.paga(
        adata_subset,
        basis="umap",
        vkey=vkey,
        color=color_col,
        palette=list(palette),
        node_colors=list(palette),
        size=args.paga_size,
        alpha=args.paga_alpha,
        min_edge_width=args.paga_min_edge_width,
        node_size_scale=args.paga_node_size_scale,
        scatter_flag=True,
        show=False,
        legend_loc=None,
        ax=ax,
    )
    _strip_axis_legends(ax)
    ax.set_title(title)


def _mark_failed_axis(ax, title, message):
    ax.clear()
    ax.text(
        0.5,
        0.5,
        f"FAILED\n{message}",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )
    ax.set_title(title)
    ax.axis("off")


def _cleanup_paga_velocity_keys(
    adata_subset, vkey, color_col=None, keep_layer=False
):
    if not keep_layer and vkey in adata_subset.layers:
        del adata_subset.layers[vkey]

    for key in (
        f"{vkey}_graph",
        f"{vkey}_graph_neg",
        f"{vkey}_graph_uncertainties",
        f"{vkey}_params",
    ):
        if key in adata_subset.uns:
            del adata_subset.uns[key]

    _clear_paga_state(adata_subset, color_col)

    key = f"{vkey}_umap"
    if key in adata_subset.obsm:
        del adata_subset.obsm[key]

    for key in list(adata_subset.obs.columns):
        if str(key).startswith(f"{vkey}_"):
            del adata_subset.obs[key]
    for key in list(adata_subset.var.columns):
        if str(key).startswith(f"{vkey}_"):
            del adata_subset.var[key]


def _save_paga_figures(
    figures,
    n_panels,
    nrows,
    ncols,
    outdir,
    base_legend_state,
    lineage_legend_state,
    color_col,
):
    names = {
        "paga": "1_velocity_paga_grid.png",
        "paga_lineage": "2_velocity_paga_lineage_grid.png",
    }
    created = []
    for key, (fig, axes) in figures.items():
        for panel_index in range(n_panels, nrows * ncols):
            row, col = divmod(panel_index, ncols)
            axes[row, col].axis("off")

        # 成功 panel は描画直後に strip 済み。ここで ax.texts を一括削除すると
        # FAILED 表示も消えるため、保存時は figure legend の追加だけにする。
        legend_state = (
            base_legend_state if key == "paga" else lineage_legend_state
        )
        _add_shared_legend(
            fig,
            legend_state.get("categories", []),
            legend_state.get("colors", []),
            title=color_col,
        )
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        path = os.path.join(outdir, names[key])
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        created.append(names[key])
    return created


def process_group(
    adata_subset,
    group_name,
    outdir,
    ode,
    K,
    args,
    device,
    color_col,
    norm_accumulator,
):
    os.makedirs(outdir, exist_ok=True)
    print(
        f"\n=== Processing {args.group_col}: {group_name} "
        f"({adata_subset.n_obs} cells) ==="
    )
    X = prepare_group_embedding(adata_subset)
    base_state, lineage_state = _prepare_palette_states(
        adata_subset, group_name, color_col
    )
    figures, nrows, ncols = _new_paga_figures(K)

    component_stats = {}
    total_sanity = None
    paga_errors = {}
    paga_vkey_fallback = False
    created_files = []

    try:
        for panel_index, spec in enumerate(_velocity_specs(K)):
            title, vkey, k = spec["title"], spec["vkey"], spec["k"]
            row, col = divmod(panel_index, ncols)
            base_ax = figures["paga"][1][row, col]
            lineage_ax = figures["paga_lineage"][1][row, col]
            print(f"  [{panel_index + 1}/{K + 1}] computing {title}")

            if k is None:
                velocity, total_sanity = compute_lincomb_total_velocity(
                    ode, X, args.velocity_t, device, args.batch_size
                )
            else:
                velocity = compute_lincomb_component_velocity(
                    ode,
                    X,
                    k,
                    args.velocity_t,
                    device,
                    args.batch_size,
                    args.component_mode,
                )

            stats = velocity_norm_stats(velocity)
            component_stats[title] = {
                key: value
                for key, value in stats.items()
                if key not in ("sum", "sum_sq")
            }
            _update_norm_accumulator(norm_accumulator, title, stats)
            adata_subset.layers[vkey] = velocity

            component_errors = {}
            try:
                _compute_velocity_projection(adata_subset, vkey, args.n_jobs)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                component_errors["velocity_graph"] = message
                _mark_failed_axis(base_ax, title, type(exc).__name__)
                _mark_failed_axis(lineage_ax, title, type(exc).__name__)
                print(f"  [ERROR] {title} velocity graph failed: {message}")
            else:
                # PAGA adjacency の行列順と category 順を一致させるため、
                # base / lineage の palette state ごとに PAGA を再計算する。
                for variant, state, ax in (
                    ("base", base_state, base_ax),
                    ("lineage", lineage_state, lineage_ax),
                ):
                    palette = _apply_palette_state(
                        adata_subset, color_col, state
                    )
                    _clear_paga_state(adata_subset, color_col)
                    try:
                        used_fallback = _compute_paga_projection(
                            adata_subset, vkey, color_col
                        )
                        paga_vkey_fallback = (
                            paga_vkey_fallback or used_fallback
                        )
                        _plot_paga_to_axis(
                            adata_subset,
                            vkey,
                            color_col,
                            palette,
                            title,
                            ax,
                            args,
                        )
                    except Exception as exc:
                        message = f"{type(exc).__name__}: {exc}"
                        component_errors[variant] = message
                        _mark_failed_axis(ax, title, type(exc).__name__)
                        print(
                            f"  [ERROR] {title} PAGA {variant} failed: "
                            f"{message}"
                        )
                _apply_palette_state(adata_subset, color_col, base_state)

            if component_errors:
                paga_errors[title] = component_errors
            _cleanup_paga_velocity_keys(
                adata_subset, vkey, color_col=color_col
            )
            del velocity
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        created_files = _save_paga_figures(
            figures,
            K + 1,
            nrows,
            ncols,
            outdir,
            base_state,
            lineage_state,
            color_col,
        )
        figures = {}
    finally:
        _close_grid_figures(figures)

    payload = {
        "group_col": args.group_col,
        "group_name": str(group_name),
        "n_cells": int(adata_subset.n_obs),
        "K": int(K),
        "velocity_t": float(args.velocity_t),
        "component_mode": args.component_mode,
        "component_norms": component_stats,
        "V_total_sanity": total_sanity,
        "paga_params": {
            "size": float(args.paga_size),
            "alpha": float(args.paga_alpha),
            "min_edge_width": float(args.paga_min_edge_width),
            "node_size_scale": float(args.paga_node_size_scale),
        },
        "paga_errors": paga_errors,
        "paga_vkey_fallback": bool(paga_vkey_fallback),
        "created_files": created_files,
    }
    _write_json(os.path.join(outdir, "component_paga_sanity.json"), payload)
    created_files.append("component_paga_sanity.json")
    print(f"Finished PAGA for {group_name}: {', '.join(created_files)}")
    return payload, created_files


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "LinComb の V_total と各 expert velocity V_k を "
            "Superclass 別 PAGA grid で比較する。"
        )
    )
    parser.add_argument(
        "--run_dir",
        required=True,
        help="対象 run dir。config/checkpoint をこの配下から自動検出する。",
    )
    parser.add_argument(
        "--model_path",
        default="",
        help="checkpoint を明示。未指定なら run_dir から自動検出。",
    )
    parser.add_argument(
        "--config",
        default="",
        help="exp_config.json を明示。未指定なら run_dir から自動検出。",
    )
    parser.add_argument(
        "--data_dir",
        default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad",
    )
    parser.add_argument(
        "--edge_tsv_path",
        default="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="未指定なら {run_dir}/viz/velocity_lincomb_component_paga/",
    )
    parser.add_argument("--group_col", default="Superclass")
    parser.add_argument(
        "--only_groups",
        default="",
        help="カンマ区切りで処理対象groupを限定。空なら全group。",
    )
    parser.add_argument("--velocity_t", type=float, default=0.0)
    parser.add_argument(
        "--max_cells",
        type=int,
        default=0,
        help="使う全細胞数の上限。0以下は制限なし。",
    )
    parser.add_argument("--min_cells", type=int, default=15)
    parser.add_argument("--n_jobs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--component_mode",
        choices=("expert", "contribution"),
        default="expert",
        help="expert=raw V_k, contribution=a_k*V_k",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--paga_size", type=float, default=50)
    parser.add_argument("--paga_alpha", type=float, default=0.1)
    parser.add_argument("--paga_min_edge_width", type=float, default=2)
    parser.add_argument("--paga_node_size_scale", type=float, default=1.5)
    return parser


@torch.no_grad()
def main():
    args = build_argparser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("[ERROR] --batch_size must be > 0")
    if args.min_cells < 2:
        raise SystemExit("[ERROR] --min_cells must be >= 2")
    if args.n_jobs <= 0:
        raise SystemExit("[ERROR] --n_jobs must be > 0")

    _apply_scvelo_compat_shims()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    scv.set_figure_params(transparent=False)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(f"--run_dir が存在しません: {run_dir}")
    config_path = find_config(run_dir, args.config)
    model_path = find_checkpoint(run_dir, args.model_path)
    data_dir = R.resolve_path(args.data_dir)
    edge_tsv_path = R.resolve_path(args.edge_tsv_path)
    output_dir = os.path.abspath(
        args.output_dir
        if args.output_dir
        else os.path.join(run_dir, "viz", ROLE)
    )
    base_group_dir = os.path.join(output_dir, "velocity_by_superclass")
    os.makedirs(base_group_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    command = " ".join(
        shlex.quote(value)
        for value in [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    )
    with open(os.path.join(output_dir, "command.txt"), "w") as f:
        f.write(f"# timestamp: {timestamp}\n# cwd: {os.getcwd()}\n{command}\n")
    log_file = open(os.path.join(output_dir, "run.log"), "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)

    def restore_stdout():
        sys.stdout = sys.__stdout__
        try:
            log_file.flush()
            log_file.close()
        except Exception:
            pass

    atexit.register(restore_stdout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lincomb-paga] config      : {config_path}")
    print(f"[lincomb-paga] checkpoint  : {model_path}")
    print(f"[lincomb-paga] data_dir    : {data_dir}")
    print(f"[lincomb-paga] edge_tsv    : {edge_tsv_path}")
    print(f"[lincomb-paga] output_dir  : {output_dir}")
    print(f"[lincomb-paga] device      : {device}")

    cfg = R.normalize_config(R.load_config(config_path))
    diffusion = R.build_diffusion(cfg["diffusion_steps"])
    gene_list = R.load_gene_list(data_dir)
    model = R.build_model(
        cfg,
        model_path,
        gene_list,
        diffusion.num_timesteps,
        edge_tsv_path,
        device,
    )
    ode, K = validate_lincomb_model(model)
    print(f"[lincomb-paga] K           : {K}")
    print(f"[lincomb-paga] velocity_t  : {args.velocity_t}")
    print(f"[lincomb-paga] mode        : {args.component_mode}")

    adata = sc.read_h5ad(data_dir)
    n_cells_full = int(adata.n_obs)
    adata, requested_groups = filter_adata_to_groups(
        adata, args.group_col, args.only_groups
    )
    n_cells_after_group_filter = int(adata.n_obs)
    if requested_groups:
        print("[lincomb-paga] only_groups : " + ", ".join(requested_groups))
    if args.max_cells > 0 and adata.n_obs > args.max_cells:
        selected = np.sort(
            np.random.choice(adata.n_obs, args.max_cells, replace=False)
        )
        adata = adata[selected].copy()
    n_cells_used = int(adata.n_obs)
    print(
        f"[lincomb-paga] cells       : {n_cells_used} "
        f"(full={n_cells_full}, after_group_filter={n_cells_after_group_filter}, "
        f"max_cells={args.max_cells or 'unlimited'})"
    )

    color_col = R.auto_label_col(adata)
    if color_col is None:
        color_col = "_velocity_group"
        adata.obs[color_col] = pd.Categorical(["all"] * adata.n_obs)
        print("[lincomb-paga] annotation column not found; using one-color fallback")
    else:
        print(f"[lincomb-paga] color_col   : {color_col}")

    if args.group_col in adata.obs.columns:
        values = adata.obs[args.group_col]
        groups = list(pd.unique(values))
    else:
        print(f"[lincomb-paga] '{args.group_col}' not found; running whole dataset")
        values = None
        groups = ["all"]

    paga_params = {
        "size": float(args.paga_size),
        "alpha": float(args.paga_alpha),
        "min_edge_width": float(args.paga_min_edge_width),
        "node_size_scale": float(args.paga_node_size_scale),
    }
    norm_accumulator = {}
    sanity_accumulator = {
        "diff_sq_sum": 0.0,
        "reference_sq_sum": 0.0,
        "max_abs_error": 0.0,
        "nonfinite_forward_values": 0,
        "nonfinite_reconstructed_values": 0,
    }
    group_results = {}
    skipped_groups = {}
    failed_groups = {}
    created_files = ["command.txt", "run.log"]
    n_cells_visualized = 0
    root_paga_vkey_fallback = False

    for group in groups:
        display_name = "NA" if pd.isna(group) else str(group)
        if values is None:
            adata_subset = adata.copy()
        else:
            mask = values.isna() if pd.isna(group) else (values == group)
            adata_subset = adata[np.asarray(mask)].copy()

        if adata_subset.n_obs < args.min_cells:
            reason = f"only {adata_subset.n_obs} cells (< min_cells={args.min_cells})"
            skipped_groups[display_name] = reason
            print(f"[lincomb-paga] skip {display_name}: {reason}")
            continue

        group_outdir = os.path.join(base_group_dir, _safe_group_name(display_name))
        try:
            payload, group_files = process_group(
                adata_subset,
                display_name,
                group_outdir,
                ode,
                K,
                args,
                device,
                color_col,
                norm_accumulator,
            )
            group_results[display_name] = payload
            root_paga_vkey_fallback = (
                root_paga_vkey_fallback
                or bool(payload.get("paga_vkey_fallback", False))
            )
            n_cells_visualized += int(adata_subset.n_obs)
            created_files.extend(
                os.path.relpath(os.path.join(group_outdir, name), output_dir)
                for name in group_files
            )

            sanity = payload.get("V_total_sanity") or {}
            sanity_accumulator["diff_sq_sum"] += float(
                sanity.get("diff_sq_sum", 0.0)
            )
            sanity_accumulator["reference_sq_sum"] += float(
                sanity.get("reference_sq_sum", 0.0)
            )
            sanity_accumulator["max_abs_error"] = max(
                sanity_accumulator["max_abs_error"],
                float(sanity.get("max_abs_error", 0.0)),
            )
            sanity_accumulator["nonfinite_forward_values"] += int(
                sanity.get("nonfinite_forward_values", 0)
            )
            sanity_accumulator["nonfinite_reconstructed_values"] += int(
                sanity.get("nonfinite_reconstructed_values", 0)
            )
        except Exception as exc:
            failed_groups[display_name] = f"{type(exc).__name__}: {exc}"
            print(
                f"[lincomb-paga][ERROR] {display_name} failed "
                f"({type(exc).__name__}: {exc}); continue"
            )
        finally:
            del adata_subset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_diff_l2 = math.sqrt(sanity_accumulator["diff_sq_sum"])
    total_reference_l2 = math.sqrt(sanity_accumulator["reference_sq_sum"])
    total_relative_l2 = total_diff_l2 / max(
        total_reference_l2, np.finfo(np.float64).eps
    )
    total_sanity = {
        "absolute_l2": float(total_diff_l2),
        "reference_l2": float(total_reference_l2),
        "relative_l2": float(total_relative_l2),
        "max_abs_error": float(sanity_accumulator["max_abs_error"]),
        "nonfinite_forward_values": int(
            sanity_accumulator["nonfinite_forward_values"]
        ),
        "nonfinite_reconstructed_values": int(
            sanity_accumulator["nonfinite_reconstructed_values"]
        ),
    }
    summary = {
        "run_dir": run_dir,
        "model_path": model_path,
        "config": config_path,
        "data_dir": data_dir,
        "edge_tsv_path": edge_tsv_path,
        "output_dir": output_dir,
        "n_cells_full": n_cells_full,
        "n_cells_after_group_filter": n_cells_after_group_filter,
        "n_cells_used": n_cells_used,
        "n_cells_visualized": n_cells_visualized,
        "K": int(K),
        "velocity_t": float(args.velocity_t),
        "component_mode": args.component_mode,
        "group_col": args.group_col,
        "only_groups": requested_groups,
        "color_col": color_col,
        "component_norms": _finalize_norm_accumulator(norm_accumulator),
        "V_total_sanity": total_sanity,
        "paga_params": paga_params,
        "paga_vkey_fallback": bool(root_paga_vkey_fallback),
        "groups": group_results,
        "skipped_groups": skipped_groups,
        "failed_groups": failed_groups,
        "created_files": created_files + ["summary.json"],
        "created_at": timestamp,
        "command": command,
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
    }
    _write_json(os.path.join(output_dir, "summary.json"), summary)

    print("\n==================== DONE ====================")
    print(f"checkpoint         : {model_path}")
    print(f"config             : {config_path}")
    print(f"n_cells_used       : {n_cells_used}")
    print(f"n_cells_visualized : {n_cells_visualized}")
    print(f"K                  : {K}")
    print(f"V_total rel L2     : {total_relative_l2:.6e}")
    print(f"PAGA vkey fallback : {root_paga_vkey_fallback}")
    print(f"output_dir         : {output_dir}")
    print(f"summary            : {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
