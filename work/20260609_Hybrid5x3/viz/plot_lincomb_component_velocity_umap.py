#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinComb の total/expert velocity を同一 UMAP 上の grid で比較する。

LinCombField:
    a = coeff_net(x, t)
    V_k = softplus(W_k x + b_k)
    V_total = sum_k a_k V_k - decay(x)

K=8 のとき V_total, V_0, ..., V_7 を 3x3 panel として、Superclass ごとに
stream / arrow x 通常配色 / lineage 配色の 4 枚を保存する。

既存の plot_velocity_umap.py は変更せず、復元は _restore.build_model() を使う。
LinComb の compute_W() は proxy なので本スクリプトでは使用しない。
"""

import argparse
import atexit
import gc
import io
import json
import math
import os
import re
import shlex
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                         # _restore / sibling viz scripts
sys.path.insert(0, os.path.join(HERE, ".."))   # run_paths

import _restore as R  # noqa: E402
from plot_lincomb_a_embedding import find_checkpoint, find_config  # noqa: E402
from plot_velocity_umap import (  # noqa: E402
    _apply_scvelo_compat_shims,
    apply_lineage_palette,
)


ROLE = "velocity_lincomb_components"


class _Tee:
    """stdout を端末と run.log の両方へ書く。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            try:
                stream.write(value)
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def _safe_group_name(value):
    name = str(value).strip() or "unnamed"
    return re.sub(r"[\\/:*?\"<>|\s]+", "_", name)


def _parse_only_groups(value):
    """Comma-separated group namesを重複なしで返す。空文字は全group。"""
    requested = []
    for item in str(value or "").split(","):
        name = item.strip()
        if name and name not in requested:
            requested.append(name)
    return requested


def filter_adata_to_groups(adata, group_col, only_groups):
    """指定groupだけにAnnDataを絞る。指定なしなら入力をそのまま返す。"""
    requested = _parse_only_groups(only_groups)
    if not requested:
        return adata, requested
    if group_col not in adata.obs.columns:
        raise SystemExit(
            f"[ERROR] --only_groups requires group column '{group_col}', "
            "but it is missing from adata.obs."
        )

    values = adata.obs[group_col].astype(object).to_numpy()
    labels = np.asarray(
        ["NA" if pd.isna(value) else str(value) for value in values],
        dtype=object,
    )
    available = set(labels.tolist())
    missing = [name for name in requested if name not in available]
    if missing:
        raise SystemExit(
            f"[ERROR] --only_groups contains unknown {group_col} values: {missing}"
        )
    mask = np.isin(labels, requested)
    return adata[np.asarray(mask)].copy(), requested


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _velocity_specs(K):
    yield {"title": "V_total", "vkey": "velocity_ode_total", "k": None}
    for k in range(K):
        yield {"title": f"V_{k}", "vkey": f"velocity_ode_k{k}", "k": k}


def validate_lincomb_model(model):
    """LinComb 専用であることと必要な実装を検証する。"""
    ode = getattr(model, "ode_model", None)
    if ode is None:
        raise SystemExit(
            "[ERROR] このモデルには ode_model がありません"
            "（plain baseline?）。LinComb run を指定してください。"
        )
    model_type = getattr(ode, "model_type", None)
    if model_type != "lincomb":
        raise SystemExit(
            "[ERROR] このスクリプトは LinComb 専用です。"
            f"ode_model.model_type='{model_type}'。"
        )
    missing = [
        name for name in ("_coeffs", "_decay", "expert_W", "expert_b", "K")
        if not hasattr(ode, name)
    ]
    if missing:
        raise SystemExit(
            "[ERROR] LinComb velocity の計算に必要な属性がありません: "
            + ", ".join(missing)
        )
    K = int(ode.K)
    if tuple(ode.expert_W.shape[:1]) != (K,) or tuple(ode.expert_b.shape[:1]) != (K,):
        raise SystemExit(
            "[ERROR] ode_model.K と expert_W/expert_b の第1次元が一致しません。"
        )
    return ode, K


def _model_time(batch, velocity_t, device):
    """plot_velocity_umap.py / R.compute_velocity と同じ (B,1) float t。"""
    return torch.full((batch, 1), float(velocity_t), dtype=torch.float32, device=device)


def _clean_velocity(value):
    value = np.asarray(value, dtype=np.float32)
    return np.nan_to_num(value, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def compute_lincomb_total_velocity(ode, X, velocity_t, device, batch_size=512):
    """V_total と、直接再構成式 vs ode.forward の sanity metric を返す。

    描画用には ode(x,t) の値を使う。これにより既存の total velocity
    と同じ経路にする。並行して a/Fx/decay から直接再構成し、差を集計する。
    """
    n, d = X.shape
    velocity = np.empty((n, d), dtype=np.float32)
    diff_sq_sum = 0.0
    ref_sq_sum = 0.0
    max_abs_error = 0.0
    nonfinite_forward = 0
    nonfinite_reconstructed = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(np.asarray(X[start:end], dtype=np.float32)).to(device)
        tb = _model_time(xb.shape[0], velocity_t, device)

        # LinCombField.forward と同じ直接式。compute_W() は使わない。
        a = ode._coeffs(xb, tb)
        Fx = F.softplus(
            torch.einsum("kij,bj->bki", ode.expert_W, xb) + ode.expert_b
        )
        reconstructed = torch.einsum("bk,bkd->bd", a, Fx) - ode._decay(xb)
        forward = ode(xb, tb)

        forward_np = forward.detach().cpu().numpy().astype(np.float32, copy=False)
        reconstructed_np = reconstructed.detach().cpu().numpy().astype(np.float32, copy=False)
        nonfinite_forward += int((~np.isfinite(forward_np)).sum())
        nonfinite_reconstructed += int((~np.isfinite(reconstructed_np)).sum())

        finite_forward = np.nan_to_num(
            forward_np.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0
        )
        finite_reconstructed = np.nan_to_num(
            reconstructed_np.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0
        )
        diff = finite_reconstructed - finite_forward
        diff_sq_sum += float(np.square(diff).sum())
        ref_sq_sum += float(np.square(finite_forward).sum())
        if diff.size:
            max_abs_error = max(max_abs_error, float(np.abs(diff).max()))
        velocity[start:end] = _clean_velocity(forward_np)

    absolute_l2 = math.sqrt(diff_sq_sum)
    reference_l2 = math.sqrt(ref_sq_sum)
    relative_l2 = absolute_l2 / max(reference_l2, np.finfo(np.float64).eps)
    sanity = {
        "absolute_l2": float(absolute_l2),
        "reference_l2": float(reference_l2),
        "relative_l2": float(relative_l2),
        "max_abs_error": float(max_abs_error),
        "nonfinite_forward_values": int(nonfinite_forward),
        "nonfinite_reconstructed_values": int(nonfinite_reconstructed),
        # root summary で group 間集計するための加算可能値。
        "diff_sq_sum": float(diff_sq_sum),
        "reference_sq_sum": float(ref_sq_sum),
    }
    return velocity, sanity


@torch.no_grad()
def compute_lincomb_component_velocity(
    ode,
    X,
    k,
    velocity_t,
    device,
    batch_size=512,
    component_mode="expert",
):
    """1つの component velocity だけを計算し、全9配列の常駐を避ける。"""
    if component_mode not in ("expert", "contribution"):
        raise ValueError(f"unknown component_mode: {component_mode}")
    if k < 0 or k >= int(ode.K):
        raise IndexError(f"component k={k} is outside 0..{int(ode.K) - 1}")

    n, d = X.shape
    velocity = np.empty((n, d), dtype=np.float32)
    Wk = ode.expert_W[k]
    bk = ode.expert_b[k]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(np.asarray(X[start:end], dtype=np.float32)).to(device)
        # forward の Fx[:, k, :] と同じ softplus(W_k x + b_k)。
        Vk = F.softplus(torch.einsum("ij,bj->bi", Wk, xb) + bk)
        if component_mode == "contribution":
            tb = _model_time(xb.shape[0], velocity_t, device)
            ak = ode._coeffs(xb, tb)[:, k:k + 1]  # raw signed a_k
            Vk = ak * Vk
        velocity[start:end] = _clean_velocity(Vk.detach().cpu().numpy())
    return velocity


def velocity_norm_stats(velocity):
    norms = np.linalg.norm(np.asarray(velocity, dtype=np.float64), axis=1)
    return {
        "n_cells": int(norms.size),
        "mean": float(norms.mean()) if norms.size else 0.0,
        "std": float(norms.std()) if norms.size else 0.0,
        "min": float(norms.min()) if norms.size else 0.0,
        "max": float(norms.max()) if norms.size else 0.0,
        "sum": float(norms.sum()),
        "sum_sq": float(np.square(norms).sum()),
    }


def _update_norm_accumulator(accumulator, title, stats):
    dst = accumulator.setdefault(title, {"n_cells": 0, "sum": 0.0, "sum_sq": 0.0})
    dst["n_cells"] += int(stats["n_cells"])
    dst["sum"] += float(stats["sum"])
    dst["sum_sq"] += float(stats["sum_sq"])


def _finalize_norm_accumulator(accumulator):
    result = {}
    for title, stats in accumulator.items():
        n = int(stats["n_cells"])
        mean = stats["sum"] / n if n else 0.0
        variance = max(stats["sum_sq"] / n - mean * mean, 0.0) if n else 0.0
        result[title] = {
            "n_cells": n,
            "mean": float(mean),
            "std": float(math.sqrt(variance)),
        }
    return result


def prepare_group_embedding(adata_subset):
    """Superclass subset に対し PCA/neighbors/UMAP を1回だけ計算する。"""
    X = R.to_dense(adata_subset.X).astype(np.float32, copy=False)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    adata_subset.X = X
    adata_subset.layers["X"] = X.copy()
    R.embed_umap(adata_subset, n_top=50)
    return X


def _default_palette(n):
    cmap = plt.get_cmap("tab20")
    return [mcolors.to_hex(cmap(i % cmap.N)) for i in range(n)]


def _prepare_palette_states(adata_subset, group_name, color_col):
    """base/lineage の category 順と palette を保存し、必要時に切り替える。"""
    source = adata_subset.obs[color_col]
    if not isinstance(source.dtype, pd.CategoricalDtype):
        source = source.astype("category")
    old_categories = list(source.cat.categories)
    old_colors = adata_subset.uns.get(f"{color_col}_colors", None)
    old_color_map = {}
    if old_colors is not None and len(old_colors) == len(old_categories):
        old_color_map = dict(zip(old_categories, list(old_colors)))

    source = source.cat.remove_unused_categories()
    adata_subset.obs[color_col] = source
    base_categories = list(source.cat.categories)
    base_colors = [old_color_map[c] for c in base_categories] if old_color_map else _default_palette(len(base_categories))
    base = {"categories": base_categories, "colors": list(base_colors)}
    _apply_palette_state(adata_subset, color_col, base)

    try:
        apply_lineage_palette(adata_subset, group_name, celltype_col=color_col)
        lineage = {
            "categories": list(adata_subset.obs[color_col].cat.categories),
            "colors": list(adata_subset.uns[f"{color_col}_colors"]),
        }
    except Exception as exc:
        print(f"  [lineage palette] fallback to base ({type(exc).__name__}: {exc})")
        lineage = {"categories": list(base["categories"]), "colors": list(base["colors"])}
    _apply_palette_state(adata_subset, color_col, base)
    return base, lineage


def _apply_palette_state(adata_subset, color_col, state):
    series = adata_subset.obs[color_col]
    if not isinstance(series.dtype, pd.CategoricalDtype):
        series = series.astype("category")
    series = series.cat.remove_unused_categories()
    present = list(series.cat.categories)
    target = [c for c in state["categories"] if c in present]
    target += [c for c in present if c not in target]
    series = series.cat.reorder_categories(target, ordered=True)
    adata_subset.obs[color_col] = series
    color_map = dict(zip(state["categories"], state["colors"]))
    colors = [color_map.get(c, "#808080") for c in target]
    adata_subset.uns[f"{color_col}_colors"] = np.asarray(colors, dtype=str)
    return colors


def _new_grid_figures(K, skip_stream=False, skip_arrow=False):
    n_panels = K + 1
    ncols = 3
    nrows = int(math.ceil(n_panels / ncols))
    figures = {}
    if not skip_stream:
        figures["stream"] = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
        )
        figures["stream_lineage"] = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
        )
    if not skip_arrow:
        figures["arrow"] = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
        )
        figures["arrow_lineage"] = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False
        )
    return figures, nrows, ncols


def _plot_kwargs(adata_subset, vkey, color_col, palette, title):
    return dict(
        adata=adata_subset,
        basis="umap",
        vkey=vkey,
        color=color_col,
        palette=list(palette),
        title=title,
        show=False,
        legend_loc=None,
        size=5,
        alpha=1,
    )


def _strip_axis_legends(ax, keep_title=True):
    """scVelo/Scanpy が panel 内に作った legend や on-data category text を消す。"""
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()

    # panel title は ax.title なので ax.texts から除かれない。
    for txt in list(ax.texts):
        try:
            txt.remove()
        except Exception:
            pass


def _add_shared_legend(fig, categories, colors, title, *, max_items=80):
    """grid 全体に1つだけ legend を図の右外へ追加する。"""
    categories = list(categories)
    colors = list(colors)
    if not categories or not colors:
        return

    shown_categories = categories[:max_items]
    shown_colors = colors[:max_items]
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=color,
            markeredgecolor=color,
            label=str(category),
        )
        for category, color in zip(shown_categories, shown_colors)
    ]
    if len(categories) > max_items:
        handles.append(
            Line2D(
                [0], [0],
                marker="",
                linestyle="",
                label=f"... +{len(categories) - max_items} more",
            )
        )

    fig.legend(
        handles=handles,
        title=title,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        borderaxespad=0.0,
    )


def _plot_to_axis(kind, adata_subset, vkey, color_col, palette, title, ax):
    kwargs = _plot_kwargs(adata_subset, vkey, color_col, palette, title)
    if kind == "stream":
        scv.pl.velocity_embedding_stream(ax=ax, recompute=False, **kwargs)
    elif kind == "arrow":
        scv.pl.velocity_embedding(
            ax=ax, recompute=False, arrow_length=3, arrow_size=2, **kwargs
        )
    else:
        raise ValueError(kind)
    ax.set_title(title)


def _plot_to_png_buffer(kind, adata_subset, vkey, color_col, palette, title):
    """scVelo が grid の ax を扱えない場合の standalone PNG fallback。"""
    kwargs = _plot_kwargs(adata_subset, vkey, color_col, palette, title)
    if kind == "stream":
        scv.pl.velocity_embedding_stream(recompute=False, **kwargs)
    elif kind == "arrow":
        scv.pl.velocity_embedding(
            recompute=False, arrow_length=3, arrow_size=2, **kwargs
        )
    else:
        raise ValueError(kind)
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=180, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    image = plt.imread(buffer, format="png")
    buffer.close()
    return image


def _render_panel(kind, adata_subset, vkey, color_col, palette, title, ax):
    try:
        _plot_to_axis(kind, adata_subset, vkey, color_col, palette, title, ax)
        _strip_axis_legends(ax)
        ax.set_title(title)
        return None
    except Exception as exc:
        print(
            f"  [plot fallback] {title} {kind}: direct ax failed "
            f"({type(exc).__name__}: {exc})"
        )
        ax.clear()
        try:
            image = _plot_to_png_buffer(
                kind, adata_subset, vkey, color_col, palette, title
            )
            ax.imshow(image)
            _strip_axis_legends(ax)
            ax.axis("off")
            return f"direct ax failed; standalone PNG used: {type(exc).__name__}: {exc}"
        except Exception as fallback_exc:
            ax.clear()
            ax.text(
                0.5,
                0.5,
                f"FAILED\n{type(fallback_exc).__name__}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            ax.axis("off")
            return (
                f"direct={type(exc).__name__}: {exc}; "
                f"fallback={type(fallback_exc).__name__}: {fallback_exc}"
            )


def _compute_velocity_projection(adata_subset, vkey, n_jobs):
    """velocity graph/embedding。並列失敗時はn_jobs=1で一度だけ再試行。"""
    try:
        scv.tl.velocity_graph(
            adata_subset,
            vkey=vkey,
            xkey="X",
            backend="loky",
            n_jobs=n_jobs,
        )
    except Exception as exc:
        if int(n_jobs) == 1:
            raise
        print(
            f"  [velocity_graph retry] {vkey}: n_jobs={n_jobs} failed "
            f"({type(exc).__name__}: {exc}); retry n_jobs=1"
        )
        _cleanup_velocity_keys(adata_subset, vkey, keep_layer=True)
        scv.tl.velocity_graph(
            adata_subset,
            vkey=vkey,
            xkey="X",
            backend="loky",
            n_jobs=1,
        )
    scv.tl.velocity_embedding(adata_subset, basis="umap", vkey=vkey)


def _cleanup_velocity_keys(adata_subset, vkey, keep_layer=False):
    if not keep_layer and vkey in adata_subset.layers:
        del adata_subset.layers[vkey]
    for key in (f"{vkey}_graph", f"{vkey}_graph_neg", f"{vkey}_params"):
        if key in adata_subset.uns:
            del adata_subset.uns[key]
    key = f"{vkey}_umap"
    if key in adata_subset.obsm:
        del adata_subset.obsm[key]
    for key in list(adata_subset.obs.columns):
        if str(key).startswith(f"{vkey}_"):
            del adata_subset.obs[key]
    for key in list(adata_subset.var.columns):
        if str(key).startswith(f"{vkey}_"):
            del adata_subset.var[key]


def _mark_failed_panel(figures, panel_index, ncols, title, message):
    row, col = divmod(panel_index, ncols)
    for _, axes in figures.values():
        ax = axes[row, col]
        ax.clear()
        ax.text(0.5, 0.5, f"FAILED\n{message}", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.axis("off")


def _save_grid_figures(
    figures,
    n_panels,
    nrows,
    ncols,
    outdir,
    *,
    base_legend_state=None,
    lineage_legend_state=None,
    color_col=None,
):
    names = {
        "stream": "1_velocity_stream_grid.png",
        "arrow": "2_velocity_arrow_grid.png",
        "stream_lineage": "3_velocity_stream_lineage_grid.png",
        "arrow_lineage": "4_velocity_arrow_lineage_grid.png",
    }
    created = []
    for key, (fig, axes) in figures.items():
        for panel_index in range(n_panels, nrows * ncols):
            row, col = divmod(panel_index, ncols)
            axes[row, col].axis("off")

        for ax in axes.ravel():
            _strip_axis_legends(ax)

        legend_state = (
            base_legend_state
            if key in ("stream", "arrow")
            else lineage_legend_state
        )
        if legend_state is not None and color_col is not None:
            _add_shared_legend(
                fig,
                legend_state.get("categories", []),
                legend_state.get("colors", []),
                title=color_col,
            )

        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        path = os.path.join(outdir, names[key])
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        created.append(names[key])
    return created


def _close_grid_figures(figures):
    for fig, _ in figures.values():
        plt.close(fig)


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
    """1 Superclass の embedding・9 velocity・4 grid を処理する。"""
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== Processing {args.group_col}: {group_name} ({adata_subset.n_obs} cells) ===")
    X = prepare_group_embedding(adata_subset)
    base_state, lineage_state = _prepare_palette_states(
        adata_subset, group_name, color_col
    )
    figures, nrows, ncols = _new_grid_figures(
        K, skip_stream=args.skip_stream, skip_arrow=args.skip_arrow
    )
    plot_requested = bool(figures)

    component_stats = {}
    total_sanity = None
    graph_errors = {}
    plot_fallbacks = {}
    created_files = []

    try:
        for panel_index, spec in enumerate(_velocity_specs(K)):
            title, vkey, k = spec["title"], spec["vkey"], spec["k"]
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
                key: value for key, value in stats.items() if key not in ("sum", "sum_sq")
            }
            _update_norm_accumulator(norm_accumulator, title, stats)

            if plot_requested:
                adata_subset.layers[vkey] = velocity
                try:
                    _compute_velocity_projection(adata_subset, vkey, args.n_jobs)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    graph_errors[title] = message
                    print(f"  [ERROR] {title} velocity graph failed: {message}")
                    _mark_failed_panel(figures, panel_index, ncols, title, type(exc).__name__)
                else:
                    row, col = divmod(panel_index, ncols)
                    base_palette = _apply_palette_state(
                        adata_subset, color_col, base_state
                    )
                    if "stream" in figures:
                        error = _render_panel(
                            "stream",
                            adata_subset,
                            vkey,
                            color_col,
                            base_palette,
                            title,
                            figures["stream"][1][row, col],
                        )
                        if error:
                            plot_fallbacks[f"{title}:stream"] = error
                    if "arrow" in figures:
                        error = _render_panel(
                            "arrow",
                            adata_subset,
                            vkey,
                            color_col,
                            base_palette,
                            title,
                            figures["arrow"][1][row, col],
                        )
                        if error:
                            plot_fallbacks[f"{title}:arrow"] = error

                    lineage_palette = _apply_palette_state(
                        adata_subset, color_col, lineage_state
                    )
                    if "stream_lineage" in figures:
                        error = _render_panel(
                            "stream",
                            adata_subset,
                            vkey,
                            color_col,
                            lineage_palette,
                            title,
                            figures["stream_lineage"][1][row, col],
                        )
                        if error:
                            plot_fallbacks[f"{title}:stream_lineage"] = error
                    if "arrow_lineage" in figures:
                        error = _render_panel(
                            "arrow",
                            adata_subset,
                            vkey,
                            color_col,
                            lineage_palette,
                            title,
                            figures["arrow_lineage"][1][row, col],
                        )
                        if error:
                            plot_fallbacks[f"{title}:arrow_lineage"] = error
                    _apply_palette_state(adata_subset, color_col, base_state)

                _cleanup_velocity_keys(adata_subset, vkey)
            del velocity
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if plot_requested:
            created_files = _save_grid_figures(
                figures,
                K + 1,
                nrows,
                ncols,
                outdir,
                base_legend_state=base_state,
                lineage_legend_state=lineage_state,
                color_col=color_col,
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
        "graph_errors": graph_errors,
        "plot_fallbacks": plot_fallbacks,
        "created_files": created_files,
    }
    _write_json(os.path.join(outdir, "component_velocity_sanity.json"), payload)
    created_files.append("component_velocity_sanity.json")
    print(f"Finished plotting for {group_name}: {', '.join(created_files)}")
    return payload, created_files


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "LinComb の V_total と各 expert velocity V_k を同一 UMAP 上の grid で比較する。"
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
        help="checkpoint を明示。未指定なら run_dir から最大 step を自動検出。",
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
        help="未指定なら {run_dir}/viz/velocity_lincomb_components/",
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
    parser.add_argument("--skip_stream", action="store_true")
    parser.add_argument("--skip_arrow", action="store_true")
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
    print(f"[lincomb-velocity] config      : {config_path}")
    print(f"[lincomb-velocity] checkpoint  : {model_path}")
    print(f"[lincomb-velocity] data_dir    : {data_dir}")
    print(f"[lincomb-velocity] edge_tsv    : {edge_tsv_path}")
    print(f"[lincomb-velocity] output_dir  : {output_dir}")
    print(f"[lincomb-velocity] device      : {device}")

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
    print(f"[lincomb-velocity] K           : {K}")
    print(f"[lincomb-velocity] velocity_t  : {args.velocity_t}")
    print(f"[lincomb-velocity] mode        : {args.component_mode}")

    adata = sc.read_h5ad(data_dir)
    n_cells_full = int(adata.n_obs)
    adata, requested_groups = filter_adata_to_groups(
        adata, args.group_col, args.only_groups
    )
    n_cells_after_group_filter = int(adata.n_obs)
    if requested_groups:
        print(
            "[lincomb-velocity] only_groups : "
            + ", ".join(requested_groups)
        )
    if args.max_cells > 0 and adata.n_obs > args.max_cells:
        selected = np.sort(
            np.random.choice(adata.n_obs, args.max_cells, replace=False)
        )
        adata = adata[selected].copy()
    n_cells_used = int(adata.n_obs)
    print(
        f"[lincomb-velocity] cells       : {n_cells_used} "
        f"(full={n_cells_full}, after_group_filter={n_cells_after_group_filter}, "
        f"max_cells={args.max_cells or 'unlimited'})"
    )

    color_col = R.auto_label_col(adata)
    if color_col is None:
        color_col = "_velocity_group"
        adata.obs[color_col] = pd.Categorical(["all"] * adata.n_obs)
        print("[lincomb-velocity] annotation column not found; using one-color fallback")
    else:
        print(f"[lincomb-velocity] color_col   : {color_col}")

    if args.group_col in adata.obs.columns:
        values = adata.obs[args.group_col]
        groups = list(pd.unique(values))
    else:
        print(
            f"[lincomb-velocity] '{args.group_col}' not found; running whole dataset"
        )
        values = None
        groups = ["all"]

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
            print(f"[lincomb-velocity] skip {display_name}: {reason}")
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
            n_cells_visualized += int(adata_subset.n_obs)
            created_files.extend(
                os.path.relpath(os.path.join(group_outdir, name), output_dir)
                for name in group_files
            )
            sanity = payload.get("V_total_sanity") or {}
            sanity_accumulator["diff_sq_sum"] += float(sanity.get("diff_sq_sum", 0.0))
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
                f"[lincomb-velocity][ERROR] {display_name} failed "
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
        "groups": group_results,
        "skipped_groups": skipped_groups,
        "failed_groups": failed_groups,
        "skip_stream": bool(args.skip_stream),
        "skip_arrow": bool(args.skip_arrow),
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
    print(f"output_dir         : {output_dir}")
    print(f"summary            : {os.path.join(output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
