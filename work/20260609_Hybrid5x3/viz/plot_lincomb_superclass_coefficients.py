#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LinComb 係数/component contribution と Superclass の対応を可視化する。"""

import argparse
import atexit
import glob
import json
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
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

import _restore as R  # noqa: E402
from plot_lincomb_a_embedding import find_checkpoint  # noqa: E402


ROLE = "lincomb_superclass_coefficients"
EPS = 1e-12


class _Tee:
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


def _write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def _config_score(path, payload):
    name = os.path.basename(path)
    score = 0
    if name == "exp_config.json":
        score += 100
    elif name == "config.json":
        score += 80
    elif "config" in name.lower():
        score += 60
    if f"{os.sep}train{os.sep}" in os.path.abspath(path):
        score += 20
    keys = set(payload) if isinstance(payload, dict) else set()
    score += 5 * len(keys & {"ode_branch", "model_type", "diffusion_steps", "K"})
    return score


def find_config(run_dir, explicit=""):
    """run_dir からモデル復元に妥当な JSON config を選ぶ。"""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError(f"--config が見つかりません: {explicit}")
        return os.path.abspath(explicit)

    patterns = (
        os.path.join(run_dir, "train", "**", "exp_config.json"),
        os.path.join(run_dir, "**", "exp_config.json"),
        os.path.join(run_dir, "train", "**", "config.json"),
        os.path.join(run_dir, "**", "config.json"),
        os.path.join(run_dir, "train", "**", "*config*.json"),
        os.path.join(run_dir, "**", "*config*.json"),
        os.path.join(run_dir, "**", "*.json"),
    )
    candidates = []
    seen = set()
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            path = os.path.abspath(path)
            if path in seen or not os.path.isfile(path):
                continue
            seen.add(path)
            try:
                with open(path) as f:
                    payload = json.load(f)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            keys = set(payload)
            if not keys.intersection({"ode_branch", "model_type", "diffusion_steps", "K"}):
                continue
            candidates.append((_config_score(path, payload), os.path.getmtime(path), path))
    if not candidates:
        raise FileNotFoundError(
            f"{run_dir} 配下に妥当な config JSON がありません。--config で指定してください。"
        )
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _resolve_input_path(explicit, config_value, config_path, label):
    value = explicit or config_value
    if not value:
        raise SystemExit(
            f"[ERROR] {label} が指定されていません。CLI または config で指定してください。"
        )

    candidates = [R.resolve_path(value)]
    if not os.path.isabs(str(value)):
        # 学習 config の相対 path は Hybrid5x3/ 基準で保存されている。
        candidates.extend(
            [
                os.path.abspath(os.path.join(HERE, "..", value)),
                os.path.abspath(os.path.join(os.path.dirname(config_path), value)),
                os.path.abspath(os.path.join(R.REPO_ROOT, value)),
            ]
        )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"{label} が見つかりません: {value}")


def validate_lincomb_model(model):
    ode = getattr(model, "ode_model", None)
    if getattr(ode, "model_type", None) != "lincomb":
        raise SystemExit("This script is only for LinCombField.")
    missing = [
        name for name in ("decompose_W", "expert_W", "expert_b", "K")
        if not hasattr(ode, name)
    ]
    if missing:
        raise SystemExit(
            "[ERROR] LinCombField に必要な属性がありません: " + ", ".join(missing)
        )
    return ode, int(ode.K)


def _ordered_groups(series):
    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = set(series.dropna().astype(str))
        groups = [str(value) for value in series.cat.categories if str(value) in observed]
    else:
        groups = sorted({str(value) for value in series.dropna().unique()})
    if series.isna().any():
        groups.append("NA")
    return groups


def _string_labels(series):
    values = series.astype(object).to_numpy()
    return np.asarray(["NA" if pd.isna(value) else str(value) for value in values], dtype=object)


def _select_top_groups(labels, group_order, top_groups):
    if not top_groups:
        return list(group_order)
    text = str(top_groups).strip()
    if text.isdigit():
        n_top = int(text)
        if n_top <= 0:
            raise SystemExit("[ERROR] --top_groups の数値は1以上にしてください。")
        counts = {group: int(np.sum(labels == group)) for group in group_order}
        selected = set(sorted(group_order, key=lambda g: (-counts[g], g))[:n_top])
        return [group for group in group_order if group in selected]
    requested = [value.strip() for value in text.split(",") if value.strip()]
    missing = [value for value in requested if value not in group_order]
    if missing:
        raise SystemExit(f"[ERROR] --top_groups に存在しない group があります: {missing}")
    return requested


def stratified_sample_indices(labels, group_order, max_cells, seed):
    """group ごとにできるだけ均等な割当てで index を選ぶ。"""
    n = len(labels)
    if max_cells <= 0 or n <= max_cells:
        return np.arange(n, dtype=int)

    pools = {group: np.flatnonzero(labels == group) for group in group_order}
    allocation = {group: 0 for group in group_order}
    remaining = int(max_cells)
    active = [group for group in group_order if len(pools[group]) > 0]

    while remaining > 0 and active:
        quotient, remainder = divmod(remaining, len(active))
        if quotient == 0:
            for group in active[:remaining]:
                allocation[group] += 1
            remaining = 0
            break

        consumed = 0
        for index, group in enumerate(active):
            capacity = len(pools[group]) - allocation[group]
            requested = quotient + (1 if index < remainder else 0)
            take = min(capacity, requested)
            allocation[group] += take
            consumed += take
        remaining -= consumed
        active = [
            group for group in active if allocation[group] < len(pools[group])
        ]
        if consumed == 0:
            break

    rng = np.random.default_rng(seed)
    selected = []
    for group in group_order:
        pool = pools[group]
        count = allocation[group]
        if count:
            selected.extend(rng.choice(pool, size=count, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


@torch.no_grad()
def compute_cell_quantities(ode, X, t_value, device, batch_size, compute_contribution):
    n = X.shape[0]
    K = int(ode.K)
    a_all = np.empty((n, K), dtype=np.float32)
    contribution_all = (
        np.empty((n, K), dtype=np.float32) if compute_contribution else None
    )

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        xb = torch.from_numpy(np.asarray(X[start:end], dtype=np.float32)).to(device)
        tb = torch.full(
            (xb.shape[0], 1), float(t_value), dtype=torch.float32, device=device
        )
        decomposition = ode.decompose_W(xb, tb)
        if not isinstance(decomposition, dict) or "a" not in decomposition:
            raise RuntimeError("ode.decompose_W(x,t) did not return a dict containing 'a'.")
        a = decomposition["a"]
        if tuple(a.shape) != (xb.shape[0], K):
            raise RuntimeError(f"unexpected coefficient shape: {tuple(a.shape)}")
        a_all[start:end] = a.detach().cpu().numpy().astype(np.float32, copy=False)

        if compute_contribution:
            fx = F.softplus(
                torch.einsum("kij,bj->bki", ode.expert_W, xb) + ode.expert_b
            )
            out_norm = torch.linalg.vector_norm(fx, dim=-1)
            contribution = torch.abs(a) * out_norm
            contribution_all[start:end] = (
                contribution.detach().cpu().numpy().astype(np.float32, copy=False)
            )

    abs_a = np.abs(a_all)
    coeff_share = abs_a / (abs_a.sum(axis=1, keepdims=True) + EPS)
    if contribution_all is None:
        contribution_share = None
    else:
        contribution_share = contribution_all / (
            contribution_all.sum(axis=1, keepdims=True) + EPS
        )
    return a_all, abs_a, coeff_share, contribution_all, contribution_share


def build_summary_table(
    labels,
    group_order,
    group_col,
    t_value,
    a,
    abs_a,
    coeff_share,
    contribution,
    contribution_share,
):
    rows = []
    K = a.shape[1]
    for group in group_order:
        mask = labels == group
        n_cells = int(mask.sum())
        coeff_share_mean = coeff_share[mask].mean(axis=0)
        dominant_coeff = int(np.argmax(coeff_share_mean))
        if contribution is not None:
            contrib_share_mean = contribution_share[mask].mean(axis=0)
            dominant_contrib = int(np.argmax(contrib_share_mean))
        else:
            dominant_contrib = -1

        for k in range(K):
            row = {
                "group_col": group_col,
                "group": group,
                "n_cells": n_cells,
                "t": float(t_value),
                "k": int(k),
                "coeff_mass_mean": float(abs_a[mask, k].mean()),
                "coeff_share_mean": float(coeff_share[mask, k].mean()),
                "coeff_signed_mean": float(a[mask, k].mean()),
                "coeff_mass_std": float(abs_a[mask, k].std()),
                "coeff_share_std": float(coeff_share[mask, k].std()),
                "coeff_signed_std": float(a[mask, k].std()),
                "dominant_by_coeff_share": int(k == dominant_coeff),
                "dominant_by_contrib_share": int(k == dominant_contrib),
            }
            if contribution is None:
                row.update(
                    {
                        "contrib_mass_mean": np.nan,
                        "contrib_share_mean": np.nan,
                        "contrib_mass_std": np.nan,
                        "contrib_share_std": np.nan,
                    }
                )
            else:
                row.update(
                    {
                        "contrib_mass_mean": float(contribution[mask, k].mean()),
                        "contrib_share_mean": float(contribution_share[mask, k].mean()),
                        "contrib_mass_std": float(contribution[mask, k].std()),
                        "contrib_share_std": float(contribution_share[mask, k].std()),
                    }
                )
            rows.append(row)
    column_order = [
        "group_col", "group", "n_cells", "t", "k",
        "coeff_mass_mean", "coeff_share_mean", "coeff_signed_mean",
        "coeff_mass_std", "coeff_share_std", "coeff_signed_std",
        "contrib_mass_mean", "contrib_share_mean",
        "contrib_mass_std", "contrib_share_std",
        "dominant_by_coeff_share", "dominant_by_contrib_share",
    ]
    return pd.DataFrame(rows)[column_order]


def build_cell_table(
    cell_indices,
    labels,
    t_value,
    a,
    abs_a,
    coeff_share,
    contribution,
    contribution_share,
):
    K = a.shape[1]
    table = pd.DataFrame(
        {
            "cell_index": cell_indices.astype(int),
            "group": labels,
            "t": float(t_value),
        }
    )
    for k in range(K):
        table[f"a_k{k}"] = a[:, k]
    for k in range(K):
        table[f"abs_a_k{k}"] = abs_a[:, k]
    for k in range(K):
        table[f"coeff_share_k{k}"] = coeff_share[:, k]
    if contribution is None:
        for k in range(K):
            table[f"contribution_k{k}"] = np.nan
        for k in range(K):
            table[f"contribution_share_k{k}"] = np.nan
        table["dominant_contrib_k"] = -1
    else:
        for k in range(K):
            table[f"contribution_k{k}"] = contribution[:, k]
        for k in range(K):
            table[f"contribution_share_k{k}"] = contribution_share[:, k]
        table["dominant_contrib_k"] = np.argmax(contribution_share, axis=1).astype(int)
    table["dominant_coeff_k"] = np.argmax(coeff_share, axis=1).astype(int)
    ordered_tail = ["dominant_coeff_k", "dominant_contrib_k"]
    return table[[c for c in table.columns if c not in ordered_tail] + ordered_tail]


def _component_colors(K):
    cmap = plt.get_cmap("tab10")
    return [cmap(k % cmap.N) for k in range(K)]


def _matrix_from_summary(summary, group_order, value_col, K):
    matrix = np.zeros((len(group_order), K), dtype=float)
    for group_index, group in enumerate(group_order):
        subset = summary[summary["group"] == group].set_index("k")
        matrix[group_index] = subset.loc[range(K), value_col].to_numpy(dtype=float)
    return matrix


def plot_stacked_bar(
    values,
    group_order,
    colors,
    path,
    title,
    ylabel,
    dpi,
    percent=False,
):
    values = np.asarray(values, dtype=float)
    if percent:
        values = values / np.maximum(values.sum(axis=1, keepdims=True), EPS)
        plot_values = values * 100.0
    else:
        plot_values = values

    fig, ax = plt.subplots(figsize=(max(10, 1.15 * len(group_order) + 3), 6.5))
    x = np.arange(len(group_order))
    bottom = np.zeros(len(group_order), dtype=float)
    for k in range(plot_values.shape[1]):
        ax.bar(
            x,
            plot_values[:, k],
            bottom=bottom,
            color=colors[k],
            label=f"k{k}",
            width=0.78,
        )
        bottom += plot_values[:, k]
    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=55, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="component", loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_violin_grid(values, labels, group_order, colors, path, title, ylabel, dpi, share_mode):
    K = values.shape[1]
    ncols = 3
    nrows = int(math.ceil(len(group_order) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 4.2 * nrows), squeeze=False
    )
    for group_index, group in enumerate(group_order):
        row, col = divmod(group_index, ncols)
        ax = axes[row, col]
        mask = labels == group
        distributions = [values[mask, k] for k in range(K)]
        try:
            violin = ax.violinplot(
                distributions,
                positions=np.arange(K),
                widths=0.82,
                showmeans=False,
                showmedians=True,
                showextrema=True,
            )
            for k, body in enumerate(violin["bodies"]):
                body.set_facecolor(colors[k])
                body.set_edgecolor(colors[k])
                body.set_alpha(0.7)
        except Exception as exc:
            print(f"[plot] violin fallback to boxplot for {group}: {type(exc).__name__}: {exc}")
            boxes = ax.boxplot(distributions, positions=np.arange(K), widths=0.65, patch_artist=True)
            for k, box in enumerate(boxes["boxes"]):
                box.set_facecolor(colors[k])
                box.set_alpha(0.7)
        ax.set_xticks(np.arange(K))
        ax.set_xticklabels([f"k{k}" for k in range(K)])
        ax.set_title(f"{group} (n={int(mask.sum())})", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        if share_mode:
            ax.set_ylim(0, 1)
    for panel_index in range(len(group_order), nrows * ncols):
        row, col = divmod(panel_index, ncols)
        axes[row, col].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_cell_table(table, output_dir):
    parquet_path = os.path.join(output_dir, "lincomb_cell_coefficients.parquet")
    try:
        table.to_parquet(parquet_path, index=False)
        print(f"[save] cell table: {parquet_path}")
        return parquet_path, "parquet"
    except Exception as exc:
        csv_path = os.path.join(output_dir, "lincomb_cell_coefficients.csv")
        print(f"[save] parquet unavailable ({type(exc).__name__}: {exc}); fallback to CSV")
        table.to_csv(csv_path, index=False)
        print(f"[save] cell table: {csv_path}")
        return csv_path, "csv"


def build_argparser():
    parser = argparse.ArgumentParser(
        description="LinComb a_k/contribution と Superclass の対応を集計・可視化する。"
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--data_dir", default="", help="未指定なら config のdata_dir")
    parser.add_argument("--edge_tsv_path", default="", help="未指定なら config の edge_tsv_path")
    parser.add_argument("--group_col", default="Superclass")
    parser.add_argument("--t", type=float, default=0.0)
    parser.add_argument("--max_cells", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--fig_dpi", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--abs_mode", choices=("abs",), default="abs")
    parser.add_argument(
        "--top_groups",
        default="",
        help="数値なら細胞数上位N group、カンマ区切りなら指定groupのみ。",
    )
    parser.add_argument("--skip_violin", action="store_true")
    parser.add_argument("--skip_contribution", action="store_true")
    return parser


@torch.no_grad()
def main():
    args = build_argparser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("[ERROR] --batch_size must be > 0")
    if args.fig_dpi <= 0:
        raise SystemExit("[ERROR] --fig_dpi must be > 0")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(f"--run_dir が存在しません: {run_dir}")
    config_path = find_config(run_dir, args.config)
    model_path = find_checkpoint(run_dir, args.model_path)
    raw_config = R.load_config(config_path)
    data_dir = _resolve_input_path(
        args.data_dir, raw_config.get("data_dir"), config_path, "data_dir"
    )
    edge_tsv_path = _resolve_input_path(
        args.edge_tsv_path,
        raw_config.get("edge_tsv_path"),
        config_path,
        "edge_tsv_path",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_base = os.path.join(os.path.abspath(args.output_dir), ROLE)
    else:
        output_base = os.path.join(run_dir, "viz", ROLE)
    output_dir = os.path.join(output_base, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    command = " ".join(
        shlex.quote(value)
        for value in [sys.executable, os.path.abspath(__file__), *sys.argv[1:]]
    )
    args_payload = dict(vars(args))
    args_payload.update({"timestamp": timestamp, "command": command, "cwd": os.getcwd()})
    _write_json(os.path.join(output_dir, "args.json"), args_payload)

    resolved = {
        "run_dir": run_dir,
        "model_path": model_path,
        "config": config_path,
        "data_dir": data_dir,
        "edge_tsv_path": edge_tsv_path,
        "output_dir": output_dir,
        "sampling_seed": int(args.seed),
    }
    _write_json(os.path.join(output_dir, "resolved_paths.json"), resolved)

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

    print(f"[lincomb-coeff] run_dir    : {run_dir}")
    print(f"[lincomb-coeff] config     : {config_path}")
    print(f"[lincomb-coeff] checkpoint : {model_path}")
    print(f"[lincomb-coeff] data_dir   : {data_dir}")
    print(f"[lincomb-coeff] edge_tsv   : {edge_tsv_path}")
    print(f"[lincomb-coeff] output_dir : {output_dir}")
    print(f"[lincomb-coeff] device     : {device}")

    cfg = R.normalize_config(raw_config)
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
    print(f"[lincomb-coeff] K          : {K}")

    adata = sc.read_h5ad(data_dir)
    if args.group_col not in adata.obs.columns:
        raise SystemExit(
            f"[ERROR] group column '{args.group_col}' が adata.obs にありません。"
        )
    n_cells_before = int(adata.n_obs)
    original_indices = np.arange(adata.n_obs, dtype=int)
    raw_labels = _string_labels(adata.obs[args.group_col])
    group_order = _ordered_groups(adata.obs[args.group_col])
    group_order = _select_top_groups(raw_labels, group_order, args.top_groups)

    group_mask = np.isin(raw_labels, group_order)
    adata = adata[group_mask].copy()
    original_indices = original_indices[group_mask]
    labels = raw_labels[group_mask]
    n_cells_after_group_filter = int(adata.n_obs)
    selected = stratified_sample_indices(
        labels, group_order, args.max_cells, args.seed
    )
    if len(selected) < adata.n_obs:
        adata = adata[selected].copy()
        original_indices = original_indices[selected]
        labels = labels[selected]
    n_cells_after = int(adata.n_obs)
    group_order = [group for group in group_order if np.any(labels == group)]
    print(
        f"[lincomb-coeff] cells      : {n_cells_after} "
        f"(full={n_cells_before}, after_group_filter={n_cells_after_group_filter})"
    )
    print(
        "[lincomb-coeff] groups     : "
        + ", ".join(f"{g}={int(np.sum(labels == g))}" for g in group_order)
    )

    X = R.to_dense(adata.X).astype(np.float32, copy=False)
    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    a, abs_a, coeff_share, contribution, contribution_share = compute_cell_quantities(
        ode,
        X,
        args.t,
        device,
        args.batch_size,
        compute_contribution=not args.skip_contribution,
    )

    summary = build_summary_table(
        labels,
        group_order,
        args.group_col,
        args.t,
        a,
        abs_a,
        coeff_share,
        contribution,
        contribution_share,
    )
    summary_path = os.path.join(output_dir, "lincomb_superclass_coefficients.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[save] summary: {summary_path}")

    cell_table = build_cell_table(
        original_indices,
        labels,
        args.t,
        a,
        abs_a,
        coeff_share,
        contribution,
        contribution_share,
    )
    cell_table_path, cell_table_format = save_cell_table(cell_table, output_dir)

    colors = _component_colors(K)
    created_figures = []
    coeff_share_matrix = _matrix_from_summary(
        summary, group_order, "coeff_share_mean", K
    )
    coeff_mass_matrix = _matrix_from_summary(
        summary, group_order, "coeff_mass_mean", K
    )
    figure_specs = [
        (
            "coeff_share_100pct_stacked_bar.png",
            coeff_share_matrix,
            f"LinComb coefficient share by Superclass, t={args.t:g}",
            "mean coefficient share",
            True,
        ),
        (
            "coeff_mass_stacked_bar.png",
            coeff_mass_matrix,
            f"LinComb coefficient mass by Superclass, t={args.t:g}",
            "mean |a_k|",
            False,
        ),
    ]
    if contribution is not None:
        figure_specs.extend(
            [
                (
                    "contrib_share_100pct_stacked_bar.png",
                    _matrix_from_summary(summary, group_order, "contrib_share_mean", K),
                    f"LinComb contribution share by Superclass, t={args.t:g}",
                    "mean contribution share",
                    True,
                ),
                (
                    "contrib_mass_stacked_bar.png",
                    _matrix_from_summary(summary, group_order, "contrib_mass_mean", K),
                    f"LinComb contribution mass by Superclass, t={args.t:g}",
                    "mean |a_k| * ||softplus(W_k x + b_k)||",
                    False,
                ),
            ]
        )
    for filename, values, title, ylabel, percent in figure_specs:
        plot_stacked_bar(
            values,
            group_order,
            colors,
            os.path.join(output_dir, filename),
            title,
            ylabel,
            args.fig_dpi,
            percent=percent,
        )
        created_figures.append(filename)

    if not args.skip_violin:
        plot_violin_grid(
            coeff_share,
            labels,
            group_order,
            colors,
            os.path.join(output_dir, "coeff_share_violin_by_superclass.png"),
            f"LinComb coefficient share distributions, t={args.t:g}",
            "|a_k| / sum_j |a_j|",
            args.fig_dpi,
            share_mode=True,
        )
        created_figures.append("coeff_share_violin_by_superclass.png")
        plot_violin_grid(
            abs_a,
            labels,
            group_order,
            colors,
            os.path.join(output_dir, "coeff_mass_violin_by_superclass.png"),
            f"LinComb absolute coefficient distributions, t={args.t:g}",
            "|a_k|",
            args.fig_dpi,
            share_mode=False,
        )
        created_figures.append("coeff_mass_violin_by_superclass.png")

    resolved.update(
        {
            "device": str(device),
            "K": int(K),
            "t": float(args.t),
            "group_col": args.group_col,
            "group_order": group_order,
            "n_cells_before": n_cells_before,
            "n_cells_after_group_filter": n_cells_after_group_filter,
            "n_cells_after": n_cells_after,
            "sampling_seed": int(args.seed),
            "cell_table": cell_table_path,
            "cell_table_format": cell_table_format,
            "created_figures": created_figures,
        }
    )
    _write_json(os.path.join(output_dir, "resolved_paths.json"), resolved)

    print("\n==================== DONE ====================")
    print(f"n_cells      : {n_cells_after}")
    print(f"groups       : {len(group_order)}")
    print(f"K            : {K}")
    print(f"summary      : {summary_path}")
    print(f"cell table   : {cell_table_path}")
    print(f"figures      : {', '.join(created_figures)}")
    print(f"output_dir   : {output_dir}")


if __name__ == "__main__":
    main()
