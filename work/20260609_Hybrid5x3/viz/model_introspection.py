#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_introspection.py  (20260609 可視化スイート)
=================================================

**有効作用素 W(x,t) の introspection** を担当する補助モジュール（plot_params.py から使用）。

2 階層の可視化のうち「effective operator visualization」側:
  - effective_W(model, x, t)        : ode_model.compute_W / 静的 W から W(x,t) を numpy で取得
  - w_metrics(Wm, mask, ...)        : モデル間比較用の共通 metric（off_on_ratio / sparsity /
                                       hoyer / effective_rank / top1pct_mass など）
  - decompose_W(model, x, t)        : branch の decompose_W（無ければ compute_W から合成）
  - plot_branch_specific_across_steps(...) : LowRank/LinComb/MatSum/LoRA の構造可視化（checkpoint
                                       step を横軸にする step-axis 版。find_viz_checkpoints の全 ckpt 使用）

設計方針:
  - ODE モデル本体には matplotlib を入れない（model class は tensor/dict のみ返す）。
    作図・CSV はこのモジュール側で行う。各 branch-specific 図には対応 CSV を必ず保存。
  - すべて numerical-safe（division by zero / empty mask / all-zero W / NaN / SVD 失敗を NaN 化）。
    metric が落ちても可視化全体は止めない。
  - W は基本 `W.mean(axis=0)`（少数 cell の平均有効作用素）で metric を計算（CSV に n_cells_used）。
  - 係数 a_k は raw signed の平均（t 方向 mean ± std を line + band）。abs も CSV に併記。
"""

import csv
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # viz/（_restore を lazy import するため）


# ============================================================
# effective W の取得
# ============================================================
@torch.no_grad()
def effective_W(model, x, t):
    """ode_model から有効作用素 W(x,t) を (B,d,d) numpy で返す。

    返り値: (W ndarray or None, is_exact bool)
      - field 系（lowrank/lincomb/matsum/lora）: ode.compute_W(x,t) を使用（lincomb は proxy）
      - GeneODE                               : 静的 W（soft=False なら W*mask）を (1,d,d) に
      - plain / compute_W 無し                : (None, True)
    """
    ode = getattr(model, "ode_model", None)
    if ode is None:
        return None, True
    is_exact = bool(getattr(ode, "W_IS_EXACT", True))
    if hasattr(ode, "compute_W"):
        try:
            W = ode.compute_W(x, t)
            return W.detach().cpu().numpy().astype(np.float64), is_exact
        except Exception as e:  # noqa: BLE001
            print(f"[introspect] compute_W failed ({type(e).__name__}: {e})")
            return None, is_exact
    if getattr(ode, "W", None) is not None:
        W = ode.W
        if getattr(ode, "soft", True) is False and getattr(ode, "mask", None) is not None:
            W = ode.W * ode.mask
        return W.detach().cpu().numpy().astype(np.float64)[None], is_exact
    return None, is_exact


def model_mask(model):
    """ode_model.mask を numpy で返す（無ければ None）。"""
    ode = getattr(model, "ode_model", None)
    m = getattr(ode, "mask", None) if ode is not None else None
    if m is None:
        return None
    return m.detach().cpu().numpy()


# ============================================================
# numerical-safe な metric 群（すべて float / NaN を返す）
# ============================================================
def _abs_ravel(W):
    return np.abs(np.asarray(W, dtype=np.float64)).ravel()


def density_gt(W, eps):
    """|W| > eps の割合（density）。"""
    w = np.abs(np.asarray(W, dtype=np.float64))
    return float(np.mean(w > eps)) if w.size else float("nan")


def fraction_lt(W, eps):
    """|W| < eps の割合（near-zero fraction）。"""
    w = np.abs(np.asarray(W, dtype=np.float64))
    return float(np.mean(w < eps)) if w.size else float("nan")


def top_mass_fraction(W, frac=0.01):
    """上位 frac の |W| が全 Σ|W| に占める割合。Σ|W|==0 は NaN。"""
    w = _abs_ravel(W)
    s = w.sum()
    if w.size == 0 or s <= 0:
        return float("nan")
    k = max(1, int(round(frac * w.size)))
    top = np.sort(w)[::-1][:k]
    return float(top.sum() / s)


def hoyer_sparsity(W):
    """Hoyer sparsity: (sqrt(n) - ||w||_1/||w||_2) / (sqrt(n) - 1), w=|W|.ravel()。"""
    w = _abs_ravel(W)
    n = w.size
    if n <= 1:
        return float("nan")
    l1 = w.sum()
    l2 = math.sqrt(float((w ** 2).sum()))
    if l2 <= 0:
        return float("nan")
    sqn = math.sqrt(n)
    return float((sqn - l1 / l2) / (sqn - 1.0))


def effective_rank(Wm, max_svd_dim=512):
    """entropy-based effective rank = exp(-Σ p_i log p_i), p=s/Σs（s=特異値）。

    大行列は row/col を max_svd_dim に間引いて SVD（full_matrices=False）。SVD 失敗は NaN。
    """
    A = np.asarray(Wm, dtype=np.float64)
    if A.ndim > 2:
        A = A.reshape(A.shape[-2], A.shape[-1])
    if A.ndim != 2 or A.size == 0:
        return float("nan")
    d0, d1 = A.shape
    if max(d0, d1) > max_svd_dim:
        ri = np.unique(np.linspace(0, d0 - 1, min(d0, max_svd_dim)).round().astype(int))
        ci = np.unique(np.linspace(0, d1 - 1, min(d1, max_svd_dim)).round().astype(int))
        A = A[np.ix_(ri, ci)]
    if not np.all(np.isfinite(A)):
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        s = np.linalg.svd(A, compute_uv=False, full_matrices=False)
    except Exception:  # noqa: BLE001
        return float("nan")
    s = s[s > 0]
    if s.size == 0:
        return float("nan")
    p = s / s.sum()
    ent = -float(np.sum(p * np.log(p)))
    return float(math.exp(ent))


def w_metrics(Wm, mask=None, eps_list=((1e-4, "1e-4"), (1e-3, "1e-3"), (1e-2, "1e-2")),
              max_svd_dim=512):
    """単一の有効作用素 Wm (d,d) に対する共通 metric 群を dict で返す（NaN-safe）。

    eps_list: [(float, label)]。density_abs_gt_{label} / fraction_abs_lt_{label} を生成。
    """
    Wm = np.asarray(Wm, dtype=np.float64)
    aW = np.abs(Wm)
    out = {"mean_abs_all": float(aW.mean()) if aW.size else float("nan")}

    # on/off mask
    if mask is not None and Wm.shape == np.asarray(mask).shape:
        m = np.asarray(mask)
        on = aW[m == 1]
        off = aW[m == 0]
        mean_on = float(on.mean()) if on.size else float("nan")
        mean_off = float(off.mean()) if off.size else float("nan")
        sum_all = float(aW.sum())
        off_mass = float(off.sum() / sum_all) if sum_all > 0 else float("nan")
    else:
        mean_on = mean_off = off_mass = float("nan")
    out["mean_abs_on"] = mean_on
    out["mean_abs_off"] = mean_off
    out["off_mass_fraction"] = off_mass
    out["off_on_ratio"] = (mean_off / mean_on
                           if (np.isfinite(mean_on) and mean_on > 0
                               and np.isfinite(mean_off)) else float("nan"))

    for eps, lab in eps_list:
        out[f"density_abs_gt_{lab}"] = density_gt(Wm, eps)
        out[f"fraction_abs_lt_{lab}"] = fraction_lt(Wm, eps)

    out["hoyer_sparsity"] = hoyer_sparsity(Wm)
    out["effective_rank"] = effective_rank(Wm, max_svd_dim=max_svd_dim)
    out["top1pct_mass_fraction"] = top_mass_fraction(Wm, frac=0.01)
    return out


# ============================================================
# decompose_W（branch decompose があれば使う / 無ければ合成）
# ============================================================
@torch.no_grad()
def decompose_W(model, x, t):
    """ode_model.decompose_W があれば使い、無ければ compute_W/静的W から最小 dict を合成。"""
    ode = getattr(model, "ode_model", None)
    if ode is None:
        return None
    if hasattr(ode, "decompose_W"):
        try:
            return ode.decompose_W(x, t)
        except Exception as e:  # noqa: BLE001
            print(f"[introspect] decompose_W failed ({type(e).__name__}: {e})")
    W, is_exact = effective_W(model, x, t)
    if W is None:
        return None
    return {"W": torch.from_numpy(W), "is_exact": is_exact}


# ============================================================
# branch-specific 可視化（step 横軸・全て defensive）
# ============================================================
def _to_t_tensor(t, B, device):
    return torch.full((B, 1), float(t), device=device)


_KIND_SHORT = {"model_init": "init", "ema": "ema", "model": "mdl"}


def _ck_axis(ck_list):
    """checkpoint 列 → x 位置(=index) と tick ラベル('kind\\nstep')。

    step が重複（init step0 と ema step0 等）しても index で必ず分離できる。
    """
    xpos = list(range(len(ck_list)))
    ticks = [f"{_KIND_SHORT.get(c['kind'], c['kind'])}\n{c['step']}" for c in ck_list]
    return xpos, ticks


def plot_k_lines_vs_step(ck_list, series, output_path, ylabel, title, *,
                         yerr=None, dashed=None, legend_labels=None,
                         xlabel="training step (checkpoint)", value_fmt=".4e",
                         eps_note=None, log=print):
    """k ごとの折れ線 vs checkpoint step（共通 helper）。

    - series:        {label: 1d array(len==len(ck_list))}。NaN 可。
    - yerr:          {label: 1d array} があれば fill_between で ±band。
    - dashed:        破線+太線にする label 集合（例 {"W0"}）。
    - legend_labels: {label: 凡例文字列}（無ければ "label final=<sci>" を自動生成）。
    - 凡例は図の外（右）、保存は bbox_inches="tight"。NaN/失敗でも落ちない。
    """
    try:
        xpos, ticks = _ck_axis(ck_list)
        dashed = dashed or set()
        legend_labels = legend_labels or {}
        fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(ck_list) + 3.0), 4.6))
        for lab, y in series.items():
            y = np.asarray(y, dtype=float)
            finite = y[np.isfinite(y)]
            fv = finite[-1] if finite.size else float("nan")
            disp = legend_labels.get(lab, f"{lab} final={fv:{value_fmt}}")
            if lab in dashed:
                line, = ax.plot(xpos, y, label=disp, lw=2.4, ls="--", color="k")
            else:
                line, = ax.plot(xpos, y, label=disp, lw=1.3, marker="o", ms=3)
            if yerr and lab in yerr:
                e = np.asarray(yerr[lab], dtype=float)
                m = np.isfinite(y) & np.isfinite(e)
                if m.any():
                    xa = np.asarray(xpos, dtype=float)
                    ax.fill_between(xa[m], (y - e)[m], (y + e)[m], alpha=0.18,
                                    color=line.get_color())
        ax.set_xticks(xpos); ax.set_xticklabels(ticks, fontsize=7)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        if eps_note:
            ax.text(0.01, 0.99, eps_note, transform=ax.transAxes, fontsize=7, va="top")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        log(f"[branch] plot_k_lines_vs_step failed ({os.path.basename(output_path)}: "
            f"{type(e).__name__}: {e})")
        plt.close("all")


def _write_csv(path, fieldnames, rows, log=print):
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    except Exception as e:  # noqa: BLE001
        log(f"[branch] CSV write failed ({os.path.basename(path)}: {e})")


def _col(step_data, key, k):
    """各 checkpoint の rec[key][k] を 1d array に（欠落は NaN）。"""
    out = []
    for r in step_data:
        v = r.get(key)
        out.append(float(v[k]) if (v is not None and k < len(v)) else float("nan"))
    return np.asarray(out, dtype=float)


def _scol(step_data, key):
    """各 checkpoint の scalar rec[key] を 1d array に（欠落は NaN）。"""
    return np.asarray([float(r[key]) if (r.get(key) is not None) else float("nan")
                       for r in step_data], dtype=float)


# ---- checkpoint ごとの branch metric 収集（matplotlib 非依存）----
@torch.no_grad()
def _branch_step_record(ode, x, t_grid, device, eps):
    """1 checkpoint 分の branch metric を numpy で収集して dict 返す。"""
    mtype = getattr(ode, "model_type", "")
    rec = {"mtype": mtype, "n_cells": int(x.shape[0]), "n_t": int(len(t_grid))}

    # 係数 a_k(x,t): signed/abs を cell 平均 → t 軸に積む
    signed, absol = [], []
    for t in t_grid:
        d = ode.decompose_W(x, _to_t_tensor(int(t), x.shape[0], device))
        a = d.get("a")
        if a is None:
            signed = []; break
        signed.append(a.mean(dim=0).detach().cpu().numpy())        # (K,) signed cell-mean
        absol.append(a.abs().mean(dim=0).detach().cpu().numpy())   # (K,) abs cell-mean
    if signed:
        S = np.asarray(signed); A = np.asarray(absol)               # (n_t, K)
        rec["mean_a"] = S.mean(axis=0); rec["std_t_a"] = S.std(axis=0)
        rec["mean_abs_a"] = A.mean(axis=0); rec["std_t_abs_a"] = A.std(axis=0)
        rec["abs_cell_mean_per_t"] = A                              # lora component norm 用

    if mtype == "lincomb":
        contrib = []
        for t in t_grid:
            d = ode.decompose_W(x, _to_t_tensor(int(t), x.shape[0], device))
            a = d["a"]                                              # (B,K)
            fx = F.softplus(torch.einsum("kij,bj->bki", ode.expert_W, x) + ode.expert_b)  # (B,K,d)
            out_norm = fx.norm(p=2, dim=-1)                         # (B,K)
            contrib.append((a.abs() * out_norm).mean(dim=0).detach().cpu().numpy())       # (K,)
        C = np.asarray(contrib)
        rec["mean_contribution"] = C.mean(axis=0); rec["std_t_contribution"] = C.std(axis=0)
        eW = ode.expert_W.detach().cpu().numpy()
        rec["density_expertW"] = np.array([density_gt(eW[k], eps) for k in range(eW.shape[0])])
    elif mtype == "matsum":
        A = ode.A.detach().cpu().numpy()
        rec["density_A"] = np.array([density_gt(A[k], eps) for k in range(A.shape[0])])
    elif mtype == "lora":
        Delta = torch.einsum("kir,kjr->kij", ode.U, ode.V).detach().cpu().numpy()  # (K,d,d)
        rec["delta_norm"] = np.linalg.norm(Delta.reshape(Delta.shape[0], -1), axis=1)  # (K,) ||Δ_k||_F
        W0 = ode.W0.detach().cpu().numpy()
        rec["W0_norm"] = float(np.linalg.norm(W0))
        rec["density_Delta"] = np.array([density_gt(Delta[k], eps) for k in range(Delta.shape[0])])
        rec["density_W0"] = density_gt(W0, eps)
    return rec


# ---- step 横軸 emit（branch 別）----
def _emit_coeff_vs_step(step_data, ck_list, mtype, output_dir, log):
    recs = [r for r in step_data if r.get("mean_a") is not None]
    if not recs:
        return
    K = max(len(r["mean_a"]) for r in recs)
    proxy = " [PROXY]" if mtype == "lincomb" else ""
    series, yerr, legend = {}, {}, {}
    for k in range(K):
        y = _col(step_data, "mean_a", k); e = _col(step_data, "std_t_a", k)
        series[f"k{k}"] = y; yerr[f"k{k}"] = e
        fy = y[np.isfinite(y)]; fe = e[np.isfinite(e)]
        legend[f"k{k}"] = (f"k{k} μ={(fy[-1] if fy.size else float('nan')):.4e} "
                           f"σ_t={(fe[-1] if fe.size else float('nan')):.4e}")
    for nm in (f"{mtype}_coeff_vs_step.png", f"{mtype}_coeff_t_vs_k.png"):
        plot_k_lines_vs_step(
            ck_list, series, os.path.join(output_dir, nm),
            ylabel="mean_t( mean_cells( a_k ) )  [signed]",
            title=f"{mtype}{proxy}: signed a_k vs training step (line=mean_t, band=±std_t)",
            yerr=yerr, legend_labels=legend, log=log)
    rows = []
    for r in step_data:
        if r.get("mean_a") is None:
            continue
        ck = r["ck"]
        for k in range(len(r["mean_a"])):
            rows.append({"branch": mtype, "checkpoint_label": ck["label"],
                         "checkpoint_step": ck["step"], "checkpoint_kind": ck["kind"], "k": k,
                         "mean_a": float(r["mean_a"][k]), "std_t_a": float(r["std_t_a"][k]),
                         "mean_abs_a": float(r["mean_abs_a"][k]),
                         "std_t_abs_a": float(r["std_t_abs_a"][k]),
                         "n_cells": r["n_cells"], "n_t": r["n_t"]})
    _write_csv(os.path.join(output_dir, "branch_coeff_vs_step.csv"),
               ["branch", "checkpoint_label", "checkpoint_step", "checkpoint_kind", "k",
                "mean_a", "std_t_a", "mean_abs_a", "std_t_abs_a", "n_cells", "n_t"], rows, log)
    log(f"[branch] {mtype}: coeff_vs_step (signed a_k, ±std_t) + branch_coeff_vs_step.csv saved{proxy}")


def _density_plot(step_data, ck_list, key, fname, label_metric, eps_label, output_dir, log,
                  extra_scalar_key=None, extra_name=None):
    recs = [r for r in step_data if r.get(key) is not None]
    if not recs:
        return
    K = max(len(r[key]) for r in recs)
    series, legend, dashed = {}, {}, set()
    for k in range(K):
        y = _col(step_data, key, k); series[f"k{k}"] = y
        fy = y[np.isfinite(y)]
        legend[f"k{k}"] = f"k{k} final={(fy[-1] if fy.size else float('nan')):.4e}"
    if extra_scalar_key is not None:
        y = _scol(step_data, extra_scalar_key); series[extra_name] = y; dashed.add(extra_name)
        fy = y[np.isfinite(y)]
        legend[extra_name] = f"{extra_name} final={(fy[-1] if fy.size else float('nan')):.4e}"
    plot_k_lines_vs_step(
        ck_list, series, os.path.join(output_dir, fname),
        ylabel=f"density(|{label_metric}| > {eps_label})",
        title=f"density |{label_metric}| > {eps_label} vs training step",
        legend_labels=legend, dashed=dashed, eps_note=f"eps = {eps_label}", log=log)


def _emit_lincomb_contrib(step_data, ck_list, output_dir, log):
    recs = [r for r in step_data if r.get("mean_contribution") is not None]
    if not recs:
        return
    K = max(len(r["mean_contribution"]) for r in recs)
    series, yerr, legend = {}, {}, {}
    for k in range(K):
        y = _col(step_data, "mean_contribution", k); e = _col(step_data, "std_t_contribution", k)
        series[f"k{k}"] = y; yerr[f"k{k}"] = e
        fy = y[np.isfinite(y)]; fe = e[np.isfinite(e)]
        legend[f"k{k}"] = (f"k{k} μ={(fy[-1] if fy.size else float('nan')):.4e} "
                           f"σ_t={(fe[-1] if fe.size else float('nan')):.4e}")
    for nm in ("lincomb_expert_contribution_vs_step.png", "lincomb_expert_contribution_t_vs_k.png"):
        plot_k_lines_vs_step(
            ck_list, series, os.path.join(output_dir, nm),
            ylabel="mean_t contribution_k",
            title="lincomb [PROXY]: expert contribution |a_k|·||softplus(W_k x+b_k)|| vs training step",
            yerr=yerr, legend_labels=legend, log=log)
    rows = []
    for r in step_data:
        if r.get("mean_contribution") is None:
            continue
        ck = r["ck"]
        for k in range(len(r["mean_contribution"])):
            rows.append({"checkpoint_label": ck["label"], "checkpoint_step": ck["step"],
                         "checkpoint_kind": ck["kind"], "k": k,
                         "mean_contribution": float(r["mean_contribution"][k]),
                         "std_t_contribution": float(r["std_t_contribution"][k]),
                         "n_cells": r["n_cells"], "n_t": r["n_t"]})
    _write_csv(os.path.join(output_dir, "lincomb_expert_contribution_vs_step.csv"),
               ["checkpoint_label", "checkpoint_step", "checkpoint_kind", "k",
                "mean_contribution", "std_t_contribution", "n_cells", "n_t"], rows, log)
    log("[branch] lincomb: expert_contribution_vs_step (+std_t) + CSV saved [PROXY]")


def _emit_density_csv(step_data, key, csv_name, eps_label, output_dir, log):
    rows = []
    for r in step_data:
        if r.get(key) is None:
            continue
        ck = r["ck"]
        for k in range(len(r[key])):
            rows.append({"checkpoint_label": ck["label"], "checkpoint_step": ck["step"],
                         "checkpoint_kind": ck["kind"], "k": k,
                         "density_abs_gt_eps": float(r[key][k]), "eps": eps_label})
    _write_csv(os.path.join(output_dir, csv_name),
               ["checkpoint_label", "checkpoint_step", "checkpoint_kind", "k",
                "density_abs_gt_eps", "eps"], rows, log)


def _emit_lincomb_expertW_sparsity(step_data, ck_list, eps_label, output_dir, log):
    _density_plot(step_data, ck_list, "density_expertW", "lincomb_expertW_k_sparsity.png",
                  "expert_W_k", eps_label, output_dir, log)
    _emit_density_csv(step_data, "density_expertW", "lincomb_expertW_k_sparsity_vs_step.csv",
                      eps_label, output_dir, log)
    log(f"[branch] lincomb: expertW_k_sparsity step-axis (|expert_W_k|>{eps_label}) + CSV saved")


def _emit_matsum_sparsity(step_data, ck_list, eps_label, output_dir, log):
    _density_plot(step_data, ck_list, "density_A", "matsum_Ak_sparsity.png",
                  "A_k", eps_label, output_dir, log)
    _emit_density_csv(step_data, "density_A", "matsum_Ak_sparsity_vs_step.csv",
                      eps_label, output_dir, log)
    log(f"[branch] matsum: Ak_sparsity step-axis (|A_k|>{eps_label}) + CSV saved")


def _emit_lora_component_norm(step_data, ck_list, output_dir, log):
    for r in step_data:
        if r.get("abs_cell_mean_per_t") is None or r.get("delta_norm") is None:
            r["_comp_mean"] = None; r["_comp_std"] = None; continue
        comp = r["abs_cell_mean_per_t"] * r["delta_norm"][None, :]   # (n_t,K)=mean_cells(|a_k|)*||Δ_k||_F
        r["_comp_mean"] = comp.mean(axis=0); r["_comp_std"] = comp.std(axis=0)
    recs = [r for r in step_data if r.get("_comp_mean") is not None]
    if not recs:
        return
    K = max(len(r["_comp_mean"]) for r in recs)
    series, yerr, legend, dashed = {}, {}, {}, set()
    for k in range(K):
        y = _col(step_data, "_comp_mean", k); e = _col(step_data, "_comp_std", k)
        series[f"k{k}"] = y; yerr[f"k{k}"] = e
        fy = y[np.isfinite(y)]
        legend[f"k{k}"] = f"k{k} final={(fy[-1] if fy.size else float('nan')):.4e}"
    w0 = _scol(step_data, "W0_norm"); series["W0"] = w0; dashed.add("W0")
    fw = w0[np.isfinite(w0)]
    legend["W0"] = f"||W0||_F final={(fw[-1] if fw.size else float('nan')):.4e}"
    for nm in ("lora_delta_component_norm_vs_step.png", "lora_delta_base_ratio_vs_t.png"):
        plot_k_lines_vs_step(
            ck_list, series, os.path.join(output_dir, nm),
            ylabel="mean_t ||a_k Δ_k||_F",
            title="lora: component ||a_k Δ_k||_F (per k) and ||W0||_F vs training step",
            yerr=yerr, legend_labels=legend, dashed=dashed, log=log)
    rows = []
    for r in step_data:
        if r.get("_comp_mean") is None:
            continue
        ck = r["ck"]
        for k in range(len(r["_comp_mean"])):
            rows.append({"checkpoint_label": ck["label"], "checkpoint_step": ck["step"],
                         "checkpoint_kind": ck["kind"], "k": k,
                         "mean_component_norm": float(r["_comp_mean"][k]),
                         "std_t_component_norm": float(r["_comp_std"][k]),
                         "W0_norm": float(r["W0_norm"]), "n_cells": r["n_cells"], "n_t": r["n_t"]})
    _write_csv(os.path.join(output_dir, "lora_delta_component_norm_vs_step.csv"),
               ["checkpoint_label", "checkpoint_step", "checkpoint_kind", "k",
                "mean_component_norm", "std_t_component_norm", "W0_norm", "n_cells", "n_t"], rows, log)
    log("[branch] lora: delta_component_norm_vs_step (per-k ||a_kΔ_k||_F + ||W0||_F) + CSV saved")


def _emit_lora_density(step_data, ck_list, eps_label, output_dir, log):
    _density_plot(step_data, ck_list, "density_Delta", "lora_Delta_k_metrics.png",
                  "Δ_k", eps_label, output_dir, log,
                  extra_scalar_key="density_W0", extra_name="W0")
    rows = []
    for r in step_data:
        ck = r["ck"]
        if r.get("density_Delta") is not None:
            for k in range(len(r["density_Delta"])):
                rows.append({"checkpoint_label": ck["label"], "checkpoint_step": ck["step"],
                             "checkpoint_kind": ck["kind"], "component": "Delta", "k": k,
                             "density_abs_gt_eps": float(r["density_Delta"][k]), "eps": eps_label})
        if r.get("density_W0") is not None:
            rows.append({"checkpoint_label": ck["label"], "checkpoint_step": ck["step"],
                         "checkpoint_kind": ck["kind"], "component": "W0", "k": "",
                         "density_abs_gt_eps": float(r["density_W0"]), "eps": eps_label})
    _write_csv(os.path.join(output_dir, "lora_Delta_k_metrics_vs_step.csv"),
               ["checkpoint_label", "checkpoint_step", "checkpoint_kind", "component", "k",
                "density_abs_gt_eps", "eps"], rows, log)
    log(f"[branch] lora: Delta_k_metrics step-axis (|Δ_k|>{eps_label}, +W0) + CSV saved")


# ---- vs diffusion-t / 単一 checkpoint（最終 ckpt のモデルで描く）----
@torch.no_grad()
def _emit_lowrank_vs_t(ode, x, t_grid, output_dir, log):
    try:
        sv_rows, means = [], []
        for t in t_grid:
            d = ode.decompose_W(x, _to_t_tensor(int(t), x.shape[0], x.device))
            U, V, W = d["U"], d["V"], d["W"]
            Wm = W.mean(dim=0).detach().cpu().numpy()
            s = np.linalg.svd(np.nan_to_num(Wm), compute_uv=False, full_matrices=False)
            sv_rows.append(s[:min(20, s.size)])
            means.append([float(U.mean()), float(V.mean()), float(W.mean()),
                          float(U.abs().mean()), float(V.abs().mean()), float(W.abs().mean())])
        # singular values（凡例は図の外）
        Kc = max(len(r) for r in sv_rows)
        sv = np.full((len(t_grid), Kc), np.nan)
        for i, r in enumerate(sv_rows):
            sv[i, :len(r)] = r
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        for j in range(min(Kc, 8)):
            ax.plot(t_grid, sv[:, j], marker="o", ms=3, label=f"σ{j}")
        ax.set_xlabel("diffusion t"); ax.set_ylabel("singular value")
        ax.set_title("lowrank: top singular values of mean W(x,t) vs t")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        fig.savefig(os.path.join(output_dir, "lowrank_singular_values_vs_t.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        # mean(U)/mean(V)/mean(W=UVᵀ)（norm ではなく mean）
        M = np.asarray(means)
        fig, ax = plt.subplots(figsize=(7.6, 4.3))
        ax.plot(t_grid, M[:, 0], marker="o", ms=3, label="mean(U)")
        ax.plot(t_grid, M[:, 1], marker="s", ms=3, label="mean(V)")
        ax.plot(t_grid, M[:, 2], marker="^", ms=3, label="mean(W=UVᵀ)")
        ax.set_xlabel("diffusion t"); ax.set_ylabel("mean value (signed)")
        ax.set_title("lowrank: mean(U), mean(V), mean(W=UVᵀ) vs t")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        for nm in ("lowrank_UV_mean_vs_t.png", "lowrank_UV_norm_vs_t.png"):
            fig.savefig(os.path.join(output_dir, nm), dpi=150, bbox_inches="tight")
        plt.close(fig)
        rows = [{"t": int(t), "mean_U": M[i, 0], "mean_V": M[i, 1], "mean_W": M[i, 2],
                 "mean_abs_U": M[i, 3], "mean_abs_V": M[i, 4], "mean_abs_W": M[i, 5],
                 "n_cells": int(x.shape[0])} for i, t in enumerate(t_grid)]
        _write_csv(os.path.join(output_dir, "lowrank_UV_mean_vs_t.csv"),
                   ["t", "mean_U", "mean_V", "mean_W", "mean_abs_U", "mean_abs_V",
                    "mean_abs_W", "n_cells"], rows, log)
        log("[branch] lowrank: singular_values_vs_t (legend outside) + UV_mean_vs_t + CSV saved")
    except Exception as e:  # noqa: BLE001
        log(f"[branch] lowrank vs-t failed ({type(e).__name__}: {e})"); plt.close("all")


@torch.no_grad()
def _emit_matsum_pairwise_corr(ode, output_dir, log):
    try:
        A = ode.A.detach().cpu().numpy()
        K = A.shape[0]
        flat = A.reshape(K, -1)
        with np.errstate(all="ignore"):
            corr = np.corrcoef(flat)
        off = corr[~np.eye(K, dtype=bool)]
        off = off[np.isfinite(off)]
        max_abs = float(np.max(np.abs(off))) if off.size else 1.0
        if max_abs <= 0:
            max_abs = 1.0
        # 下三角のみ表示（対角線・上三角は mask）
        disp = np.full((K, K), np.nan)
        for i in range(K):
            for j in range(i):
                disp[i, j] = corr[i, j]
        masked = np.ma.masked_invalid(disp)   # 対角/上三角は mask → 描画されない（背景=白）
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        im = ax.imshow(masked, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"matsum: A_k pairwise Pearson correlation\n"
                     f"(lower-tri, diag hidden, off-diag scale ±{max_abs:.3f})")
        ax.set_xlabel("k"); ax.set_ylabel("k")
        ax.set_xticks(range(K)); ax.set_yticks(range(K))
        for nm in ("matsum_Ak_pairwise_corr.png", "matsum_Ak_pairwise_cosine.png"):
            fig.savefig(os.path.join(output_dir, nm), dpi=150, bbox_inches="tight")
        plt.close(fig)
        rows = [{"k1": i, "k2": j, "pearson_corr": float(corr[i, j])}
                for i in range(K) for j in range(i)]
        _write_csv(os.path.join(output_dir, "matsum_Ak_pairwise_corr.csv"),
                   ["k1", "k2", "pearson_corr"], rows, log)
        log("[branch] matsum: pairwise Pearson correlation (diag hidden, off-diag scale) + CSV saved")
    except Exception as e:  # noqa: BLE001
        log(f"[branch] matsum pairwise corr failed ({type(e).__name__}: {e})"); plt.close("all")


# ---- driver: find_viz_checkpoints の全 ckpt を使って branch-specific を描く ----
def plot_branch_specific_across_steps(ckpts, cfg, gene_list, diffusion, x_real, edge_tsv_path,
                                      output_dir, device, n_t_grid=8, max_cells=4,
                                      sparsity_eps=1e-3, eps_label=None, log=print):
    """checkpoint step を横軸にした branch-specific 可視化（全 ckpt を build して描く）。

    - 係数 a_k / contribution / density / LoRA component norm は **step 横軸**。
    - LowRank の singular/UV mean と MatSum の pairwise corr は **最終 checkpoint** で描く（vs t / 単一）。
    - plain / decompose_W 非対応(geneode) / ckpt 不足でも落ちず skip。
    """
    if eps_label is None:
        eps_label = f"{sparsity_eps:g}"
    if not ckpts:
        log("[branch] no checkpoints → skip branch-specific"); return
    import _restore as R  # lazy（plot_params が先に sys.path を整える）
    branch = (cfg.get("ode_branch") or cfg.get("model_type") or "").lower()
    T = diffusion.num_timesteps
    t_grid = np.unique(np.linspace(0, T - 1, n_t_grid).round().astype(int))
    n = min(max_cells, x_real.shape[0])
    x = torch.from_numpy(np.asarray(x_real[:n])).float().to(device)

    step_data, last_model = [], None
    for ck in ckpts:
        try:
            model = R.build_model(cfg, ck["path"], gene_list, T, edge_tsv_path, device)
        except Exception as e:  # noqa: BLE001
            log(f"[branch] build_model failed {ck['label']} ({type(e).__name__}: {e}); skip"); continue
        ode = getattr(model, "ode_model", None)
        if ode is None or not hasattr(ode, "decompose_W"):
            del model; continue                          # plain / geneode → branch-specific 無し
        try:
            rec = _branch_step_record(ode, x, t_grid, device, sparsity_eps)
        except Exception as e:  # noqa: BLE001
            log(f"[branch] metric failed {ck['label']} ({type(e).__name__}: {e}); skip"); del model; continue
        rec["ck"] = ck
        step_data.append(rec)
        if last_model is not None:
            del last_model
        last_model = model                               # 最終 ckpt のモデルだけ保持（vs-t / 単一用）

    if not step_data:
        log(f"[branch] no decompose_W branch across checkpoints (branch={branch}) → skip")
        if last_model is not None:
            del last_model
        return
    mtype = step_data[-1]["mtype"]
    ck_list = [r["ck"] for r in step_data]
    log(f"[branch] across-steps: branch={mtype}, {len(ck_list)} checkpoints, eps={eps_label}")

    # step 横軸
    if mtype in ("lincomb", "matsum", "lora"):
        _emit_coeff_vs_step(step_data, ck_list, mtype, output_dir, log)
    if mtype == "lincomb":
        _emit_lincomb_contrib(step_data, ck_list, output_dir, log)
        _emit_lincomb_expertW_sparsity(step_data, ck_list, eps_label, output_dir, log)
    elif mtype == "matsum":
        _emit_matsum_sparsity(step_data, ck_list, eps_label, output_dir, log)
    elif mtype == "lora":
        _emit_lora_component_norm(step_data, ck_list, output_dir, log)
        _emit_lora_density(step_data, ck_list, eps_label, output_dir, log)

    # vs diffusion-t / 単一 checkpoint（最終 ckpt のモデル）
    if last_model is not None:
        final_ode = last_model.ode_model
        if mtype == "lowrank":
            _emit_lowrank_vs_t(final_ode, x, t_grid, output_dir, log)
        elif mtype == "matsum":
            _emit_matsum_pairwise_corr(final_ode, output_dir, log)
        del last_model
