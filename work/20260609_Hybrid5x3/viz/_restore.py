"""
_restore.py  (20260609 可視化スイート共有ヘルパ)
==============================================

3 dir（Hybrid5x3 / HybridNormModes / MathMLPHybrid）の checkpoint を、それぞれの config json
（exp_config / field_config / hybrid_config）を正規化して `build_denoiser` で一本化再構築する。

- normalize_config(): 3 形式の config を共通キーに正規化（ode_branch/hybrid_norm_mode/rank/K/...）
- build_diffusion(): diffusion_steps から GaussianDiffusion を生成（num_timesteps 取得用）
- build_model(): build_denoiser で復元 + checkpoint ロード（hybrid は load_hybrid_state_dict）
- load_gene_list(): h5ad の gene_name（backed read で軽量）
- compute_velocity(): model.ode_model(x, t) で velocity（plain は None）

リモート/ローカル両対応: data/edge は local_paths.resolve_path で解決。
"""

import os
import sys
import json
import glob
import re
import traceback
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
# viz -> Hybrid5x3 -> work -> repo
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for _p in ("/home/suzuki/Projects/scDiffusion", REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import scanpy as sc
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from local_paths import resolve_path  # noqa: E402
from guided_diffusion import dist_util  # noqa: E402
from guided_diffusion.script_util import (  # noqa: E402
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)
from ODE.ode_20260609_hybrid5x3 import build_denoiser  # noqa: E402
from ODE.ode_20260609_mathmlp import load_hybrid_state_dict, clean_state_dict  # noqa: E402


# ------------------------------------------------------------------
def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            return json.load(f)
    return {}


def normalize_config(cfg, **cli_overrides):
    """3 形式の config を共通キーに正規化。CLI で非 None のものは上書き。"""
    def b(v):
        return str(v).lower() in ("true", "1", "yes", "y", "t") if not isinstance(v, bool) else v

    soft = cfg.get("SoftReg", cfg.get("soft", True))
    out = dict(
        ode_branch=(cfg.get("ode_branch") or cfg.get("model_type") or "geneode"),
        hybrid_norm_mode=cfg.get("hybrid_norm_mode", "ratio_reg"),
        rank=int(cfg.get("rank", 16)),
        K=int(cfg.get("K", 8)),
        soft=b(soft),
        use_mask=b(cfg.get("use_mask_reg", soft)),
        use_decay=b(cfg.get("use_decay", True)),
        time_dim=int(cfg.get("time_dim", 64)),
        field_hidden=int(cfg.get("field_hidden", 256)),
        field_dropout=float(cfg.get("field_dropout", 0.0)),
        lowrank_penalty_subsample=int(cfg.get("lowrank_penalty_subsample", 8)),
        ratio_reg_weight=float(cfg.get("ratio_reg_weight", 1.0)),
        ratio_reg_target=float(cfg.get("ratio_reg_target", 1.0)),
        hybrid_scale_init=float(cfg.get("hybrid_scale_init", 1.0)),
        hybrid_scale_eps=float(cfg.get("hybrid_scale_eps", 1e-8)),
        diffusion_steps=int(cfg.get("diffusion_steps", 1000)),
    )
    for k, v in cli_overrides.items():
        if v is not None:
            out[k] = v
    return out


def build_diffusion(diffusion_steps):
    defaults = model_and_diffusion_defaults()
    defaults["diffusion_steps"] = int(diffusion_steps)
    _, diffusion = create_model_and_diffusion(**defaults)
    return diffusion


def load_gene_list(data_dir):
    adata = sc.read_h5ad(resolve_path(data_dir), backed="r")
    return list(adata.var["gene_name"].unique())


def build_model(cfg_norm, model_path, gene_list, diffusion_num_timesteps, edge_tsv_path, device):
    """build_denoiser で復元し checkpoint をロードした eval モデルを返す。"""
    model = build_denoiser(
        ode_branch=cfg_norm["ode_branch"],
        gene_list=gene_list,
        edge_tsv_path=resolve_path(edge_tsv_path),
        timesteps=diffusion_num_timesteps,
        hybrid_norm_mode=cfg_norm["hybrid_norm_mode"],
        rank=cfg_norm["rank"], K=cfg_norm["K"], soft=cfg_norm["soft"],
        ode_reg_lambda=0.0,
        time_dim=cfg_norm["time_dim"], field_hidden=cfg_norm["field_hidden"],
        field_dropout=cfg_norm["field_dropout"],
        lowrank_penalty_subsample=cfg_norm["lowrank_penalty_subsample"],
        use_decay=cfg_norm["use_decay"],
        ratio_reg_weight=cfg_norm["ratio_reg_weight"],
        ratio_reg_target=cfg_norm["ratio_reg_target"],
        hybrid_scale_init=cfg_norm["hybrid_scale_init"],
        hybrid_scale_eps=cfg_norm["hybrid_scale_eps"],
        device=device,
    )
    sd = dist_util.load_state_dict(model_path, map_location="cpu")
    if hasattr(model, "ode_model"):
        load_hybrid_state_dict(model, sd, strict=False, log=print)
    else:  # plain Cell_Unet baseline
        model.load_state_dict(clean_state_dict(sd), strict=False)
    model.to(device).eval()
    return model


def has_ode_branch(model):
    return hasattr(model, "ode_model")


@torch.no_grad()
def compute_velocity(model, X_np, velocity_t, device, batch_size=256):
    """model.ode_model(x, t) で velocity を計算。plain（ode_model 無）は None。"""
    if not has_ode_branch(model):
        return None
    n = X_np.shape[0]
    V = np.zeros_like(X_np, dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = torch.from_numpy(np.asarray(X_np[s:e])).float().to(device)
        tb = torch.full((xb.shape[0], 1), float(velocity_t), device=device)
        vb = model.ode_model(xb, tb).detach().cpu().numpy()
        V[s:e] = np.asarray(vb, dtype=np.float32)
    return V


def to_dense(X):
    return np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)


def sanitize(arr):
    """NaN/Inf を 0 に置換（PCA/scvelo が NaN を拒否するため）。"""
    return np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def prep_scvelo_layers(adata):
    """元 velocity_by_Superclass.py 踏襲: velocity を渡す前に layers['X']=X.copy() を用意し xkey='X' を返す。

    scvelo の xkey は adata.layers[xkey] を参照するため、"X" という名前の layer を作ることで
    spliced/Ms を要求されずに「モデルで計算した velocity」を作図関数に流せる（元コードの解決法）。
    """
    X = to_dense(adata.X)
    if "X" not in adata.layers:
        adata.layers["X"] = X.copy()
    if "velocity_ode" in adata.layers:
        V = np.asarray(adata.layers["velocity_ode"], dtype=np.float32)
        V[~np.isfinite(V)] = 0.0          # 元コードと同じ: 非有限を 0 に
        adata.layers["velocity_ode"] = V
    return "X"


def auto_label_col(adata, preferred=("celltype", "Subclass", "Superclass", "ClassAnn")):
    for c in preferred:
        if c in adata.obs.columns:
            return c
    return None


# ------------------------------------------------------------------
# 学習ステップ軸: checkpoint 探索 + パラメタ抽出
# ------------------------------------------------------------------
def _step_of(path):
    m = re.findall(r"(\d+)\.pt$", os.path.basename(path))
    return int(m[-1]) if m else -1


def find_checkpoints(model_path, max_ckpts=8):
    """model_path のディレクトリから複数 checkpoint を step 順に返す: [(step, path)]。

    raw params の model*.pt を優先（無ければ ema_*.pt）。max_ckpts 超なら等間隔に間引く。
    """
    d = os.path.dirname(model_path)
    cks = sorted(glob.glob(os.path.join(d, "model*.pt")), key=_step_of)
    if not cks:
        cks = sorted(glob.glob(os.path.join(d, "ema_*.pt")), key=_step_of)
    cks = [(_step_of(p), p) for p in cks if _step_of(p) >= 0]
    if len(cks) > max_ckpts:
        idx = np.linspace(0, len(cks) - 1, max_ckpts).round().astype(int)
        cks = [cks[i] for i in sorted(set(idx))]
    return cks


def extract_param_groups(sd, cfg):
    """checkpoint state_dict から branch に応じた学習パラメタ群を抽出。

    返り値: (groups: OrderedDict{label: 1d np.array}, scalars: dict{label: float})
    静的 W 系は per-k × on/off に分割。MLP/cellunet は pool。取りこぼしは 'other' に集約。
    """
    sd = clean_state_dict(sd)
    branch = cfg.get("ode_branch", "geneode")
    mask = sd["ode_model.mask"].detach().cpu().numpy() if "ode_model.mask" in sd else None
    groups = OrderedDict()
    scalars = {}
    used = set()

    def npy(k):
        used.add(k)
        return sd[k].detach().cpu().numpy()

    def split(label, W):
        if mask is not None and W.shape[-2:] == mask.shape:
            groups[f"{label}_on"] = W[..., mask == 1].ravel()
            groups[f"{label}_off"] = W[..., mask == 0].ravel()
        else:
            groups[label] = W.ravel()

    # --- 静的 W 系（per-k × on/off）---
    if branch == "geneode" and "ode_model.W" in sd:
        split("W", npy("ode_model.W"))
    elif branch == "lincomb" and "ode_model.expert_W" in sd:
        eW = npy("ode_model.expert_W")
        for k in range(eW.shape[0]):
            split(f"expertW{k}", eW[k])
    elif branch == "matsum" and "ode_model.A" in sd:
        A = npy("ode_model.A")
        for k in range(A.shape[0]):
            split(f"A{k}", A[k])
    elif branch == "lora":
        if "ode_model.W0" in sd:
            split("W0", npy("ode_model.W0"))
        if "ode_model.U" in sd and "ode_model.V" in sd:
            U, V = npy("ode_model.U"), npy("ode_model.V")
            for k in range(U.shape[0]):
                split(f"Delta{k}", U[k] @ V[k].T)

    # --- gamma / b / expert_b ---
    for key, label in (("ode_model.gamma", "gamma"), ("ode_model.b", "b"),
                       ("ode_model.expert_b", "expert_b")):
        if key in sd:
            groups[label] = npy(key).ravel()

    # --- MLP nets（weight+bias を pool）---
    for net in ("coeff_net", "U_net", "V_net", "time_emb"):
        keys = [k for k in sd if k.startswith(f"ode_model.{net}.")]
        if keys:
            groups[net] = np.concatenate([npy(k).ravel() for k in keys])

    # --- Cell_Unet ---
    ml_keys = [k for k in sd if k.startswith("ml_model.")]
    if ml_keys:
        groups["cellunet"] = np.concatenate([npy(k).ravel() for k in ml_keys])

    # --- scalar（log_scale）---
    if "log_scale" in sd:
        used.add("log_scale")
        scalars["scale"] = float(np.exp(sd["log_scale"].detach().cpu().numpy().ravel()[0]))

    # --- catch-all: 未使用の float テンソル（mask 等は除く）---
    leftover = []
    for k, v in sd.items():
        if k in used or k.endswith(".mask") or k == "ode_model.scale":
            continue
        try:
            a = v.detach().cpu().numpy().ravel()
        except Exception:
            continue
        if a.size > 1 and np.issubdtype(a.dtype, np.floating):
            leftover.append(a)
    if leftover:
        groups["other"] = np.concatenate(leftover)

    return groups, scalars


def embed_umap(adata, n_top=50):
    """小サンプルでも安全に PCA/neighbors/UMAP（n_comps/n_neighbors を obs 数にクランプ）。"""
    ncomp = max(2, min(n_top, adata.n_vars - 1, adata.n_obs - 1))
    sc.tl.pca(adata, svd_solver="arpack", n_comps=ncomp)
    npcs = min(40, adata.obsm["X_pca"].shape[1])
    nnb = max(2, min(15, adata.n_obs - 1))
    sc.pp.neighbors(adata, n_neighbors=nnb, n_pcs=npcs)
    sc.tl.umap(adata)


def scvelo_stream(adata, outdir, prefix, color, n_jobs=32):
    """元 velocity_by_Superclass.py の scvelo 使い方をそのまま踏襲。

    layers['X'] を xkey に、velocity を layers['velocity_ode'] (vkey) に置き、
    velocity_graph→velocity_embedding→stream/arrow/grid を描く（backend='loky', n_jobs=32）。
    成功で True。安全網として例外時のみ UMAP scatter にフォールバック（False）。
    """
    import scvelo as scv
    os.makedirs(outdir, exist_ok=True)
    xkey = prep_scvelo_layers(adata)  # layers['X']=X.copy(), returns 'X'
    palette = adata.uns.get(f"{color}_colors", None) if color else None
    palette = list(palette) if palette is not None else None
    common = dict(basis="umap", vkey="velocity_ode", color=color, palette=palette,
                  show=False, legend_loc="right margin", size=5, alpha=1)
    try:
        scv.tl.velocity_graph(adata, vkey="velocity_ode", xkey=xkey, backend="loky", n_jobs=n_jobs)
        scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity_ode")
        scv.pl.velocity_embedding_stream(adata, **common)
        plt.savefig(os.path.join(outdir, f"{prefix}velocity_stream.png"), dpi=300, bbox_inches="tight"); plt.close()
        scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, **common)
        plt.savefig(os.path.join(outdir, f"{prefix}velocity_arrow.png"), dpi=300, bbox_inches="tight"); plt.close()
        scv.pl.velocity_embedding_grid(adata, **common)
        plt.savefig(os.path.join(outdir, f"{prefix}velocity_grid.png"), dpi=300, bbox_inches="tight"); plt.close()
        return True
    except Exception as e:
        print(f"[velocity] scvelo failed ({type(e).__name__}: {e}). "
              f"（ローカル numpy {np.__version__}+scvelo の非互換時の安全網）→ UMAP scatter にフォールバック。")
        print("[velocity] ---- full traceback (どの scvelo 関数で落ちたか) ----")
        traceback.print_exc()
        print("[velocity] ----------------------------------------------------")
        try:
            sc.pl.umap(adata, color=color, show=False)
            plt.savefig(os.path.join(outdir, f"{prefix}umap_by_group.png"), dpi=200, bbox_inches="tight"); plt.close()
        except Exception as e2:
            print(f"[velocity] fallback UMAP も失敗: {e2}")
        return False
