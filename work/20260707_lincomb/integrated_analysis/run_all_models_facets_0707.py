#!/usr/bin/env python3
"""Create only all_models_facets.png for the 20260707 LinComb matrix.

This script is intentionally independent of the training/sampling/viz
pipeline. It reads:

  work_root/runs/<model>/<run_id>/exp_config.json
  work_root/samples/<model>/<run_id>/samples.npz

and writes only:

  output_root/<YYYYMMDD_HHMMSS>/all_models_facets.png
  output_root/<YYYYMMDD_HHMMSS>/selected_runs.json
  output_root/<YYYYMMDD_HHMMSS>/run_config.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import struct
import sys
import zipfile
from datetime import datetime
from pathlib import Path


DEFAULT_DATA_DIR = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"
DEFAULT_ANNOTATION_PRIORITY = ["Superclass", "Subclass", "ClassAnn", "celltype", "final_annotation"]
MODEL_ORDER = [
    "hybrid_reverse_lincomb",
    "lincomb_only_raw",
    "lincomb_softmax_gate",
    "lincomb_sparse_reg",
    "lincomb_entropy_reg",
    "hybrid_ts_soft_lincomb",
]
SAMPLE_KEY = "cell_gen"
GEN_COLOR = "#D62728"

# Keep these aligned with the 20260609 per-model facets defaults.
PM_SIZE_REAL = 3
PM_SIZE_GEN = 14
PM_ALPHA_REAL = 0.3
PM_ALPHA_GEN = 0.9


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_repo_root(start):
    path = Path(start).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "local_paths.py").exists():
            return candidate
    return None


def resolve_local_path(value):
    if not value:
        return value
    path = Path(value)
    if path.exists():
        return str(path)
    repo_root = find_repo_root(Path(__file__).resolve())
    if repo_root and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        from local_paths import resolve_path as repo_resolve_path  # type: ignore

        resolved = repo_resolve_path(str(value))
        if resolved:
            return resolved
    except Exception:
        pass
    if repo_root:
        remote_root = "/home/suzuki/Projects/scDiffusion"
        text = str(value)
        if text.startswith(remote_root):
            candidate = repo_root / os.path.relpath(text, remote_root)
            if candidate.exists():
                return str(candidate)
        if not path.is_absolute():
            candidate = repo_root / text
            if candidate.exists():
                return str(candidate)
    return str(value)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def ensure_unique_output_dir(output_root):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = now_stamp()
    candidate = root / stamp
    if not candidate.exists():
        candidate.mkdir()
        return candidate
    for i in range(1, 100):
        candidate = root / f"{stamp}_{i:02d}"
        if not candidate.exists():
            candidate.mkdir()
            return candidate
    raise RuntimeError(f"could not create a unique output directory under {root}")


def npz_array_shape(npz_path, key=SAMPLE_KEY):
    """Read a .npy header from a .npz without importing numpy."""
    member = f"{key}.npy"
    try:
        with zipfile.ZipFile(npz_path, "r") as archive:
            if member not in archive.namelist():
                return None, f"missing '{key}' in npz"
            with archive.open(member, "r") as handle:
                magic = handle.read(6)
                if magic != b"\x93NUMPY":
                    return None, f"invalid npy member for '{key}'"
                major, minor = struct.unpack("BB", handle.read(2))
                if major == 1:
                    header_len = struct.unpack("<H", handle.read(2))[0]
                elif major in (2, 3):
                    header_len = struct.unpack("<I", handle.read(4))[0]
                else:
                    return None, f"unsupported npy version {major}.{minor}"
                header = handle.read(header_len).decode("latin1")
                info = ast.literal_eval(header)
                return tuple(info.get("shape", ())), None
    except Exception as exc:
        return None, f"cannot inspect npz ({type(exc).__name__}: {exc})"


def manifest_sample_status(manifest):
    statuses = []

    def walk(obj, path):
        if isinstance(obj, dict):
            if "status" in obj and any("sample" in part.lower() or "sampling" in part.lower() for part in path):
                statuses.append(str(obj.get("status", "")).strip().lower())
            for key, value in obj.items():
                walk(value, path + [str(key)])
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                walk(value, path + [str(index)])

    walk(manifest, [])
    statuses = [status for status in statuses if status]
    if not statuses:
        return True, "manifest has no interpretable sample/sampling status"
    bad = {"failed", "failure", "error", "errored", "aborted", "cancelled", "canceled"}
    if any(status in bad or any(word in status for word in bad) for status in statuses):
        return False, f"manifest sample status indicates failure: {statuses}"
    return True, f"manifest sample status accepted: {statuses}"


def inspect_manifest(run_dir):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return True, "", ""
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:
        return False, f"cannot read manifest.json ({type(exc).__name__}: {exc})", str(manifest_path)
    ok, reason = manifest_sample_status(manifest)
    return ok, reason, str(manifest_path)


def discover_latest_runs(work_root):
    work_root = Path(work_root).resolve()
    records = []
    for model in MODEL_ORDER:
        model_records = []
        selected = None
        run_parent = work_root / "runs" / model
        if run_parent.is_dir():
            run_dirs = sorted((p for p in run_parent.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
        else:
            run_dirs = []
        if not run_dirs:
            records.append({
                "model": model,
                "run_id": "",
                "run_dir": "",
                "sample_path": "",
                "sample_manifest_path": "",
                "manifest_path": "",
                "exp_config_path": "",
                "available_generated_cells": 0,
                "generated_dim": None,
                "used_generated_cells": 0,
                "used_real_cells": 0,
                "skip_reason": "no run directory",
                "selected": False,
                "selection_status": "skipped",
                "passed_over": [],
            })
            continue
        for run_dir in run_dirs:
            run_id = run_dir.name
            exp_config = run_dir / "exp_config.json"
            sample_path = work_root / "samples" / model / run_id / "samples.npz"
            sample_manifest = work_root / "samples" / model / run_id / "sample_manifest.json"
            manifest_ok, manifest_reason, manifest_path = inspect_manifest(run_dir)
            reason = ""
            shape = None
            if not exp_config.exists():
                reason = "missing exp_config.json"
            elif not sample_path.exists():
                reason = "missing samples.npz"
            elif not manifest_ok:
                reason = manifest_reason
            else:
                shape, shape_reason = npz_array_shape(sample_path)
                if shape_reason:
                    reason = shape_reason
                elif len(shape) != 2:
                    reason = f"cell_gen is not 2D: shape={shape}"
            selected_ok = not reason
            rec = {
                "model": model,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "sample_path": str(sample_path),
                "sample_manifest_path": str(sample_manifest) if sample_manifest.exists() else "",
                "manifest_path": manifest_path,
                "exp_config_path": str(exp_config) if exp_config.exists() else "",
                "available_generated_cells": int(shape[0]) if shape else 0,
                "generated_dim": int(shape[1]) if shape and len(shape) == 2 else None,
                "used_generated_cells": 0,
                "used_real_cells": 0,
                "skip_reason": reason,
                "selected": selected_ok,
                "selection_status": "selected" if selected_ok else "skipped",
                "passed_over": [],
            }
            model_records.append(rec)
            if not reason:
                selected = rec
                break
        if selected is None:
            last = model_records[-1] if model_records else {}
            records.append({
                "model": model,
                "run_id": "",
                "run_dir": "",
                "sample_path": "",
                "sample_manifest_path": "",
                "manifest_path": "",
                "exp_config_path": "",
                "available_generated_cells": 0,
                "generated_dim": None,
                "used_generated_cells": 0,
                "used_real_cells": 0,
                "skip_reason": last.get("skip_reason", "no valid run"),
                "selected": False,
                "selection_status": "skipped",
                "passed_over": model_records,
            })
        else:
            selected["passed_over"] = [rec for rec in model_records if rec is not selected]
            records.append(selected)
    return records


def print_dry_run(records):
    print("model\tselected_run_id\trun_dir\tsample_path\tgenerated_cells\tskip_reason")
    for rec in records:
        print(
            "\t".join([
                rec["model"],
                rec.get("run_id", ""),
                rec.get("run_dir", ""),
                rec.get("sample_path", ""),
                str(rec.get("available_generated_cells", 0) or 0),
                rec.get("skip_reason", ""),
            ])
        )


def import_runtime_deps():
    import anndata as ad
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc

    return ad, plt, np, sc


def to_dense(np, x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def sanitize(np, x):
    x = np.asarray(x, dtype=np.float32)
    x[~np.isfinite(x)] = 0.0
    return x


def choose_annotation_col(obs_columns):
    for column in DEFAULT_ANNOTATION_PRIORITY:
        if column in obs_columns:
            return column
    return None


def load_real_data(ad, np, data_dir):
    resolved = resolve_local_path(data_dir)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"real data h5ad not found: {data_dir} -> {resolved}")
    real = ad.read_h5ad(resolved)
    annotation_col = choose_annotation_col(list(real.obs.columns))
    real.X = sanitize(np, to_dense(np, real.X))
    real.obs = real.obs.copy()
    real.obs["origin"] = "Real"
    real.obs["config_label"] = "__real__"
    if annotation_col:
        real.obs["annotation"] = real.obs[annotation_col].astype(str).values
    else:
        real.obs["annotation"] = "real"
    real.var_names = [f"g{i}" for i in range(real.n_vars)]
    return real, annotation_col, resolved


def subsample_real(np, real, n, seed):
    if not n or n <= 0 or real.n_obs <= n:
        return real.copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(real.n_obs, size=n, replace=False))
    return real[idx].copy()


def load_generated(ad, np, rec, n, seed, n_vars_expected):
    data = np.load(rec["sample_path"], allow_pickle=False)
    if SAMPLE_KEY not in data:
        raise ValueError(f"missing '{SAMPLE_KEY}' in npz")
    x = sanitize(np, to_dense(np, data[SAMPLE_KEY]))
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"cell_gen is empty or not 2D: shape={x.shape}")
    if x.shape[1] != n_vars_expected:
        raise ValueError(f"gene dimension mismatch: generated {x.shape[1]} != real {n_vars_expected}")
    available = int(x.shape[0])
    if n and n > 0 and x.shape[0] > n:
        rng = np.random.default_rng(seed)
        x = x[np.sort(rng.choice(x.shape[0], size=n, replace=False))]
    gen = ad.AnnData(x)
    gen.var_names = [f"g{i}" for i in range(gen.n_vars)]
    gen.obs["origin"] = "Generated"
    gen.obs["config_label"] = rec["model"]
    gen.obs["annotation"] = "__generated__"
    return gen, available


def build_combined(ad, np, real, gen):
    combined = ad.concat([real, gen], axis=0, join="inner", index_unique=None)
    combined.X = sanitize(np, to_dense(np, combined.X))
    for column in ("origin", "config_label", "annotation"):
        combined.obs[column] = combined.obs[column].astype(str)
    return combined


def compute_umap(np, sc, combined, n_pcs, n_neighbors, min_dist, seed):
    np.random.seed(seed)
    sc.settings.verbosity = 0
    n_pcs = int(min(n_pcs, max(2, min(combined.n_obs - 1, combined.n_vars - 1))))
    sc.pp.pca(combined, n_comps=n_pcs, random_state=seed)
    sc.pp.neighbors(combined, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
    sc.tl.umap(combined, min_dist=min_dist, random_state=seed)
    return combined


def palette(n):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_hex

    if n <= 0:
        return []
    if n <= 10:
        return [to_hex(plt.get_cmap("tab10")(i)) for i in range(n)]
    if n <= 20:
        return [to_hex(plt.get_cmap("tab20")(i)) for i in range(n)]
    cmap = plt.get_cmap("gist_ncar")
    return [to_hex(cmap(x)) for x in __import__("numpy").linspace(0.02, 0.98, n)]


def make_real_color_map(categories):
    cats = sorted(set(map(str, categories)))
    colors = palette(len(cats))
    return {cat: colors[i] for i, cat in enumerate(cats)}


def legend_handles(real_cmap):
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=real_cmap.get(str(label), "#000000"), markeredgecolor="none",
               label=str(label))
        for label in sorted(real_cmap)
    ]
    handles.append(
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=GEN_COLOR, markeredgecolor="none", label="Generated")
    )
    return handles


def xy(combined):
    return combined.obsm["X_umap"][:, 0], combined.obsm["X_umap"][:, 1]


def masks(combined):
    origin = combined.obs["origin"].values
    return origin == "Real", origin == "Generated"


def draw_panel(ax, np, combined, real_cmap, model, gen_count):
    x, y = xy(combined)
    real_mask, gen_mask = masks(combined)
    annotation = combined.obs["annotation"].values
    colors = np.array([real_cmap.get(str(value), "#999999") for value in annotation])
    ax.scatter(
        x[real_mask], y[real_mask],
        s=PM_SIZE_REAL, c=colors[real_mask], alpha=PM_ALPHA_REAL,
        linewidths=0, rasterized=True,
    )
    ax.scatter(
        x[gen_mask], y[gen_mask],
        s=PM_SIZE_GEN, c=GEN_COLOR, alpha=PM_ALPHA_GEN,
        linewidths=0, rasterized=True,
    )
    ax.set_title(f"{model}  (gen n={int(gen_count)})", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_facets(plt, np, panels, real_cmap, output_path):
    n = len(panels)
    ncol = min(3, n)
    nrow = int(math.ceil(n / ncol))
    fig = plt.figure(figsize=(3.4 * ncol, 3.4 * nrow))
    for index, panel in enumerate(panels, 1):
        ax = fig.add_subplot(nrow, ncol, index)
        draw_panel(ax, np, panel["combined"], real_cmap, panel["model"], panel["gen_count"])
    fig.legend(
        handles=legend_handles(real_cmap),
        title="Superclass / Generated(red)",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )
    fig.suptitle(
        "Per-model: real (Superclass colors) + each model's generated (red). "
        "Independent UMAP per panel.",
        y=1.003,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_full(args, records):
    ad, plt, np, sc = import_runtime_deps()
    output_dir = ensure_unique_output_dir(args.output_root)
    created_at = datetime.now().isoformat(timespec="seconds")

    real, annotation_col, data_dir_resolved = load_real_data(ad, np, args.data_dir)
    real_pm = subsample_real(np, real, args.per_model_real_cells, args.seed)
    real_cmap = make_real_color_map(real.obs["annotation"].unique())
    panels = []

    for rec in records:
        if rec.get("skip_reason"):
            continue
        try:
            gen, available = load_generated(
                ad, np, rec, args.per_model_gen_cells, args.seed, real_pm.n_vars
            )
            combined = build_combined(ad, np, real_pm, gen)
            combined = compute_umap(
                np, sc, combined,
                n_pcs=args.n_pcs,
                n_neighbors=args.n_neighbors,
                min_dist=args.min_dist,
                seed=args.seed,
            )
            rec["available_generated_cells"] = available
            rec["used_generated_cells"] = int(gen.n_obs)
            rec["used_real_cells"] = int(real_pm.n_obs)
            panels.append({
                "model": rec["model"],
                "gen_count": int(gen.n_obs),
                "combined": combined,
            })
            print(f"[panel] {rec['model']}: real={real_pm.n_obs} gen={gen.n_obs}")
        except Exception as exc:
            rec["skip_reason"] = f"{type(exc).__name__}: {exc}"
            rec["used_generated_cells"] = 0
            rec["used_real_cells"] = 0
            print(f"[skip] {rec['model']}: {rec['skip_reason']}")

    run_config = vars(args).copy()
    run_config.update({
        "created_at": created_at,
        "work_root": str(Path(args.work_root).resolve()),
        "output_dir": str(output_dir),
        "data_dir_resolved": data_dir_resolved,
        "annotation_col": annotation_col,
        "model_order": MODEL_ORDER,
        "sample_key": SAMPLE_KEY,
        "point_style": {
            "real_size": PM_SIZE_REAL,
            "real_alpha": PM_ALPHA_REAL,
            "generated_size": PM_SIZE_GEN,
            "generated_alpha": PM_ALPHA_GEN,
            "generated_color": GEN_COLOR,
        },
    })

    selected_payload = {
        "created_at": created_at,
        "selection_rule": (
            "for each model, inspect runs/<model>/<run_id> newest by run_id and select "
            "the first run whose exp_config.json and samples/<model>/<run_id>/samples.npz "
            "exist; if manifest.json exists, skip only when sample/sampling status clearly "
            "indicates failure/error"
        ),
        "records": records,
    }
    write_json(output_dir / "run_config.json", run_config)
    write_json(output_dir / "selected_runs.json", selected_payload)

    if not panels:
        raise RuntimeError(f"no usable models; metadata written to {output_dir}")

    image_path = output_dir / "all_models_facets.png"
    plot_facets(plt, np, panels, real_cmap, image_path)
    print(f"[done] image: {image_path}")
    print(f"[done] metadata: {output_dir / 'selected_runs.json'}")
    return output_dir, image_path, records


def build_parser():
    parser = argparse.ArgumentParser(description="Create 20260707 all_models_facets.png only.")
    parser.add_argument("--work_root", default="work/20260707_lincomb")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output_root", default="")
    parser.add_argument("--per_model_real_cells", type=int, default=50000)
    parser.add_argument("--per_model_gen_cells", type=int, default=3000)
    parser.add_argument("--n_pcs", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--min_dist", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    args.work_root = str(Path(args.work_root).resolve())
    if not args.output_root:
        args.output_root = str(Path(args.work_root) / "integrated_analysis" / "outputs")
    records = discover_latest_runs(args.work_root)
    if args.dry_run:
        print_dry_run(records)
        return
    if not any(not rec.get("skip_reason") for rec in records):
        raise SystemExit("ERROR: no usable models; run with --dry_run to inspect skip reasons")
    run_full(args, records)


if __name__ == "__main__":
    main()
