#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_all_viz.py  (20260609)
==========================

1 つの checkpoint に対し役割別の可視化 4 種を順次実行する（出力は {out}/{role}/ に分離）:
  1) plot_loss.py          → {out}/loss      (loss 曲線)
  2) plot_params.py        → {out}/params    (W 可視化 / パラメタ分布 / t×step ヒートマップ)
  3) eval_model_io.py      → {out}/eval_io   (入出力評価メトリクス: corr/ + alignment_mse/)
  4) plot_velocity_umap.py → {out}/velocity  (Superclass 別 velocity UMAP + real vs gen UMAP)

  python run_all_viz.py --model_path <ckpt> --config <exp_config.json> \
      --sample_path <samples.npz> --loss_path <loss_details.csv> \
      --data_dir <h5ad> --edge_tsv_path <tsv> --output_dir <dir> [--skip loss,velocity]
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))   # Hybrid5x3/（run_paths）
import run_paths  # noqa: E402
PY = sys.executable or "python"


def run(label, cmd):
    print("\n" + "=" * 70)
    print(f"### {label}")
    print("  " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    print(f">>> {label}: exit={rc}")
    return rc


def main():
    a = create_argparser().parse_args()
    skip = set(s.strip() for s in a.skip.split(",") if s.strip())
    # --output_dir 未指定の単体実行時は --model_path から {base}/viz を導出（各 viz スクリプトと整合）
    if a.output_dir:
        out = a.output_dir
    else:
        b = run_paths.infer_run_base(a.model_path) if a.model_path else ""
        out = os.path.join(b, "viz") if b else os.path.join(HERE, "viz_out")
    os.makedirs(out, exist_ok=True)
    common = ["--model_path", a.model_path, "--config", a.config,
              "--data_dir", a.data_dir, "--edge_tsv_path", a.edge_tsv_path]
    rcs = {}

    if "loss" not in skip:
        cmd = [PY, os.path.join(HERE, "plot_loss.py"),
               "--loss_path", a.loss_path, "--output_dir", os.path.join(out, "loss")]
        rcs["plot_loss"] = run("1/4 plot_loss", cmd)

    if "params" not in skip:
        cmd = [PY, os.path.join(HERE, "plot_params.py"), *common,
               "--output_dir", os.path.join(out, "params")]
        rcs["plot_params"] = run("2/4 plot_params", cmd)

    if "eval_io" not in skip:
        cmd = [PY, os.path.join(HERE, "eval_model_io.py"), *common,
               "--output_dir", os.path.join(out, "eval_io"),
               "--weighting", a.weighting, "--num_t_points", str(a.num_t_points),
               "--num_noise_draws", str(a.num_noise_draws), "--max_cells", str(a.analyze_max_cells)]
        rcs["eval_model_io"] = run("3/4 eval_model_io", cmd)

    if "velocity" not in skip:
        cmd = [PY, os.path.join(HERE, "plot_velocity_umap.py"), *common,
               "--sample_path", a.sample_path,   # real vs gen UMAP（凡例なし）も velocity 側で描く
               "--output_dir", os.path.join(out, "velocity"),
               "--group_col", a.group_col, "--velocity_t", str(a.velocity_t),
               "--max_cells", str(a.max_cells)]
        rcs["plot_velocity_umap"] = run("4/4 plot_velocity_umap", cmd)

    print("\n" + "=" * 70)
    print("[run_all_viz] summary:")
    for k, v in rcs.items():
        print(f"  {k}: {'OK' if v == 0 else f'FAIL(exit={v})'}")
    print(f"[run_all_viz] output: {out}")
    if any(v != 0 for v in rcs.values()):
        sys.exit(1)


def create_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--config", default="")
    p.add_argument("--sample_path", default="")
    p.add_argument("--loss_path", default="")
    p.add_argument("--data_dir", default="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad")
    p.add_argument("--edge_tsv_path", default="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv")
    p.add_argument("--output_dir", default="")
    p.add_argument("--skip", default="", help="comma: loss,params,eval_io,velocity")
    p.add_argument("--velocity_t", type=float, default=0.0)
    p.add_argument("--max_cells", type=int, default=3000)
    p.add_argument("--analyze_max_cells", type=int, default=1000)
    p.add_argument("--num_t_points", type=int, default=16)
    p.add_argument("--num_noise_draws", type=int, default=4)
    p.add_argument("--weighting", default="both")
    p.add_argument("--group_col", default="Superclass")
    return p


if __name__ == "__main__":
    main()
