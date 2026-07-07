#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline_hybridnorm.py  (20260609)
======================================

HybridNormModes（GeneODE + ODE_ML_HybridNorm、3 mode）を train→sample→viz で一気通貫する。
viz は ../20260609_Hybrid5x3/viz/run_all_viz.py を共用（config は hybrid_config.json）。

  python run_pipeline_hybridnorm.py --hybrid_norm_mode ratio_reg \
      --lr_anneal_steps 1 --batch_size 8 --diffusion_steps 1000 --num_samples 16 --max_cells 800
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "..", "20260609_Hybrid5x3", "viz", "run_all_viz.py")
sys.path.insert(0, os.path.join(HERE, "..", "20260609_Hybrid5x3"))  # run_paths を共有
import run_paths  # noqa: E402
PY = sys.executable or "python"
DATA = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"
TSV = "/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv"


def run_capture(label, cmd, dry):
    print("\n" + "=" * 70 + f"\n### {label}\n  " + " ".join(cmd))
    if dry:
        return ""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout[-1500:])
    if p.returncode != 0:
        print(f"[pipeline] {label} FAILED (exit={p.returncode})"); sys.exit(p.returncode)
    return p.stdout


def grep(text, key):
    v = ""
    for line in text.splitlines():
        if key in line:
            v = line.split(key, 1)[1].strip().strip("'\"")
    return v


def main():
    # pipeline 未知フラグ（extra）は cell_train_20260609.py にそのまま forward する
    #   → lr / weight_decay / ema_rate / seed / ode_reg_norm / model_name 等の任意 keyword を指定可能。
    a, extra = create_argparser().parse_known_args()
    exp_id = f"hybridnorm__{a.hybrid_norm_mode}"
    out = run_paths.run_base(HERE, exp_id, create=not a.dry_run)   # {HERE}/runs/{model}/{YYYYMMDD}（dry-run は作らない）

    train = [PY, os.path.join(HERE, "cell_train_20260609.py"),
             "--data_dir", a.data_dir, "--edge_tsv_path", a.edge_tsv_path,
             "--hybrid_norm_mode", a.hybrid_norm_mode, "--SoftReg", a.SoftReg,
             "--reverse_coef", a.reverse_coef,
             "--ode_reg_lambda", str(a.ode_reg_lambda),
             "--lr_anneal_steps", str(a.lr_anneal_steps), "--batch_size", str(a.batch_size),
             "--diffusion_steps", str(a.diffusion_steps),
             "--log_interval", str(a.log_interval),
             "--save_interval", str(a.save_interval),
             *extra,  # pipeline 未知フラグ（lr/seed/... 任意の cell_train keyword）を素通し
             "--output_dir", os.path.join(out, "train")]
    t = run_capture("1/3 cell_train (HybridNorm)", train, a.dry_run)
    mp, cfg, mdir = grep(t, "TRAINED_MODEL_PATH="), grep(t, "HYBRID_CONFIG="), grep(t, "MODEL_DIR=")
    loss = os.path.join(mdir, "loss_details.csv") if mdir else ""

    sample = [PY, os.path.join(HERE, "cell_sample_20260609.py"),
              "--model_path", mp or "<M>", "--hybrid_config", cfg or "<C>",
              "--data_dir", a.data_dir, "--edge_tsv_path", a.edge_tsv_path,
              "--num_samples", str(a.num_samples), "--batch_size", str(a.sample_batch_size),
              "--diffusion_steps", str(a.diffusion_steps), "--output_dir", os.path.join(out, "sample")]
    s = run_capture("2/3 cell_sample (HybridNorm)", sample, a.dry_run)
    npz = grep(s, "SAMPLE_FILE_PATH=")

    if not a.skip_viz:
        viz = [PY, VIZ, "--model_path", mp or "<M>", "--config", cfg or "<C>",
               "--sample_path", npz or "<N>", "--loss_path", loss or "<L>",
               "--data_dir", a.data_dir, "--edge_tsv_path", a.edge_tsv_path,
               "--output_dir", os.path.join(out, "viz"),
               "--max_cells", str(a.max_cells), "--analyze_max_cells", str(a.analyze_max_cells),
               "--num_t_points", str(a.num_t_points)]
        run_capture("3/3 run_all_viz (HybridNorm)", viz, a.dry_run)
    print(f"\n[pipeline] DONE: {exp_id} -> {out}")


def create_argparser():
    # allow_abbrev=False: `--lr` を `--lr_anneal_steps` の略記と誤認させない（extra へ正しく流す）
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--hybrid_norm_mode", default="ratio_reg")
    p.add_argument("--reverse_coef", default="False")
    p.add_argument("--SoftReg", default="True")
    p.add_argument("--ode_reg_lambda", type=float, default=5.0)
    p.add_argument("--lr_anneal_steps", type=int, default=4)  # >=2: loss曲線/複数ckpt用
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    # checkpoint/log 頻度（既定 1000。短い test では save_interval<=step 数にする）
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--log_interval", type=int, default=1000)
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--sample_batch_size", type=int, default=8)
    p.add_argument("--max_cells", type=int, default=0)  # velocity/UMAP の cell 上限。0=制限なし（全 cell）
    p.add_argument("--analyze_max_cells", type=int, default=400)
    p.add_argument("--num_t_points", type=int, default=8)
    p.add_argument("--skip_viz", action="store_true")
    p.add_argument("--data_dir", default=DATA)
    p.add_argument("--edge_tsv_path", default=TSV)
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


if __name__ == "__main__":
    main()
