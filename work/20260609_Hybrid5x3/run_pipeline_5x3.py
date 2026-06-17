#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_pipeline_5x3.py  (20260609)
===============================

単一 config を「cell_train → cell_sample → 可視化(3種)」で一気通貫する。

  python run_pipeline_5x3.py --ode_branch lora --hybrid_norm_mode ratio_reg \
      --lr_anneal_steps 1 --batch_size 8 --diffusion_steps 1000 \
      --num_samples 16 --max_cells 800

出力は runs/{exp_id}/ 配下（train / sample / viz）。--dry-run でコマンドのみ表示。
data/edge は未指定でも local_paths fallback でローカル解決される。
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_paths  # noqa: E402
PY = sys.executable or "python"
DATA = "/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad"
TSV = "/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv"


def run_capture(label, cmd, dry):
    print("\n" + "=" * 70)
    print(f"### {label}")
    print("  " + " ".join(cmd))
    if dry:
        return ""
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print(f"[pipeline] {label} FAILED (exit={proc.returncode})")
        sys.exit(proc.returncode)
    return proc.stdout


def grep_val(text, key):
    val = ""
    for line in text.splitlines():
        if key in line:
            val = line.split(key, 1)[1].strip().strip("'\"")
    return val


def main():
    a = create_argparser().parse_args()
    exp_id = f"{a.ode_branch}__{a.hybrid_norm_mode}"
    # {HERE}/runs/{model_name}/{YYYYMMDD}/{train,sample,viz}/...（runs/ 配下に集約）
    out_dir = run_paths.run_base(HERE, exp_id, create=not a.dry_run)  # dry-run では dir を作らない

    # 1) TRAIN
    train = [PY, os.path.join(HERE, "cell_train_5x3.py"),
             "--data_dir", a.data_dir, "--edge_tsv_path", a.edge_tsv_path,
             "--ode_branch", a.ode_branch, "--hybrid_norm_mode", a.hybrid_norm_mode,
             "--rank", str(a.rank), "--K", str(a.K),
             "--SoftReg", str(a.SoftReg), "--ode_reg_lambda", str(a.ode_reg_lambda),
             "--lr_anneal_steps", str(a.lr_anneal_steps), "--batch_size", str(a.batch_size),
             "--diffusion_steps", str(a.diffusion_steps), "--log_interval", "1",
             "--save_interval", "1", "--output_dir", os.path.join(out_dir, "train")]
    t_out = run_capture("1/3 cell_train_5x3", train, a.dry_run)
    model_path = grep_val(t_out, "TRAINED_MODEL_PATH=")
    exp_config = grep_val(t_out, "EXP_CONFIG=")
    model_dir = grep_val(t_out, "MODEL_DIR=")
    loss_path = os.path.join(model_dir, "loss_details.csv") if model_dir else ""
    if not a.dry_run:
        print(f"[pipeline] model={model_path}\n[pipeline] config={exp_config}\n[pipeline] loss={loss_path}")

    # 2) SAMPLE
    sample = [PY, os.path.join(HERE, "cell_sample_5x3.py"),
              "--model_path", model_path or "<MODEL>", "--exp_config", exp_config or "<CONFIG>",
              "--data_dir", a.data_dir, "--edge_tsv_path", a.edge_tsv_path,
              "--num_samples", str(a.num_samples), "--batch_size", str(a.sample_batch_size),
              "--diffusion_steps", str(a.diffusion_steps),
              "--output_dir", os.path.join(out_dir, "sample")]
    s_out = run_capture("2/3 cell_sample_5x3", sample, a.dry_run)
    sample_path = grep_val(s_out, "SAMPLE_FILE_PATH=")
    if not a.dry_run:
        print(f"[pipeline] sample={sample_path}")

    # 3) VIZ
    if not a.skip_viz:
        viz = [PY, os.path.join(HERE, "viz", "run_all_viz.py"),
               "--model_path", model_path or "<MODEL>", "--config", exp_config or "<CONFIG>",
               "--sample_path", sample_path or "<NPZ>", "--loss_path", loss_path or "<LOSS>",
               "--data_dir", a.data_dir, "--edge_tsv_path", a.edge_tsv_path,
               "--output_dir", os.path.join(out_dir, "viz"),
               "--max_cells", str(a.max_cells), "--analyze_max_cells", str(a.analyze_max_cells),
               "--num_t_points", str(a.num_t_points)]
        if a.skip:
            viz += ["--skip", a.skip]
        run_capture("3/3 run_all_viz", viz, a.dry_run)

    print(f"\n[pipeline] DONE: {exp_id} -> {out_dir}")


def create_argparser():
    p = argparse.ArgumentParser()
    # branch/mode
    p.add_argument("--ode_branch", default="lora")
    p.add_argument("--hybrid_norm_mode", default="ratio_reg")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--SoftReg", default="True")
    p.add_argument("--ode_reg_lambda", type=float, default=5.0)
    # train
    p.add_argument("--lr_anneal_steps", type=int, default=4)  # >=2: loss曲線/複数ckpt用
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--diffusion_steps", type=int, default=1000)
    # sample
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--sample_batch_size", type=int, default=8)
    # viz
    p.add_argument("--max_cells", type=int, default=800)
    p.add_argument("--analyze_max_cells", type=int, default=400)
    p.add_argument("--num_t_points", type=int, default=8)
    p.add_argument("--skip", default="", help="viz の一部スキップ: viz,analyze,velocity")
    p.add_argument("--skip_viz", action="store_true")
    # paths
    p.add_argument("--data_dir", default=DATA)
    p.add_argument("--edge_tsv_path", default=TSV)
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


if __name__ == "__main__":
    main()
