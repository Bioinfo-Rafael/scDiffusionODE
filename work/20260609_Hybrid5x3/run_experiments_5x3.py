#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Launcher for the 17 experiments: 5 ODE-branch × 3 hybrid-mode (=15) + 2 baselines (20260609).

各実験を cell_train_5x3.py の subprocess として順次実行する。
dry-run でコマンド生成と検証だけ行える。実学習しない検証は --dry-run で。

  # dry-run（コマンド・出力path一意性・必須ファイル存在を検証、実行しない）
  python run_experiments_5x3.py --dry-run

  # 全 17 実行
  python run_experiments_5x3.py --gpu 0

  # 一部だけ
  python run_experiments_5x3.py --only lowrank__ratio_reg,baseline_cellunet
  python run_experiments_5x3.py --skip-existing
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(HERE, "cell_train_5x3.py")
PYTHON = sys.executable or "python"
sys.path.insert(0, HERE)
import run_paths  # noqa: E402

# data/edge のローカル fallback（リモートでは no-op）
_REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from local_paths import resolve_path

# ============================================================
# 全実験で統一する共通条件
# ============================================================
COMMON = dict(
    data_dir="/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad",
    edge_tsv_path="/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv",
    batch_size=128,
    microbatch=-1,
    lr="1e-4",
    weight_decay=0.0001,
    lr_anneal_steps=30000,
    schedule_sampler="uniform",
    diffusion_steps=1000,
    save_interval=5000,
    log_interval=1000,
    seed=1234,
    ode_reg_norm="l1",
    ratio_reg_weight=1.0,
    ratio_reg_target=1.0,
    rank=16,
    K=8,
    time_dim=64,
    field_hidden=256,
    field_dropout=0.0,
    lowrank_penalty_subsample=8,
    use_decay="True",
    hybrid_scale_init=1.0,
    hybrid_scale_eps="1e-8",
)

ODE_BRANCHES = ["geneode", "lowrank", "lincomb", "matsum", "lora"]
HYBRID_MODES = ["ratio_reg", "normed_learned_scale", "none"]

# 15 実験は SoftReg=True / ode_reg_lambda=5 で統一
MATRIX_REG = dict(SoftReg="True", ode_reg_lambda=5.0)


def build_matrix():
    """17 実験定義（15 + 2 baseline）を list[dict] で返す。"""
    exps = []
    # --- 15: 5 ODE branch × 3 hybrid mode ---
    for branch in ODE_BRANCHES:
        for mode in HYBRID_MODES:
            exp_id = f"{branch}__{mode}"
            exps.append(dict(
                exp_id=exp_id,
                ode_branch=branch,
                hybrid_norm_mode=mode,
                **MATRIX_REG,
            ))
    # --- baseline 1: plain Cell_Unet (ODE/hybrid なし、reg なし) ---
    exps.append(dict(
        exp_id="baseline_cellunet",
        ode_branch="plain",
        hybrid_norm_mode="none",     # plain では無視されるが値は要る
        SoftReg="False",
        ode_reg_lambda=0.0,
    ))
    # --- baseline 2: GeneODE + Cell_Unet の素朴な内分比 blend（penalty なし）---
    exps.append(dict(
        exp_id="baseline_geneode_blend",
        ode_branch="geneode",
        hybrid_norm_mode="none",
        SoftReg="False",
        ode_reg_lambda=0.0,
    ))
    return exps


def build_command(exp, runs_dir, when=None):
    """1 実験の python コマンド（list[str]）を生成する。出力 dir / model_name は exp_id で一意。

    出力構造: {runs_dir}/runs/{exp_id}/{YYYYMMDD_HHMMSS}/train/...（runs/ 配下に集約）。
    when は matrix 全実験で共有する batch 時刻（17 実験が同じ HHMMSS stamp 下に並ぶ）。
    dry-run でも呼ばれるため create=False でディレクトリは作らない（実行時 makedirs に委ねる）。
    """
    base = run_paths.run_base(runs_dir, exp["exp_id"], when=when, create=False)
    out_dir = os.path.join(base, "train")
    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--output_dir", out_dir,
        "--model_name", f"hybrid5x3_{exp['exp_id']}",
        "--ode_branch", exp["ode_branch"],
        "--hybrid_norm_mode", exp["hybrid_norm_mode"],
        "--SoftReg", str(exp["SoftReg"]),
        "--ode_reg_lambda", str(exp["ode_reg_lambda"]),
    ]
    for k, v in COMMON.items():
        cmd += [f"--{k}", str(v)]
    return cmd, out_dir


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="コマンド生成・検証のみ（実行しない）")
    ap.add_argument("--only", default="", help="カンマ区切りの exp_id だけ実行")
    ap.add_argument("--skip-existing", action="store_true", help="出力 dir が既にあればスキップ")
    ap.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES に設定する値")
    ap.add_argument("--runs-dir", default=HERE, help="各実験の出力先ルート（{root}/{exp_id}/{date}/train）")
    args = ap.parse_args()

    exps = build_matrix()
    if args.only:
        wanted = set(s.strip() for s in args.only.split(",") if s.strip())
        exps = [e for e in exps if e["exp_id"] in wanted]
        missing = wanted - {e["exp_id"] for e in exps}
        if missing:
            print(f"[ERROR] unknown exp_id(s): {sorted(missing)}")
            sys.exit(2)

    batch_dt = datetime.now()   # 全実験で共有する batch 時刻（runs/{exp_id}/{stamp} を 17 個で揃える）
    ts = batch_dt.strftime("%Y%m%d_%H%M%S")
    exp_root = os.path.join(HERE, "experiments", f"hybrid5x3_{ts}")
    summary_csv = os.path.join(exp_root, "summary.csv")
    summary_jsonl = os.path.join(exp_root, "summary.jsonl")
    ghash = git_hash()
    # NOTE: ディレクトリ作成は dry-run を抜けた execute 直前で行う（dry-run では 0 個作らない）

    # ---- resolve data/edge paths (remote = no-op, local = re-root to repo) ----
    COMMON["data_dir"] = resolve_path(COMMON["data_dir"])
    COMMON["edge_tsv_path"] = resolve_path(COMMON["edge_tsv_path"])

    # ---- validation ----
    print(f"=== experiment matrix: {len(build_matrix())} total (15 + 2 baseline) ===")
    if args.only:
        print(f"=== running subset: {len(exps)} ===")
    problems = []
    for f in (COMMON["data_dir"], COMMON["edge_tsv_path"]):
        if not os.path.exists(f):
            problems.append(f"missing required file: {f}")
    out_dirs = {}
    cmds = []
    for e in exps:
        cmd, out_dir = build_command(e, args.runs_dir, when=batch_dt)
        if out_dir in out_dirs:
            problems.append(f"duplicate output_dir: {out_dir} ({e['exp_id']} vs {out_dirs[out_dir]})")
        out_dirs[out_dir] = e["exp_id"]
        cmds.append((e, cmd, out_dir))

    print(f"git_hash={ghash}")
    for e, cmd, out_dir in cmds:
        print(f"\n[{e['exp_id']}] -> {out_dir}")
        print("  " + " ".join(cmd))

    if problems:
        print("\n[VALIDATION PROBLEMS]")
        for p in problems:
            print("  - " + p)
        # dry-run でなければ必須ファイル欠如は致命的
        if not args.dry_run and any(p.startswith("missing required") for p in problems):
            print("[ABORT] required files missing; not executing.")
            sys.exit(3)

    if args.dry_run:
        print(f"\n[dry-run] {len(cmds)} commands generated. No training executed.")
        print(f"[dry-run] summary would be written under {exp_root}")
        return

    # ---- execute ----
    os.makedirs(args.runs_dir, exist_ok=True)
    os.makedirs(exp_root, exist_ok=True)
    env = dict(os.environ)
    if args.gpu != "":
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    rows = []
    for e, cmd, out_dir in cmds:
        if args.skip_existing and os.path.isdir(out_dir) and os.listdir(out_dir):
            print(f"\n[SKIP existing] {e['exp_id']}")
            continue
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "train.log")
        start = datetime.now()
        print(f"\n[START {start.isoformat()}] {e['exp_id']}")
        with open(log_path, "w") as logf:
            logf.write("COMMAND: " + " ".join(cmd) + "\n")
            logf.write(f"GIT: {ghash}\nSTART: {start.isoformat()}\n\n")
            logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        end = datetime.now()
        trained = _extract(log_path, "TRAINED_MODEL_PATH=")
        row = dict(
            exp_id=e["exp_id"], ode_branch=e["ode_branch"], hybrid_norm_mode=e["hybrid_norm_mode"],
            model_name=f"hybrid5x3_{e['exp_id']}", output_dir=out_dir,
            exit_code=proc.returncode, start=start.isoformat(), end=end.isoformat(),
            duration_sec=round((end - start).total_seconds(), 1), git_hash=ghash,
            trained_model_path=trained, log=log_path, command=" ".join(cmd),
        )
        rows.append(row)
        with open(summary_jsonl, "a") as jf:
            jf.write(json.dumps(row) + "\n")
        status = "OK" if proc.returncode == 0 else f"FAIL(exit={proc.returncode})"
        print(f"[END {end.isoformat()}] {e['exp_id']} -> {status}")

    if rows:
        with open(summary_csv, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\n=== DONE. {len(rows)} runs. summary: {summary_csv} ===")
    n_fail = sum(1 for r in rows if r["exit_code"] != 0)
    if n_fail:
        print(f"[WARN] {n_fail} experiment(s) failed; see summary.")
        sys.exit(1)


def _extract(log_path, key):
    """学習ログ末尾から KEY='value' を拾う。"""
    try:
        val = ""
        with open(log_path) as f:
            for line in f:
                if key in line:
                    after = line.split(key, 1)[1].strip()
                    val = after.strip("'\"")
        return val
    except Exception:
        return ""


if __name__ == "__main__":
    main()
