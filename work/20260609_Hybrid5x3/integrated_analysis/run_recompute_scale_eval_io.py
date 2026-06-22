#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_recompute_scale_eval_io.py  (20260609 / integrated_analysis)
================================================================

学習済みの **scale_model** run（variant = scale_model_x / scale_model_ml_emb）について、
修正済み `viz/eval_model_io.py`（scale_model の component decomposition を実 forward に一致させた版）の
入出力評価だけを **作り直す** CLI。training / sampling / params / loss / velocity / UMAP は一切呼ばない。

出力は既存 run を上書きせず、通常 pipeline と同じ `runs/<config_label>/<timestamp>_<suffix>/` 形式の
**新しい timestamp run dir** に保存する（既定 suffix=eval_corrected）。その中に通常と同じ構造で
`viz/eval_io/<ts>_analyze/`（timestep_metrics.csv / analysis_config.json / alignment_mse/ / corr/）を作り、
新 run dir 直下に参照元を示す `source_run_metadata.json` と `recompute_command.txt` を残す。

source run は `integrated_umap_utils.select_runs()` で選ぶ（各 config_label について採用条件
= samples.npz + exp_config.json + checkpoint を満たす最新 run）。

使い方:
  cd work/20260609_Hybrid5x3
  export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"

  # scale_model 10 構成（variant 指定。既定もこの 10 個）
  python integrated_analysis/run_recompute_scale_eval_io.py \
      --runs_root runs --run_suffix ALL100k \
      --include_variants scale_model_x,scale_model_ml_emb \
      --weighting both --num_noise_draws 4 --max_cells 1000 --seed 1234

  # 完全 label 指定でも可
  python integrated_analysis/run_recompute_scale_eval_io.py \
      --runs_root runs --run_suffix ALL100k \
      --include_labels geneode__scale_model_x,lowrank__scale_model_ml_emb,... \
      --weighting both --num_noise_draws 4 --max_cells 1000 --seed 1234

  # 探索だけ（実行対象 / source run / checkpoint / 新規作成予定 run dir / command を表示。何も作らない）
  python integrated_analysis/run_recompute_scale_eval_io.py --run_suffix ALL100k \
      --include_variants scale_model_x,scale_model_ml_emb --dry_run
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import integrated_umap_utils as U  # noqa: E402

WORK_ROOT = U.HYBRID_DIR                     # work/20260609_Hybrid5x3
if WORK_ROOT not in sys.path:
    sys.path.insert(0, WORK_ROOT)
import run_paths  # noqa: E402

# eval_model_io.py のデフォルトと同じ canonical path（resolve_path がローカル再 root する）
DEFAULT_EDGE_TSV = "/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv"
SCALE_VARIANTS = ("scale_model_x", "scale_model_ml_emb")
DEFAULT_SCALE_LABELS = [l for l in U.ALL_CONFIG_LABELS
                        if U.split_config_label(l)[1] in SCALE_VARIANTS]


def build_argparser():
    p = argparse.ArgumentParser(
        description="scale_model run の eval_model_io（修正版 decomposition）だけを新 run dir に作り直す。")
    p.add_argument("--runs_root", default="runs",
                   help="source run を探す runs/ ルート（相対なら work_root 基準）")
    p.add_argument("--run_suffix", default="ALL100k",
                   help="timestamp dir 名にこの suffix を含む run を優先（無ければ最新）")
    # 実行対象の制御
    p.add_argument("--include_labels", default="",
                   help="実行する config_label をこれだけに限定（カンマ区切り）")
    p.add_argument("--include_variants", default="",
                   help="variant 名でまとめて指定（カンマ区切り。例 scale_model_x,scale_model_ml_emb）")
    p.add_argument("--exclude_labels", default="",
                   help="除外する config_label（カンマ区切り）")
    # eval_model_io へ渡す設定
    p.add_argument("--data_dir", default=U.DEFAULT_DATA_DIR,
                   help="real data h5ad（exp_config に data_dir があればそちらを優先）")
    p.add_argument("--edge_tsv_path", default=DEFAULT_EDGE_TSV,
                   help="edge tsv（exp_config に edge_tsv_path があればそちらを優先）")
    p.add_argument("--output_suffix", default="eval_corrected",
                   help="新 run dir の timestamp 末尾 suffix（runs/<label>/<ts>_<suffix>/）")
    p.add_argument("--weighting", choices=["weighted", "unweighted", "both"], default="both")
    p.add_argument("--num_t_points", type=int, default=16)
    p.add_argument("--num_noise_draws", type=int, default=4)
    p.add_argument("--max_cells", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--dry_run", action="store_true",
                   help="実行対象 / source run / 新規作成予定 run dir / command を表示するだけ（何も作らない）")
    return p


def _csv(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def resolve_labels(a):
    """include_labels / include_variants / exclude_labels から対象 config_label を解決。

    優先順位: include_labels > include_variants > 既定（scale_model 10 個）。最後に exclude を差し引く。
    順序は ALL_CONFIG_LABELS 順に正規化（未知ラベルは末尾）。
    """
    inc_labels = _csv(a.include_labels)
    inc_variants = set(_csv(a.include_variants))
    exc_labels = set(_csv(a.exclude_labels))
    if inc_labels:
        targets = inc_labels
    elif inc_variants:
        targets = [l for l in U.ALL_CONFIG_LABELS if U.split_config_label(l)[1] in inc_variants]
    else:
        targets = list(DEFAULT_SCALE_LABELS)
    targets = [l for l in targets if l not in exc_labels]
    known = [l for l in U.ALL_CONFIG_LABELS if l in targets]
    extra = [l for l in targets if l not in U.ALL_CONFIG_LABELS]
    ordered, seen = [], set()
    for l in known + extra:
        if l not in seen:
            seen.add(l)
            ordered.append(l)
    return ordered, exc_labels


def _abs_from_exp(exp_config_path, key, default):
    """exp_config.json の data_dir / edge_tsv_path を絶対パス化して返す（無ければ default）。

    相対パス（例 ../../work/...）は training cwd = work_root 基準で記録されているので work_root と結合。
    /home/suzuki などの絶対パスはそのまま（eval_model_io 側の resolve_path がローカル再 root する）。
    """
    try:
        with open(exp_config_path) as f:
            d = json.load(f)
    except Exception:
        return default
    val = d.get(key)
    if not val:
        return default
    if os.path.isabs(val):
        return val
    return os.path.normpath(os.path.join(WORK_ROOT, val))


def main():
    a = build_argparser().parse_args()
    targets, exc = resolve_labels(a)

    runs_root = a.runs_root if os.path.isabs(a.runs_root) else os.path.join(WORK_ROOT, a.runs_root)
    selected, skipped = U.select_runs(runs_root, a.run_suffix, config_labels=targets)

    batch_dt = datetime.now()
    batch_ts = batch_dt.strftime("%Y%m%d_%H%M%S")
    eval_script = os.path.join(WORK_ROOT, "viz", "eval_model_io.py")
    py = sys.executable

    print(f"[recompute] runs_root={runs_root}")
    print(f"[recompute] run_suffix={a.run_suffix}  batch_ts={batch_ts}  output_suffix={a.output_suffix}")
    print(f"[recompute] targets ({len(targets)}): {targets}")
    if exc:
        print(f"[recompute] excluded: {sorted(exc)}")
    if skipped:
        print(f"[recompute] skipped ({len(skipped)}):")
        for label, v in skipped.items():
            print(f"    {label:32s} reason={v['reason']}")

    plan = []
    for label in targets:
        if label not in selected:
            continue
        rec = selected[label]
        new_run_dir = run_paths.run_base(WORK_ROOT, label, when=batch_dt,
                                         create=(not a.dry_run), suffix=a.output_suffix)
        eval_out = os.path.join(new_run_dir, "viz", "eval_io")
        data_dir = _abs_from_exp(rec["exp_config_path"], "data_dir", a.data_dir)
        edge_tsv = _abs_from_exp(rec["exp_config_path"], "edge_tsv_path", a.edge_tsv_path)
        cmd = [py, eval_script,
               "--model_path", rec["checkpoint_path"],
               "--config", rec["exp_config_path"],
               "--data_dir", data_dir,
               "--edge_tsv_path", edge_tsv,
               "--output_dir", eval_out,
               "--weighting", a.weighting,
               "--num_t_points", str(a.num_t_points),
               "--num_noise_draws", str(a.num_noise_draws),
               "--max_cells", str(a.max_cells),
               "--batch_size", str(a.batch_size),
               "--seed", str(a.seed)]
        plan.append((label, rec, new_run_dir, eval_out, cmd))

    print(f"\n=== plan ({len(plan)} run) ===")
    for label, rec, new_run_dir, eval_out, cmd in plan:
        print(f"\n--- {label} ---")
        print(f"  source_run_dir : {rec['run_dir']}")
        print(f"  checkpoint     : {rec['checkpoint_path']}  (step={rec['checkpoint_step']}, {rec['checkpoint_kind']})")
        print(f"  exp_config     : {rec['exp_config_path']}")
        print(f"  NEW run dir    : {new_run_dir}")
        print(f"  eval output    : {eval_out}")
        print(f"  command        : {' '.join(cmd)}")

    if a.dry_run:
        print(f"\n[dry-run] {len(plan)} run を実行予定（実行しない）。ディレクトリも作成していません。")
        return

    self_cmd = "python " + " ".join([os.path.relpath(os.path.abspath(__file__), WORK_ROOT)] + sys.argv[1:])
    n_ok = n_fail = 0
    for label, rec, new_run_dir, eval_out, cmd in plan:
        print(f"\n########## {label} ##########")
        meta = {
            "config_label": label,
            "run_suffix": a.run_suffix,
            "created_at": batch_ts,
            "recompute_kind": "scale_model_corrected_eval",
            "source_run_dir": rec["run_dir"],
            "source_checkpoint_path": rec["checkpoint_path"],
            "source_checkpoint_step": rec["checkpoint_step"],
            "source_checkpoint_kind": rec["checkpoint_kind"],
            "source_exp_config_path": rec["exp_config_path"],
            "source_sample_path": rec.get("sample_path"),
            "eval_output_dir": eval_out,
            "eval_command": cmd,
        }
        with open(os.path.join(new_run_dir, "source_run_metadata.json"), "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        with open(os.path.join(new_run_dir, "recompute_command.txt"), "w") as f:
            f.write("# recompute launcher:\n" + self_cmd + "\n\n"
                    "# eval_model_io command:\n" + " ".join(cmd) + "\n")
        rc = subprocess.call(cmd, cwd=WORK_ROOT)
        if rc == 0:
            n_ok += 1
            print(f"[recompute] OK   {label} -> {new_run_dir}")
        else:
            n_fail += 1
            print(f"[recompute] FAIL {label} (exit={rc})")

    print(f"\n[recompute] done. ok={n_ok} fail={n_fail}  batch_ts={batch_ts}")
    print(f"[recompute] outputs under runs/<label>/{batch_ts}_{a.output_suffix}/viz/eval_io/<ts>_analyze/")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
