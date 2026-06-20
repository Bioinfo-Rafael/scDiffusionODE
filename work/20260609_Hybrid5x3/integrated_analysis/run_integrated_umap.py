#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_integrated_umap.py  (20260609 / integrated_analysis)
========================================================

`runs/` の 20 構成（5 ODE branch × 4 variant）の generated を使って 2 系統の UMAP を作る CLI。

  A) per-model 個別比較: 各 config_label について real(既定 50000) + その model の gen(既定 3000)
     → per_model/<label>.png ×20 + per_model/all_models_facets.png
  B) integrated 統合比較: real(既定 全部) + 全 model の gen(各 500)
     → integrated/{integrated_real_annotation,integrated_origin,
        integrated_branch_marker_variant_color,integrated_summary_side_by_side}.png

既存 pipeline / training / sampling / viz は **一切呼ばない・変更しない**。runs/ は読むだけ。

使い方:
  cd work/20260609_Hybrid5x3
  python integrated_analysis/run_integrated_umap.py --runs_root runs \
      --output_root integrated_analysis/outputs --run_suffix ALL100k \
      --per_model_real_cells 50000 --per_model_gen_cells 3000 \
      --integrated_real_cells 0 --integrated_gen_per_model 500 --seed 0

  # 探索だけ（read-only。selected/skipped と config preview）
  python integrated_analysis/run_integrated_umap.py --runs_root runs \
      --output_root integrated_analysis/outputs --run_suffix ALL100k --dry_run

出力: {output_root}/{YYYYMMDD_HHMMSS}/
"""

import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import integrated_umap_utils as U  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser(
        description="全モデル統合/個別 UMAP（per-model 20枚+facets / integrated overlay+summary）。既存コードは変更しない。")
    # 共通
    p.add_argument("--runs_root", default="runs")
    p.add_argument("--output_root", default=os.path.join("integrated_analysis", "outputs"))
    p.add_argument("--run_suffix", default="ALL100k",
                   help="timestamp dir 名にこの suffix を含む run を優先（無ければ最新）")
    p.add_argument("--data_dir", default=U.DEFAULT_DATA_DIR,
                   help="real data h5ad（resolve_path でローカル解決）")
    p.add_argument("--seed", type=int, default=0)
    # A) per-model
    p.add_argument("--per_model_real_cells", type=int, default=50000, help="per-model の real 件数")
    p.add_argument("--per_model_gen_cells", type=int, default=3000, help="per-model の generated 件数")
    # B) integrated
    p.add_argument("--integrated_real_cells", type=int, default=0, help="integrated の real 件数（0=全部）")
    p.add_argument("--integrated_gen_per_model", type=int, default=500, help="integrated の model ごと generated 件数")
    # どちらを作るか
    p.add_argument("--only", choices=["per_model", "integrated"], default=None,
                   help="片方だけ作る（既定は両方）")
    # UMAP / annotation
    p.add_argument("--n_pcs", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--min_dist", type=float, default=0.5)
    p.add_argument("--annotation_priority", default=",".join(U.DEFAULT_ANNOTATION_PRIORITY),
                   help="real annotation 列の優先順位（カンマ区切り。既定 Superclass 先頭）")
    p.add_argument("--dry_run", action="store_true",
                   help="探索だけ。selected/skipped と config preview を表示・保存。UMAP/作図しない")
    return p


def _params_from_args(a, out_dir):
    d = U.default_params()
    d.update({
        "out_dir": out_dir, "runs_root": a.runs_root, "output_root": a.output_root,
        "run_suffix": a.run_suffix, "data_dir": a.data_dir, "seed": a.seed,
        "per_model_real_cells": a.per_model_real_cells, "per_model_gen_cells": a.per_model_gen_cells,
        "integrated_real_cells": a.integrated_real_cells,
        "integrated_gen_per_model": a.integrated_gen_per_model,
        "n_pcs": a.n_pcs, "n_neighbors": a.n_neighbors, "min_dist": a.min_dist,
        "annotation_priority": [s.strip() for s in a.annotation_priority.split(",") if s.strip()],
    })
    return d


def main():
    a = build_argparser().parse_args()
    ts = U.now_stamp()

    if a.dry_run:
        out_dir = os.path.join(a.output_root, f"{ts}_dryrun")
        params = _params_from_args(a, out_dir)
        print(f"[dry-run] runs_root={a.runs_root} run_suffix={a.run_suffix}")
        selected, skipped, config = U.discover_and_save(out_dir, params, log=print)
        print("\n=== SELECTED (使う予定の run) ===")
        for label, rec in selected.items():
            print(f"  {label:32s} step={rec['checkpoint_step']:>7} ({rec['checkpoint_kind']}) "
                  f"<- {rec['run_dir_rel']}")
        print("\n=== SKIPPED ===")
        for label, v in skipped.items():
            print(f"  {label:32s} reason={v['reason']}")
        print(f"\n[dry-run] selected={len(selected)} skipped={len(skipped)}")
        print("\n=== plan ===")
        print(f"  per-model     : real {a.per_model_real_cells} + gen {a.per_model_gen_cells}/model"
              f"  -> per_model/<label>.png x{len(selected)} + all_models_facets.png")
        print(f"  integrated    : real {'all' if a.integrated_real_cells<=0 else a.integrated_real_cells}"
              f" + gen {a.integrated_gen_per_model}/model -> integrated/*.png (overlay 3 + summary)")
        print("\n=== selected_runs_config.json (preview) ===")
        preview = {k: config[k] for k in ("created_at", "runs_root", "run_suffix", "data_dir",
                                          "data_dir_resolved", "selection_rule", "n_selected", "n_skipped")}
        preview["models"] = {l: {"checkpoint_step": r["checkpoint_step"],
                                 "sample_path_rel": r["sample_path_rel"]}
                             for l, r in list(selected.items())[:3]}
        preview["models_note"] = f"... ({len(selected)} models total; full detail in JSON)"
        print(json.dumps(preview, indent=2, default=str, ensure_ascii=False))
        print(f"\n[dry-run] metadata written under: {out_dir}")
        return

    out_dir = os.path.join(a.output_root, ts)
    params = _params_from_args(a, out_dir)
    do_pm = a.only in (None, "per_model")
    do_int = a.only in (None, "integrated")
    out_dir, selected, skipped = U.run_full(params, do_per_model=do_pm, do_integrated=do_int, log=print)

    print(f"\n[done] outputs -> {out_dir}")
    print(f"[done] selected={len(selected)} skipped={len(skipped)}")
    if do_pm:
        print("[done] per_model/   : <label>.png x{n} + all_models_facets.png".format(n=len(selected)))
    if do_int:
        print("[done] integrated/  : integrated_real_annotation.png / integrated_origin.png / "
              "integrated_branch_marker_variant_color.png / integrated_summary_side_by_side.png")
    print("[done] metadata     : selected_runs_config.json / selected_runs.csv / skipped_runs.csv / run_config.json")
    print("[done] color maps   : color_map_real_annotation.csv / model_color_map.csv")


if __name__ == "__main__":
    main()
