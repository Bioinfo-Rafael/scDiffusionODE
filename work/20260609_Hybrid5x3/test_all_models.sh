#!/bin/bash
# test_all_models.sh (20260609 / Hybrid5x3)
# このディレクトリの【全 17 モデル】(5 ODE枝 × 3 hybrid mode = 15 + baseline 2) を
# train→sample→viz で一気通貫テストする。各 config は run_pipeline_5x3.py を呼ぶ。
# 出力は runs/{model}/{date}/ 配下。最小サイズ（env で上書き可）。
#
#   bash test_all_models.sh              # 全 17 を最小実行
#   DRY=1 bash test_all_models.sh        # コマンドだけ表示（--dry-run）
#   STEPS=4 MAXCELLS=120 bash test_all_models.sh
#
# 注: conda env の python を使うこと（例: export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"）。

set +e
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$WORK/run_pipeline_5x3.py"

# 最小設定（env で上書き可）。STEPS>=2: loss曲線/複数ckpt
STEPS="${STEPS:-4}"; BS="${BS:-8}"; DIFF="${DIFF:-1000}"
NSAMP="${NSAMP:-16}"; SBS="${SBS:-8}"
MAXCELLS="${MAXCELLS:-120}"; AMAX="${AMAX:-100}"; NT="${NT:-4}"
[ -n "$DRY" ] && DRYFLAG="--dry-run" || DRYFLAG=""

COMMON="--lr_anneal_steps $STEPS --batch_size $BS --diffusion_steps $DIFF \
        --num_samples $NSAMP --sample_batch_size $SBS \
        --max_cells $MAXCELLS --analyze_max_cells $AMAX --num_t_points $NT $DRYFLAG"

SUMMARY=""; FAIL=0
run () {  # $1=label ; rest=run_pipeline args
  label="$1"; shift
  echo ""; echo "##############################################################"
  echo "### ${label}"
  echo "##############################################################"
  python "$PIPE" $COMMON "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then SUMMARY="${SUMMARY}  OK   : ${label}\n"; else SUMMARY="${SUMMARY}  FAIL : ${label} (exit=${rc})\n"; FAIL=1; fi
}

# --- 15: 5 ODE 枝 × 3 hybrid mode ---
for B in geneode lowrank lincomb matsum lora; do
  for M in ratio_reg normed_learned_scale none; do
    run "${B}__${M}" --ode_branch "$B" --hybrid_norm_mode "$M"
  done
done

# --- baseline 1: plain Cell_Unet（ODE/hook/reg なし）---
run "baseline_cellunet" --ode_branch plain --hybrid_norm_mode none --SoftReg False --ode_reg_lambda 0

# --- baseline 2: GeneODE+Cell_Unet 素朴 blend（penalty なし）---
run "baseline_geneode_blend" --ode_branch geneode --hybrid_norm_mode none --SoftReg False --ode_reg_lambda 0

echo ""; echo "=================== SUMMARY (Hybrid5x3: 15 + 2 baseline = 17) ==================="
printf "%b" "$SUMMARY"
echo "outputs under work/20260609_Hybrid5x3/runs/<model>/<date>/"
exit $FAIL
