#!/bin/bash
# 20260609 全 3 dir の train→sample→viz を最小設定で実行する master。
# 成功した図は runs/{exp_id}/viz/ に残る（削除しない）。
#   bash work/run_all_20260609_pipelines.sh
# 引数で最小サイズを上書き可（例: MAXCELLS=400）。

set +e  # 1 つ失敗しても次へ
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 最小設定（環境変数で上書き可）
STEPS="${STEPS:-4}"; BS="${BS:-8}"; DIFF="${DIFF:-1000}"   # STEPS>=2: loss曲線/複数ckpt
NSAMP="${NSAMP:-16}"; SBS="${SBS:-8}"
MAXCELLS="${MAXCELLS:-150}"; AMAX="${AMAX:-120}"; NT="${NT:-4}"

COMMON="--lr_anneal_steps $STEPS --batch_size $BS --diffusion_steps $DIFF --num_samples $NSAMP --sample_batch_size $SBS --max_cells $MAXCELLS --analyze_max_cells $AMAX --num_t_points $NT"

SUMMARY=""
FAIL=0

run () {  # $1=label $2=dir $3=script ; rest=script-specific args
  label="$1"; d="$2"; script="$3"; shift 3
  echo ""; echo "##################################################################"
  echo "### ${label}"
  echo "##################################################################"
  ( cd "$WORK/$d" && python "$script" $COMMON "$@" )
  rc=$?
  if [ "$rc" -eq 0 ]; then SUMMARY="${SUMMARY}  OK   : ${label}\n"; else SUMMARY="${SUMMARY}  FAIL : ${label} (exit=${rc})\n"; FAIL=1; fi
}

# 3 pipeline とも save_interval/log_interval を CLI で持つ（既定 1000）。短い smoke test で毎 step
# checkpoint を残すため 1 を明示（さもないと init+最終の 2 点だけになる）。
run "Hybrid5x3 (lora/ratio_reg)"  20260609_Hybrid5x3      run_pipeline_5x3.py        --ode_branch lora --hybrid_norm_mode ratio_reg --save_interval 1 --log_interval 1
run "HybridNorm (ratio_reg)"      20260609_HybridNormModes run_pipeline_hybridnorm.py --hybrid_norm_mode ratio_reg --save_interval 1 --log_interval 1
run "MathMLP (lowrank)"           20260609_MathMLPHybrid   run_pipeline_mathmlp.py    --model_type lowrank --save_interval 1 --log_interval 1

echo ""; echo "=================== SUMMARY ==================="
printf "%b" "$SUMMARY"
echo "figures kept under each work/20260609_*/runs/<exp_id>/viz/"
exit $FAIL
