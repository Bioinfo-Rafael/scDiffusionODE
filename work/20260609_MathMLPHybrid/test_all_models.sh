#!/bin/bash
# test_all_models.sh (20260609 / MathMLPHybrid)
# このディレクトリの【全 4 モデル】= math field 4 系統
# (lowrank / lincomb / matsum / lora) を train→sample→viz で一気通貫テストする。
# 各 model_type は run_pipeline_mathmlp.py を呼ぶ。出力は runs/{model}/{date}/ 配下。
#
#   bash test_all_models.sh              # 全 4 を最小実行
#   DRY=1 bash test_all_models.sh        # コマンドだけ表示（--dry-run）
#   STEPS=4 MAXCELLS=120 bash test_all_models.sh
#
# 注: conda env の python を使うこと（例: export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"）。

set +e
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$WORK/run_pipeline_mathmlp.py"

STEPS="${STEPS:-4}"; BS="${BS:-8}"; DIFF="${DIFF:-1000}"
NSAMP="${NSAMP:-16}"; SBS="${SBS:-8}"
MAXCELLS="${MAXCELLS:-120}"; AMAX="${AMAX:-100}"; NT="${NT:-4}"
[ -n "$DRY" ] && DRYFLAG="--dry-run" || DRYFLAG=""

COMMON="--lr_anneal_steps $STEPS --batch_size $BS --diffusion_steps $DIFF \
        --num_samples $NSAMP --sample_batch_size $SBS \
        --max_cells $MAXCELLS --analyze_max_cells $AMAX --num_t_points $NT $DRYFLAG"

SUMMARY=""; FAIL=0
run () {
  label="$1"; shift
  echo ""; echo "##############################################################"
  echo "### ${label}"
  echo "##############################################################"
  python "$PIPE" $COMMON "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then SUMMARY="${SUMMARY}  OK   : ${label}\n"; else SUMMARY="${SUMMARY}  FAIL : ${label} (exit=${rc})\n"; FAIL=1; fi
}

for MT in lowrank lincomb matsum lora; do
  run "mathmlp__${MT}" --model_type "$MT"
done

echo ""; echo "=================== SUMMARY (MathMLP: 4 field types) ==================="
printf "%b" "$SUMMARY"
echo "outputs under work/20260609_MathMLPHybrid/runs/<model>/<date>/"
exit $FAIL
