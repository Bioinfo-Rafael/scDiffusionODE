#!/bin/bash
# run_all_pipelines.sh  (20260609 / Hybrid5x3)
# ============================================================================
# 5 ODE 枝 × 4 hybrid 変種 = 【20 構成】を train→sample→viz で一気通貫テストする。
# 各 config は run_pipeline_5x3.py を呼ぶ（内部で cell_train → cell_sample → run_all_viz）。
#
#   変種(4):
#     none             通常の内分比 blend（r·ode+(1-r)·ml、ratio penalty なし）
#     ratio_reg        比の正則化（log-norm-ratio penalty を loss に加算）
#     scale_model_x        scale_model mode・scale 入力 = 遺伝子ベクトル x
#     scale_model_ml_emb   scale_model mode・scale 入力 = Cell_Unet 中間表現 ml_emb
#
#   ※ 旧 test_all_models.sh は別マトリクス（5×3 + baseline2 = 17、normed_learned_scale 含む）。
#      本スクリプトは scale_model mode を含む新マトリクス。
#
# 使い方:
#   export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"   # ← conda env python を最優先
#   bash run_all_pipelines.sh                       # 全 20 を最小設定で実行
#   DRY=1 bash run_all_pipelines.sh                 # コマンドだけ表示（--dry-run、dir も作らない）
#   BRANCHES="lora" bash run_all_pipelines.sh       # 1 モデル(lora)の 4 変種だけ
#   BRANCHES="geneode lora" bash run_all_pipelines.sh
#   STEPS=5 MAXCELLS=300 SKIP=velocity bash run_all_pipelines.sh   # サイズ/skip 上書き
#
# 注意:
#   - 必ず `bash` で実行（zsh だと文字列 $COMMON が word-split されない）。
#   - python は conda env の scdiffusion を使うこと（SMA venv が PATH 上の python を隠すため）。
#   - data/edge は未指定でも local_paths.resolve_path でローカル解決される。
# ============================================================================

set +e
WORK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPE="$WORK/run_pipeline_5x3.py"
cd "$WORK"

# --- 設定（env で上書き可）。STEPS>=2: loss 曲線/複数 checkpoint 用 ---
STEPS="${STEPS:-3}"; BS="${BS:-8}"; DIFF="${DIFF:-1000}"
NSAMP="${NSAMP:-8}"; SBS="${SBS:-8}"
# test は毎 step checkpoint を残したいので 1（pipeline 既定 1000 に依存しない。env で上書き可）
SAVEINT="${SAVEINT:-1}"; LOGINT="${LOGINT:-1}"
MAXCELLS="${MAXCELLS:-300}"; AMAX="${AMAX:-150}"; NT="${NT:-4}"
BRANCHES="${BRANCHES:-geneode lowrank lincomb matsum lora}"
[ -n "$DRY" ] && DRYFLAG="--dry-run" || DRYFLAG=""
[ -n "$SKIP" ] && SKIPFLAG="--skip $SKIP" || SKIPFLAG=""

COMMON="--lr_anneal_steps $STEPS --batch_size $BS --diffusion_steps $DIFF \
        --save_interval $SAVEINT --log_interval $LOGINT \
        --num_samples $NSAMP --sample_batch_size $SBS \
        --max_cells $MAXCELLS --analyze_max_cells $AMAX --num_t_points $NT $DRYFLAG $SKIPFLAG"

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

for B in $BRANCHES; do
  run "${B}__none"               --ode_branch "$B" --hybrid_norm_mode none        --scale_model_type none
  run "${B}__ratio_reg"          --ode_branch "$B" --hybrid_norm_mode ratio_reg   --scale_model_type none
  run "${B}__scale_model_x"      --ode_branch "$B" --hybrid_norm_mode scale_model --scale_model_type simple --scale_input_source x
  run "${B}__scale_model_ml_emb" --ode_branch "$B" --hybrid_norm_mode scale_model --scale_model_type simple --scale_input_source ml_emb
done

echo ""; echo "=================== SUMMARY (Hybrid5x3: 5 ODE × 4 variants = 20) ==================="
printf "%b" "$SUMMARY"
echo "outputs under work/20260609_Hybrid5x3/runs/<model>/<YYYYMMDD_HHMMSS>/{train,sample,viz}/"
exit $FAIL
