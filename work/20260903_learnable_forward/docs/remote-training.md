# リモートでpullし、Model A・Bをバックグラウンド学習する

SSHでログイン済みのLinux/GPUサーバー上で、以下をまとめて実行する。
`REPO` は既存configのサーバーパス、`GPU=0` は使用GPU。環境に応じてこの2箇所を変更する。
Condaの `scdiffusion` 環境とデータ/GRNファイルが用意されていることを前提とする。

既存 `scripts/launch.py` で **Model A → Model B の順番**に、同じGPU上で学習する。
両方とも新しい共通batch IDを使い、旧run/checkpointは指定しない。
各モデル30,000 steps、batch size 128、GRN weight 5、Model AはK=64
（各値は現在のconfig既定値）。Model AのB初期値は0.01 Iである。
Model Aが失敗するとlauncherも停止し、その場合Model Bは開始しない。
この手順は両モデルの学習までで、sampling/analysisは実行しない。

```bash
bash <<'BASH'
set -euo pipefail

REPO=/home/suzuki/Projects/scDiffusion
BRANCH=work/20260816-hill-variants
GPU=0

cd "$REPO"
git fetch origin
git switch "$BRANCH"
git pull --ff-only origin "$BRANCH"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scdiffusion
PYTHON_BIN="$(command -v python)"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

SUITE="$REPO/work/20260903_learnable_forward"
DATA="$REPO/work/20260215_embryonic/data/Embryonic.h5ad"
GRN="$REPO/external_data/tf_target_edges.tsv"
BATCH_ID="ab-aux-$(date +%Y%m%d-%H%M%S)-$$"
LOG_DIR="$SUITE/runs/launch_logs/$BATCH_ID"

test -f "$DATA"
test -f "$GRN"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'
"$PYTHON_BIN" "$SUITE/scripts/verify_protected.py"
mkdir -p "$LOG_DIR"
git rev-parse HEAD > "$LOG_DIR/git_commit.txt"

ARGS=(
  "$SUITE/scripts/launch.py"
  --model stationary_qd
  --model free_affine
  --batch-id "$BATCH_ID"
  --set device=cuda
  --set "data_dir=$DATA"
  --set "edge_tsv_path=$GRN"
)
"$PYTHON_BIN" "${ARGS[@]}" --dry-run > "$LOG_DIR/plan.json"

nohup "$PYTHON_BIN" -u "${ARGS[@]}" \
  > "$LOG_DIR/train.log" 2>&1 < /dev/null &
LAUNCH_PID=$!
printf '%s\n' "$LAUNCH_PID" > "$LOG_DIR/launcher.pid"

printf 'Batch ID: %s\nLauncher PID: %s\n' "$BATCH_ID" "$LAUNCH_PID"
printf 'Model A: %s/runs/model_a_stationary_qd_aux/%s\n' "$SUITE" "$BATCH_ID"
printf 'Model B: %s/runs/model_b_free_affine_dense/%s\n' "$SUITE" "$BATCH_ID"
printf 'Log: %s/train.log\nPlan: %s/plan.json\n' "$LOG_DIR" "$LOG_DIR"
printf 'Monitor: tail -f "%s/train.log"\n' "$LOG_DIR"
BASH
```

`nohup` 起動後はSSHを切断しても学習が継続する。表示された `Monitor:` のコマンドでlogを追跡でき、`tail -f` のCtrl+Cは監視だけを終了する。PIDはlauncherのPIDであり、GPUで計算するtraining子プロセスのPIDとは異なる。

起動直後のPID表示は学習成功を保証しない。logに `[1/2]` が出てAが始まり、Aの正常終了後に `[2/2]` が出てBが始まる。

Gitのローカル変更がpullと競合する場合やbranchが分岐している場合は、`set -e` / `--ff-only` により起動前に停止する。強制reset・stash・旧runの削除は行わない。

**旧dense Model Aのstep-5000 checkpointからresumeしてはいけない。** 新parameterizationを採用した理由は[設計判断メモ](decisions/20260905-model-a-auxiliary-subspace.md)、実装時の検証は[validation](../validation/README.md)を参照。
