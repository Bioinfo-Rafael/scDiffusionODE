# 2026-07-07 実験script

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連する実行entrypointと共通helperを置きます。通常は `run_pipeline_0707.py` から使用します。

- `run_pipeline_0707.py`: 1 configについてtrain → sample → viz → summaryを実行
- `run_matrix_0707.py`: 複数configを順番に実行
- `train_0707.py`: 学習、`T_s` 解決、config/manifest/checkpoint保存
- `sample_0707.py`: configとcheckpointを復元してsampling
- `summarize_results_0707.py`: manifestを比較用CSV/JSONへ集約
- `common.py`: config、path、manifest、git state、model buildの共通helper
- `../utils/regime_time.py`: AnnDataから `lambda_max` と `T_s` を推定

全実験に共通する標準条件は `../configs/base.json` で指定します。現在は各モデルを
`lr_anneal_steps=100000` で学習し、各学習済みモデルから `num_samples=3000` 細胞を
samplingします。sampling時のミニバッチは `sample_batch_size=50` です。個別の実験
configに同じキーを書いた場合は、個別configの値が優先されます。

`run_pipeline_0707.py` は以下を扱います。

- `--dry-run`: ファイルを作らずコマンドを確認
- `--resume <path|auto>`: checkpointから再開
- `--skip-existing`: 完了済みstageをskip
- `--force-stage train|sample|viz`: 指定stageだけ実行

## 実行前の準備

以下のコマンドはリポジトリrootから実行します。

```bash
cd /Users/cls-lab/Git/scDiffusionODE
```

ローカルでは、必要な依存関係が入ったPythonを明示できます。

```bash
PYTHON="$HOME/miniconda3/envs/scdiffusion/bin/python"
```

以下の例で `python` が環境違いになる場合は、`python` を `$PYTHON` に置き換えてください。

## 全実験を一括実行

次のコマンドは、6つのconfigを順番にtrain → sample → viz → summaryまで実行します。

```bash
python work/20260707_lincomb/scripts/run_matrix_0707.py \
  --config-dir work/20260707_lincomb/configs
```

実行される実験:

1. `hybrid_reverse_lincomb`
2. `lincomb_only_raw`
3. `lincomb_softmax_gate`
4. `lincomb_sparse_reg`
5. `lincomb_entropy_reg`
6. `hybrid_ts_soft_lincomb`

全コマンドだけを事前確認する場合:

```bash
python work/20260707_lincomb/scripts/run_matrix_0707.py \
  --config-dir work/20260707_lincomb/configs \
  --dry-run
```

## 各実験を個別に実行

### 1. `hybrid_reverse_lincomb`

既存schedulerのODE/CellUnet係数を反転したhybridです。既定では `hybrid_norm_mode="none"` を使います。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_reverse_lincomb.json
```

### 2. `lincomb_only_raw`

CellUnetと混合せず、raw signed coefficientのLinComb fieldだけをdenoiserに使います。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/lincomb_only_raw.json
```

### 3. `lincomb_softmax_gate`

LinComb係数を正・総和1のsoftmax gateとして使います。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/lincomb_softmax_gate.json
```

temperatureを変更する例:

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/lincomb_softmax_gate.json \
  --gate_temperature 0.5
```

### 4. `lincomb_sparse_reg`

raw coefficientへ `mean(abs(a))` のL1正則化を加えます。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/lincomb_sparse_reg.json
```

正則化weightを変更する例:

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/lincomb_sparse_reg.json \
  --sparse_lambda 0.005
```

### 5. `lincomb_entropy_reg`

softmax gateへ正のentropy penaltyを加え、sampleごとのexpert使用を尖らせます。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/lincomb_entropy_reg.json
```

正則化weightを変更する例:

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/lincomb_entropy_reg.json \
  --entropy_lambda 0.005
```

### 6. `hybrid_ts_soft_lincomb`

training時にAnnDataから `lambda_max` と `T_s` を推定し、sigmoid regime gateを使います。Regime IではCellUnet、Regime II/IIIではODE/LinCombを強くします。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json
```

`T_s` を自動推定せず手動指定する例:

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json \
  --t_s 600 \
  --gate_tau 20
```

## 1実験のdry-run

出力directoryを作らず、resolved config、run path、train/sample/vizコマンドを表示します。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json \
  --dry-run
```

## Stageごとの実行

### Trainingだけを実行

```bash
python work/20260707_lincomb/scripts/train_0707.py \
  --config work/20260707_lincomb/configs/lincomb_only_raw.json
```

完了時に表示される `RUN_DIR` と `TRAINED_MODEL_PATH` を後続処理で使用します。

### Samplingだけを実行

```bash
python work/20260707_lincomb/scripts/sample_0707.py \
  --run-dir work/20260707_lincomb/runs/<experiment>/<run_id>
```

### Visualizationだけを実行

```bash
python work/20260707_lincomb/viz/run_all_viz_0707.py \
  --run-dir work/20260707_lincomb/runs/<experiment>/<run_id>
```

### Result summaryだけを再生成

```bash
python work/20260707_lincomb/scripts/summarize_results_0707.py
```

## 再開・再実行

manifestに記録されたcheckpointからtrainingを再開します。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json \
  --run-dir work/20260707_lincomb/runs/hybrid_ts_soft_lincomb/<run_id> \
  --resume auto
```

完了済みstageをskipして不足分だけ進めます。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json \
  --run-dir work/20260707_lincomb/runs/hybrid_ts_soft_lincomb/<run_id> \
  --skip-existing
```

指定stageだけを再実行します。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json \
  --run-dir work/20260707_lincomb/runs/hybrid_ts_soft_lincomb/<run_id> \
  --force-stage viz
```

## 主な出力先

```text
work/20260707_lincomb/runs/<experiment>/<run_id>/
work/20260707_lincomb/samples/<experiment>/<run_id>/
work/20260707_lincomb/viz/<experiment>/<run_id>/
work/20260707_lincomb/logs/<experiment>/<run_id>/
work/20260707_lincomb/results/
```

script本体は実験コードなので編集可能ですが、変更後は `tests/` とdry-runを実行してください。
