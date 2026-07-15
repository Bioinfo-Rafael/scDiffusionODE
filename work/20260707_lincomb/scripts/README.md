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

## `build_model_from_config` から作られる6モデルの内訳

調べた範囲では、6モデルは `run_matrix_0707.py` に並んでいるこの6 configです。`run_pipeline_0707.py` 自体は各 config について `train_0707.py` を呼び、そこで `build_model_from_config()` が呼ばれます。

参照箇所: `run_matrix_0707.py`, `train_0707.py`, `common.py`

### 全体の流れ

`train_0707.py` で、

1. `base.json` と各 config を merge
2. AnnData から `gene_list = adata.var["gene_name"]`
3. diffusion を作って `timesteps = diffusion.num_timesteps`
4. `hybrid_ts_soft_lincomb` だけ `t_s="auto"` を実データから整数に解決
5. `build_model_from_config(config, gene_list, timesteps, device)` を呼ぶ

という流れです。

`build_model_from_config()` から `build_denoiser_0707()` に渡る主な共通値はこうです。

```text
gene_list        = AnnData の gene_name 一覧
edge_tsv_path    = tf_target_edges.tsv
timesteps        = 1000
K                = 8
use_mask         = true
soft             = true
time_dim         = 64
field_hidden     = 256
field_dropout    = 0.0
use_decay        = true
gate_temperature = 1.0
off_mask_lambda  = 5.0
ratio_reg_weight = 0.0
ratio_reg_target = 1.0
scale_model_type = "none"
```

重要なのは、`base.json` に `ode_branch: "lincomb"` はありますが、`build_model_from_config()` は `ode_branch` を渡していません。つまり0707の6本は全部、固定で `ConfigurableLinCombField` を使います。

### 6モデルごとの差分

| config | `build_denoiser_0707` に渡る差分 | できるモデル |
|---|---|---|
| `lincomb_only_raw` | `denoiser_mode="lincomb_only"`, `gate_mode="raw"` | `LinCombOnlyDenoiser(ConfigurableLinCombField)` |
| `lincomb_softmax_gate` | `denoiser_mode="lincomb_only"`, `gate_mode="softmax"` | softmax係数版の LinComb-only |
| `lincomb_sparse_reg` | `denoiser_mode="lincomb_only"`, `gate_mode="raw"`, `sparse_lambda=0.01` | raw係数 + L1 sparse gate正則化 |
| `lincomb_entropy_reg` | `denoiser_mode="lincomb_only"`, `gate_mode="softmax"`, `entropy_lambda=0.01` | softmax係数 + entropy正則化 |
| `hybrid_reverse_lincomb` | `denoiser_mode="hybrid"`, `hybrid_norm_mode="none"`, `reverse_coef=true`, `gate_mode="raw"` | LinComb + `Cell_Unet` の reverse blend |
| `hybrid_ts_soft_lincomb` | `denoiser_mode="hybrid"`, `regime_gate_mode="Ts_I_vs_II_III"`, `regime_gate_type="sigmoid"`, `t_s=<auto解決された整数>`, `gate_tau=20.0`, `gate_mode="raw"` | LinComb + `Cell_Unet` の T_s sigmoid regime gate |

### LinComb部分で何が起きるか

`build_denoiser_0707()` はまず必ず `build_lincomb_field_0707()` を呼びます。そこで `use_mask=true` なので `tf_target_edges.tsv` から `(遺伝子数, 遺伝子数)` の mask を作り、`ConfigurableLinCombField` を作ります。

中身はざっくりこれです。

```text
logits = coeff_net([x, time_emb(t)])       # (B, K)
a_k    = logits              if gate_mode="raw"
       = softmax(logits/T)   if gate_mode="softmax"

f_k(x) = softplus(W_k x + b_k)
out    = sum_k a_k f_k(x) - decay(x)
```

つまり各サンプル・各 timestep ごとに、8個の expert ODE field を係数 `a_k(x,t)` で混ぜるモデルです。`raw` の場合は係数が負にも大きくもなれます。`softmax` の場合は8 expertの確率的な重みになります。

正則化は `off_mask_lambda=5.0` が全モデル共通なので、全6本で `expert_W` の TF-target mask 外成分に L1 penalty がかかります。さらに sparse/entropy の2本だけ、forward中に cache した gate正則化が `off_mask_penalty()` に足されます。

参照: `ODE/ode_20260707_lincomb.py`, `guided_diffusion/train_util.py`

### lincomb_only系

`denoiser_mode="lincomb_only"` の4本は、`LinCombOnlyDenoiser` が返ります。

```text
model.forward(x,t) = ConfigurableLinCombField(x,t)
```

なので diffusion の denoiser は Cell_Unet を使わず、LinComb field だけです。

### hybrid系

`denoiser_mode="hybrid"` の2本は、追加で

```python
ml_model = Cell_Unet(input_dim=len(gene_list))
```

を作って、`UnifiedODEMLHybrid` になります。

通常の hybrid blend は、

```text
r(t) = 1 - t / (T - 1)

reverse_coef=false:
out = r * ode_out + (1-r) * ml_out

reverse_coef=true:
out = (1-r) * ode_out + r * ml_out
```

なので `hybrid_reverse_lincomb` は通常と係数が逆です。`t=0` 付近では ML/Cell_Unet が強く、`t=999` 付近では ODE/LinComb が強くなります。

`hybrid_ts_soft_lincomb` は `r(t)` ではなく、

```text
w_ode(t) = sigmoid((t_s - t) / gate_tau)
out = w_ode * ode_out + (1-w_ode) * ml_out
```

を使います。つまり `t <= t_s` 側では LinComb が強く、`t > t_s` 側では Cell_Unet が強くなります。`t_s` は実行時に AnnData から推定され、`exp_config.json` と `ts_estimate.json` に保存されます。
