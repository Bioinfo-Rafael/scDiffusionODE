# CellUnet–single ODE consistency regularization (2026-08-30)

## 目的と構造

通常の CellUnet diffusion の denoising 出力を変えず、training 中だけ同じ
`x_t` を single ODE に入力し、CellUnet と ODE の vector field を近づけます。
`CellUNetODERegularized20260830.forward()` の返り値は常に `cell_out` だけです。
`model.eval()` では ODE forward 自体を呼びません。ODE の sampling への影響は、
training 中に更新された CellUnet parameter を介するものだけです。

共有ファイルは変更せず、wrapper、4 ODE、factory、TrainLoop subclass、config、
launcher、test のすべてをこの directory に閉じ込めています。wrapper は既存
soft constraint と互換にするため `ml_model` と `ode_model` を保持します。

## Loss

各 sample について

```text
cell_ode_per_sample[b]
  = mean_gene((CellUnet(x_t, t)[b] - ODE(x_t, t)[b]) ** 2)
```

とし、diffusion の schedule sampler と同じ `weights[b]` を適用します。

```text
L_consistency = mean_b(weights[b] * cell_ode_per_sample[b])
L_total = L_diffusion
        + ode_reg_lambda * L_ODE_soft
        + cell_ode_reg_lambda_20260830 * L_consistency
```

`lambda=0` では consistency の loss 寄与は厳密に 0 です。今回の12条件は
`0.1, 1.0, 10.0` を使用します。

詳細lossは各optimizer stepでmemory bufferへ記録し、既定100 stepごと、および
checkpoint・正常終了・例外終了時に `loss_components_20260830.csv` へappend/flush
します。console loggerは従来どおり `log_interval=1000` で、CSV頻度とは独立です。
CSV列は以下です（末尾に旧analysis互換aliasも保持します）。

| CSV column | definition |
|---|---|
| `training_step` | 完了したoptimizer step（1始まり） |
| `diffusion_loss` | sampler-weighted diffusion loss |
| `ode_offmask_base_raw` | edge parameterのoff-mask base $R_{base}$ |
| `ode_offmask_after_internal_lambda` | $5R_{base}$ |
| `ode_regularization_final_weighted` | $1\times5R_{base}$ |
| `cell_ode_consistency_raw_20260830` | sampler weight適用前のconsistency |
| `cell_ode_consistency_sampler_weighted_20260830` | sampler weight適用後 |
| `cell_ode_consistency_final_weighted_20260830` | $\lambda_{cons}$ 適用後 |
| `total_loss` | 3つの最終寄与の和 |
| `off_mask_lambda`, `ode_reg_lambda`, `cell_ode_reg_lambda_20260830` | 各係数 |
| `learning_rate` | そのoptimizer updateで使用したLR |

## 既存 soft constraint の weight 構造

soft constraint の reduction と係数は既存実験から変更していません。

```text
base = mean_all_entries(abs((1 - mask) * penalty_parameter))  # ode_reg_norm=l1
L_ODE_soft = off_mask_lambda * base                           # ODE 内部: 5.0
total contribution = ode_reg_lambda * L_ODE_soft              # TrainLoop: 1.0
```

従って既定値では **edge parameter off-mask regularization** のbaseに対する実効係数は
`5.0 * 1.0 = 5.0` です。対象parameterはODEごとに異なり、centeredは`alpha`、
shiftedは`rho`、hill/simpleは`W`です。

| ODE family | off-mask regularization対象 |
|---|---|
| `centered_signed_hill` | `alpha` |
| `shifted_hill_rho` | `rho` |
| `hill_after_linear` | `W` |
| `simple_softplus` | `W` |
新しい consistency 係数はこの二つから独立で、既存 soft constraint を消したり
置き換えたりしません。

## 4つの single ODE

すべて `parameter[target, source]`、すなわち target `i` が source `j` から受ける
項を `[i,j]` に置きます。既存 `build_edge_mask()` は `[source,target]` なので、
factory 境界で一度だけ転置します。全モデルで `K=1`、`gate_mode=none` であり、
LinComb expert、softmax gate、gate network はありません。

### centered_signed_hill

```text
f_i(x) = b_i + sum_j A_ij tanh(alpha_ij log(max(x_j,eps)/theta_ij)) - delta_i x_i
A, theta, delta = softplus(raw) + eps
```

soft constraint 対象は signed parameter `alpha` です。20260816 の direct Hill を
20260817 の single-ODE 化に合わせて移植しました。

### shifted_hill_rho

```text
f_i(x) = b_i + sum_j A_ij (exp(rho_ij)-1)
                   * max(x_j,eps)^2 / (theta_ij^2 + max(x_j,eps)^2)
         - delta_i x_i
```

Hill coefficient は `n=2` 固定で、`rho=0` の regulation contribution は厳密に
0 です。soft constraint 対象は `rho` です。実装元は同じく 20260817 single ODE
です。

### hill_after_linear

```text
z_i = softplus(sum_j W_ij x_j + b_i)
f_i(x) = V_i z_i^2 / (K_i^2 + z_i^2) - delta_i x_i
K, V, delta = softplus(raw) + eps
```

Hill coefficient は `n=2` 固定、soft constraint 対象は `W` です。
`work/20260803_ODE_hill_exp/models/ode_fields.py` の single mode の式、初期化、
全要素 mean penalty を踏襲しています。

### simple_softplus

```text
s = 1 / sqrt(d)
f_i(x) = softplus(s * (sum_j W_ij x_j + b_i)) - softplus(gamma_i) x_i
```

soft constraint 対象は `W` です。`ODE/ode_20260421_regODEMLratio.py::GeneODE`
を基にしていますが、旧 `x @ W` の向きは変更せずに再利用せず、今回専用実装で
`F.linear(x,W,b) = x @ W.T` として `[target,source]` に統一しました。

## CellUnet / diffusion 条件の継承元

比較軸以外は `work/20260817-singleODE` ブランチの
`work/20260817_singleODE/configs/base.json`（その training defaults の元は
`work/20260803_ODE_hill_exp/configs/base.json`）を継承しています。

- Embryonic AnnData、既存 preprocessing 済み入力と `gene_name` 順序
- CellUnet `[2000,1000,500,500]`、dropout `0.1`
- diffusion 1000 step、linear beta、epsilon prediction、MSE、fixed variance
- uniform sampler、batch 128、microbatch `-1`
- AdamW、LR `1e-4`、weight decay `1e-4`
- EMA `0.9999`、seed 1234
- ancestral sampling、3,000 samples、sample batch 50、clip off

データ、HVG/preprocessing、diffusion/model mean/loss、architecture、optimizer、
sampling、seed は比較軸として変更していません。

今回の共通かつ意図的な変更は、全12条件のtraining lengthを30,000から100,000へ
延長することだけです。local TrainLoopのlinear annealingは最初のupdateで`1e-4`を
使い、100,000番目のupdate完了後にoptimizer LRが厳密に0になります。

## Training / sampling条件の再監査

`20260817` は現在branchの作業treeにはsourceがないため、local branch
`work/20260817-singleODE` のcommit `ed376ec`にあるbase configを直接比較しました。
3 suite共通の完全なpathは次のとおりです。

- `data_dir=/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad`
- `edge_tsv_path=/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv`

| condition | 20260803_ODE_hill_exp | 20260817_singleODE | 20260830 final |
|---|---|---|---|
| `data_dir` | Embryonic.h5ad | 同左 | 同左 |
| `edge_tsv_path` | tf_target_edges.tsv | 同左 | 同左 |
| `cell_unet_hidden_num` | `[2000,1000,500,500]` | 同左 | 同左 |
| `diffusion_steps` | 1000 | 1000 | 1000 |
| `noise_schedule` | linear | linear | linear |
| `timestep_respacing` | `""` | `""` | `""` |
| `learn_sigma` | false | false | false |
| `use_kl` | false | false | false |
| `predict_xstart` | false | false | false |
| `rescale_timesteps` | false | false | false |
| `rescale_learned_sigmas` | false | false | false |
| `schedule_sampler` | uniform | uniform | uniform |
| `lr` | 1e-4 | 1e-4 | 1e-4 |
| `weight_decay` | 1e-4 | 1e-4 | 1e-4 |
| `batch_size` | 128 | 128 | 128 |
| `microbatch` | -1 | -1 | -1 |
| `ema_rate` | 0.9999 | 0.9999 | 0.9999 |
| `use_fp16` | false | false | false |
| `total_steps` | 30,000 | 30,000 | **100,000** |
| `lr_anneal_steps` | 30,000 | 30,000 | **100,000** |
| `log_interval` | 1000 | 1000 | 1000 |
| `save_interval` | 5000 | 5000 | 5000 |
| `seed` | 1234 | 1234 | 1234 |
| `SoftReg` / `use_mask_reg` | true / true | true / true | true / true |
| `off_mask_lambda` | 5.0 | 5.0 | 5.0 |
| `ode_reg_lambda` / norm | 1.0 / l1 | 1.0 / l1 | 1.0 / l1 |
| `positive_epsilon` | 1e-6 | 1e-6 | 1e-6 |
| `raw_delta_init` | 0.1 | 0.1 | 0.1 |
| `num_samples` | 3000 | 3000 | 3000 |
| `sample_batch_size` | 50 | 50 | 50 |
| `use_ddim` | false | false | false |
| `clip_denoised` | false | false | false |

20260830のODE固有値は、centered/shiftedについて20260817の
`K=1`, `gate_mode=none`, `regulation_A_init=1`, `theta_init=1`,
`target_chunk_size=16`, `hill_n=2`を維持します。hill-after-linearは20260803由来の
`hill_n=2`, `hill_K_init=1`, `hill_V_init=1`、simple-softplusは歴史的な
`gamma=0.1`, input scale `1/sqrt(d)`を維持します。

### Final fixed conditions

| setting | value |
|---|---:|
| Training steps | **100000** |
| LR | **1e-4** |
| LR anneal steps | **100000** |
| Batch / microbatch | **128 / -1** |
| Diffusion steps | **1000** |
| EMA | **0.9999** |
| `off_mask_lambda` | **5.0** |
| `ode_reg_lambda` | **1.0** |
| Cell–ODE lambda | **0.1 / 1 / 10** |

### Checkpoint容量概算

`save_interval=5000`は変更していません。step 0を含め最終100,000まで約21組です。
FP32では1組（raw model + EMA + Adamの2 moment）は概ねparameter byte数の4倍です。
入力gene数を`d=2000`と仮定すると、centered/shiftedは約0.78 GiB/組（約16.4
GiB/run）、hill/simpleは約0.66 GiB/組（約13.9 GiB/run）、12 run合計は概ね
182 GiB + serialization overheadです。実容量はAnnDataの`n_vars`に依存します。
容量が問題なら`save_interval=10000`で概ね半減できますが、比較条件維持のため今回は
5000のままです。

## Canonical 12条件

| # | experiment | ODE | `cell_ode_reg_lambda_20260830` |
|---:|---|---|---:|
| 1 | `01_centered_signed_hill_lambda0p1` | centered_signed_hill | 0.1 |
| 2 | `02_centered_signed_hill_lambda1` | centered_signed_hill | 1.0 |
| 3 | `03_centered_signed_hill_lambda10` | centered_signed_hill | 10.0 |
| 4 | `04_shifted_hill_rho_lambda0p1` | shifted_hill_rho | 0.1 |
| 5 | `05_shifted_hill_rho_lambda1` | shifted_hill_rho | 1.0 |
| 6 | `06_shifted_hill_rho_lambda10` | shifted_hill_rho | 10.0 |
| 7 | `07_hill_after_linear_lambda0p1` | hill_after_linear | 0.1 |
| 8 | `08_hill_after_linear_lambda1` | hill_after_linear | 1.0 |
| 9 | `09_hill_after_linear_lambda10` | hill_after_linear | 10.0 |
| 10 | `10_simple_softplus_lambda0p1` | simple_softplus | 0.1 |
| 11 | `11_simple_softplus_lambda1` | simple_softplus | 1.0 |
| 12 | `12_simple_softplus_lambda10` | simple_softplus | 10.0 |

launcher と config validator の両方がこの順序・ODE・lambda の組を固定します。

## Commands

全12条件を train → sample → analysis の順で実行:

```bash
/path/to/scdiffusion/bin/python work/20260830/scripts/launch.py \
  --batch-id main-20260830
```

全diffusion timestepのanalysisを行う本番pipelineでは `--analysis-full` を追加します。
全12条件のper-run analysis後、launcherは12条件summaryも自動生成します。

background 実行:

```bash
/path/to/scdiffusion/bin/python work/20260830/scripts/launch.py \
  --batch-id main-20260830 --analysis-full --background
```

1条件のみ:

```bash
/path/to/scdiffusion/bin/python work/20260830/scripts/launch.py \
  --experiment 01_centered_signed_hill_lambda0p1 --batch-id main-20260830
```

同じ run directory の最新 complete raw checkpoint から resume:

```bash
/path/to/scdiffusion/bin/python work/20260830/scripts/launch.py \
  --experiment 01_centered_signed_hill_lambda0p1 \
  --batch-id main-20260830 --resume auto
```

sampling only / analysis only:

```bash
/path/to/scdiffusion/bin/python work/20260830/scripts/launch.py \
  --experiment 01_centered_signed_hill_lambda0p1 \
  --batch-id main-20260830 --sample-only

/path/to/scdiffusion/bin/python work/20260830/scripts/launch.py \
  --experiment 01_centered_signed_hill_lambda0p1 \
  --batch-id main-20260830 --analysis-only
```

実行計画だけを確認:

```bash
/path/to/scdiffusion/bin/python work/20260830/scripts/launch.py \
  --batch-id main-20260830 --dry-run
```

## Test / smoke

PyTorch を含む repository の環境で実行します。

```bash
/path/to/scdiffusion/bin/python -m unittest -v \
  work/20260830/tests/test_experiment_20260830.py

/path/to/scdiffusion/bin/python work/20260830/tests/smoke.py
```

test は共有3ファイルの baseline SHA-256、既存 CellUnet import/forward、4 ODE の
shape/single性、matrix convention、mask 転置、penalty 対象、wrapper の Cell-only
出力、ODE 非混合、両 branch gradient、eval 時 ODE 非呼出し、lambda=0、sampler
weight、独立 logging key、12 config を検証します。smoke は全4 ODE で実際の
Gaussian diffusion `training_losses`、backward、eval forward を通します。
さらに専用 TrainLoop を microbatch 2 で1 step実行し、optimizer、EMA、checkpoint、
既存 loss CSV と新しい component CSV の生成まで確認します。

## Outputs

すべて `work/20260830/runs/<experiment>/<batch-id>/` 以下です。

```text
exp_config.json
model_info.json
logs/train/
checkpoints/segment_NNN/model/{model,ema,opt}*.pt
checkpoints/segment_NNN/model/loss_details.csv
checkpoints/segment_NNN/model/loss_components_20260830.csv
samples/*.npz
analysis/detailed_20260830/{analysis_config.json,cell_indices.npy}
analysis/detailed_20260830/csv/*.csv
analysis/detailed_20260830/figures/*.png
```

## Detailed post-hoc analysis

training/optimizer/samplingコードを変更せず、checkpointを復元して解析する実装を
`analysis/` に追加しています。通常metricは `torch.no_grad()`、gradient解析は
`torch.autograd.grad()` のみを使い、optimizerを作成・更新しません。詳細なmetric、
loss weightの定義、16 figure一覧は
`work/20260830/analysis/README.md` を参照してください。

同一12条件では同じdataset cell index、noise seed、diffusion timestepを使います。
defaultは2,048 cellsと既存timestep grid `0,249,499,616,749,999`、quickは128 cells、
fullは最大4,096 cellsと全diffusion timestepです。
metricとgradient CSVはdiffusion timestep/checkpoint groupごとに途中保存され、同じ
analysis条件で再実行すると完了済みgroupをskipします。loss historyは各optimizer
stepの最大100,000行を読み、rolling median/Q25/Q75を計算します。

```bash
# analysis smoke test
/path/to/scdiffusion/bin/python work/20260830/tests/analysis_smoke.py

# single run analysis
/path/to/scdiffusion/bin/python work/20260830/scripts/analyze.py \
  --run-dir work/20260830/runs/01_centered_signed_hill_lambda0p1/main-20260830

# all 12 conditions quick
/path/to/scdiffusion/bin/python work/20260830/scripts/analyze.py \
  --all-runs --batch-id main-20260830 --quick

# all 12 conditions full (4096 cells, diffusion_timestep 0..999)
/path/to/scdiffusion/bin/python work/20260830/scripts/analyze.py \
  --all-runs --batch-id main-20260830 --full

# completed per-run CSVからsummaryだけ再生成
/path/to/scdiffusion/bin/python work/20260830/scripts/analyze.py \
  --all-runs --batch-id main-20260830 --summary-only

# checkpoint gradient analysisだけ再実行
/path/to/scdiffusion/bin/python work/20260830/scripts/analyze.py \
  --run-dir work/20260830/runs/01_centered_signed_hill_lambda0p1/main-20260830 \
  --gradient-only --gradient-timesteps 0,499,999 --force
```

custom timestepは `--timesteps 0,1,2,5-100:5,200-999:25` のように指定できます。
CSVではoptimizer軸を `training_step`、forward diffusion軸を
`diffusion_timestep` と明記しています。
