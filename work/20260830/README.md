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

ログ名は `diffusion_loss`、`ode_soft_constraint`、
`ode_soft_constraint_weighted`、`cell_ode_consistency_20260830`、
`cell_ode_consistency_weighted_20260830`、`total_loss` です。checkpoint 保存時に
`loss_components_20260830.csv` にも独立列で記録します。

## 既存 soft constraint の weight 構造

soft constraint の reduction と係数は既存実験から変更していません。

```text
base = mean_all_entries(abs((1 - mask) * penalty_parameter))  # ode_reg_norm=l1
L_ODE_soft = off_mask_lambda * base                           # ODE 内部: 5.0
total contribution = ode_reg_lambda * L_ODE_soft              # TrainLoop: 1.0
```

従って既定値では off-mask base に対する実効係数は `5.0 * 1.0` です。
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
- EMA `0.9999`、30,000 steps、seed 1234
- ancestral sampling、3,000 samples、sample batch 50、clip off

データ、HVG/preprocessing、diffusion/model mean/loss、architecture、optimizer、
sampling、seed は比較軸として変更していません。

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

background 実行:

```bash
/path/to/scdiffusion/bin/python work/20260830/scripts/launch.py \
  --batch-id main-20260830 --background
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
analysis/sample_summary.json
```
