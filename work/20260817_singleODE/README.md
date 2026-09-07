# 2026-08-17 direct-Hill single ODE comparison

`work/20260816` の2つのdirect-Hill式を、20260803の
`Standard Hybrid + single ODE` と同じ「1個のODE + CellUNet」構造で比較する
独立suiteです。`work/20260816` のコード、run、checkpointは変更しません。

## Single ODE

全条件で1個だけのODEを使用します。

\[
\frac{dx_i}{dt}=b_i+\sum_j A_{ij}H_{ij}(x_j)-\delta_i x_i
\]

Centered signed Hill:

\[
H_{ij}(x_j)=\tanh\left(\alpha_{ij}\log\frac{x_j}{\theta_{ij}}\right)
\]

Learnable shifted Hill:

\[
H_{ij}(x_j)=\operatorname{expm1}(\rho_{ij})
\frac{x_j^2}{\theta_{ij}^2+x_j^2}
\]

`A = softplus(raw_A) + positive_epsilon` は非負magnitudeで、符号は
`alpha` / `rho` だけが持ちます。off-mask正則化もno-regulationを直接表す
`alpha` / `rho` に適用します。非正のdiffusion stateにはlog/power計算時だけ
`clamp_min(x, positive_epsilon)` を使います。`Wx+b -> softplus -> Hill` は
ありません。

このsuiteにはODE内部のLinCombがありません。

- `K = 1`
- `gate_mode = none`
- ODE componentは1個
- `coeff_net` / ODE用time embedding / softmax gateは作成しない

## Four experiments

以下の順番です。

1. `linear_singleODE_centered_signed_hill`
2. `linear_singleODE_shifted_hill_rho`
3. `ts_soft_tau80_singleODE_centered_signed_hill`
4. `ts_soft_tau80_singleODE_shifted_hill_rho`

Linear Hybrid:

\[
output=\left(1-\frac{t}{T-1}\right)ODE
+\frac{t}{T-1}CellUNet
\]

TS_soft_tau80 Hybrid:

\[
w_{ODE}(t)=\operatorname{sigmoid}\left(\frac{t_s-t}{80}\right)
\]

\[
output=w_{ODE}(t)ODE+(1-w_{ODE}(t))CellUNet
\]

TS条件は `regime_gate_mode=Ts_I_vs_II_III`、`regime_gate_type=sigmoid`、
`t_s=auto`、`gate_tau=80.0` です。全条件の `total_steps` と
`lr_anneal_steps` は30,000です。

## Commands

4モデルを順番に学習し、sampling、通常解析、20-step解析まで実行:

```bash
python work/20260817_singleODE/scripts/launch.py --background \
  --batch-id single_ode_hill_20260817
```

学習だけを実行:

```bash
python work/20260817_singleODE/scripts/launch.py --train-only --background \
  --batch-id single_ode_hill_20260817
```

既存の学習済みrunにsamplingだけを実行:

```bash
python work/20260817_singleODE/scripts/launch.py --sample-only \
  --batch-id single_ode_hill_20260817
```

通常解析と20-step解析を実行:

```bash
python work/20260817_singleODE/scripts/launch.py --analysis-only \
  --batch-id single_ode_hill_20260817
```

出力先:

```text
work/20260817_singleODE/runs/<experiment>/<batch-id>/
work/20260817_singleODE/batches/<batch-id>/
```

Local verification:

```bash
python -m unittest -v work/20260817_singleODE/tests/test_models.py
python work/20260817_singleODE/scripts/launch.py --smoke \
  --batch-id smoke_single_ode_20260817
```

Remote GPU:

```bash
git fetch origin
git checkout work/20260816-hill-variants
git pull --ff-only origin work/20260816-hill-variants
python work/20260817_singleODE/scripts/launch.py --background \
  --batch-id single_ode_hill_20260817
```
