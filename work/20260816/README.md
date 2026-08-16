# 2026-08-16 direct regulator Hill ODE comparison

`work/20260803_ODE_hill_exp` の学習条件、K=8 LinComb + softmax gate、CellUNet、
optimizer、TF-target mask（行列規約 `[target, source]`）、off-mask 正則化を維持し、
ODE式と Hybrid coefficient だけを比較する独立実験です。全条件で

\[
\frac{dx_i}{dt}=b_i+\sum_j A_{ij}H_{ij}(x_j)-\delta_i x_i
\]

を使います。`A = softplus(raw_A) + positive_epsilon` は非負の regulation
magnitude で、符号は `alpha` または `rho` だけが持ちます。`theta` と `delta`
も同じ positive parameterization です。20260803の `W` / `r` と同様に、
off-mask正則化はno-regulationを直接表すsigned parameter（`alpha` / `rho`）へ
適用します。

## ODE variants

Centered signed Hill:

\[
H_{ij}(x_j)=\tanh\left(\alpha_{ij}\log\frac{x_j}{\theta_{ij}}\right)
\]

`alpha > 0` はactivation、`alpha < 0` はinhibition、`alpha = 0` は厳密な
no regulationです。外側の `1/2` はありません。

Learnable shifted Hill:

\[
H_{ij}(x_j)=\operatorname{expm1}(\rho_{ij})
\frac{x_j^n}{\theta_{ij}^n+x_j^n}
\]

`rho` の符号がactivation/inhibitionを決め、`rho = 0` は厳密なno regulation
です。20260803の設定を引き継ぎ、`n = 2.0` は固定、`theta` の物理初期値は
`1.0` です。

diffusion 中の非正値に対してのみ、両ODEで
`x_pos = clamp_min(x, positive_epsilon)` をlog/power計算前に使用します。
これはnumerical guardであり、生物学的変換ではありません。各regulator
`x_j` に直接Hill応答を適用し、`Wx+b -> softplus -> Hill` のlatent variableは
存在しません。

## Four experiments and order

1. `linear_centered_signed_hill`
2. `linear_shifted_hill_rho`
3. `ts_soft_tau80_centered_signed_hill`
4. `ts_soft_tau80_shifted_hill_rho`

Linear条件は既存Hybridの `r(t) = 1 - t/(T-1)`、TS条件は20260801/20260803
と同じ `Ts_I_vs_II_III`、sigmoid、`t_s=auto`、`gate_tau=80.0` を使用します。
どちらもODE内部のLinComb係数ではなく、ODEとCellUNetのHybrid係数です。

## Commands

単一モデル学習（30,000 steps）:

```bash
python work/20260816/scripts/train.py \
  --config work/20260816/configs/linear_centered_signed_hill.json
```

4モデルを上記の順で連続学習:

```bash
python work/20260816/scripts/launch.py --train-only --background \
  --batch-id hill_variants_20260816
```

学習、sampling、通常解析を一括実行する場合は `--train-only` を外します。
出力は `work/20260816/runs/<experiment>/<batch-id>/`、launcher状態とlogは
`work/20260816/batches/<batch-id>/` に保存されます。

Sampling:

```bash
python work/20260816/scripts/sample.py \
  --run-dir work/20260816/runs/linear_centered_signed_hill/hill_variants_20260816
```

学習済みcheckpointの20-timestep post-hoc解析:

```bash
python work/20260816/scripts/analyze_corr_norm_grid.py \
  --run-dir work/20260816/runs/linear_centered_signed_hill/hill_variants_20260816 \
  --step 20
```

解析gridは `diffusion.num_timesteps` から生成し、0と最終有効timestepを必ず
含みます。解析ディレクトリには `timestep_metrics.csv`、
`diagnostic_manifest.json`、correlation/raw norm/weighted norm plotを保存します。

Local verification:

```bash
python -m unittest -v work/20260816/tests/test_models.py
python work/20260816/scripts/launch.py --smoke --batch-id smoke_20260816
```

Remote GPUでpull後に実行:

```bash
git fetch origin
git checkout work/20260816-hill-variants
git pull --ff-only origin work/20260816-hill-variants
python work/20260816/scripts/launch.py --train-only --background \
  --batch-id hill_variants_20260816
```
