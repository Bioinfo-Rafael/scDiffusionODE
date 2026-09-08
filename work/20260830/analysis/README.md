# Detailed analysis outputs

通常metricは `torch.no_grad()` で計算し、gradient解析だけを別関数で
`torch.autograd.grad()` により計算します。optimizerは作成せず、`backward()` と
optimizer stepは実行しません。各gradient checkpointの前後でstate fingerprintを
比較します。

## training実装review

`CellUNetODERegularized20260830` が `ml_model` と `ode_model` を子moduleとして
登録しているため、既存 `MixedPrecisionTrainer(model)` の `model.parameters()` と
AdamWには両方が含まれます。consistencyは
`(cell_out - ode_out).square()` でありdetachはありません。ODE parameterは通常の
`nn.Parameter` なので `requires_grad=True` です。eval分岐はCell出力直後にreturnし、
ODE forwardを呼びません。config validatorは4 ODE × lambda 0.1/1/10の12条件を
固定します。

## loss mapping

```text
ode_regularization_base_before_off_mask_lambda
  = mean((1-mask) * abs(penalty_parameter))

ode_regularization_raw
  = base（weight適用前）

ode_regularization_weighted_once_by_off_mask_lambda
  = off_mask_lambda(5.0) * raw
  = saved ode_soft_constraint

ode_regularization_weighted
  = ode_reg_lambda(1.0) * weighted_once_by_off_mask_lambda
  = final total-loss contribution

cell_ode_consistency_raw_20260830
  = sampler weight適用前のper-cell gene MSE平均

cell_ode_consistency_sampler_weighted_20260830
  = schedule sampler weight適用後

cell_ode_consistency_weighted_20260830
  = cell_ode_reg_lambda_20260830 * sampler-weighted consistency
```

`training_step` はoptimizer/training軸、`diffusion_timestep` はforward diffusionの
`0..999` 軸として全CSV・figureで区別します。

## per-run CSV

- `diffusion_metrics_by_timestep.csv`
- `cell_ode_metrics_by_timestep.csv`
- `loss_history.csv`
- `loss_fraction.csv`
- `gradient_metrics.csv`

## figures

`01`–`12`はrunごと、`13`–`16`は12条件summaryとして生成します。

1. `01_cell_target_metrics_vs_t.png`
2. `02_cell_ode_corr_cos_vs_t.png`
3. `03_cell_ode_mse_nmse_vs_t.png`（full range、linear）
4. `04_cell_ode_norm_vs_t.png`
5. `05_cell_ode_norm_ratio_vs_t.png`
6. `06_cell_ode_metrics_zoom.png` と `06_cell_ode_metrics_zoom_tge1.png`
7. `07_cell_ode_metrics_log.png`（非正値はlog表示のみ除外）
8. `08_loss_components_raw.png`
9. `09_loss_components_weighted_log.png`
10. `10_loss_contribution_fraction.png`
11. `11_gradient_norms.png`
12. `12_gradient_cosine.png`
13. `13_lambda_comparison.png`
14. `14_ode_family_comparison.png`
15. `15_condition_summary_heatmap_raw.png`
16. `16_condition_summary_heatmap_standardized.png`（列ごとのpopulation z-score）

full plotはoutlierを含む全範囲、zoomはtitleとfilenameに `t>=1` を明記します。
percentileによるylim切断はしません。

loss figure `08` と `09` はraw scatterを表示せず、既定100 optimizer stepの
rolling meanを線、rolling mean +/- population stdを帯として表示します。CSVには
mean/stdに加え、既存解析との互換性のためmedian/Q25/Q75も残します。

## hill_after_linear parameter distributions

`scripts/plot_hill_after_linear_parameters.py` は `hill_after_linear` の6条件を、
行にconsistency weight `10, 1, 0.1, 0.01, 0.001, 10 -> 0.001 (exp)`、列に
raw `model000000.pt` と5,000 step刻みのEMA
`ema_0.9999_005000.pt`〜`ema_0.9999_030000.pt`として比較します。`runs` と
`runs2` から、それぞれ3条件すべてに共通し必要checkpointが揃う最新batchを
自動選択します。`model000000.pt` は厳密な学習前stateではなく、legacy TrainLoopが
最初のoptimizer update後にstep 0という名前で保存したraw checkpointです。
再現性のため `--runs-batch-id` と `--runs2-batch-id` でbatchを
明示することもできます。

既定出力は `01`〜`03` の `W` 全体、mask内、mask外（mask=0の対角成分を含む）と、
`11`〜`13` のCellUnet全parameter、全weight、全biasの6 PNGです。200 binsを使い、
6 PNGすべてで `x=-0.5..0.5`, `y=0..30 percent/bin` とbin境界を共有します。
x範囲外の値はbarから除外しますが、CSVとpanel内のmean/stdには含めます。
`--all-parameters` を付けると、
ODEの `b`, `raw_K`, `raw_V`, `raw_delta`、式で使う変換後の `K`, `V`, `delta`、
およびstate dict内のCellUnet各学習parameterもそれぞれ1 PNGで保存します。
`--independent-x` を付けた場合だけPNGごとに横軸を決めます。すべてのpanelはraw
pointを描かず、histogram、mean、population std、要素数を表示します。CSVには
表示対象のfull-range要約統計、metadataには選択batchと全checkpoint pathを記録します。

```bash
/path/to/scdiffusion/bin/python \
  work/20260830/scripts/plot_hill_after_linear_parameters.py \
  --output-dir work/20260830/analysis_results/hill_after_linear_parameter_distributions/example
```

作成済みの `std_over_mean/` 内の6 CSVは、次の専用scriptで元parameter groupごとに
1枚の折れ線図へ変換できます。横軸はtraining step、縦軸は `std_over_mean`、線は
6種類のconsistency weightです。CSVの7点をそのまま結び、smoothingや補間はしません。

```bash
/path/to/scdiffusion/bin/python work/20260830/scripts/plot_std_over_mean.py \
  --input-dir /path/to/std_over_mean
```
