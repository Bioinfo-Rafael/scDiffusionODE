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
  = sampler-weighted per-cell gene MSE

cell_ode_consistency_weighted_20260830
  = cell_ode_reg_lambda_20260830 * raw consistency
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
