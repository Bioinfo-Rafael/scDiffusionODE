# 2026-07-07 visualization

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連する可視化runnerと生成結果の保存先です。Python scriptは編集可能ですが、実験ごとの画像・CSV・cacheは生成物なので手動編集しません。

- `run_all_viz_0707.py`: 既存可視化と0707専用可視化を一括実行
- `plot_gate_diagnostics_0707.py`: coefficient、entropy、Superclass、gate curveを保存
- `run_velocity_suite_0707.py`: UMAP/velocity graphを共有してtotal/component UMAPとPAGAを生成

主な生成物:

- loss plot
- parameter / effective W visualization
- model I/O diagnostics
- LinComb coefficient diagnostics
- gate curve
- velocity UMAP / component velocity UMAP
- component PAGA
- Superclass coefficient / contribution plot

`hybrid_ts_soft_lincomb` では `gate_curve.csv`、互換名 `regime_gate_curve.csv`、`gate_curve.png` を保存します。`w_ode` はODE/LinComb側、`1-w_ode` はCellUnet側の重みです。

現行CSVの物理列:

```text
t, alpha_bar, w_ode
```

`regime_label` は `t > T_s` ならRegime I、`t <= T_s` ならRegime II/IIIとして、この3列とconfigの `T_s` から一意に導出できます。

```bash
python work/20260707_lincomb/viz/run_all_viz_0707.py \
  --run-dir work/20260707_lincomb/runs/<experiment>/<run_id>
```

