# 2026-07-07 比較結果

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連する複数run・複数experimentの集約結果を保存します。

- `comparison_summary.csv`: 表計算・解析用の比較表
- `comparison_summary.json`: 同じ比較情報のJSON
- `experiment_summary.csv`: experiment単位の集約を置く場合の標準名
- `comparison_report.md`: 論文・報告向けの比較メモを置く場合の標準名
- `baseline_registry.json`: 新規学習しない既存baselineのpathと扱い

`hybrid_ts_soft_lincomb` の行には `lambda_max`、`T_s`、`gate_tau`、gate端点の重みを含めます。論文用・報告用の比較表はここを元データとし、run個別の詳細は `../runs/`、図は `../viz/` を参照してください。

`baseline_registry.json` は参照先を変える場合のみ慎重に編集し、自動生成されたsummaryは手動修正せず再生成します。

```bash
python work/20260707_lincomb/scripts/summarize_results_0707.py
```

