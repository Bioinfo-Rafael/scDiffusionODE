# 2026-07-07 tests

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連するunit testとsmoke testを保存します。test codeは編集可能ですが、期待値を変更するときは仕様変更との対応を明記してください。

主な確認項目:

- `regime_gate_mode="none"` が従来出力と一致すること
- `Ts_I_vs_II_III` のsigmoid gateの向き
- `t > T_s` でCellUnet、`t <= T_s` でODE/LinCombが強いこと
- hard gateの境界
- raw L1 sparse / softmax entropy正則化の値とgradient
- sparse/dense AnnDataでの `lambda_max` / `T_s` 推定
- checkpoint save/restoreとmetadata
- tiny training smoke test
- pipeline dry-run

```bash
$HOME/miniconda3/envs/scdiffusion/bin/python \
  -m unittest discover -s work/20260707_lincomb/tests -v
```

`hybrid_ts_soft_lincomb` を変更した場合は、上記testに加えてpipelineの `--dry-run` も確認してください。

