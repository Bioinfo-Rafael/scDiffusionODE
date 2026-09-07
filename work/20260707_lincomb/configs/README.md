# 2026-07-07 実験config

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連する共通設定と実験別設定を保存します。JSONは実験条件なので、変更時は差分を確認してください。

- `base.json`: データ、学習、LinComb、正則化、保存先などの共通設定
- `hybrid_reverse_lincomb.json`: 既存schedulerのODE/CellUnet係数を反転
- `hybrid_ts_soft_lincomb.json`: `T_s` soft regime gateを使うhybrid
- `lincomb_only_raw.json`: raw coefficientのLinComb-only baseline
- `lincomb_softmax_gate.json`: LinComb係数をsoftmax gateとして使用
- `lincomb_sparse_reg.json`: raw coefficientへL1 sparse regularizationを追加
- `lincomb_entropy_reg.json`: softmax gateへentropy regularizationを追加

`hybrid_ts_soft_lincomb.json` の中心設定は次です。

```json
{
  "denoiser_mode": "hybrid",
  "regime_gate_mode": "Ts_I_vs_II_III",
  "regime_gate_type": "sigmoid",
  "t_s": "auto",
  "gate_tau": 20.0
}
```

`t_s="auto"` の場合、学習入力と同じAnnData layerから `lambda_max` を推定し、`alpha_bar[t] * lambda_max ≈ 1` に最も近いtimestepを `T_s` にします。`t > T_s` ではCellUnet、`t <= T_s` ではODE/LinCombが強くなります。

実行例:

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json --dry-run
```

