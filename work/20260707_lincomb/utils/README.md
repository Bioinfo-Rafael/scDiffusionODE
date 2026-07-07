# 2026-07-07 regime time utility

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連する軽量utilityを置きます。

- `regime_time.py`: AnnDataから `lambda_max` を推定し、`alpha_bar[t] * lambda_max ≈ 1` から `T_s` を決定
- `__init__.py`: utility package定義

`estimate_lambda_max_from_adata()` はcellを先にsampleしてから必要部分だけdense化します。`estimate_ts_from_lambda()` はdiffusion schedule上で境界timestepを選びます。

`hybrid_ts_soft_lincomb` の `t_s="auto"` でtraining前に呼ばれます。実験コードなので編集可能ですが、変更後は `../tests/test_regime_time.py` を実行してください。

