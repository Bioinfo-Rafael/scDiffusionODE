# 2026-07-07 LinComb / hybrid regime gate 実験 suite

このディレクトリは、2026-07-07 LinComb / hybrid regime gate 実験に関連するコード、設定、run、可視化、比較結果をまとめるrootです。

## 対象実験

- `hybrid_reverse_lincomb`: 既存schedulerのODE/CellUnet係数を反転
- `lincomb_only_raw`: raw coefficientのLinComb-only baseline
- `lincomb_softmax_gate`: 正・総和1のsoftmax gate
- `lincomb_sparse_reg`: raw coefficientへL1正則化
- `lincomb_entropy_reg`: softmax gateへentropy正則化
- `hybrid_ts_soft_lincomb`: `T_s` に基づくsoft regime gateを使用

`hybrid_ts_soft_lincomb` では、次のODE側重みを使います。

```python
w_ode = sigmoid((t_s - t) / gate_tau)
out = w_ode * ode_term + (1.0 - w_ode) * cellunet_term
```

```text
Regime I        : t > T_s  → w_odeが小さい → CellUnetが強い
Regime II, III  : t <= T_s → w_odeが大きい → ODE/LinCombが強い
t = T_s         : w_ode ≈ 0.5
```

`T_s` は、モデル入力と同じAnnData空間から推定した `lambda_max` とdiffusion scheduleを使い、`alpha_bar[t] * lambda_max ≈ 1` となるtimestepとして求めます。

## ディレクトリ

- `configs/`: 共通設定と実験別config
- `scripts/`: train → sample → viz → summaryのentrypoint
- `utils/`: `lambda_max` / `T_s` 推定helper
- `runs/`: training run本体と再現用metadata
- `samples/`: sampling出力
- `viz/`: 可視化runnerと生成画像・CSV
- `logs/`: stageごとの実行log
- `results/`: 複数runの比較結果とbaseline registry
- `tests/`: unit / smoke test

生成物である `runs/`、`samples/`、`logs/`、`viz/<experiment>/` は原則として手動編集しません。

## 実行例

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json
```

複数実験を順番に実行します。

```bash
python work/20260707_lincomb/scripts/run_matrix_0707.py \
  --config-dir work/20260707_lincomb/configs
```

このコマンドは、各configについて `run_pipeline_0707.py` を呼び出し、1条件ごとに
`train → sample → viz → summary` を完了してから次の条件へ進みます。実行順は次です。

1. `hybrid_reverse_lincomb`
2. `lincomb_only_raw`
3. `lincomb_softmax_gate`
4. `lincomb_sparse_reg`
5. `lincomb_entropy_reg`
6. `hybrid_ts_soft_lincomb`

実ファイルを作らず、解決後の設定とコマンドを確認します。

```bash
python work/20260707_lincomb/scripts/run_pipeline_0707.py \
  --config work/20260707_lincomb/configs/hybrid_ts_soft_lincomb.json \
  --dry-run
```

詳細は各サブディレクトリの `README.md` を参照してください。
