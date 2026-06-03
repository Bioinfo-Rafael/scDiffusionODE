# LegacyFiles/metrics/
このディレクトリには、旧metrics関連コードを保存しています。

## 移動元

```text
metrics/ -> LegacyFiles/metrics/
```

## 移動理由
GitHub移行後のroot直下の構造を簡潔にするため、目標構造に含まれていなかった `metrics/` を退避しました。

移動前に、本体コード側から `metrics/` ディレクトリが参照されていないか確認しました。

grepで検出されたのは、`metrics/gene_network_metrics.py` 内部の `sklearn.metrics` 参照であり、root側の本体コードから `metrics/` を直接参照するものではありませんでした。
