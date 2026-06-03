# external_data/
このディレクトリには、scDiffusionで使用する外部参照データを保存しています。

## 重要ファイル
以下の2つのTSVファイルは、GitHub移行後も残す方針です。

```text
Mouse_tf_target_edges.tsv
tf_target_edges.tsv
```

## .gitignore 方針
このディレクトリ内のTSVファイルは除外しない方針です。

特に以下は `.gitignore` に追加しないでください。

```text
external_data/
external_data/*.tsv
*.tsv
```
