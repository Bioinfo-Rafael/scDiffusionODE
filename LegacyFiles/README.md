# LegacyFiles/
このディレクトリには、GitHub移行前にroot直下から退避した古いファイルや、旧実験スクリプト・旧実験結果を保存しています。

## 目的
root直下を整理しつつ、研究履歴を削除せずに残すための退避先です。

## 主な内容

```text
exp_script/   古い実験スクリプト・notebook
exp_result/   古い実験結果
metrics/      旧metrics関連コード
```
また、root直下にあった古い `.py`、`.sh`、`.ipynb`、`.log`、test/debug/check系ファイルもここに退避する方針です。

## 注意
このディレクトリは削除対象ではありません。

`.gitignore` でも `LegacyFiles/` は除外しない方針です。
