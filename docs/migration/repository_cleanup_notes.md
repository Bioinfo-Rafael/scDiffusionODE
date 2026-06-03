# GitHub移行前リポジトリ整理記録
最終更新: 2026-06-03 07:53:38

## 目的
この文書は、`scDiffusion-github/` をGitHubへ移行する前に行った整理作業の記録です。

目的は、研究コードや実験履歴を削除せずに保存しつつ、GitHub上で見通しのよい構造に整理することです。

## 基本方針

- 削除ではなく、原則として退避・整理を行いました。
- 本体コードはroot直下に残しました。
- 最近の実験作業は `work/` に残しました。
- 外部参照データは `external_data/` に残しました。
- 古いスクリプトや古い実験結果は `LegacyFiles/` に集約しました。
- 環境情報、説明資料、参照資料、移行記録は `docs/` に集約しました。
- `git add`、`git commit`、`git push` は実行していません。

## 削除について
この整理作業では、ファイルやディレクトリを意図的に削除していません。

行った作業は、主に以下です。

```text
移動
整理
Markdownによる説明文書の追加
.gitignore の見直し
```

## root直下に残すディレクトリ

```text
guided_diffusion/
ODE/
VAE/
data_preparation/
external_data/
docs/
work/
LegacyFiles/
```

## docs/ に移動したもの

```text
environment/        -> docs/environment/
documentation/      -> docs/documentation/
file_changes/       -> docs/file_changes/
reference_files/    -> docs/reference_files/
migration_snapshot/  -> docs/migration/
```

## LegacyFiles/ に移動したもの

```text
exp_script/ -> LegacyFiles/exp_script/
exp_result/ -> LegacyFiles/exp_result/
metrics/    -> LegacyFiles/metrics/
```
また、root直下にあった古い `.py`、`.sh`、`.ipynb`、`.log`、test/debug/check系ファイルも、存在するものは `LegacyFiles/` に退避する方針としました。

## external_data/ について
以下の2つの大きなTSVファイルは、GitHub移行後も残す方針です。

```text
external_data/Mouse_tf_target_edges.tsv
external_data/tf_target_edges.tsv
```
そのため、`.gitignore` では `external_data/*.tsv` や `*.tsv` を除外しない方針にしています。

## .gitignore の方針
`.gitignore` では、Pythonキャッシュ、Jupyterチェックポイント、一時ファイル、ローカル環境、巨大な生成物などを中心に除外します。

一方で、以下は除外しない方針です。

```text
external_data/
external_data/*.tsv
LegacyFiles/
work/
docs/
guided_diffusion/
ODE/
VAE/
```

## Git操作について
この整理作業中には以下を実行していません。

```text
git add
git commit
git push
```
このディレクトリは、作業時点ではまだGitリポジトリとして初期化されていない可能性があります。

## 次に行う確認
GitHub公開前に、以下を確認します。

- root直下の構造
- repository全体サイズ
- 20MB超ファイル一覧
- `.gitignore` の影響
- `external_data/*.tsv` が除外されていないこと
- `git add --dry-run .`
