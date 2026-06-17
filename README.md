# scDiffusion
このリポジトリは、`scDiffusion` のGitHub移行用に整理した研究コードリポジトリです。

最終更新: 2026-06-03 07:53:38

## 概要
このリポジトリには、scRNA-seqデータに対する diffusion model 関連の研究コード、ODE/VAE関連コード、データ準備コード、外部参照データ、最近の実験作業ディレクトリ、移行前の旧ファイル退避先が含まれています。

## 主要ディレクトリ

```text
guided_diffusion/   diffusion model本体コード
ODE/                ODE関連コード
VAE/                VAE関連コード
data_preparation/   データ準備用コード・ノートブック
external_data/      外部参照データ
docs/               環境・説明・移行記録
work/               最近の実験作業ディレクトリ
LegacyFiles/        旧スクリプト・旧実験結果・root退避ファイル
```

## GitHub移行前の整理方針
GitHub移行前の整理では、削除ではなく退避を基本方針にしました。

root直下に散らばっていた古い学習スクリプト、サンプリングスクリプト、分類器関連スクリプト、notebook、log、test/debug/check系ファイルなどは、原則として `LegacyFiles/` に退避しています。

環境情報、説明資料、参照ファイル、移行スナップショットは `docs/` 配下に整理しています。

## 最近の実験（20260609）
`work/20260609_*` に ODE×ML hybrid denoising model 群（5 ODE 枝 × hybrid mode）と役割別の可視化スイートがある。
hybrid mode は `none` / `ratio_reg` / `scale_model`（方向×scalar scale。新規）＋ `normed_learned_scale`（deprecated）。
詳細・実行手順は [`work/20260609_Hybrid5x3/README.md`](work/20260609_Hybrid5x3/README.md) と
[`work/20260609_Hybrid5x3/PIPELINE_TEST.md`](work/20260609_Hybrid5x3/PIPELINE_TEST.md)、
ODE モデル本体は [`ODE/README_20260609.md`](ODE/README_20260609.md) を参照。

## 注意
この整理作業では、`git add`、`git commit`、`git push` はまだ実行していません。

詳細は以下を参照してください。

```text
docs/migration/repository_cleanup_notes.md
```
