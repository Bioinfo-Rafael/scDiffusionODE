# 実装・検証記録（2026-10-08 JST）

## 状態

ソースはローカルcheckoutの `work/20261008_ode_optimizer/` に準備済み。リモート接続先が未指定のため、`~/Projects/scDiffusion-github`への配置、リモートでのimport/回帰テスト/CPU smoke testは未実施。ユーザーのローカル実行不要という指定に従い、ローカル学習・sampling・runtime testも実行していない。実際の科学的結果・UMAP画像が得られたとは主張しない。

## 実施済みの静的検査

- Python 15ファイルをAST parse: PASS。
- JSON config 5ファイル（共通base+指定4条件）: PASS。
- 作業開始時の既存911ファイルのSHA-256再照合: PASS、変更0。
- 新規ディレクトリ外への追加ファイル: 0。
- 作業開始前からある既存checkoutの編集はそのまま保持。

機械可読結果: `static_checks.json`。この検査はimportやruntime smokeの代替ではない。

## 用意した実行テスト

`tests/test_suite.py`に9つの回帰テストと1つの完全pipeline smoke testを実装。optimizer集合の排他/網羅、λ=0の勾配、両optimizerとEMAのresume一致、実SGD更新式、初期化/データ順一致、旧samplerとの全1000更新および途中状態一致、FP16拒否、dry-run無変更、既存ファイル非変更を検証する。完全smokeでは4条件のCPU学習・sampling・図01〜12・24枚のUMAP・4×2比較PNG・座標からの再描画・λ=0予測差0を検証する。

`python -B work/20261008_ode_optimizer/tests/smoke.py` が実際の回帰+完全smoke実行コマンド。成功/失敗は `validation/smoke_result.json` に保存する。現時点では未実行。

## 追加ファイル一覧

- `.gitignore`
- `README.md`
- `analysis/__init__.py`
- `analysis/parameter_plots.py`
- `configs/base.json`
- `configs/lambda0.json`
- `configs/lambda0p001.json`
- `configs/lambda0p1.json`
- `configs/lambda1.json`
- `scripts/analyze.py`
- `scripts/common.py`
- `scripts/launch.py`
- `scripts/plot_comparison.py`
- `scripts/plot_umap.py`
- `scripts/sample.py`
- `scripts/train.py`
- `scripts/verify_protected.py`
- `tests/smoke.py`
- `tests/test_suite.py`
- `training/__init__.py`
- `training/data.py`
- `training/train_loop.py`
- `validation/protected_files.json`
- `validation/static_checks.json`
- `validation/IMPLEMENTATION_REPORT.md`（この記録）

## UMAP再作図の修正

20260911/src/umap_adapter.py、20260913_2step/common.py・analysis/umaps.py、20260915_x0predict/common.py・analysis/umaps.pyを確認。これらのReal参照集団はSuperclass == Erythropoieticの全細胞。今回の全細胞から最大3000件という旧選択を修正し、同じErythropoietic全件選択へ変更した。既存exp_config.jsonは変更せず、そのumap_real_cells制限を作図で無視する。

更新: scripts/plot_umap.py、scripts/plot_comparison.py、tests/test_suite.py、README.md、この記録。追加: scripts/replot_umaps.py、tests/test_umap_reference.py。

AST構文検査・git diff --checkはPASS。ローカルで学習/sampling/UMAP実行は行っていない。新しい4件の参照選択/25PNG再描画テストはリモートで実行するために追加（実行成功の主張はしない）。replot_umaps.pyはUMAP再fitと25PNG生成だけを実行し、元のsampling NPZやcheckpointを変更しない。

## X_STARTでの新規4条件実験

既存のpredict_xstart=trueが学習/sampling/解析で共通のdiffusion factoryに伝わることを確認。booleanの厳密検証とModelMeanType.START_X/EPSILONの整合性検査を追加し、model_info・sampling metadata・analysis metadata/summaryにprediction_targetを明記した。旧既定falseは保持し、新規batch-idと--set predict_xstart=trueで再学習する。

tests/test_xstart.pyに4条件のtarget設定、真のclean XへのMSE、samplerのX_START解釈、λ=0のCell勾配、X_START runの分割resume一致、異targetからのresume拒否を検証する4テストを追加。AST構文検査とgit diff --checkを実施。ローカル学習・runtimeテストは未実行。リモート起動コマンドの冒頭で専用CPUテストを実行する。
