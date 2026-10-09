# Mouse Gastrulation erythroid: 共通RNA Velocity評価

全89,267細胞で一度学習したモデルの予測を、erythroid 9,815細胞の**3-fold**で評価します。
学習コードや元の`MouseGastrulation.h5ad`は変更しません。Task IのCBDir/ICVCoh、Task IIのCTO/TSCは
無改変の公式VeloEVを使用します。Task IVは対象外です。

## 研究プロトコルと出典

- [Wu et al., Genome Biology (2026)](https://doi.org/10.1186/s13059-026-04182-z)、Data 3。
- [Supplementary Notes](https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs13059-026-04182-z/MediaObjects/13059_2026_4182_MOESM3_ESM.pdf)：S10 Data 3 p.8の4遷移、S1のUMAP方向評価、S4のpseudotime代用。
- [Supplementary Tables](https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs13059-026-04182-z/MediaObjects/13059_2026_4182_MOESM1_ESM.xlsx)：S1のData 3はcelltype、time、FoldNum=3。
- 公式Benchmark commit **`b8bc1312e8dc54ab5347159f447707bc06404ad2`**：
  [fold分割](https://github.com/edawu11/Benchmark-RNA-Velocity/blob/b8bc1312e8dc54ab5347159f447707bc06404ad2/preprocessing/general/utils.py)、
  [前処理Notebook](https://github.com/edawu11/Benchmark-RNA-Velocity/blob/b8bc1312e8dc54ab5347159f447707bc06404ad2/preprocessing/general/preprocessing_general.ipynb)。
- VeloEV submodule commit **`719aafa3aebe488bf557c484ca700b78a0b90c95`**：
  [postprocess](https://github.com/edawu11/VeloEV/blob/719aafa3aebe488bf557c484ca700b78a0b90c95/veloev/postprocessing/postprocess.py)、
  [evaluation](https://github.com/edawu11/VeloEV/blob/719aafa3aebe488bf557c484ca700b78a0b90c95/veloev/evaluation/evaluation.py)。

公式`split_anndata_stratified()`をそのまま呼び出します。
`StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`で`celltype`を層化し、各foldは**test_idx**側です。
9815細胞は3272/3272/3271細胞の互いに重ならないfoldとなり、和集合が9815細胞です。
入力行順を保持して分割します。同じseedでも入力順が違うとfoldは変わります。
各fold内でHVG・PCA・neighbors・UMAPを独立に計算します。

| 入力 | metadataのinference_protocol | 学習・推定 |
| --- | --- | --- |
| `--prediction` 1ファイル | `full-trained` | 全細胞で一度学習済み。foldでは抽出・評価だけ |
| `--fold-predictions` 3ファイル | `per-fold` | 各foldで別々に推定。各ファイルはそのfoldの細胞のみ |

全細胞学習済みモデルを3回再学習しません。残り2-foldを訓練集合とする交差検証でもありません。
従来の一括評価は`--evaluation full`で明示的に選べます。

## 指標・時間・正解遷移

`config.py`に次の遷移を固定しています。根拠はSupplementary Note S10です。
資料の`Erythroid 1`等に対し、実データのラベルは`Erythroid1`等です。

```text
Blood progenitors 1 -> Blood progenitors 2
Blood progenitors 2 -> Erythroid1
Erythroid1 -> Erythroid2
Erythroid2 -> Erythroid3
```

| 指標 | 公式呼出し | 評価内容 |
| --- | --- | --- |
| CBDir | `single_metric('cbdir')` | UMAP座標と投影velocityによる4遷移の方向一致 |
| ICVCoh | `single_metric('icvcoh')` | 同celltype内のUMAP velocityの一貫性 |
| CTO | `single_metric('cto')` | 隣接する観測stage間の推定時刻順序 |
| TSC | `single_metric('tsc')` | 正解stageと推定時刻のSpearman相関 |

stageはE6.5, E6.75, E7.0, E7.25, E7.5, E7.75, E8.0, E8.25, E8.5のみ受理します。
元stageを維持し、胚日を数値`stage_day`へ変換します。CTOは各foldの観測胚日の昇順隣接ペアです。
モデル時刻は`latent_time`/`model_time`を自動検出、別名は`--time-key`で指定します。
時刻がなければ公式postprocessの`scv.tl.velocity_pseudotime()`を使います。
事前計算したvelocity由来時刻は`--time-key COLUMN --time-kind precomputed_velocity_pseudotime`で区別します。
渡した時刻は`candidate_time`になり、公式postprocessはpseudotimeを再計算しません。返値との完全一致も検証します。
rootはscVeloの自動推定（terminal_statesの既定random_state=0）です。
正解stageによるroot選択・符号反転・時刻の修正は行いません。

各foldを独立ディレクトリへ渡すため、VeloEV内部の`k_fold=0`は「今回渡す1つのfold」を意味します。
共通入口が3回実行してfold 0/1/2と対応付けます。独自のmetric実装はありません。
`metrics_summary.csv`は3値の算術平均と母標準偏差 **ddof=0** です。

## 前処理profile

**normalized（既定・全細胞学習済みモデル用）**：既存S/Uは独立に全遺伝子で1e4正規化済みです。
9815細胞抽出・fold分割後、各foldのsplicedのコピーにlog1p、Seurat HVG（目標2000）、
PCA 30（ARPACK、zero_center=True、seed=1234）、neighbors 30（euclidean、UMAP方式、seed=1234）、
UMAP（2次元、min_dist=.5、spread=1、seed=1234）を適用します。
参照X/S/Uは元スケールの全遺伝子を保持します。raw `min_shared_counts=20`フィルタは適用しません。
モデル遺伝子へ照合後、同じfoldのgeometryでscVelo momentsを計算します。
scVeloが再正規化を要求する場合はスケール変更を避けるため停止します。

**official-raw（論文再現優先）**：`--profile official-raw --input /absolute/path/raw_erythroid.h5ad`。
9815細胞、正確なID、5 celltype、stage、非負整数のX/S/Uを要求します。
既存geometryを持ち込まず、公式Notebookの計算cell全体を無改変で実行します。
`filter_and_normalize(min_shared_counts=20,n_top_genes=2000)`、必要時のHVG再選択、
`moments(n_neighbors=30,n_pcs=30)`、公式PCA距離neighbor indices、UMAP seed=1234です。
raw層・Ms/Mu・PCA/UMAP・neighbors・HVGを保持します。公式のfold別正規化は既存の独立1e4正規化とは異なり、
既存モデルのvelocityをこのスケールと偽って渡すことはできません。

原本は`vendor/benchmark_source/`（LICENSEと`sources.json`に出典/hash）へ保存しています。
`upstream.py`はハッシュ照合と実行コンテキストの設定を行います。
raw再現全体は[再現実験README](../../../work/20261009_reproduce/README.md)を参照してください。

## 環境構築・評価データ作成（remote Linux）

専用Python **3.12.14**、scVelo 0.3.3、Scanpy 1.11.5、NumPy 2.3.5、SciPy 1.16.3、
scikit-learn 1.8.0等を`requirements.lock`で固定しています。学習環境へinstall/updateしません。
下記はuv導入済みのホスト用です。

```bash
cd /home/suzuki/Projects/scDiffusion-github
git submodule update --init --recursive
uv python install 3.12.14
PYTHON_BIN="$(uv python find 3.12.14)" bash data_preparation/20261007/benchmark/setup_env.sh
python data_preparation/20261007/benchmark/prepare.py
```

`prepare.py`/`run.py`は呼出し元Pythonでは標準ライブラリだけを使い、
`data_preparation/20261007/data/benchmark/.venv-eval/bin/python`へsubprocessで移ります。
既定入力は`data_preparation/20261007/data/MouseGastrulation.h5ad`、
出力は`data_preparation/20261007/data/benchmark/references/normalized-3fold/`。
既存出力を上書きせず、入力hashを処理前後で検証します。参照作成は初回だけです。
`prepare.py --output NEW_DIRECTORY`で別の保存先を指定できます。生成物は既存`data/` ignore規則の対象です。

## 各workからの使い方

```text
work/20261009_xxx/train.py -> checkpoint（全89267細胞で1回学習）
work/20261009_xxx/predict.py -> runs/velocity_prediction.h5ad
共通benchmark/run.py -> erythroid 9815細胞 -> 3つのtest fold -> 公式VeloEV
```

推論で`velocity`（cell×gene）、`cell_ids`、`gene_ids`、`checkpoint_path`を取得済みの位置へ追加する例です。
IDはcheckpoint/学習データから取得します。全89267細胞予測と、9815細胞すべての予測に対応します。
shape・ID重複・欠損・NaN/Inf・annotation・スケールを検証し、黙った補完はしません。

```python
from pathlib import Path
import subprocess
import sys
import anndata as ad
import pandas as pd

HERE = Path(__file__).resolve().parent  # work/20261009_xxx/
ROOT = HERE.parents[1]
out = HERE / "runs"
out.mkdir(exist_ok=True)
prediction_path = out / "velocity_prediction.h5ad"
genes_path = out / "gene_ids.txt"
if prediction_path.exists() or genes_path.exists():
    raise FileExistsError("Use new prediction filenames")
pred = ad.AnnData(X=None,
    obs=pd.DataFrame(index=pd.Index(cell_ids)),
    var=pd.DataFrame(index=pd.Index(gene_ids)))
pred.layers["velocity"] = velocity
# モデルが直接推定した時刻がある場合のみ:
# pred.obs["latent_time"] = model_latent_time
pred.uns["benchmark_velocity"] = {
    "definition": "ds_dt",
    "expression_scale": "spliced_independent_normalize_total_1e4",
    "time_direction": "forward",
    "training_n_cells": 89267,
    "time_unit": "model ODE time (arbitrary unit)",
    "inference_description": "ODE evaluated on normalized spliced X",
    "checkpoint": str(Path(checkpoint_path).resolve()),
}
pred.write_h5ad(prediction_path)
genes_path.write_text("\n".join(gene_ids) + "\n")
subprocess.run([
    sys.executable, str(ROOT / "data_preparation/20261007/benchmark/run.py"),
    "--prediction", str(prediction_path), "--genes", str(genes_path),
    "--method", "experiment_001",
    # 実験内へ保存する場合（未作成ディレクトリ）:
    # "--output-dir", str(out / "benchmark"),
], check=True)
```

宣言は実際のモデル定義に合わせてください。ODE場のds/dtとSTART_X/denoiser予測は異なります。
`work/20261009_newBenchmark`では指定された実験上の仮定としてCellUNet生出力をds/dtと宣言しています。
契約の検証はその生物学的妥当性を証明しません。subset後に1e4への再正規化もしません。

export済み入力を評価するコマンド:

```bash
cd /home/suzuki/Projects/scDiffusion-github
python data_preparation/20261007/benchmark/run.py \
  --prediction work/20261009_xxx/runs/velocity_prediction.h5ad \
  --genes work/20261009_xxx/runs/gene_ids.txt --method experiment_001
```

`--genes`省略時は各referenceの全遺伝子が必須です。subsetモデルは明示的にリストを渡します。
予測var_namesとの集合一致を検証し、cell/gene順序はIDで並べ直します。
各foldのgeometry HVGとモデル遺伝子の一致/相違・重複数をmetadataへ保存します。
velocity graphはモデル遺伝子を使うため、遺伝子集合の異なるモデルは同条件ではありません。
fold別推定は`--fold-predictions FOLD0.h5ad FOLD1.h5ad FOLD2.h5ad`と対応する`--reference`で渡します。
各予測のtraining_n_cellsはそのfoldの細胞数、expression_scaleはprofileと一致させます。

従来の一括評価:

```bash
python data_preparation/20261007/benchmark/prepare.py --evaluation full
python data_preparation/20261007/benchmark/run.py --evaluation full \
  --prediction work/20261009_xxx/runs/velocity_prediction.h5ad \
  --genes work/20261009_xxx/runs/gene_ids.txt --method experiment_001_full
```

fullは従来の`data/benchmark/erythroid.h5ad`と単一metrics.csvを使います。
fullモードには`--output-dir`と`--time-kind`はありません。
移動した`work/20261009_newBenchmark/evaluate.py`は動作を維持するためfullを明示しています。

## 出力と比較条件

```text
data/benchmark/references/normalized-3fold/
  manifest.json, prepare.log
  fold_0/ ... fold_2/
    reference.h5ad, cell_ids.txt, gene_ids.txt, hvg_ids.txt, metadata.json
  official/processed/         # official-rawのみ。元Notebookの出力

data/benchmark/results/<method>/    # または --output-dir
  metrics_per_fold.csv             # 3行、4指標・cell/gene数
  metrics_summary.csv              # 4行、mean/std/ddof/n_folds
  metadata.json, run.log
  fold_0/ ... fold_2/
    metrics.csv, cbdir_transitions.csv, cell_times.csv
    metadata.json, run.log, cell_ids.txt, gene_ids.txt, hvg_ids.txt
    processed/adata_run_candidate_full.h5ad
    postprocess/candidate_full.pkl
    evaluation/{cbdir,icvcoh,cto,tsc}_df.csv
```

metadataは入力/実装hash、環境版、commit、分割/前処理パラメータ、学習条件、時刻の種類、条件差を含みます。
失敗時は完了済みfoldの値と中間ファイルを残し、status=failed、summaryは作りません。
CBDirで境界がない遷移は個別CSVを残して失敗とし、NaNを除いた部分平均を成功扱いしません。

分割方法が一致しても、入力ID/行順、raw counts、HVG、依存版、学習条件が違えば論文と同条件ではありません。
normalized profileはraw前処理を再現しません。raw profileでも論文の元fold/HVG artifactとの同一性は未確認です。
詳細は[再現実験README](../../../work/20261009_reproduce/README.md)を参照してください。
公式postprocessのgraphはMs上でsqrt_transform=False。公式neighbor indicesはPCA全対距離で自己近傍を含みます。
各foldのpostprocessは新規subprocess、NumPy/Python seed=1234、thread=1、n_jobs=1です。
異なるOS/BLASのbit単位一致は保証しません。公式処理はdense行列を使い、全53801遺伝子のfold行列1枚で約0.7GB、
複数コピーと全細胞入力の読込分のRAMが必要です。

## テストと未検証事項

既定は小規模synthetic単体テストです。実際のPCA/UMAP・scVelo postprocessの統合テストはremoteでopt-inします。

```bash
cd /home/suzuki/Projects/scDiffusion-github
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg \
  data_preparation/20261007/data/benchmark/.venv-eval/bin/python -m pytest \
  data_preparation/20261007/benchmark/tests -q \
  -o cache_dir=data_preparation/20261007/data/benchmark/pytest-cache
# remoteでのみ、既存の公式4指標・pseudotime等のsynthetic統合テストも実行:
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg \
  data_preparation/20261007/data/benchmark/.venv-eval/bin/python -m pytest \
  data_preparation/20261007/benchmark/tests -q --run-integration \
  -o cache_dir=data_preparation/20261007/data/benchmark/pytest-cache
```

単体テストは公式test_idx分割、fold独立性、ID/時間/遺伝子照合、欠損・NaN、hash不一致、集計、失敗記録、
上書き拒否、原本関数の呼出し順、事前計算時刻保持、比較閾値を検証します。
今回ローカルで実データ前処理・推定・評価は行っていません。
過去の33テスト成功は旧full実装の記録であり、今回の3-fold実データ成功を意味しません。

2026-10-09の専用Python 3.12.14で**51 passed / 4 skipped**（約3秒）。
skipはremote opt-inの統合テストです。4公式metric関数は事前計算した小規模synthetic fixtureで実際に呼び出しました。
overflow異常系の意図したRuntimeWarningが1件あります。Python構文・README例・shell構文・CLIヘルプも確認済みです。
ローカルには元のMouseGastrulation.h5adがないため、実物の同一性・raw比較・論文値一致はremoteで確認してください。
