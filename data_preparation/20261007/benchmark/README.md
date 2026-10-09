# Mouse Gastrulation erythroid: 共通RNA Velocity評価

全89,267細胞で一度学習したモデルを、erythroid lineageの9,815細胞で評価します。
学習・既存データ準備コードは変更しません。Task IのCBDir/ICVCoh、Task IIのCTO/TSCを
**固定した公式VeloEVの関数で計算**します。3-fold再学習、Task IVは行いません。
この実装は共通の評価プロトコルであり、論文の数値の完全再現ではありません。

## 出典と確認できた定義

- [Wu et al., Genome Biology (2026)](https://doi.org/10.1186/s13059-026-04182-z)、MethodsのCBDir/ICVCoh/CTO/TSC、式(1)–(4)。
- [Supplementary Notes](https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs13059-026-04182-z/MediaObjects/13059_2026_4182_MOESM3_ESM.pdf)：Note S10 **Data 3、p.8**が下記4遷移を明記。別論文からの推測ではありません。Note S1はUMAP空間での方向評価、Note S4 p.4は時刻のない手法への`scvelo.tl.velocity_pseudotime()`適用を説明。
- [Supplementary Tables](https://media.springernature.com/original/springer-static/esm/art%3A10.1186%2Fs13059-026-04182-z/MediaObjects/13059_2026_4182_MOESM1_ESM.xlsx)：Table S1のData 3はcluster key=`celltype`、time key=`time`、FoldNum=3。S3/S4はTask I/IIの結果。
- [公式前処理](https://github.com/edawu11/Benchmark-RNA-Velocity/blob/b8bc1312e8dc54ab5347159f447707bc06404ad2/preprocessing/general/preprocessing_general.ipynb)：`min_shared_counts=20`、HVG=2,000、momentsのk=30/PC=30、UMAP seed=1234。参照commitは`b8bc1312e8dc54ab5347159f447707bc06404ad2`。
- [公式VeloEV postprocess](https://github.com/edawu11/VeloEV/blob/719aafa3aebe488bf557c484ca700b78a0b90c95/veloev/postprocessing/postprocess.py)と[evaluation](https://github.com/edawu11/VeloEV/blob/719aafa3aebe488bf557c484ca700b78a0b90c95/veloev/evaluation/evaluation.py)。submodule commit=`719aafa3aebe488bf557c484ca700b78a0b90c95`。論文中のVeloEV v1.0表記に対し、このcommitのPython package metadataは0.0.1なので、バージョン名でなくcommitを識別子にします。

`config.py`の正解遷移：

```text
Blood progenitors 1 -> Blood progenitors 2
Blood progenitors 2 -> Erythroid1
Erythroid1 -> Erythroid2
Erythroid2 -> Erythroid3
```

補足資料では`Erythroid 1`等と表記されていますが、データの実ラベルは空白なしです。
保存済み`../inspect_gastrulation.ipynb`のobs/category出力で確認しました。

| 指標 | 公式関数 | この実装で渡す値 |
| --- | --- | --- |
| CBDir | `single_metric('cbdir')` → `calculate_cbdir` | UMAP発現座標・UMAP投影velocity、上記4遷移 |
| ICVCoh | `single_metric('icvcoh')` → `calculate_icvcoh` | 同celltype内のUMAP投影velocity |
| CTO | `single_metric('cto')` → `calculate_cto` | 隣接する正解stage間の推定時刻順序 |
| TSC | `single_metric('tsc')` → `calculate_tsc` | 推定時刻と正解stageのSpearman相関 |

stageはNotebookで確認した9ラベルのみ受理します：
`E6.5, E6.75, E7.0, E7.25, E7.5, E7.75, E8.0, E8.25, E8.5`。
`stage`を保持し、対応する胚日をfloatの`stage_day`にします。文字列の辞書順やcategory番号は使いません。
CTOは式(3)に従い、**実際のerythroid subsetに存在する時刻群**の昇順隣接ペアを使用し、
実際のペアをmetadataに保存します。未観測stageの細胞は作りません。
公開Notebookの例は汎用前処理で、Data 3の専用time変換コード・全foldのHVGリストは確認できていません。

## ファイルと環境

```text
benchmark/
  config.py              固定プロトコル、標準ライブラリだけのsubprocess入口
  common.py              検証・来歴記録
  prepare.py             評価専用reference作成
  run.py                 全work共通の評価入口
  setup_env.sh            専用venv構築
  requirements.lock      推移的依存を含む全Python packageの固定版
  tests/test_benchmark.py
  vendor/VeloEV/         改変しないGit submodule
../data/benchmark/
  .venv-eval/             評価専用Python
  erythroid.h5ad          全遺伝子を保持した評価reference
  results/<method>/      各実行の結果
```

生成物は既存の`../.gitignore`の`data/`規則ですべてGit管理外です。
`git submodule add`が作るrootの`.gitmodules`以外は新しい`benchmark/`内の変更です。
commit/pushはしません。submodule追加により`.gitmodules`とgitlinkはstageされます。

CPython **3.12.14**を使用します。主要依存はVeloEV公式requirementsと同じ
scVelo 0.3.3 / Scanpy 1.11.5 / NumPy 2.3.5 / SciPy 1.16.3 / pandas 2.3.3 /
scikit-learn 1.8.0。AnnData 0.12.6、numba 0.63.1等もlockしています。
学習環境へのinstall/updateは一切行いません。VeloEVはsite-packagesへコピーせず、
実行時にcommitとtracked sourceの無改変を検証してsubmoduleからimportします。

### リモートの環境構築（コピー実行）

Python 3.12.14が導入済みなら、その実行ファイルを`PYTHON_BIN`へ指定できます。
以下は`uv`導入済みホストで専用Pythonを用意する例です。システムのPythonは置換しません。

```bash
cd /home/suzuki/Projects/scDiffusion-github
uv python install 3.12.14
PYTHON_BIN="$(uv python find 3.12.14)" bash data_preparation/20261007/benchmark/setup_env.sh
```

`setup_env.sh`は既存venvのPython版とsubmodule commitが違えば停止します。
依存はlock通りに専用venvへinstallし、`pip check`を実施します。
Linux上での動作と実データのメモリ使用量はローカルmacOSの検証には含まれません。

### 評価データ生成（コピー実行）

```bash
cd /home/suzuki/Projects/scDiffusion-github
python data_preparation/20261007/benchmark/prepare.py
```

呼び出し元の`python`は標準ライブラリのみ使用し、処理は自動的に`.venv-eval/bin/python`へ移ります。
相対既定パスはscriptの位置基準です。入力は`../data/MouseGastrulation.h5ad`、
出力は`../data/benchmark/erythroid.h5ad`です。入力変更は行わず、既存出力も上書きしません。
入力SHA256を処理前後で比較し、出力にはcell/gene ID、celltype、stage、全遺伝子のS/Uを保持します。
元の保存済みPCA/UMAP、rootや古いvelocity graphは再利用しません。

評価専用のgeometryは次の固定手順です。

1. 5 celltypeを抽出して9,815細胞を要求。元の89,267細胞のIDリストをreferenceへ保存。
2. normalized splicedの**別コピー**にlog1p。Scanpy `highly_variable_genes(flavor='seurat', n_top_genes=2000)`。同率等で実数が変わる場合も、選択IDと実数を保存。
3. HVGのみでPCA 30次元、ARPACK、zero_center=True、seed=1234。
4. kNN=30、PCA 30次元、euclidean、Scanpy UMAP方式、seed=1234。
5. UMAP 2次元、min_dist=0.5、spread=1.0、seed=1234。

referenceのX/S/Uは正規化済み線形発現量のままです。log1pはgeometry作成コピーのみです。
評価時はモデル遺伝子へ厳密照合した後、既存geometryを使って公式`scv.pp.moments()`でMsを作成します。
scVeloのheuristicが再正規化を要求する入力は停止し、velocityのスケールを黙って変えません。

公式postprocessingは`Ms`上でvelocity graph（sqrt_transform=False）、UMAPへの投影、
必要ならvelocity pseudotimeを計算します。CBDir/ICVCoh用neighbor indicesは
VeloEVの`fill_in_neighbors_indices()`がPCA座標から計算する30近傍をそのまま使います。
この公式関数は全細胞対距離をdense化し、自己近傍も含みます。独自の近傍実装に置き換えていません。

## 予測AnnDataの契約

`obs_names`と`var_names`は学習データの正確なIDにしてください。
barcodeのsuffix削除、gene symbolへの再変換、IDの自動一意化、交差集合、ゼロ埋めは行いません。

- 行は全89,267細胞、または正確なerythroid 9,815細胞。集合はreferenceのIDと厳密一致が必要です。
- 列は全reference遺伝子（既定）、または`--genes`で指定した**モデルが使った完全な遺伝子リスト**。
- `layers['velocity']`はその列の順序に対応した`ds/dt`。dense/sparseとも可。
- Xは不要です。評価側のreferenceを使用します。
- 時刻があれば`obs['latent_time']`または`obs['model_time']`。任意名は`--time-key`で指定します。
  両方あれば明示指定が必要です。指定列の欠損・NaNや定数時刻はfallbackせず停止します。
- 任意のcelltype/stage列がある場合はreferenceとの一致も検証します。
- 全予測行のvelocity/timeについてNaN/Infを拒否し、float32変換のoverflowも検出します。

`uns['benchmark_velocity']`には次を必須とします。これは呼び出し側による宣言であり、
checkpointを再学習・解析して学習細胞数や速度の生物学的妥当性を証明するものではありません。

```python
prediction.uns['benchmark_velocity'] = {
    'definition': 'ds_dt',
    'expression_scale': 'spliced_independent_normalize_total_1e4',
    'time_direction': 'forward',
    'training_n_cells': 89267,
    'time_unit': 'model ODE time (arbitrary unit)',
    'inference_description': 'ODE field evaluated on unnoised normalized spliced X',
    'checkpoint': '/absolute/path/to/checkpoint',
}
```

`time_direction='forward'`はモデルの定義に従います。正解stageや結果の相関を使って符号を反転させません。
正規化前counts、log1p、z-score、潜在空間、denoised x0を直接渡すことはできません。
遺伝子subsetにしても、1e4への再正規化はしません（元の全遺伝子分母のスケールを保持）。
必要な座標変換はモデル側で導関数の変換を導出し、その実装・単位をinference_descriptionに記録してください。

### 既存workの出力調査

- `work/20260830/scripts/train.py`は`load_data(..., preprocess=False)`。
- `work/20260830/hematopoietic_viz/core.py::compute_vector_fields`のODE出力は
  `model.ode_model(clean_X, None)`。対応する`models/ode_fields_20260830.py`はproduction−decay×X型のベクトル場を返します。
  MouseGastrulation正規化Xで学習・推論し、gene順がcheckpointと一致する場合に、その座標のODE微分として出力できます。
  モデル時間と実際の胚日との校正はありません。
- 同じ可視化関数のML出力はnoisy Xに対する`model.ml_model(...)`です。
  `work/20260929noneqThermo/cellunet_adapter.py`もSTART_Xをそのままvelocityとしています。
  **START_X/denoiser予測がds/dtであることは確認できません**。この評価に流用するための自動変換は実装しません。
- `work/20260913_2step/analysis/umaps.py`等のjoint embeddingは別目的です。
  そのUMAPを評価用geometryとして使い回しません。

### 各workからの出力例

以下は既存推論後、`velocity`（cell×gene）、`cell_ids`、`model_gene_ids`が得られた位置へ追加するexport例です。
これらのIDは学習・checkpointから取得し、shapeから推測して付け直さないでください。
学習コードへの変更は今回加えていません。

```python
from pathlib import Path
import anndata as ad
import pandas as pd

out_dir = Path('data_preparation/20261007/data/benchmark/predictions')
out_dir.mkdir(parents=True, exist_ok=True)
prediction = ad.AnnData(
    X=None,
    obs=pd.DataFrame(index=pd.Index(cell_ids)),
    var=pd.DataFrame(index=pd.Index(model_gene_ids)),
)
prediction.layers['velocity'] = velocity
# モデルが直接出力する場合だけ指定。diffusion timestepをlatent timeとして使わない。
# prediction.obs['latent_time'] = model_latent_time
prediction.uns['benchmark_velocity'] = {
    'definition': 'ds_dt',
    'expression_scale': 'spliced_independent_normalize_total_1e4',
    'time_direction': 'forward',
    'training_n_cells': 89267,
    'time_unit': 'model ODE time (arbitrary unit)',
    'inference_description': 'model.ode_model evaluated on normalized spliced X',
    'checkpoint': str(Path(checkpoint_path).resolve()),
}
output = out_dir / 'scDiffusionODE.h5ad'
if output.exists():
    raise FileExistsError(output)
prediction.write_h5ad(output)
(out_dir / 'scDiffusionODE.genes.txt').write_text('\n'.join(model_gene_ids) + '\n')
```

### 評価実行（コピー実行）

上記exportで用意したファイルを、学習環境から次のように評価できます。

```bash
cd /home/suzuki/Projects/scDiffusion-github
python data_preparation/20261007/benchmark/run.py \
  --prediction data_preparation/20261007/data/benchmark/predictions/scDiffusionODE.h5ad \
  --genes data_preparation/20261007/data/benchmark/predictions/scDiffusionODE.genes.txt \
  --method scDiffusionODE
```

`--velocity-key`の既定は`velocity`。モデル時刻が別名なら`--time-key your_time_column`を追加します。
全細胞予測でもerythroid予測でも同じコマンドです。必ず9,815細胞へ照合して評価します。
任意のworkのPythonからも、既存学習プロセスへのpackage追加なしで呼び出せます。

```python
import subprocess
from pathlib import Path
repo = Path('/home/suzuki/Projects/scDiffusion-github')
bench = repo / 'data_preparation/20261007/benchmark'
subprocess.run([
    str(bench.parent / 'data/benchmark/.venv-eval/bin/python'),
    str(bench / 'run.py'),
    '--prediction', str(bench.parent / 'data/benchmark/predictions/scDiffusionODE.h5ad'),
    '--genes', str(bench.parent / 'data/benchmark/predictions/scDiffusionODE.genes.txt'),
    '--method', 'scDiffusionODE_run2',
], check=True)
```

時刻がなければ、公式postprocess内の`scv.tl.velocity_pseudotime()`を使用します。
root/endはscVeloのterminal_statesの自動推定（random_state=0）です。入力に保存されたrootやcell_fateは持ち込みません。
グローバルseed=1234、thread数=1、n_jobs=1、cell/gene順・package版を固定し、stageでrootを選びません。
scVeloのVPT固有値計算はARPACKの初期ベクトルを指定しておらず、NumPy seedのみでは
同一プロセス内の過去の計算状態をリセットできません。公式postprocessを毎回新規subprocessで
実行することで、その状態も分離します。Syntheticで同じ入力からの2回の時刻・スコア一致を検証しています。
異なるOS/BLAS/CPU間のbit単位一致は保証しません。
モデル時刻は`candidate_time`へ、velocityは`candidate_velocity`へ変換します。
内部method名`candidate`はVeloEVの特定手法向け分岐を避け、利用者のmethod名は結果ディレクトリ・metadataに残します。
公式が例外をwarningに変える場合も、postprocess出力の時刻・shape・ID・有限値を再検証して失敗として扱います。

## 使用遺伝子と論文との比較条件

`--genes`省略時は全reference遺伝子が必須です。予測がHVG subsetなら、その**checkpointの遺伝子リスト**を渡します。
宣言したリストに対する欠損・余分な遺伝子はエラーです。予測列の並べ替えはIDで安全に揃えます。
geometryは同一referenceで共通に保ち、velocity graphは宣言したモデル遺伝子すべてを使います。

公式HVGの正確なIDリストが手元にある場合だけ、`--official-hvg-genes path/to/verified_ids.txt`で
同一集合か、共通遺伝子数はいくつかを記録できます。この引数は評価遺伝子を変更しません。
リストがなければ`not_verified`と記録し、公式HVGと一致するとは報告しません。
公式はfoldごとのHVGなので、リストの由来・foldも外部で管理してください。

主な相違は、全細胞で一度学習、erythroid全体で一度評価、既に独立1e4正規化した入力、
raw countに対するmin_shared_countsフィルタの不適用、正規化データ由来のgeometry HVG、
モデル遺伝子集合でのvelocity graph、PCA seedの明示固定です。
論文のfold平均・分散と単一実行のスコアを同じ実験条件の値として比較しないでください。
元のscVelo fullとZenodo erythroidのID差は既存調査でも記録されているため、
この実装はfullのIDを正本にしており、Zenodo barcodeへの置換はしません。

## 出力

`../data/benchmark/results/<method>/`は既存なら停止します。再実行は別のmethod名で行ってください。

| ファイル | 内容 |
| --- | --- |
| `metrics.csv` | method名と4指標 |
| `evaluation/{cbdir,icvcoh,cto,tsc}_df.csv` | VeloEVが直接出力したCSV（内部名candidate、fullの1列） |
| `cbdir_transitions.csv` | 各遷移を1件ずつ公式calculate_cbdirへ渡した返値。式の再実装なし |
| `cell_times.csv` | cell ID、元stage、stage_day、推定時刻 |
| `metadata.json` | 成否、時刻の種類、細胞/遺伝子数、入力hash、パラメータ、commit、実行環境、比較条件の差 |
| `gene_ids.txt`, `geometry_gene_ids.txt` | 評価velocityと共通geometryそれぞれの遺伝子ID |
| `processed/adata_run_candidate_full.h5ad` | 公式へ渡したAnnData（S、Ms、velocity、model timeがあればその列） |
| `postprocess/candidate_full.pkl` | 公式postprocessの返す中間成果物。pseudotimeもこの中に保存 |
| `run.log` | 標準出力・warning・例外traceback |

CBDirの4遷移のうち境界近傍がなくスコア未定義の遷移があれば、個別CSVを残して失敗します。
VeloEVがNaNを無視して部分平均だけ返す場合に成功扱いしないためです。
失敗時は`metadata.status=failed`とログを確認してください。`metrics.csv`は成功時のみ作成します。

## テスト・確認範囲

```bash
cd /home/suzuki/Projects/scDiffusion-github
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg \
  data_preparation/20261007/data/benchmark/.venv-eval/bin/python -m pytest \
  data_preparation/20261007/benchmark/tests -q \
  -o cache_dir=data_preparation/20261007/data/benchmark/pytest-cache \
  --basetemp=data_preparation/20261007/data/benchmark/test-tmp
```

Syntheticは全110細胞中100細胞、48遺伝子で、正規化発現と解析的なds/dtを作成します。
I/O、full/subset予測、dense/sparse、cell/geneの順序変更、宣言した遺伝子subset、
欠損・重複・不明ID、NaN/Inf、float32 overflow、時刻異常、誤スケール、誤った学習細胞数、
annotation不一致、shape違反を確認します。
公式postprocessと4指標の実計算はモデル時刻あり/なしの両方で実行し、
結果保存・入力hash不変・既存結果の上書き拒否も検証します。

2026-10-08のローカルmacOS arm64 / 専用Python 3.12.14では**33テスト成功**。
`setup_env.sh`、`pip check`、学習環境相当の別Pythonからの両CLI起動も確認しました。
float32 overflow異常系の意図したRuntimeWarningが1件あり、公式のAnnData API非推奨warningは各実行ログに保持します。
実装開始前のworktreeと比較し、既存Git追跡995ファイルのSHA256はすべて不変でした。
学習コード、`prepare_mouse_gastrulation.py`、既存README/Notebook、元からあったユーザー編集も保持しています。
監査結果・テストログは`../data/benchmark/implementation_audit.json`、`tests.log`に保存しました。
`MouseGastrulation.h5ad`と学習済み予測はローカルにないため、実9,815細胞での評価・実モデルのスコア・
実際のcheckpointの微分定義と単位は未検証です。上記コマンドをリモートで実行してください。

メモリ上の注意：公式VeloEVはvelocity、Ms、Sをdense化し、scVeloも作業用コピーを作ります。
全53,801遺伝子ではfloat32行列1枚だけで約2.1 GB、公式近傍用全対距離でも追加メモリを使います。
入力full AnnDataの読込も含め、十分なRAMが必要です。小RAM向けのbacked/streaming動作は保証しません。
遺伝子を勝手に間引いてメモリ不足を回避する処理はありません。

## `work/`からの標準的な使い方

各実験を`work/20261009_xxx/`で行い、学習・推論後にvelocityをAnnDataへ保存して共通入口を呼び出します。

```text
work/20261009_xxx/
├── train.py
├── predict.py
├── evaluate_velocity.py
└── runs/
    ├── checkpoint.pt
    └── velocity_prediction.h5ad

data_preparation/20261007/benchmark/run.py
    └── erythroid 9,815細胞への照合 → 公式VeloEV → results/<method>/
```

### 初回だけ行う準備

```bash
cd /home/suzuki/Projects/scDiffusion-github
git submodule update --init --recursive
uv python install 3.12.14
PYTHON_BIN="$(uv python find 3.12.14)" bash data_preparation/20261007/benchmark/setup_env.sh
python data_preparation/20261007/benchmark/prepare.py
```

`erythroid.h5ad`が存在する場合、`prepare.py`は上書きせず停止します。環境構築とデータ準備は各`work/`で繰り返しません。

### 各`work/`からの評価

モデルから`velocity`（`[n_cells, n_genes]`の`ds/dt`）、`cell_ids`、`gene_ids`、`checkpoint_path`を取得します。
全89,267細胞分でもerythroid 9,815細胞分でも入力できますが、評価対象は必ず9,815細胞へ照合されます。

```python
from pathlib import Path
import subprocess
import sys
import anndata as ad
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
prediction = ad.AnnData(
    X=None,
    obs=pd.DataFrame(index=pd.Index(cell_ids)),
    var=pd.DataFrame(index=pd.Index(gene_ids)),
)
prediction.layers["velocity"] = velocity
prediction.uns["benchmark_velocity"] = {
    "definition": "ds_dt",
    "expression_scale": "spliced_independent_normalize_total_1e4",
    "time_direction": "forward",
    "training_n_cells": 89267,
    "time_unit": "model ODE time (arbitrary unit)",
    "inference_description": "ODE evaluated on normalized spliced X",
    "checkpoint": str(Path(checkpoint_path).resolve()),
}
out = HERE / "runs"
out.mkdir(exist_ok=True)
prediction_path = out / "velocity_prediction.h5ad"
genes_path = out / "gene_ids.txt"
prediction.write_h5ad(prediction_path)
genes_path.write_text("\n".join(gene_ids) + "\n")
benchmark = ROOT / "data_preparation/20261007/benchmark/run.py"
subprocess.run([
    sys.executable, str(benchmark),
    "--prediction", str(prediction_path), "--genes", str(genes_path),
    "--method", "experiment_001",
], check=True)
```

`run.py`は評価専用venvをsubprocessで起動します。モデル時刻を出力する場合は`obs["latent_time"]`または
`obs["model_time"]`へ保存し、時刻がない場合は公式`scv.tl.velocity_pseudotime()`を使用します。
`ds_dt`などの契約は、実際のモデル出力が満たす場合だけ宣言してください。

### 結果と注意点

```text
data_preparation/20261007/data/benchmark/results/experiment_001/
├── metrics.csv
├── cbdir_transitions.csv
├── cell_times.csv
├── metadata.json
├── run.log
├── evaluation/
└── postprocess/
```

`metrics.csv`にはCBDir、ICVCoh、CTO、TSCを保存します。`metadata.json`には細胞数・遺伝子数、時刻の種類、
パラメータ、入力hash、VeloEV commit、条件差を保存します。同じ`--method`名での再実行は上書きせずエラーになります。

ODEの`ds/dt`は評価できますが、CellUNetの`x_start`予測やdenoiser出力をそのままvelocityとして扱うことはできません。
PCA/UMAPは2,000 HVG、velocity graphはモデルのgene集合を使うため、遺伝子数の異なるモデルは同一条件ではありません。
syntheticデータによる33テストと公式VeloEVの4指標計算は確認済みですが、実9,815細胞・checkpointでの評価はリモート環境で実行してください。

結果を各実験ディレクトリへ直接保存する`--output-dir`は現時点では未実装です。現在は`--method`で実験名を分けてください。
