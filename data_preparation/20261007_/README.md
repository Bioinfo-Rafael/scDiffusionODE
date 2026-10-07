# Gastrulation dataset check — 2026-10-07

mouse gastrulation全体として取得するscVeloデータと、benchmark Data 3の対応を調べるためのダウンロード・読取専用診断コードです。annotation移植、統合、filter、normalize、HVG選択、モデル学習、velocity推論、評価指標計算は行いません。データ内容に関する結論は実行後の観測値から判断してください。

## リモートで実行

配置先は `/home/suzuki/Projects/scDiffusion-github/data_preparation/20261007_` です。ローカルではデータ取得もNotebook実行もしていません。

```bash
cd /home/suzuki/Projects/scDiffusion-github/data_preparation/20261007_
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
python -m jupyterlab inspect_gastrulation.ipynb
```

Notebookを上から順に実行してください。保存先の引数や環境変数の指定は不要です。download_data.pyは自身の配置場所を基準にします。Notebookは現在地・親ディレクトリ・その配下の `data_preparation/20261007_` を探索し、最後に上記リモート配置先を確認します。リポジトリ名をローカルの `scDiffusionODE` と仮定しません。

必要ならリモートで非対話実行できます（既存Notebookの出力は上書きしません）。

```bash
jupyter nbconvert --to notebook --execute inspect_gastrulation.ipynb \
  --output inspect_gastrulation.executed.ipynb --ExecutePreprocessor.timeout=-1
```

## ファイル

- `download_data.py`: Zenodo APIからファイル名で解決しchecksum検証後に展開。scVeloの取得結果をそのまま保存。既存の完成h5adは再取得しない。各ファイルのパス・バイト数・shape・n_obs・n_varsを表示。
- `inspect_gastrulation.ipynb`: A–Iの構造、obs、ID、annotation、gene、前処理の診断、対応発現値、図、summary。
- `inspection_helpers.py`: sparseを維持したブロック統計、ID対応、比較、作図helper。
- `requirements.txt`: リモート用依存関係。厳密なlockではありません。Notebook冒頭で実際のバージョンを表示します。
- `data/`: 2つのh5ad、ZIP、scVelo取得元cache、出典情報JSON。
- `figures/`: NotebookのPNG出力。

## データソースと留意点

1. [Benchmark論文](https://link.springer.com/article/10.1186/s13059-026-04182-z)、[公式repository](https://github.com/edawu11/Benchmark-RNA-Velocity)、[Zenodo record 18051944](https://zenodo.org/records/18051944)。対象は `03_Gastrulation_erythroid.h5ad.zip`。一時URLを固定せずAPIのfile entryを使います。
2. [scvelo.datasets.gastrulation](https://scvelo.readthedocs.io/en/stable/scvelo.datasets.gastrulation.html) の戻り値を `gastrulation_full.h5ad` として保存します。`file_path` はcache位置のみを指定します。[公式実装](https://github.com/theislab/scvelo/blob/main/scvelo/datasets/_datasets.py)には取得関数内部の `var_names_make_unique()` が含まれます。今回のコードは戻り値に追加の変更を加えず、writerによる文字列の自動categorical化も無効化します。「full」は調査上の呼称であり、実際のshape・annotationから対象範囲を確認してください。

## 統計・比較の読み方

- 全matrixのdense化は行いません。moments、gene mean/variance、library sizeは全行を小さいブロックで走査します。分散は母分散です。非有限値を件数表示し、momentsから除外します。library sizeの非有限行は別途報告します。
- median/quantileはゼロを含む一様ランダム座標100,000件（復元抽出）の推定値です。整数らしさとヒストグラムには、有限nonzero値の一様reservoir sample（最大100,000件）を使用します。seed固定、整数判定の絶対許容誤差は1e-6です。
- integer-likeやlog1p等のmetadataは前処理の手掛かりであり、raw countであることの証明ではありません。sampleでの完全一致も全matrixの一致を意味しません。
- common ID数はunique集合の積です。coverageは `common unique / n_obs` と明示し、重複がある場合に備えてrow membershipも表示します。重複IDは変更せず、値比較から曖昧なIDを全て除外します。対応cell/geneを明示的に同一順序へ揃えます。
- annotationは文字列の実値を比較。category順序は無視し、片方missing・両方missing・双方有効を分離します。一致率の分母は双方有効な件数です。大きすぎるcrosstabは理由を表示して省略します。
- 500 cells × 500 genes以下の対応subsetのみdense化します。相関が定義できない定数ベクトルはNaN。scale ratioは `ery/full`（分母非ゼロ）。zeroの多さによる見かけの一致を確認するため、少なくとも片方がnonzeroの一致率も表示します。
- gene meanのfull vs ery図はそれぞれの全cell集団を使うため、集団構成の違いを含みます。matched scatterとは異なります。

## メモリ・再実行

AnnDataを `scanpy.read_h5ad` で読み込むため、Xだけでなく全layersを保持するRAMが必要です。全体がdenseならその保存サイズ相当のRAMが必要で、これを小RAMで実行できるとは保証しません。`backed='r'` にしてもlayersは通常RAMに読み込まれるため、ここでは安易なbacked対応はしていません。全layersの走査には時間がかかりますが、統計結果をNotebook内で再利用します。scVelo取得時はその戻り値をRAMに保持し、原cacheと保存版の両方をdiskに残します。

完成h5adが既存なら内容を差し替えません。再取得が必要な場合は対象ファイルを明示的に退避してから実行してください。処理途中の `.part` ファイルは完成ファイルとして扱いません。ダウンロードや診断の実行確認はリモートで行ってください。
