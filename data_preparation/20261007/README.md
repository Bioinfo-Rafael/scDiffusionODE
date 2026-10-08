# @Rafaメモ：
- benchmark論文のdata 3についてZenodoから入手したものはscveloのデータの完全に一部であることがわかった。
- 前処理はraw countのまま

# chatGPT

結論：はい。FullからErythroidの5種類のcelltypeを切り出せばよいと判断できます。発現行列もraw count相当です。

更新後のNotebookを確認しました。

## 1. FullとErythroidの対応

| 検証項目                  | 結果              |
| --------------------- | --------------- |
| Fullの細胞数              | 89,267          |
| Erythroidの細胞数         | 9,815           |
| Fullから5種類のcelltypeを抽出 | 9,815           |
| `stage × celltype`の分布 | 完全一致            |
| Barcode一致             | 9,755 / 9,815   |
| 発現量完全一致               | 9,755 / 9,755   |
| `celltype`一致率         | 100%            |
| `stage`一致率            | 100%            |
| 共通遺伝子                 | 53,801 / 53,801 |

特に、9,755細胞では53,801遺伝子すべてについて、X・spliced・unsplicedが完全一致しています。

残り60細胞はbarcodeが一致しませんでしたが、`-1`というsuffixが付いたIDです。こちらはID変換を試す余地があります。

さらに5種類のcelltype別細胞数だけでなく、stageとのクロス集計まで完全一致しています。

したがって、以下で評価用subsetを作る方針で進められます。

```
ery_celltypes = [    "Blood progenitors 1",    "Blood progenitors 2",    "Erythroid1",    "Erythroid2",    "Erythroid3",]ery_eval = full[    full.obs["celltype"].isin(ery_celltypes)].copy()print(ery_eval.shape)# (9815, 53801)
```

60細胞の同一性は未検証ですが、評価用の集団としては非常に強い一致が確認されています。

## 2. preprocessing状態

| 項目                | Full    | Erythroid |
| ----------------- | ------- | --------- |
| `X`               | Count相当 | Count相当   |
| `X == spliced`    | 完全一致    | 完全一致      |
| `X == raw_counts` | layerなし | 完全一致      |
| `unspliced`       | あり      | あり        |
| log1p済み           | 形跡なし    | 形跡なし      |
| Library size正規化   | 形跡なし    | 形跡なし      |
| HVG subset        | 未実施と推定  | 未実施と推定    |
| PCA / UMAP        | 計算済み    | 計算済み      |

注意点として、`X`はspliced countであって、spliced + unsplicedではありません。

scDiffusionODEの学習前には、Fullの89,267細胞についてnormalize、log1p、HVG選択などを実施する必要があります。

# Gastrulation dataset check — 2026-10-07

mouse gastrulation全体として取得するscVeloデータと、benchmark Data 3の対応を調べるためのダウンロード・読取専用診断コードです。annotation移植、統合、filter、normalize、HVG選択、モデル学習、velocity推論、評価指標計算は行いません。データ内容に関する結論は実行後の観測値から判断してください。

## リモートで実行

配置先は `/home/suzuki/Projects/scDiffusion-github/data_preparation/20261007` です。ローカルではデータ取得もNotebook実行もしていません。

```bash
cd /home/suzuki/Projects/scDiffusion-github/data_preparation/20261007
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_data.py
python -m jupyterlab inspect_gastrulation.ipynb
```

Notebookを上から順に実行してください。保存先の引数や環境変数の指定は不要です。download_data.pyは自身の配置場所を基準にします。Notebookは現在地・親ディレクトリ・その配下の `data_preparation/20261007` を探索し、最後に上記リモート配置先を確認します。リポジトリ名をローカルの `scDiffusionODE` と仮定しません。

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

## 2026-10-08 追加解析（J–R）

既存Notebookの23セル（空セル3つを含む）と全実行結果を保持し、末尾に18セルを追加しました。処理は新しい `additional_inspection.py` に分離しています。新たなNotebookは作成していません。追加解析は未実行です。

確認した既存出力はFull 89,267×53,801、Ery 9,815×53,801で、gene IDと順序は一致、obs_namesの直接一致は0件でした。Fullには重複を含むbarcode列があり、元のD/Gでは細胞対応を比較できていません。既存の整数判定はnonzero sampleに基づき、行列の完全一致を証明するものではありません。

リモートで既存Notebookを開き、kernelを再起動して先頭のセットアップcodeセル（読込・helper import）を実行した後、J–Rを順に実行できます。A–Iも再実行する場合は先頭から通して実行してください。Aの `diagnostics` がkernel内にあれば再利用し、なければPで必要な統計を計算します。別のデータを同じkernelに読み直した場合は、古い `diagnostics` を再利用しないようkernelを再起動してください。

| 節 | 追加内容 |
| --- | --- |
| J | Full barcodeとEry indexの厳密照合。一意候補、重複候補、候補なしの数・一致率。重複barcodeのsample別分布。 |
| K | barcode候補をX/spliced/unsplicedでそれぞれ独立検証。256→2,048→全共通geneの段階的照合と、一対一で検証済みの対応表。 |
| L | 5 celltypeのFull内細胞数、Eryとのcelltype別細胞数・stage×celltype集計の比較。 |
| M | Kで確定した対応のみでcelltype/stage一致率、不一致全件、confusion matrix、cluster.sub/haem_subclustとEry celltypeのcrosstab。 |
| N | 保存済みPCA/UMAPによる10図。座標再計算なし、全カテゴリ名を図外の凡例に表示。 |
| O | 全要素のsparse差分によるX/spliced/raw_countsの一致検証。異なる要素数、最大絶対誤差、平均絶対誤差。 |
| P | 既存のinteger-like、library size、nonzero統計、metadataをもとに事実・解釈・未確認事項を分離。 |
| Q | 取得時JSON、現在のloader source、元cacheのshapeを読取り確認。loader自体は呼び出さない。 |
| R | 上記の計算結果からsummary表を作成。未確認値は補完しない。 |

### 対応確定の範囲とメモリ

barcodeは末尾suffixなどを加工せず比較します。Fullのsample/stage/celltypeは候補を絞る根拠に使いません。gene IDは一意に対応するものだけを使い、両行列の列順を明示的に揃えます。候補が一意になっても、残りの全共通geneを検証し終えるまで確定しません。全matrixが使用できない場合は利用したmatrix名を表示します。

一致判定の許容誤差は0です。完全一致の範囲は全共通geneと両側で利用できるmatrixであり、欠けているgene/layerについての一致を意味しません。X・spliced・unsplicedを別々に検証するため、いずれかだけが不一致でもmatrix別の結果が残ります。発現量が異なる候補を近似一致で割り当てることはしません。

複数候補、多対一の衝突、ゼロ値しかない対応は未確定として残します。非有限値を検出した候補も確定しません。完全一致候補を持つ細胞数と、一対一に確定した細胞数を区別します。棄却された候補は検証を早期終了するため、そのgene検証数は候補別CSVを参照してください。

発現比較と全要素差分は最大32行ずつ処理し、全発現matrixはdense化しません。小さなdense入力ブロックはsparseへ変換して比較します。PCA/UMAP描画でのみ保存済み座標の2列を使用します。追加解析は全AnnDataをRAM上に保持する前提で、元の読込時のメモリ要件は変わりません。

### 保存する図

以下は `figures/` 配下に保存します（対象matrix/annotationがない場合は理由を表示してskip）。

- `full_saved_pca_celltype.png`
- `full_saved_pca_stage.png`
- `full_saved_umap_celltype.png`
- `full_saved_umap_stage.png`
- `full_saved_umap_erythroid_lineage.png`
- `full_saved_umap_haem_subclust.png`
- `ery_saved_pca_celltype.png`
- `ery_saved_pca_stage.png`
- `ery_saved_umap_celltype.png`
- `ery_saved_umap_stage.png`
- `matched_celltype_vs_celltype.png`
- `matched_stage_vs_stage.png`
- `matched_cluster_sub_vs_celltype.png`
- `matched_haem_subclust_vs_celltype.png`

Full/Ery間で同名カテゴリの色を揃えますが、保存済みPCA/UMAPの座標系が共通とは仮定しません。lineage図の他celltype・欠損annotationは灰色で表示します。

`reports/` にbarcode重複分布、全候補のmatrix別検証結果、全Ery細胞の解決状態、一対一対応表、geneの列順、annotation不一致全件、集計表、前処理の根拠、最終summaryのCSVを保存します。AnnData本体やobs/varには書き戻しません。生成物はGit ignore対象です。図を共有する際は必要なPNGだけ `git add -f` で追加してください。

### 細胞数の差について確認できたこと

[元論文](https://www.nature.com/articles/s41586-019-0933-9)の116,312細胞と、既存出力の89,267細胞は27,045細胞異なります。[scVelo v0.3.3のloader](https://github.com/theislab/scvelo/blob/v0.3.3/scvelo/datasets/_datasets.py)と[v0.3.4](https://github.com/theislab/scvelo/blob/v0.3.4/scvelo/datasets/_datasets.py)は、[Figshare file 28095525](https://ndownloader.figshare.com/files/28095525)を読み、gene名を一意化して返します。loader本体に明示的なcell filterはありません。本ディレクトリのdownload_data.pyにも追加cell filterはありません。

[benchmark README](https://github.com/edawu11/Benchmark-RNA-Velocity#-dataset-information)はData 3の元データ取得方法をgastrulation_erythroidと記載しています。今回のEryはZenodoのprocessed版です。配布Fullファイル作成時の具体的なcell選別理由は、確認した説明・loaderからは特定できません。QC・doublet除去などの原因は未確認です。Qで元cacheも89,267細胞と確認できれば、差が今回の保存前から存在することを検証できます。

保存済みNotebookのkernelはanndata 0.8.0 / scanpy 1.9.3 / scvelo 0.3.3でした。ダウンロード時のログ（scvelo 0.3.4）とは異なるため、Qでは現在のkernelと取得時JSONを分けて表示します。保存済みmetadataを今回の実行環境として上書きしていません。

### 検証範囲

NotebookのJSON/nbformat schema、全codeセルとhelperのPython構文、追加セルからのhelper参照、helperのimportを確認しました。元の23セル・実行結果・Notebook metadataが変更前と一致することも確認しています。追加依存はなく、requirements.txtのpackageを使用します。ローカルの依存環境は指定requirementsより古いため、import成功は指定環境での解析動作の保証ではありません。

ローカルではh5adの取得・追加解析関数の実行はしていません。barcodeによる実際の対応数、発現の完全一致数、annotation一致率、PNG描画、全要素比較、元cacheのshape確認はリモートで未実行です。追加セルの出力は空のままにしています。
