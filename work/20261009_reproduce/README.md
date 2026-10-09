# Data 3 × scVelo stochastic の再現実験

[Wu et al., Genome Biology (2026)](https://doi.org/10.1186/s13059-026-04182-z)の
Mouse Gastrulation erythroidをrawから3-fold処理し、scVelo stochasticのCBDir/ICVCoh/CTO/TSCを
[共通benchmark](../../data_preparation/20261007/benchmark/README.md)で評価します。
既存の学習環境・学習コード・入力データ・過去の結果は変更しません。
**実データの前処理・推定・評価はremote Linuxで実行してください。** CLIもLinux以外での実行を拒否します。

## 公式コードと変更点

Benchmarkの参照commitは`b8bc1312e8dc54ab5347159f447707bc06404ad2`です。

- [公式stochastic](https://github.com/edawu11/Benchmark-RNA-Velocity/blob/b8bc1312e8dc54ab5347159f447707bc06404ad2/methods/RNA-only/03_run_scvelo_stc.py)
- [公式前処理Notebook](https://github.com/edawu11/Benchmark-RNA-Velocity/blob/b8bc1312e8dc54ab5347159f447707bc06404ad2/preprocessing/general/preprocessing_general.ipynb)
- [公式fold分割](https://github.com/edawu11/Benchmark-RNA-Velocity/blob/b8bc1312e8dc54ab5347159f447707bc06404ad2/preprocessing/general/utils.py)
- [Task I報告値](https://github.com/edawu11/Benchmark-RNA-Velocity/tree/b8bc1312e8dc54ab5347159f447707bc06404ad2/Figure_reproduction/Fig3/data03)
- [Task II報告値](https://github.com/edawu11/Benchmark-RNA-Velocity/tree/b8bc1312e8dc54ab5347159f447707bc06404ad2/Figure_reproduction/Fig4/data03)

原本3ファイルとLICENSEを`data_preparation/20261007/benchmark/vendor/benchmark_source/`へ
**byte単位で無改変**保存し、`sources.json`のcommit・SHA256を実行前に検証します。
共通`upstream.py`がNotebookの計算cell（0始まりcell 3）全体を実行し、推定は原本から
`run_scvelo_st`のASTをそのまま取り出して実行します。推定関数本体への変更はありません。

原本の末尾には、定義名`run_scvelo_st`に対し`run_scvelo_stc`を呼ぶ名前の不一致があります。
アダプタは実在する関数を呼び、原本のargparse・ハードコードされたパス・末尾のループを代替します。
推定の順序は原本どおり`velocity(mode="stochastic")` → `velocity_graph()` →
`velocity_pseudotime()` → `scvelo_stc_time`保存です。アルゴリズム・引数は変更しません。

原本の`--seed`は読み取られるだけでRNGに設定されません。
こちらは前処理前および推定の3-foldループ前にPython/NumPy seed=1234を明示します。
foldごとの途中リセットはしません。この実行制御の差はmetadataに記録します。
Notebookの欠損cluster除外は、Data 3の全9815細胞とannotationを検証して不足なら停止する検証に置き換えます。
古いgeometryは新規AnnDataへ持ち込みません。元Notebookが利用し得る`initial_size_*`等のraw由来obsは保持します。
入力obsにroot関連列がある場合も原本に合わせて保持し、その入力自体を保存します。評価側では推定済み時刻を維持します。
raw S/UのCSR化は値を変えず、元Notebookの`.toarray()`に対応させます。

## 入力と前処理

既定では`scvelo.datasets.gastrulation_erythroid(file_path=NEW_PATH)`からData 3を取得します。
scVelo 0.3.3のダウンロード元はFigshare file **27686871**です。
loader自身が`var_names_make_unique()`を呼ぶため、ダウンロード原本とloader返却AnnDataを両方残します。
アダプタによるbarcode suffix削除、gene symbol変換、ID交差による評価遺伝子の間引きは行いません。
取得元の同一性・行順が論文で実際に使用されたartifactと同一かは未確認です。
手元に原本raw erythroidがある場合は`--raw-input /absolute/path/raw.h5ad`で指定できます。

`--existing-input`（既定: `data_preparation/20261007/data/MouseGastrulation.h5ad`）と次を比較します。

- fullから抽出したerythroidとのcell ID集合・順序・件数・hash。
- gene ID集合・順序・共通遺伝子数・hash。
- S/Uの非整数要素数、行和分布、独立1e4正規化の整合性、保存された前処理宣言。
- 正確な共通ID上のS/U差、およびrawを全raw遺伝子分母で1e4にした場合の差。celltype/stageの一致。

比較だけは共通ID上で行い、その範囲を記録します。gene集合が違えばrawの分母も異なる可能性があります。
差が小さくても同一原本の証明にはしません。比較結果は`input_comparison.json`です。
入力が一致しなくても、公式rawを正本として別条件の再現実験を続行します。
元の正規化データをrawと偽って再正規化しません。raw X/S/Uの非負整数検証に失敗した場合は停止します。

公式`StratifiedKFold(n_splits=3,shuffle=True,random_state=42)`の**test_idx**を使い、
入力行順を保った3272/3272/3271細胞のfoldで別々に推定します。
前処理はNotebookどおり、min_shared_counts=20、HVG=2000（不足時は元Notebookの分岐）、
momentsのneighbors=30/PC=30、公式PCA距離neighbors、UMAP seed=1234です。
foldごとのID・HVG・reference・元Notebook出力を残します。既存モデルの全細胞学習を3回繰り返すことはありません。

## 共通評価への接続

`reproduce.py::adapt_prediction`は原本推定AnnDataからvelocityとvarのvelocity gene mask、
`scvelo_stc_time`を保持してインターフェースの契約を付けます。
公式の推定結果・velocity graph・pseudotimeは別の原本出力にそのまま保存します。
共通評価には`inference_protocol=per-fold`、`time_kind=precomputed_velocity_pseudotime`を渡します。
参照のMs/Mu・geometry・neighborsは維持し、momentsを再計算しません。

無改変VeloEV submoduleのcommitは`719aafa3aebe488bf557c484ca700b78a0b90c95`です。
CBDir/ICVCohは公式UMAP方向評価、CTOは観測stageの隣接ペア、TSCはstageとのSpearman相関です。
正解の4遷移とstage→胚日の固定変換は共通READMEを参照してください。
VeloEVのpostprocessは方向評価用graphをsqrt_transform=Falseで再計算します。
これは原本推定時のgraphと異なる可能性がありますが、公式postprocessの処理を変更していません。
時刻は`candidate_time`として渡すため再計算されず、postprocess返値と完全一致することを確認します。
TSCの符号を揃える操作、結果を見たroot選択、stage変更はありません。

各foldを単独artifactとして公式`k_fold=0`へ渡し、共通入口が3foldを集計します。
summaryのstdは**ddof=0**です。非有限velocity/timeや未定義CBDir境界は失敗として記録します。
VeloEVのNaN補完に依存して黙って成功させることはしません。

## remote Linuxでの実行

uv導入済み、既存の`MouseGastrulation.h5ad`が準備済みの前提です。
本実験の通常起動は、入力比較・raw準備・推定・共通評価・論文値比較をまとめて行います。

```bash
cd /home/suzuki/Projects/scDiffusion-github
git submodule update --init --recursive
uv python install 3.12.14
PYTHON_BIN="$(uv python find 3.12.14)" bash data_preparation/20261007/benchmark/setup_env.sh
python work/20261009_reproduce/reproduce.py
```

標準ライブラリのみの入口が既存の専用`.venv-eval`へ移るため、学習環境にパッケージを追加しません。
出力既定値はscript位置基準の`work/20261009_reproduce/runs/scvelo_stochastic/`です。
存在すれば停止し、失敗結果も上書きしません。再試行時は新しい出力名を指定します。

```bash
python work/20261009_reproduce/reproduce.py \
  --output work/20261009_reproduce/runs/scvelo_stochastic_retry01
```

raw原本を指定する場合:

```bash
python work/20261009_reproduce/reproduce.py \
  --raw-input /absolute/path/erythroid_raw.h5ad \
  --existing-input data_preparation/20261007/data/MouseGastrulation.h5ad \
  --output work/20261009_reproduce/runs/scvelo_stochastic_local_raw
```

推定完了後、共通評価だけ別名で再実行する例（保存済み予測を再推定しない）:

```bash
python data_preparation/20261007/benchmark/run.py \
  --reference work/20261009_reproduce/runs/scvelo_stochastic/references \
  --fold-predictions \
    work/20261009_reproduce/runs/scvelo_stochastic/inference_fold_0/prediction.h5ad \
    work/20261009_reproduce/runs/scvelo_stochastic/inference_fold_1/prediction.h5ad \
    work/20261009_reproduce/runs/scvelo_stochastic/inference_fold_2/prediction.h5ad \
  --velocity-key scvelo_stc_velocity --time-key scvelo_stc_time \
  --time-kind precomputed_velocity_pseudotime --method scvelo_stc_recheck
```

## 出力と論文値比較

```text
runs/scvelo_stochastic/
  metadata.json, run.log, input_comparison.json
  gastrulation_erythroid.h5ad, loader_returned.h5ad  # loader使用時
  references/
    manifest.json, prepare.log
    fold_0/ ... fold_2/       # reference.h5ad、cell/gene/HVG、metadata
    official/processed/
      adata_preprocessed_{0,1,2}.h5ad
      adata_run_scvelo_stc_{0,1,2}.h5ad  # 公式推定出力、graph/time保持
  inference_fold_0/ ... inference_fold_2/
    prediction.h5ad, metadata.json, run.log
  evaluation/
    metrics_per_fold.csv, metrics_summary.csv, metadata.json, run.log
    fold_0/ ... fold_2/       # VeloEV入力AnnData・pickle・CSV・時刻・log
  metrics_per_fold.csv       # evaluationからのコピー
  metrics_summary.csv
  comparison_with_paper.csv
  comparison_{cbdir,icvcoh,cto,tsc}.png
```

`paper_expectations.csv`は公式Fig3/Fig4 CSVのscvelo_stc行を元にしています（小数桁を保持）。

| Metric | Fold 0 | Fold 1 | Fold 2 |
| --- | ---: | ---: | ---: |
| CBDir | 0.18536694 | 0.20260206 | 0.099310525 |
| ICVCoh | 0.523451775 | 0.483732258 | 0.523129301 |
| CTO | 0.721900383 | 0.705395872 | 0.534423917 |
| TSC | 0.810866629 | 0.811225770 | -0.438577898 |

比較CSVはfold別の公式値・再現値・符号付き差・絶対誤差、`within_1e_5`（第一基準）と
`within_1e_4`（補助基準）を含みます。PNGは各指標の3foldを比較し、負のTSCもそのまま描きます。
閾値以内でも入力・前処理・foldの同一性を確認するまで完全再現とは判定しません。
metadataの`input_preprocessing_fold_equivalence`は現時点では未検証です。

## 条件差・差分調査・検証状況

既存評価専用環境を使います。公式[実行環境対応表](https://github.com/edawu11/Benchmark-RNA-Velocity/blob/b8bc1312e8dc54ab5347159f447707bc06404ad2/methods/run_all_RNA_only.sh)の
scVelo stochasticはpy310_pt212です。その[環境](https://github.com/edawu11/Benchmark-RNA-Velocity/blob/b8bc1312e8dc54ab5347159f447707bc06404ad2/envs/py310_pt212/environment.yml)は
Python 3.10.0 / NumPy 1.26.0 / SciPy 1.15.2 / Scanpy 1.11.0 / scikit-learn 1.4.2 / scVelo 0.3.3。
今回のlockはPython 3.12.14 / NumPy 2.3.5 / SciPy 1.16.3 / Scanpy 1.11.5 / scikit-learn 1.8.0 / scVelo 0.3.3です。
依存衝突の証拠がないため別環境は増設していませんが、原論文と同じ環境ではありません。
元Notebookの入力Xやraw層、gene IDの命名と順序、RNGの状態も数値差の候補です。

差が出た場合は次の順に、保存したartifactを比較します。

1. `input_comparison.json`とrawのSHA256、cell/gene IDと入力行順。
2. reference manifestのfold IDリストとHVGリスト。論文元artifactが得られれば集合と順序を比較。
3. `adata_preprocessed_*.h5ad`のX/S/U、Ms/Mu、X_pca、X_umap、neighbors。
4. `adata_run_scvelo_stc_*.h5ad`のvelocity、velocity gene mask、graph、pseudotime。
5. 共通評価のprocessed AnnDataとpostprocess pickle、cell_times.csv、遷移別CBDirとmetric CSV。
6. Python/package/OS/BLAS差、元の乱数状態。差を隠す符号調整等はしない。

ローカルではsyntheticの軽量単体テスト・構文確認のみ実施します。
入力実物の比較、rawの前処理、9815細胞の推定・評価、公式値一致は**未実行・未検証**です。
共通READMEの`--run-integration`付きテストと、このREADMEの実データコマンドはremoteで実行してください。

2026-10-09の専用venvによる軽量テストは51成功、統合テスト4件はskipです。
公式metric関数の小規模fixtureテストを含みますが、stochastic推定自体や実データスコアの成功を示すものではありません。
