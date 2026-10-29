# Surface divergence実装・検証（2026-10-28 JST）

等値面でのsurface divergence、Jacobian項・曲率項、遺伝子座標別の厳密寄与、5枚のoverview図を実装した。**実行済みの結果は人工データの数値検証であり、実D20の生物学的解析結果ではない。** ユーザーの最新指示に従い、リモートでのpull・本解析起動はユーザーが行う。既存checkpoint・学習コード・他作業の変更は編集していない。

## 実行範囲と成功・失敗

macOS arm64、Python 3.11.15、Julia 1.10.10、PyTorch 2.7.1、CellRank 2.0.7、公式ddHodge 0.7.1固定commitで検証した。既存依存以外のpackage追加はない。全テストは**39 passed、1 skipped（45.73秒）**。skipは実checkpoint統合テストのみ。Numbaの配列連続性に関する性能warning 2件は計算失敗ではない。`pip check`も成功した。

人工データは200細胞・10座標・各クラス40細胞。スカラーは公式Hodge/CellRankを全200細胞で実際に計算し、160細胞fit/40細胞holdoutの連続補間を使用した。概要解析は各クラス8、計40細胞、32 probes、seed 28。詳細寄与は各クラス1、計5細胞を両手法で共通に選んだ。Hodgeでは40/40、CellRankでは34/40がretractionと微分に成功した。CellRankの6失敗は`line_search_or_domain_failure`で、Blood progenitors 1に1、Erythroid3に5。失敗をゼロ値に置き換えず、CSV/AnnDataにNaNと理由を残した。

終了statusは`completed_with_cell_failures`。launcherのバックグラウンド実行もローカルで完了し、`exit_code=0`を確認した。90個の細胞キャッシュ（概要80、詳細10）は再開前後でhash・mtimeが不変。probe条件変更での再開を拒否し、以前のmanifestを維持することも確認した。`--replot`後に全成果物hashを照合し、PNGがちょうど5枚でPDF/SVGの重複がないことを確認した。5枚を目視で確認した。

## Divergenceと分解の観測値

次の値は人工データにおける成功した概要標本の統計である。根拠は`outputs/divergence_analysis/smoke/{hodge,cellrank}_overview.csv`、`class_summary.csv`、`manifest.json`。任意モデルrate単位であり、生物学的時間単位ではない。

| 指標 | Hodge | CellRank |
|---|---:|---:|
| 全対象有効細胞数（中央値の母集団） | 200 | 200 |
| 連続スカラー中央値c | -0.28701461 | 0.62308776 |
| retraction・divergence成功 | 40/40 | 34/40 |
| D_S平均 | 0.38046942 | -2.30912318 |
| D_S中央値 | 0.38646929 | -1.62009902 |
| D_J平均 | -1.33038016 | -1.35540457 |
| D_C平均（負号込み） | 1.71084958 | -0.95371861 |
| probe SE中央値 | 0.03840059 | 0.03746311 |
| スカラーholdout NRMSE | 0.00057473 | 0.02436701 |

共通成功34細胞だけに揃えるとHodgeのD_S平均は0.38268953、D_J平均は-1.33201376、D_C平均は1.71470329となる。CellRankは同じ34細胞のため上表から変わらない。この共通集合でのD_S平均差は2.69181271で、D_Jの差0.02339081とD_Cの差2.66842190の和に等しい。従って、この人工例の手法差は**分解式上は主として曲率項の差で説明される**。ただしD_Cには曲率だけでなくF·nと勾配normも入り、手法ごとに移動先yも異なるため、曲率単独の因果効果を分離した実験ではない。

共通34細胞でのPearson r=-0.06463、Spearman rho=-0.08877。比較候補は事前指定したHodge–CellRankの1組のみ（探索順位は1/1）で、複数相関から最大値を選んでいない。両者はF、元細胞、射影式を共有するため、仮に高相関でも独立した生物学的証拠とはならない。今回の人工例ではこの二つの相関から強い細胞間一致は認められない。

## 大きい遺伝子座標寄与と成立範囲

根拠は`{method}_gene_ranking.csv`、`{method}_gene_contributions.npz`、`{method}_representatives.csv`。候補10座標を共通代表5細胞の平均絶対寄与で順位付けした。SYN IDは人工座標であり、実在遺伝子ではない。

| 手法 | 順位/候補数 | 人工座標 | 符号付き平均 | 平均絶対寄与 |
|---|---:|---|---:|---:|
| Hodge | 1/10 | SYN0001 | 0.160789 | 0.160789 |
| Hodge | 2/10 | SYN0002 | 0.149532 | 0.149532 |
| Hodge | 3/10 | SYN0009 | -0.067721 | 0.067721 |
| CellRank | 1/10 | SYN0008 | -0.359995 | 0.359995 |
| CellRank | 2/10 | SYN0009 | -0.323022 | 0.323022 |
| CellRank | 3/10 | SYN0006 | -0.293616 | 0.293616 |

順位相関はrho=0.06667（10座標、共通代表5細胞、1手法対）。top20を指定してもD=10なのでk=10となり、重複10/10・Jaccard=1は**全候補を選んだ自明な結果**である。これを順位の一致の証拠には使わない。Figure 5とmanifestにもこの制約を明示した。人工fixtureには生物学的GRNがないため、GRNパネルは概要推定と代表細胞の厳密値の差・SEに置き換えた。実遺伝子の上位候補・因果効果についての結果はまだない。

代表細胞での遺伝子寄与総和と独立traceの最大残差は両手法とも2.78×10^-16以下、方向微分によるD_J+D_C照合の最大残差はHodge 8.88×10^-16、CellRank 3.55×10^-15。これは採用した関数を数値的に整合して計算した証拠である。補間の曲率が真の細胞状態空間の曲率であることや、分化経路へ戻る生物学的復元力を証明しない。特にCellRankのErythroid3で5/8が失敗しており、成功した3細胞だけのクラス平均を全クラス平均とは扱えない。

## 完成図とファイル

ローカル検証出力は`outputs/divergence_analysis/smoke/`。大きい生成物・キャッシュは既存のoutputs ignore方針を保ち、Gitにはコード・再現手順・この検証記録とJSONを保存する。リモートで実行すると同じ形式の実データ成果物が生成される。

1. [Figure 1: 全体像](outputs/divergence_analysis/smoke/figure1_analysis_overview.png)
2. [Figure 2: Surface divergence](outputs/divergence_analysis/smoke/figure2_surface_divergence.png)
3. [Figure 3: Jacobian・曲率分解](outputs/divergence_analysis/smoke/figure3_divergence_decomposition.png)
4. [Figure 4: 遺伝子座標寄与](outputs/divergence_analysis/smoke/figure4_gene_contributions.png)
5. [Figure 5: 共通集合比較・数値精度](outputs/divergence_analysis/smoke/figure5_method_comparison.png)

同ディレクトリに`annotated_divergence.h5ad`、手法別CSV/NPZ、`class_summary.csv`、`manifest.json`を保存した。細胞ごとの成功・失敗、元cell ID・gene ID・入力順序、採用c、package/model/source hashはこれらが正本。機械可読な軽量検証要約は[DIVERGENCE_VALIDATION.json](DIVERGENCE_VALIDATION.json)。

## 独立数値検証と再現

- 平面s=x、F=(-x,-2y)でD_S=-2、D_C=0。
- 円/球面D=2/3/10/1024、定数F=2e1、y=e1でD_J=0、D_C=D_S=-2(D-1)。曲率項が非ゼロ。
- D=3/10でfull autograd Jacobianとsurface式を照合。非対称JではJnとJᵀnを別々に検証。
- PCA–RBFのHessianを独立torch.func.hessianと比較。
- 元のLoRAFieldを人工重みで構築し、状態依存係数を含むJVPをjacrev・中央差分と比較。重み不変を確認。実checkpointの検証とは区別する。
- D=1024でstreaming JVP/VJPの座標総和と解析値が一致。全細胞×D×D配列なし。
- D=10の固定行列、24 seedsでHutchinson RMSEは8 probesの3.995869から512 probesの0.326491へ減少。全seedで単調減少するとは主張しない。
- eligible全母集団の中央値、not_sampled/失敗の区別、共通代表優先、セル単位cache再利用を検証。

```bash
# repository root、環境構築はREADME参照
MPLBACKEND=Agg python -m pytest work/20261028_Projection/tests -q
python work/20261028_Projection/scripts/run_divergence_analysis.py --smoke
python work/20261028_Projection/scripts/run_divergence_analysis.py --smoke --resume
python work/20261028_Projection/scripts/run_divergence_analysis.py --replot work/20261028_Projection/outputs/divergence_analysis/smoke
# 実D20（環境・データを持つリモートでユーザーが起動）
PYTHON=python bash work/20261028_Projection/scripts/run_divergence_background.sh --device auto
```

ローカル実行Pythonは`/tmp/scprojection-venv/bin/python`、`JULIA=/tmp/julia-1.10.10/bin/julia`、`JULIA_DEPOT_PATH=/tmp/scprojection-julia-depot`。最初のsmokeで公式スカラーを計算し、最終launcher検証では同一全200細胞の保存済みスカラーを再利用した。再利用元を指定する場合は`--hodge-dir /path/to/hodge --cellrank-dir /path/to/cellrank`を追加し、再開時も同じ指定を渡す。

`check_model.py --device cpu`はローカルで`D20_lora_consistency_lambda1/effective_config.json`が存在しないためFileNotFoundError。実D20によるdivergence、実遺伝子順位、GPU上のAD、全training細胞のスカラー計算の時間・メモリ、Linux上の本解析完走は未検証。既存チェックポイントとデータがある環境での実行後に、実データmanifestと失敗率を確認する必要がある。


2026-10-29追記：本報告の検証後、divergence entry pointのスカラー推定母集団もbenchmarkの5クラスに限定した。上記の過去の測定値は変更していない。母集団変更の回帰検証は`tests/test_scalar_population.py`を参照。

母集団変更後の全テスト：43 passed, 1 skipped（実checkpoint）, 2 Numba performance warnings、66.79秒。追加テストは対象5クラス＋対象外2細胞をCLIへ渡し、両projectorのfit入力が対象5細胞だけであること、全7細胞の旧artifactを再利用しないこと、manifestの件数・ID・入力hashを検証する。公式数値計算自体は既存テストで検証し、この追加テストではprojectorを記録用stubに置換して入力境界を検証した。実データでの再実行結果は未検証。既存scdiffusion Python 3.9で変更Pythonファイルの構文検査も通過。
