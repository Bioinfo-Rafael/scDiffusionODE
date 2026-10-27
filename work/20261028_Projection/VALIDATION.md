# 実装・検証記録（2026-10-28 JST）

実装は `work/20261028_Projection/` 内に追加した。既存学習コード・モデル・checkpointを変更せず、Git push・リモート長時間ジョブは行っていない。作業開始時に存在した他ディレクトリの変更は本作業の変更ではない。

## 実行した検証

macOS arm64、Python 3.11.15、Julia 1.10.10。Pythonは作業用 `/tmp/scprojection-venv`、Juliaは `/tmp/julia-1.10.10/bin/julia` と分離したdepotを用いた。既存Python環境へパッケージを追加していない。長期利用時はREADMEの `.venv` 構築コマンドを使う。

```bash
JULIA=/tmp/julia-1.10.10/bin/julia JULIA_DEPOT_PATH=/tmp/scprojection-julia-depot MPLBACKEND=Agg /tmp/scprojection-venv/bin/python -m pytest work/20261028_Projection/tests -q
```

**22 passed, 1 skipped**。skipは `test_real_checkpoint_end_to_end` のみ。公式ddHodge・CellRankの人工テストは実際に実行し、mockや代替solverを使っていない。Numbaの非連続配列に関するperformance warningが2件あり、計算失敗はない。`pip check` は `No broken requirements found`。両exampleの `--help` も正常に終了した。

| 検証対象 | 内容 | 結果 |
|---|---|---|
| 共通射影 | 平面s=x、F=(1,2)、面上/面外、単一/バッチ | F_c=(0,2)、位置yの一致 |
| Velocity評価位置 | F(x,y)=(y,x)の平面射影 | 元のxではなく移動先yでFを評価 |
| 曲面 | 単位円へのnormal retraction | 解析的なradial解・接線性に一致 |
| 失敗処理 | 範囲外c、NaN、負の発現、勾配消失、支持領域外、反復上限、非負領域内に解なし | 正常Velocityを返さず例外 |
| 連続拡張 | PCA→Gaussian RBF→元空間gradient | finite differenceと一致、定数・大きいholdout誤差を拒否 |
| Hodge | 二次U、F=-x、D≠N、ssa=false/full dimension | 符号と加法定数を除いた解析Uに一致 |
| Hodge alignment | 公式ssa=true、rdim=2 | 公式workflowが完走し、同じ解析Uに一致 |
| Hodge接続 | 公式u→連続U→未観測の面外発現x+0.01の射影 | 位置移動・残差・接線性を検証 |
| Hodge失敗 | 非連結graph、Juliaなし | 停止、公式呼出し失敗の理由をunsへ保存 |
| CellRank P | 80細胞、公式VelocityKernel | 疎・非負・行確率、元neighbors等を保持 |
| 吸収時間 | 3状態の解析chain、terminalを変更 | [2,1,0]→[1,0,0]に一致 |
| 競合terminal | 4状態chain、A/Bの条件付き吸収時間 | A=[5/3,2,0,inf]、B=[3,2,inf,0]に一致 |
| CellRank全工程 | 単一/複数terminal、外挿eligible区別、面外x+0.005 | 生値・連続値・P保存、射影成立 |
| 吸収の未定義領域 | terminalへ到達不能な閉クラス | 公式solve前に停止 |
| 保存/再読込 | 両手法のNPZ/JSON/h5ad、同じCallable | 同じF_cを再現、上書き・改竄・順序違い・異なるFを拒否 |
| D20 adapter（人工module） | timestep49、単一/バッチ、parameter不変 | 成功。ただし実checkpoint検証とは区別 |

## 数値結果と成立範囲

数値の正本は [VALIDATION.json](VALIDATION.json)。N=80, D=3、`rng=default_rng(13)`、発現は `[0.5,2.5]` の一様乱数。両手法ともfit64細胞・holdout16細胞。これは人工データ1組であり、多seed・多datasetの性能評価ではない。

| 指標 | Hodge | CellRank |
|---|---:|---:|
| 生のpotentialと解析Uの加法定数除去後の最大絶対誤差 | 3.41949e-13 | 該当せず |
| 公式potential stationarity相対残差 | 2.50477e-13 | 該当せず |
| 連続近似のfit RMSE | 0.00196735 | 0.0419140 |
| 連続近似のholdout RMSE | 0.00624099 | 0.0711289 |
| holdout NRMSE（対象離散値の範囲で割る） | 0.00105417 | 0.0711289 |
| 連続値と離散値の全80細胞での最大絶対差 | 0.0108992 | 0.159617 |
| Pの最大行和誤差 | 該当せず | 3.33067e-16 |
| 吸収確率方程式の最大残差 | 該当せず | 1.89963e-8 |
| 条件付きtime方程式の相対残差 | 該当せず | 3.25367e-8 |

Hodgeはk=12, rdim=3, ssa=false、Uの離散範囲は5.92026。CellRankはF=(1,0.2,0.1)、x₁>2.2の14細胞をterminalとし、T_min=0、T_max=4.16203 steps。Pは80×80、nnz=1096（density=0.17125）、自己遷移0。吸収chainの数値比較は別の3/4状態の解析fixtureを使用した。

観測された誤差は、Hodgeの**公式離散potential復元**と今回の**連続近似**が異なる精度であることを示す。特にCellRankの連続値には最大約0.16の差があるため、離散τと同じとみなしてはならない。holdout NRMSEは既定上限0.25を下回るが、等値面の局所gradientや生物学的解釈の正しさを証明しない。両手法のスカラーの尺度・定義が異なるので、このRMSEの大小を手法の科学的優劣と解釈しない。

## 実checkpointの確認状況

次を実行した:

```bash
/tmp/scprojection-venv/bin/python work/20261028_Projection/scripts/check_model.py
```

既存10/27ローダーのimportまでは成功したが、以下が存在せず `FileNotFoundError` で停止した。

```text
work/20261010_model_comparison/runs/model_expansion_20261012/
  D20_lora_consistency_lambda1/effective_config.json
```

したがって**実checkpointの復元・実Velocity・実データ上のHodge/CellRank全工程は未実行**。既存のpytest用checkpointは実D20検証の代わりに使っていない。全89,000細胞の所要時間、ピークメモリ、補間誤差、terminalの妥当性も未確認。`outputs/validation_20261028/` にこのセッションのpytestとmodel checkログを置いた（outputsはGit管理対象外）。

## 残る制約

- 原論文Methodsのpotential式と公式ddHodgeの符号表示差について、著者への確認はしていない。公式固定commitの符号を維持した。
- normal retractionは一般の最近点射影でなく、非連結等値面や臨界点で到達性・一意性を保証しない。
- PCAで落とした方向、holdout16細胞で捉えられない誤差、support anchorによる外挿判定は追加の近似。
- 条件付きtimeが非有限・低吸収確率の領域はfit対象外。main列の該当細胞値は外挿。
- 元ODE出力を生物学的dx/dtとみなす仮定は検証していない。
- Linux/CUDA、89,000細胞、別Python/Julia versionでの動作はこの記録の対象外。

## 作成ファイル一覧

既存ファイルの変更はない。以下は本作業で新規作成したファイル（outputsの実行ログとcacheは除外）。

```text
.gitignore
.python-version
CellRank/README.md
CellRank/__init__.py
CellRank/cellrank_core.py
CellRank/projector.py
Hodge/README.md
Hodge/__init__.py
Hodge/ddhodge_bridge.py
Hodge/julia_env/Manifest.toml
Hodge/julia_env/Project.toml
Hodge/projector.py
Hodge/run_ddhodge.jl
README.md
SOURCE_REUSE_MAP.md
VALIDATION.json
VALIDATION.md
common/__init__.py
common/base.py
common/levelset_projection.py
common/persistence.py
common/scalar_extension.py
common/velocity_adapter.py
config.json
outputs/.gitignore
requirements.txt
scripts/check_model.py
scripts/example_cellrank.py
scripts/example_hodge.py
tests/conftest.py
tests/test_cellrank.py
tests/test_hodge.py
tests/test_integration.py
tests/test_persistence.py
tests/test_projection.py
```
