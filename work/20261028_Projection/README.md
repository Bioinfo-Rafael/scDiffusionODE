# 等値面上のdivergence・遺伝子別寄与とVelocity射影

新しい解析の入口は **`scripts/run_divergence_analysis.py`**。既存D20 LoRA（EMA、timestep 49）を読み、全training細胞の公式Hodge/CellRankスカラーを準備し、5クラスの有効細胞から中央値cを選び、retraction先yでsurface divergenceと遺伝子座標別寄与を計算する。最初に見る図は **Figure 1**。完成図は全体でPNG **5枚だけ**で、手法・クラスごとの画像は増やさない。

環境構築済みならrepository rootから次で実行できる。`python` は後述の固定依存を満たす環境を使う。既存環境を変更したくない場合は後述の専用venvを使い、`python` をそのPythonパスに置き換える。

```bash
# 人工データで全工程を確認（実データの代用品にはしない）
python work/20261028_Projection/scripts/run_divergence_analysis.py --smoke
# 実D20: GPUを自動選択。CPUを明示するなら --device cpu
python work/20261028_Projection/scripts/run_divergence_analysis.py
# 出力から再作図。モデル・スカラー推定・微分は再実行しない
python work/20261028_Projection/scripts/run_divergence_analysis.py --replot work/20261028_Projection/outputs/divergence_analysis/real
```

ユーザー側でリモートのpull・起動を行う場合は次を使う。launcherは環境構築やpullを自動実行せず、指定Python・Juliaを確認してから新しいjobディレクトリを作る。

```bash
git pull --ff-only origin main
# 必要なら export JULIA=/absolute/path/to/julia
PYTHON=python bash work/20261028_Projection/scripts/run_divergence_background.sh --device auto
# 専用venvを使う場合:
# PYTHON=work/20261028_Projection/.venv/bin/python bash work/20261028_Projection/scripts/run_divergence_background.sh
```

起動時に`PID`、`JOB_DIR`、`LOG`、`OUTPUT`を表示する。`JOB_DIR`には`launcher.pid`、`console.log`、`source_commit`、`source_status`、`command.txt`を保存し、処理終了後に`exit_code`を作る。結果は`JOB_DIR/analysis/`。終了コード0でも個別細胞の失敗はあり得るため`analysis/manifest.json`のstatus・成功率を確認する。

```bash
# 起動時に表示されたパスを代入
job_dir=/absolute/path/to/job_directory
tail -f "$job_dir/console.log"
ps -p "$(cat "$job_dir/launcher.pid")" -o pid,etime,args
cat "$job_dir/exit_code"  # 終了後のみ存在、0=処理完了
# 中断後、同じ引数・コード・データで再開（計算済み細胞を再利用）
nohup python -u work/20261028_Projection/scripts/run_divergence_analysis.py --output "$job_dir/analysis" --resume > "$job_dir/resume.log" 2>&1 < /dev/null &
```

直接実行の既定出力は`outputs/divergence_analysis/real/`、人工検証は`.../smoke/`。既存ディレクトリへの通常実行は拒否する。`--resume`は入力・モデル・数値コード・サンプリング条件と保存スカラーhashが一致する場合だけ利用できる。変更時は新しい`--output`を使う。既存全細胞スカラーの場所は`--hodge-dir`と`--cellrank-dir`で指定できる。対象集団が異なる保存結果は全training細胞で明示的に再計算する。公式計算途中で失敗した未完成スカラーは再計算が必要で、完成済みスカラーと細胞単位の微分キャッシュを再利用する。

## Divergence解析の定義と保存内容

5クラスは`data_preparation/20261007/benchmark/config.py`の定義をimportする。Blood progenitors 1/2、Erythroid1/2/3の全細胞を結果AnnDataへ残す。スカラー推定は5クラスに絞らず、元の全training細胞を使う。CellRank terminalは`celltype == 'Erythroid3'`の全細胞。各手法のcは**サンプリング前の全対象有効細胞**における連続スカラーの中央値で、CellRankのineligible細胞を除く。

概要計算は既定で各クラス128細胞、計640細胞までを層別抽出する（`--samples-per-class 0`で全対象）。これは計算量を制御するためであり、全対象の微分を計算したとは扱わない。遺伝子詳細は各クラス2成功細胞、計10細胞まで（`--representatives-per-class`）を用い、両手法共通の成功細胞を優先する。Figure 5の比較は同一元cell IDの共通成功集合に限定する。失敗率が異なるため、手法ごとの成功集合平均と共通集合平均を別々に保存する。

`y = projected_field.surface_points(x)`を既存実装で求め、既存`projected_field(y)`で接線Velocityを照合する。以下の微分の独立変数は**面上のy**であり、retractionを含む合成関数は微分しない。\(g=\nabla s, n=g/\|g\|, P=I-nn^\top, H_s=\nabla^2s\)として、

\[
D_S=\operatorname{tr}(P J_{PF})=D_J+D_C,\quad
D_J=\operatorname{tr}(P J_F),\quad
D_C=-\frac{F\cdot n}{\|g\|}\operatorname{tr}(P H_s).
\]

依頼文のunderbraceと加算式の符号を整合させるため、**D_Cには負号を含める**。遺伝子iの寄与は

\[
C_i^{(J)}=(P J_F P)_{ii},\qquad
C_i^{(C)}=-\frac{F\cdot n}{\|g\|}(P H_s P)_{ii},\qquad
C_i=C_i^{(J)}+C_i^{(C)},\quad \sum_i C_i=D_S.
\]

`analysis/surface_divergence.py`はPyTorch公式JVP/VJPで元の状態依存LoRA forwardを微分する。既存Hill ODE専用の解析Jacobianは流用しない。モデルはeval・固定重みのままで、微分区間だけ`inference_mode(False)`と`enable_grad()`を使い、optimizerやparameter更新を行わない。既存スカラーの勾配を再利用し、Gaussian RBFのHessianをPCA空間で解析計算する（H=B H_z Bᵀ）。従ってD×D Hessianを作る必要がない。

概要のD_JはRademacher probeのHutchinson推定（既定64、`--probes`、seed 28）で、D_Cは採用した連続補間関数の解析trace。各細胞のprobe値、標準誤差、seed、probe数をNPZへ保存する。`analysis/gene_contributions.py`は座標JVPを少数ずつstreamし、JnとJᵀnを両方使って非対称Jacobianの対角寄与を厳密計算する。作業メモリは概ねO(batch×D + D×PCA次元)で、全細胞×D×D配列を保持しない。座標JVPの回数は代表細胞あたりDなので、詳細対象数には計算時間上の制約がある。

加算一致は、定義したD_Sから同じ和を引くだけでなく、`d(PF)`の方向微分を別経路で縮約したtraceと照合する。overviewのSEはD_Jのprobe不確実性のみで、補間・モデル・射影の不確実性を含まない。スカラーのholdout NRMSEは値の近似誤差であり、勾配や曲率の誤差上限ではない。補間のPCA次元・center・seed・誤差・支持距離、surface残差、接線誤差、変位、勾配norm、失敗理由を保存する。

| 保存先（出力ディレクトリ内） | 内容 |
|---|---|
| `annotated_divergence.h5ad` | 5クラス全細胞。`projection_{hodge,cellrank}_{surface_divergence,jacobian_term,curvature_term,retraction_success}`とstatus、理由、診断量。未計算値はNaN |
| `{method}_overview.csv` | 全対象行、元cell ID、scalar/eligible/sample、成功・失敗・未計算の区別、概要推定値 |
| `{method}_representatives.csv` | 代表成功細胞の厳密計算と独立traceとの差、概要推定の誤差/SE |
| `{method}_gene_contributions.npz` | C_J/C_C/C_totalとcell ID、gene ID、元入力順序、クラス |
| `{method}_gene_ranking.csv`, `class_summary.csv` | 符号付き平均・平均絶対寄与と順位、クラス別D_S/D_J/D_Cの平均・中央値 |
| `grn_sensitivities.csv` | hash確認済みtraining GRNが使える場合のみ、候補辺のJ[target,source]。因果効果ではない |
| `manifest.json`, `identity.json`, `scalar_identity.json` | 完了/失敗stage、checkpoint/input/source hash、母集団・抽出ID、c、補間設定、version、device、成功数、比較統計 |
| `cache/{method}/` | 細胞ごとのy、probe、詳細寄与。再開時に再利用 |

`analysis_status`は`not_sampled`、`scalar_ineligible`、`surface_undefined`（勾配消失）、`retraction_failed`、`derivative_failed`、`ok`を区別する。retraction成功flagは未試行NaN、失敗0、成功1。成功条件付きの層別標本から全細胞平均を無条件に推定しない。

## 5枚のoverview図

1. `figure1_analysis_overview.png` — 対象クラス、スカラー分布、中央値c、射影成功率、D_Sの全体像。まずこれを見る。
2. `figure2_surface_divergence.png` — 元細胞UMAP上のD_S、クラス別分布、失敗率。両手法で共通のゼロ中心色尺度。
3. `figure3_divergence_decomposition.png` — D_J/D_C/D_S分布、クラス別平均・中央値、方向微分による加算残差。
4. `figure4_gene_contributions.png` — 代表細胞での寄与上位20遺伝子（`--top-genes`で変更）、符号付き平均と平均絶対値、クラス別heatmap。
5. `figure5_method_comparison.png` — 共通成功細胞の相関、共通代表細胞の遺伝子順位一致、検証済みGRN候補の局所感度または数値精度パネル。

UMAPの座標は**元の入力細胞xの既存UMAP**。色は移動先yでの計算値で、y自体のembeddingではない。元UMAPがなければそのパネルに未提供と記し、再推定しない（人工fixtureのみ検証用UMAPを作る）。寄与は発現座標・尺度依存の局所的な数値量であり、遺伝子の因果効果や分化経路への生物学的復元力を意味しない。単位はモデルの任意rate単位で、生物学的時間への換算は未検証。

検証結果・人工例の数値・未実施範囲は[DIVERGENCE_VALIDATION.md](DIVERGENCE_VALIDATION.md)、公式機能との境界は[SOURCE_REUSE_MAP.md](SOURCE_REUSE_MAP.md)を参照。今回の解析による追加packageはない。以下は既存Callable・スカラー準備APIの説明である。

## Hodge・CellRankによる等値面へのVelocity射影API

このディレクトリは、変更しない学習済みVelocity `F(x)` とAnnDataからスカラー場を構成し、ユーザーが選んだ等値面へ移動した点での接線Velocity `F_c(x)` をCallableとして返す。Hodgeは公式ddHodge.jlのポテンシャル、CellRankは公式VelocityKernelの遷移行列と公式平均吸収ステップ数を使う。両者の離散結果から連続スカラーを作る部分と等値面射影は本実装の追加処理であり、原論文の機能として扱わない。

最初に `scripts/check_model.py` で既存D20を確認する。次に `scripts/example_hodge.py` または `scripts/example_cellrank.py` を実行し、保存されたAnnDataを**外部で可視化してから** `c` を指定する。環境構築後、repository rootでの最短例は次の通り。

```bash
work/20261028_Projection/.venv/bin/python work/20261028_Projection/scripts/check_model.py
work/20261028_Projection/.venv/bin/python work/20261028_Projection/scripts/example_hodge.py
# Erythroid3はここでユーザーが明示する例であり、コードに既定terminal stateはない。
work/20261028_Projection/.venv/bin/python work/20261028_Projection/scripts/example_cellrank.py --terminal-obs-key celltype --terminal-values Erythroid3
```

| 手法 | `adata.obs` のメインキー | 公式計算の生値 |
|---|---|---|
| Hodge | `projection_hodge` = 連続近似U(x_i) | `projection_hodge_raw` = ddh.u（符号・加法定数・尺度を変更しない） |
| CellRank | `projection_cellrank` = 連続近似τ(x_i) | `projection_cellrank_absorption_steps` = 公式の平均吸収ステップ数 |

```python
# repository rootからPythonを起動した場合
import sys
sys.path.insert(0, 'work/20261028_Projection')
from Hodge.projector import HodgeProjector
from CellRank.projector import CellRankProjector

projector = HodgeProjector(velocity=velocity_fn, adata=adata)
projector.fit_scalar()
# この段階で別セルなどから外部可視化する。
# sc.pl.umap(adata, color='projection_hodge')  # X_umapが存在する場合
# c = ...  # 可視化後にユーザーが選択した値
projected_field = projector.get_projected_field(c=c)
v = projected_field(x)                # (D,) または (N,D)、元の遺伝子順序
```

```python
y = projected_field.surface_points(x)
v, diagnostics = projected_field.evaluate(x)
```

`c=0.5` を既定値にしない。Hodgeは正規化しないため、とくに0.5が有効範囲にあるとは限らない。`projector.scalar.valid_range` を確認する。CellRankでは `fit_scalar(terminal_obs_key='celltype', terminal_values=['Erythroid3'])` のように対象を必ず明示する。

## 環境と再現コマンド

Python **3.11**、Julia **1.10.10**。Pythonの直接・間接依存は `requirements.txt` に固定し、Julia依存はPkgで実際に生成した `Hodge/julia_env/Manifest.toml` に固定した。ddHodge **0.7.1 / d5d6b09431c0ccbb16e0b63414a5038511c19b85**、CellRank **2.0.7 / tag v2.0.7**を使用する。CellRankの別versionは明示的に拒否する。Juliaも実際のddHodge revisionを確認する。

```bash
python3.11 -m venv work/20261028_Projection/.venv
work/20261028_Projection/.venv/bin/python -m pip install -r work/20261028_Projection/requirements.txt
work/20261028_Projection/.venv/bin/python -m pip check
# Julia 1.10.10をPATHに置く。別の場所ならexport JULIA=/path/to/juliaでもよい。
julia --startup-file=no --project=work/20261028_Projection/Hodge/julia_env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

`Manifest.toml` は削除・再解決しない。CPUで動く。Julia/CellRankの初回コンパイルとダウンロードには追加時間が必要。Linux/GPU環境では同じ依存が解決することと数値許容差を別途検証すること。ここで検証したプラットフォームはmacOS arm64である。

### 人工データのテスト

```bash
MPLBACKEND=Agg work/20261028_Projection/.venv/bin/python -m pytest work/20261028_Projection/tests -q
# Juliaが未導入でも共通部分とCellRankは動くが、Hodgeテストはskipされる。
# Hodgeの検証完了を主張する場合はskipの内訳を必ず確認する。
```

### 実checkpointの確認

```bash
work/20261028_Projection/.venv/bin/python work/20261028_Projection/scripts/check_model.py --device cpu
```

`config.json` の `run_root` / `data_dir` は既存10/27解析と同じ既定値。D20条件名は `D20_lora_consistency_lambda1`。`effective_config.json`、最終checkpoint、training AnnData、training metadata、GRN、hashが揃っていることが必要。このローダーはリモートからの自動取得・再学習をしない。バックグラウンド実行には冒頭の明示launcherを使う。異なる配置では別のJSONを作り `--config /path/to/config.json` を指定する。

実データで両手法の小規模統合テストを明示実行する場合、configに `real_test_terminal_obs_key`、`real_test_terminal_values`（複数なら `real_test_lineage` も）を指定する。先頭128細胞にその群が存在しなければ停止する。このsubsetの結果は全細胞結果を保証しない。

```bash
PROJECTION_REAL_CONFIG=/absolute/path/to/config.json MPLBACKEND=Agg work/20261028_Projection/.venv/bin/python -m pytest work/20261028_Projection/tests/test_integration.py::test_real_checkpoint_end_to_end -q
```

### Hodge: fit → 外部可視化 → 選択したcで再読込

```bash
work/20261028_Projection/.venv/bin/python work/20261028_Projection/scripts/example_hodge.py --output work/20261028_Projection/outputs/hodge
# annotated.h5adを外部で確認してから実行する。0.5は選択値の例。
work/20261028_Projection/.venv/bin/python work/20261028_Projection/scripts/example_hodge.py --resume work/20261028_Projection/outputs/hodge --c 0.5
```

既定は全training細胞、`k=12, rdim=4, seed=0, ssa=true, maxiter=5000`。`--max-cells 512` は明示的に先頭512細胞へ限定するsmoke用オプションで、全細胞のpotentialとは異なる。`--k`、`--rdim` を指定可能。低次元人工データでは `rdim<=D` を明示する。現在の公式workflowは2Dのcurl面も作るため `rdim>=2` が必要。

### CellRank: fit → 外部可視化 → 選択したcで再読込

```bash
work/20261028_Projection/.venv/bin/python work/20261028_Projection/scripts/example_cellrank.py --terminal-obs-key celltype --terminal-values Erythroid3 --output work/20261028_Projection/outputs/cellrank
work/20261028_Projection/.venv/bin/python work/20261028_Projection/scripts/example_cellrank.py --resume work/20261028_Projection/outputs/cellrank --c 0.5
```

既定は全training細胞、`n_neighbors=30, use_rep='X', similarity='cosine', softmax_scale=10, model='deterministic', seed=0, solver='gmres', tol=1e-8, use_petsc=False, n_jobs=1`。ConnectivityKernelとは混合しない。複数競合terminalを指定する場合は、例えば `--terminal-values Erythroid3 OtherLabel --lineage Erythroid3` とする。`OtherLabel` は実在するラベルに置き換える。スクリプトは全ラベルを文字列として受け取る。整数カテゴリにはPython APIを使う。

`--c` は `--resume` と組み合わせる。fitと同時にcを自動決定する経路はない。examplesは現存する第1細胞の発現を問い合わせの例にしているが、Callableは任意の非負有限入力にも使える。ただし支持領域・収束条件を満たさなければ `ProjectionError` となる。

## 既存Velocityへの接続

確認した既存ファイルは `work/20261010_model_comparison/adapters/{model,checkpoint,data}.py`、`work/20261027_matsum_lora_analysis/load_model.py`、`work/20260817_vector_field_analysis/vector_field_adapter.py`、`work/20261013_D20_noeqThermo/src/d20_adapter.py`。

`common/velocity_adapter.load_d20()` は既存10/27の `load_trained_model('lora',...)` を再利用する。このローダーは元の `checkpoint.resolve/read/check_ids`、`data.read_data/mask_for`、`model.build_field`、source compatibility検証を使ってODE branchだけを復元する。10/13のアダプタは実測細胞のみを許しCellUNetも構築するため、任意点Callableの入口としては利用しない。8/17のアダプタは別モデルのHill解析用であり、D20の数式へ流用しない。

既定はEMA、固定diffusion timestep **49**、clean training X。1,024 HVGの**線形正規化発現**、遺伝子IDと順序を元のtraining data/hashから検証する。raw count/log1p入力や再正規化は行わない。任意の `velocity_fn` にはこれらのprovenanceを自動推定できないため、呼び出し側が同じ前処理と順序を保証し、`provenance={...}` を与える。`feature_ids` 属性があるCallableではAnnDataとの順序を必ず照合する。

ODEのSTART_X/consistency学習出力を生物学的な **dx/dt** と解釈すること自体が実験的仮定である。射影の数値検証はこの生物学的妥当性を証明しない。学習コード・モデルパラメータ・checkpointは変更しない。通常のVelocity評価は`torch.inference_mode()`、今回のJVP/VJP計算は固定重みで微分を有効にして評価する。

## Hodgeの数学と公式実装

流れは `(x_i,v_i) → ddHodge → U_i → U(x) → S_c → F_c(x)`。

- 原論文: [Maehara & Ohkawa, Nature Communications 2025](https://www.nature.com/articles/s41467-025-67782-6)
- 公式source: [ddHodge.jl・採用commit](https://github.com/kazumits/ddHodge.jl/tree/d5d6b09431c0ccbb16e0b63414a5038511c19b85)
- 計算入口: `Hodge/projector.py:HodgeProjector.fit_scalar()`
- 公式呼出し: `Hodge/run_ddhodge.jl` の `kNNGraph(X,k)`、`ddHodgeWorkflow(g,X,V;...)`、`ddh.u`

Juliaへの入力は **D×N**。PythonのN×D配列をC-orderの1D配列としてHDF5へ保存し、Juliaで `reshape(flat,D,N)` とする。HDF5の言語間軸順解釈に依存せず、D≠Nのテストで確認する。

公式ではエッジi→jに対して `w_ij=(v_i+v_j)^T(x_j-x_i)/2`、`(d0*u)_ij=u_j-u_i`、**`u = argmin ||d0*u+w||²`**を計算する。したがってgradient componentは **−∇U**。論文Resultsの `F=-∇U` と式(17)の `w+d0*u` に一致する。一方、公開論文Methods式(2)の表示は `||δ0*u−w||²` で、同じ式(3)のw・同じ差分方向で読めば公式実装と符号が逆である。この表示上の不一致を隠さず、ここでは**固定commitの `lssolve(d0,-w)` の出力を正本**とする。原著の誤植かどうかは著者未確認である。

`U=||x||²/2, F=-x` の人工系では、trapezoidエッジ積分は二次Uに対して厳密であり、加法定数を除いた公式uと解析Uを比較できる。正規化・反転はしない。graphが非連結の場合、成分間で加法定数が不定になるため停止する。公式solverはpotentialの収束flagを返さないので、既存 `ddh.d0` と `ddh.w` を用いて `||d0ᵀ(d0*u+w)|| / max(||d0ᵀw||,eps) <= 1e-5` を検査する。これは解の残差検査であり、solverの再実装ではない。curl成分がある場合 `||d0*u+w||` 自体が0になる必要はない。

## CellRankの数学と公式実装

流れは `(x_i,v_i) → P_ij → T_i → τ_i → S_c → F_c(x)`。

- 原論文: [Weiler et al., CellRank 2, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02303-9)
- 公式source: [CellRank v2.0.7](https://github.com/scverse/cellrank/tree/v2.0.7)
- [VelocityKernel API](https://cellrank.readthedocs.io/en/stable/api/_autosummary/kernels/cellrank.kernels.VelocityKernel.html)、[GPCCA API](https://cellrank.readthedocs.io/en/stable/api/_autosummary/estimators/cellrank.estimators.GPCCA.html)、[RNA velocity tutorial](https://cellrank.readthedocs.io/en/stable/notebooks/tutorials/kernels/200_rna_velocity.html)
- 計算入口: `CellRank/projector.py:CellRankProjector.fit_scalar()`
- 薄い公式wrapper: `CellRank/cellrank_core.py`

`scanpy.pp.neighbors(use_rep='X')` の近傍を作り、明示した同一feature順序のexpression/velocity layerを `VelocityKernel` に渡す。`compute_transition_matrix(model='deterministic')` はその生のF(x_i)を使う。概念式 `P_ij ∝ exp(β sim(F(x_i),x_j−x_i))` のsimilarity、softmax、遷移計算は全て公式側。行和・非負性・非有限値・疎性・自己遷移を記録する。元AnnDataのneighbors等を壊さないため、公式関数はscratch AnnData上で動く。

`GPCCA.set_terminal_states({name: cell_IDs})` でterminal細胞を明示し、自動macrostates/terminal推定は行わない。`compute_fate_probabilities(keys=None)` と **`compute_absorption_times(keys=None)`** は全競合terminalを保持したまま実行し、結果のDataFrame `absorption_times['<lineage>_mean']` から指定lineageを読む。単に `keys=[lineage]` を渡すと他terminalをtransient扱いに変えるため、この実装では行わない。

公式v2.0.7の意味を `cellrank/_utils/_utils.py:_calculate_lineage_absorption_time_means` で確認した。単一terminalではQをtransient部分行列として `(I−Q)T=1`、terminalでT=0。複数terminalでは各lineage jへの吸収確率 `h_j=(I−Q)^{-1}R_j` と `D_j=diag(h_j)` を用いて `D_j^{-1}(I−Q)D_j T_j=1` を公式が解く。これは**そのlineageへ吸収される条件での平均ステップ数**。他のterminalに既にある細胞では選択lineageのtimeはinf、確率は0となる。全terminalへの無条件吸収時間を混ぜて使わない。

到達不能な非終末クラスはグラフ到達性検査で停止する。公式solverの結果も確率方程式と `h*T−Q(h*T)=h` の残差で検証する。残差検査でMFPTを再計算しない。確率の許容誤差は `max(1e-6,100*tol)`、値のclipや自前solverへの切替はしない。公式APIが失敗する場合は例外で停止する。

有限・非負Tかつ選択lineageの吸収確率 `>=1e-6` の細胞のみをfit対象とする。その対象で `τ=1−(T−T_min)/(T_max−T_min)` を定義し、定数Tや12細胞未満なら停止する。T_min/T_maxと対象cell IDを保存する。このτは **CellRank absorption-time-derived pseudotime** であり、実時間でも、CellRankの標準pseudotimeでも、PseudotimeKernelでもない。

メイン列は全細胞で同じ連続関数を評価する。`projection_cellrank_eligible=False` の細胞における値は**除外領域への外挿**であり、そこに有効な離散τが観測されたことを意味しない。`projection_cellrank_absorption_probability` とeligible列を併せて可視化する。正規化後の連続近似は0–1を少し越える場合があり、微分可能性を維持するためclipしない。

## 連続拡張と等値面射影（今回の独自処理）

`common/scalar_extension.py` は固定seedで有効細胞の20%をvalidationに取り分ける。fit側の最大4,096細胞でscikit-learn PCAをfitし、最大128個のfit側centerでsmooth Gaussian RBF + affine termをridge回帰する。validation細胞をPCA・center・係数のfitに使わない。全体を再fitしないので、検証した関数そのものが保存される。

`z=Bᵀ(x−μ)`、Bの列は正規直交。`s(x)=a₀+aᵀ(z/scale)+Σ_j w_j exp(−||z−z_j||²/(2σ²))`。σはcenter間の正距離の中央値、scaleはfit zのRMS。係数はバッチごとの十分統計を集計し、SciPyの `linalg.solve` で解く。微分はこのGaussian関数を解析微分し `∇_x s = B ∇_z s` とする。finite differenceとの比較テストを含む。ニューラルネットワークや既存Velocityの学習は行わない。

fit/validation RMSE、全細胞最大誤差、validation NRMSE（離散対象値の範囲で割ったRMSE）、PCA説明分散率、fit/validation rowを保存する。既定NRMSE上限は0.25。この閾値を通過することは、局所等値面や勾配が科学的に正確である保証ではない。目的に応じて閾値を厳しくし、離散値とメイン列を比較すること。失敗した場合、生の公式値はAnnDataに残し、`uns['projection_analysis'][method]` にfailure理由を残すが、Callableは作らない。

PCAで失われた方向にはスカラーは変化せず、勾配の成分もない。従ってその方向の元Velocityは接線射影で残り得る。これはモデルの性質について得た知見ではなく、採用した連続拡張の制約である。回帰・PCA・外挿は公式ddHodge/CellRankの推定誤差に追加される。

支持領域の診断は、有効細胞から固定seedで選んだ最大4,096個の**元発現空間anchorへの最近傍Euclidean距離**。全分布に対する密度推定・全細胞への厳密な最近傍距離ではない。支持半径はfit対象各細胞からanchorの第2近傍までの距離の99 percentileの2倍。これは数値的な外挿ガードであり、統計的信頼領域ではない。ユーザーが `max_distribution_distance` を明示指定することもできる。xと更新候補yが支持領域を外れた場合は停止する。

射影は指定値cに対し、次の**normal retraction**を局所的に行う。

\[
 y_{k+1}=y_k-\frac{s(y_k)-c}{\|\nabla s(y_k)\|^2}\nabla s(y_k),\quad y_0=x.
\]

非負性・支持半径・残差減少を満たすまでstepを半分にする（最大30回）。近傍の局所解を見つける手法であり、**一般にEuclidean最近点射影ではない**。多成分等値面のどの成分へ到達するか、解の一意性、全てのx,cでの存在・収束は保証しない。既に `|s(x)-c|<=1e-8` の場合は位置を動かさないが、接線を定義できる有効勾配が必要。

\[
 F_c(x)=F(y)-\nabla s(y)\frac{\nabla s(y)^\top F(y)}{\|\nabla s(y)\|^2},\quad y=\Pi_{S_c}(x).
\]

ここでFは**移動先yで**再評価する。元のFの値やパラメータを書き換えない。cはfit対象細胞で評価した連続値のmin–max内に限定する（その範囲でも到達性は保証しない）。非有限値、shape不正、負の発現（既定）、勾配ノルム `<1e-10`、100反復以内に非収束、line search失敗、外挿、接線誤差過大は例外。バッチの1行でも失敗すると、正常なVelocity配列は返さない。`ProjectionError.diagnostics` または `field.last_diagnostics` で失敗理由を調べる。非負制約によって数学的には存在する等値面へ到達できない場合も、勝手なclippingで正常化しない。

`evaluate(x)` の診断:

| キー | 意味 |
|---|---|
| `surface_residual` | `abs(s(y)-c)` |
| `tangent_error` | `abs(grad_s(y) @ F_c(x))`（Velocity計算成功時） |
| `displacement` | `norm(y-x)` |
| `distribution_distance` / `input_distribution_distance` | y / xから元発現空間のsupport anchorへの距離 |
| `gradient_norm` | yでの勾配ノルム |
| `converged`, `status`, `iterations` | 行ごとの成否・理由・反復数 |

この数値診断は**近似したスカラー場に対する**精度である。生物学的に正しい等値面への射影であることは未検証。

## AnnData・中間結果・再読込

`fit_scalar()` はメモリ上のAnnDataだけを更新し、元h5adを自動保存しない。既存の出力obs/obsp/同手法のuns entryがあれば上書きを拒否する。HodgeとCellRankは同じAnnData内に共存できる。元のX、layers、neighbors、obsm、他unsは保存する。view/backed AnnDataは受け付けず `.copy()` を要求する。

`adata.uns['projection_analysis'][method]['metadata_json']` に以下をJSON文字列で保存する。これはh5adでnested null/listが変質しないための形式で、`json.loads(...)` で取り出せる。

- method / 実際のpackage version / 全feature ID順序 / cell ID / コード各ファイルSHA256
- Velocityのprovenance（標準D20ではcheckpoint SHA256、EMA/raw、training SHA256、input SHA256等）
- 現在のAnnDataのX（float64 C-order）＋ordered IDに対するSHA256。元h5adファイルのhashとは区別する
- Julia/ddHodge commit、Manifest hash、stdout/stderr、seed、graph/local dimension/solver設定
- CellRankのterminal cell ID、選択lineage、正規化定数・対象cell ID、公式solver・近傍設定、Pと吸収計算の検査値
- 補間係数の設定・誤差・有効範囲・非有限値や除外件数・射影solverの既定設定

外部h5adファイルを汎用APIへ入力する場合、`provenance={'input_h5ad_sha256': sha256(path), ...}` を明示する。標準ローダーは元training/sourceのファイルhashを既に記録する。入力値だけから物理ファイルのhashを推測しない。

```python
from common.base import sha256
path = projector.save('work/20261028_Projection/outputs/unique_run')
# 保存済みannotated.h5adは自由に可視化できる。
import anndata as ad
saved_adata = ad.read_h5ad(path / 'annotated.h5ad')
restored = HodgeProjector.load_saved(path, velocity_fn, saved_adata)
field_again = restored.get_projected_field(c=c)
```

保存先は新規ディレクトリだけを許可する。`scalar.npz` はPCA basis/mean、RBF center/係数/幅/scale、support anchor、split row。`official.npz` は公式計算に使ったX,V、Hodgeではu,w,graph edges,div,rot,solver残差、CellRankでは全lineageのtimes/probabilitiesと選択値、離散τ、valid mask。CellRankのPは `transition.npz` と `adata.obsp['projection_cellrank_transition']` へCSRで保存する。`metadata.json` は上記に加えて保存した全artifactのhashを含む。公式計算の中間結果を再読込でき、再実行なしに同じ連続スカラーを復元する。

Velocity自体をシリアライズしない。再読込時には同じ学習済みCallableを再度渡す。D20はcheckpoint hash/weights/timestepを照合し、汎用Callableも含め保存した全XでF(X)を再評価して一致を検査する。任意の未知点で2つのCallableが同じことは有限参照点から証明できないため、呼び出し側が同じモデルを再使用する責任を持つ。保存後にgene/cell順序やXが変わった場合は停止する。

cを選んだ後に別の新規directoryへ `projector.save(...)` すると、選択したcと射影solver設定も `metadata.json['selected_projection']` に保存する。各問い合わせの診断は `evaluate()` の戻り値であり、履歴を自動的に全件保存する仕組みはない。

## リソースと未検証範囲

N=89,000、D=1,024のfloat64 Xだけで約729 MB（約695 MiB）、Vも同量。scratch AnnData、PCA演算、公式kernelのmoments等によりピークは複数GB以上になり得る。ddHodgeはpotential以外の局所Hessian/Jacobianも公式workflowで計算し、ssaの固有値計算も行うため特に重い。**この実装で89,000細胞全体の実測時間・ピークメモリは未計測**。小規模subsetから評価すること。

本実装はN×Nの密補間行列を作らない。RBFの係数行列は最大 `(1+PCA次元+center数)²`、特徴量は512細胞ずつ。CellRankは疎Pと公式gmresを既定とする。公式direct solverは密化するため2,000細胞を超えたら拒否する。ただし公式近傍計算・Julia内部のメモリ使用量をこのwrapperが上限保証するものではない。

人工系の成功はD20の生物学的方向、補間精度、terminal指定の妥当性、分化時刻、外挿を保証しない。結果と実行環境の監査は `VALIDATION.md`、原著者コードとの境界は `SOURCE_REUSE_MAP.md` を参照する。
