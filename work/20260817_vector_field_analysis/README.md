# Trained ODE vector-field analysis

`work/20260803_ODE_hill_exp` の学習済みrunから、Hybrid全体ではなく
`model.ode_model` だけを復元し、

```text
V(x) = trained_ODE(x)
```

を元のD次元gene-expression空間のvector fieldとして解析します。training、
sampling、checkpoint、source runのmanifestは変更しません。DynamoのSparseVFC
等による再推定も行いません。

現在受け付ける対象は意図的に次の1条件だけです。

- `model_family=standard_hybrid_single`
- `ode_type=hill_after_linear`
- single ODE branch（LinComb gate、Hybrid係数、diffusion timestepを不使用）

## Dynamo source audit

2026-08-18にDynamo公式repository
[`aristoteleo/dynamo-release`](https://github.com/aristoteleo/dynamo-release/tree/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650)
のcommit `69ba903000a1dcd5aa7d8b36a9318c1b22c5d650` を確認しました。

Dynamoの公開 `vector_calculus.py` entry pointはAnnData内のDynamo `VecFld` と
再構成済みvector-field classを前提とします。また、このprojectの
`scdiffusion` environmentにはDynamo自体が入っていません。そのためDynamoを
runtime dependencyとして追加したり、SparseVFC stateを擬装したりしていません。
代わりに、学習済みODEをDynamoと同じrow-vector `func(X)` および
`get_Jacobian(method)(X)` interfaceへ接続し、fit後の数式と集約だけを最小限
実装しています。Dynamoのコードを一括copyしたものではありません。

| Analysis | Dynamo reference | scDiffusionODE implementation |
|---|---|---|
| Vector-field interface / velocity | [`dynamo/vectorfield/scVectorField.py::BaseVectorField.func, get_V`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/scVectorField.py#L545-L597) | `vector_field_adapter.py::TrainedODEVectorField.func`, raw `model.ode_model(x)` only |
| Jacobian | [`DifferentiableVectorField.get_Jacobian`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/scVectorField.py#L787-L804) and [`SvcVectorField.get_Jacobian`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/scVectorField.py#L1055-L1084) | `TrainedODEVectorField.get_Jacobian`, `jacobian_tensor`; exact Hill derivative or opt-in `torch.func.jacrev` |
| Divergence | [`dynamo/vectorfield/utils.py::compute_divergence`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/utils.py#L699-L725) | `TrainedODEVectorField.evaluate`: exact `tr(J)` without retaining per-cell full Jacobians |
| Acceleration | [`utils.py::acceleration_, compute_acceleration`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/utils.py#L728-L793) | `TrainedODEVectorField.evaluate`: exact `a=J@V`; `dynamo_analysis.py::evaluate_dataset` saves norms/cosine |
| Sensitivity | [`utils.py::compute_sensitivity`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/utils.py#L845-L868) | `dynamo_analysis.py::sensitivity_aggregates`; same normalized `(I-J)^-1` formula, explicit opt-in/dimension guard |
| Fixed-point seeds/search | [`BaseVectorField.find_fixed_points`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/scVectorField.py#L599-L637) and [`utils.py::find_fixed_points`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/utils.py#L1215-L1276) | `representative_real_cells`, `find_fixed_points`; PCA k-means/annotation representatives、real-data domain判定、重複除去を採用 |
| Fixed-point stability | [`dynamo/vectorfield/FixedPoints.py::FixedPoints`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/FixedPoints.py#L7-L90) | `dynamo_analysis.py::_classify_eigenvalues`; small Dはfull eig、large DはARPACKのleftmost/rightmost eigs |
| Acceleration gene ranking | [`rank_vf.py::rank_acceleration_genes`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/rank_vf.py#L372-L403) | `dynamo_analysis.py::acceleration_gene_summary`; cell平均のraw/absolute accelerationをrank |
| Jacobian gene/pair ranking | [`rank_vf.py::rank_jacobian_genes`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/rank_vf.py#L440-L569) | `jacobian_gene_summary`, `jacobian_top_interactions`; target/source軸平均とpositive/negative/absolute pair ranking |
| Sensitivity gene ranking | [`rank_vf.py::rank_sensitivity_genes`](https://github.com/aristoteleo/dynamo-release/blob/69ba903000a1dcd5aa7d8b36a9318c1b22c5d650/dynamo/vectorfield/rank_vf.py#L572-L660) | opt-in sensitivity aggregateに同じsource/target reductionを適用 |

### Fixed-point solverのadapter差分

Dynamoの汎用utilityはMINPACK `fsolve` の数値差分Jacobianを使います。復元ODEは
float32で評価され、D=1024ではこの数値差分がゼロになる場合があるため、ここでは
同じ `V(x*)=0` 探索をtrained Hill ODEの厳密JacobianとNewton–GMRESで解きます。
これはvector fieldの再推定ではありません。解はreal-dataの各geneのmin/max domain
外なら棄却し、Dynamoと同じく近接解を重複除去します。全seedの成否と理由は
`fixed_point_attempts.csv` に残ります。

## 実行方法

`scdiffusion` environmentでrepository rootから実行します。

```bash
conda activate scdiffusion

python work/20260817_vector_field_analysis/analyze_vector_field.py \
  --run-dir work/20260803_ODE_hill_exp/runs/standard_hybrid_single__hill_after_linear/<batch-id> \
  --max-real-cells 2000 \
  --max-generated-cells 2000 \
  --jacobian-cells 200 \
  --device cuda
```

run directoryの `exp_config.json` / `manifest.json` からcheckpoint、real h5ad、
generated NPZを自動復元します。既定の保存先は次のtimestamped directoryで、既存
directoryは上書きしません。

```text
work/20260817_vector_field_analysis/outputs/
  <experiment>__<checkpoint>__<timestamp>/
```

任意の場所へ保存する場合は、まだ存在しないpathを `--output-dir` に渡します。

```bash
python work/20260817_vector_field_analysis/analyze_vector_field.py \
  --run-dir "$RUN_DIR" \
  --output-dir work/20260817_vector_field_analysis/outputs/my_analysis \
  --max-real-cells 2000 \
  --max-generated-cells 2000 \
  --jacobian-cells 200 \
  --device cuda
```

background実行例です。

```bash
mkdir -p work/20260817_vector_field_analysis/logs
nohup python work/20260817_vector_field_analysis/analyze_vector_field.py \
  --run-dir "$RUN_DIR" \
  --max-real-cells 2000 \
  --max-generated-cells 2000 \
  --jacobian-cells 200 \
  --device cuda \
  > work/20260817_vector_field_analysis/logs/vector_field_analysis.log 2>&1 < /dev/null &
echo $!
```

`--jacobian-method analytical` が既定です。独立検証または小規模解析では
`--jacobian-method autograd` により `torch.func.jacrev` を使えます。
`--jacobian-cells 200` はrealとgeneratedからそれぞれ最大200 cellを選びます。

Sensitivityは1024×1024 dense inverseをcellごとに行うため既定では実行しません。
小さいモデルで明示的に実行する例は次の通りです。D=1024で実行するには
`--sensitivity-max-dim` も意図的に引き上げる必要があります。

```bash
python work/20260817_vector_field_analysis/analyze_vector_field.py \
  --run-dir "$RUN_DIR" \
  --sensitivity-cells 2 \
  --sensitivity-max-dim 256 \
  --device cpu
```

fixed-point探索を省略したい場合は `--fixed-point-seeds 0` を指定します。

## Erythropoietic UMAP visualization

既存解析に `erythropoietic_umap.py` を追加し、
`work/20260609_Hybrid5x3/viz/plot_velocity_umap.py` の
`_plot_stream_arrow()` と `apply_lineage_palette()`（Erythropoieticの既存celltype順、
`magma` paletteを含む）を直接再利用します。既存のPCA、Jacobian、fixed point、
real/generated解析はそのまま残します。

処理は元h5adを `sc.read_h5ad(data_dir)` で改めて読み、最初に
`Superclass == "Erythropoietic"` を抽出します。したがって
`--max-real-cells` で全Superclassから選んだreal cellは使いません。必要な場合だけ
`--umap-max-cells N` で、この抽出後の集団を固定seedでsubsampleできます（既定値
`0` は全Erythropoietic cell）。そのsubsetに対して旧実装と同じ
PCA (`arpack`, 50 components) → neighbors (`n_neighbors=15`, `n_pcs=40`) → UMAPを
1回だけ行い、同一の `X_umap` を8図で共有します。

subsetのcell順を確定してから同じ `adata_sub.X` を
`TrainedODEVectorField.evaluate()` に1回渡し、返った `V(x)`、`||V||`、`tr(J)`、
`||JV||`、`cosine(V,JV)` を位置で `layers` / `obs` に代入します。代入前に
gene順、cell ID順、metric長、`velocity_ode.shape == adata_sub.X.shape` を検証します。
metricは1024次元expression空間で計算し、UMAP空間では再計算しません。velocity
だけは旧実装と同じ `scv.tl.velocity_graph(..., xkey="X", backend="loky")` →
`scv.tl.velocity_embedding(..., basis="umap")` で投影します。

通常の実行コマンドに追加引数は不要です。全Erythropoietic cellを使うことを明示する
場合は次のように指定します。

```bash
python work/20260817_vector_field_analysis/analyze_vector_field.py \
  --run-dir "$RUN_DIR" \
  --max-real-cells 2000 \
  --max-generated-cells 2000 \
  --umap-max-cells 0 \
  --jacobian-cells 200 \
  --device cuda
```

生成物はanalysis output以下の次のdirectoryに保存します。

```text
umap_by_superclass/Erythropoietic/
├── 1_velocity_stream.png
├── 2_velocity_arrow.png
├── 3_velocity_stream_lineage.png
├── 4_velocity_arrow_lineage.png
├── 5_speed_umap.png
├── 6_divergence_umap.png
├── 7_acceleration_norm_umap.png
├── 8_cosine_velocity_acceleration_umap.png
└── erythropoietic_umap_metrics.csv
```

変更対象は `analyze_vector_field.py`、新規 `erythropoietic_umap.py`、このREADME、
および新しい成果物を検証する `tests/smoke.py` です。

## 生成物

各analysis directoryには少なくとも次を保存します。

```text
cell_metrics.csv
summary_metrics.csv
acceleration_gene_summary.csv
jacobian_cell_selection.csv
jacobian_aggregates.npz
jacobian_gene_summary.csv
jacobian_top_interactions.csv
fixed_point_attempts.csv                 # searchを実行した場合
fixed_points.csv                         # 見つかった場合のみ
fixed_point_eigenvalues.csv              # 見つかった場合のみ
real_velocity_pca.png
generated_velocity_pca.png
divergence_pca.png
acceleration_norm_pca.png
speed_pca.png
real_vs_generated_metrics.png
pca_projection.csv
pca_model.npz
analysis_manifest.json
```

`cell_metrics.csv` はreal/generated各cellの `||V||`、`tr(J)`、`||J V||`、
`cosine(V, J V)` を持ちます。速度または加速度が厳密にzeroでcosineが未定義の
場合だけ、`cosine_defined=false` としcosine値を0にします。

`jacobian_aggregates.npz` はdataset別のmean Jacobianとmean absolute Jacobian
だけを持ちます。cellごとのD×D×N tensorは保存しません。`W[target, source]` と
同じく、Jacobianもrowがtarget、columnがsourceです。

PCAはreal dataにだけfitし、stateは通常のPCA transform、velocityは
`V_PCA = V @ components.T` で線形projectします。全vector-field metricと
fixed-point探索はPCAではなく元のgene-expression空間で計算します。

## 検証

数式テストはclosed-form Hill Jacobianと `torch.func.jacrev`、divergence、
acceleration/JVP、fixed-point stabilityを比較します。

```bash
python work/20260817_vector_field_analysis/tests/test_vector_field.py
```

CLI smoke testは学習もsamplingも行わず、tiny h5ad、generated NPZ、直接serialize
したmodel stateを一時directoryに作り、post-hoc CLIと成果物を検証します。

```bash
python work/20260817_vector_field_analysis/tests/smoke.py
```

いずれの経路もNaN/Infを `nan_to_num` 等で修復しません。入力、checkpoint、ODE
出力、Jacobian、metric、PCA、fixed pointに非finite値があれば明示的に失敗し、
作成済みanalysis directoryのmanifestを `status=failed` にします。
