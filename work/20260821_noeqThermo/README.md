# Erythropoietic nonequilibrium landscape / flux解析

## 目的

`work/20260817_vector_field_analysis/`で復元・監査済みの学習済みODEを、論文で用いられたDynamoの連続ベクトル場と非平衡landscape/flux解析へ接続します。対象は常に次の実細胞です。

```text
Superclass == Erythropoietic
```

既存解析を置き換えません。2種類の量を明確に分離します。

- 既存解析: 1024次元gene-expression空間の`V(x)`、`tr(J)`、`JV`など
- 本解析: 固定された2次元UMAP上で再構成した`F_UMAP(z)`、curl、固定点、確率分布、potential、probability flux、LAP

UMAP上のcurlや固定点を、1024次元gene空間の量として解釈してはいけません。

## 論文と公式実装

- Zhu et al., *PNAS* 2024, [doi:10.1073/pnas.2401540121](https://doi.org/10.1073/pnas.2401540121), [公式コード](https://github.com/Zhu-1998/celldevelopment)
- Zhu and Wang, *Advanced Science* 2024, [doi:10.1002/advs.202308879](https://doi.org/10.1002/advs.202308879), [公式コード](https://github.com/Zhu-1998/cellcycle)
- [Dynamo v1.4.1](https://github.com/aristoteleo/dynamo-release/tree/v1.4.1)

公式コードとの行単位の対応、確認commit、意図的な修正は[`paper_reference/UPSTREAM_AUDIT.md`](paper_reference/UPSTREAM_AUDIT.md)に記録しています。

## 数学的パイプライン

全モデルで、同じh5ad、同じcell順、同じgene順を検証します。最初のモデルのh5adからErythropoietic細胞を一度だけ抽出し、既存実装と同じPCA (`arpack`, 最大50成分) → neighbors (`n_neighbors=15`, `n_pcs=40`) → UMAPを一度だけ実行します。

```text
adata.X（gene順固定）
  -> 復元済み model.ode_model(x) = V(x)
  -> 既存scVelo velocity_graph / velocity_embedding
  -> 全モデル共通X_umap上のper-cell velocity_umap
  -> dyn.vf.VectorField(..., basis="umap")
  -> adata.uns["VecFld_umap"]
  -> Dynamo vector_field_function(z)
```

連続場`F(z)`から、著者コードと同じEuler–Maruyama法を使います。

```text
z(t+dt) = z(t) + F(z) dt + sqrt(2 D dt) N(0, I)
P_ss = num_tra / sum(num_tra)
U = -log(P_ss)
J = <F> P_ss - D grad(P_ss)
F_gradient = -D grad(U)
F_rotational = J / P_ss
```

境界は反射境界です。simulation中は全trajectory履歴を保持せず、現在位置と`num_tra`、`total_Fx`、`total_Fy`だけを保持します。checkpointにはRNG状態も保存するため、同じoutput directoryで再実行すると決定論的に再開できます。

## 全モデル共通のUMAPとSDEパラメータ

`dt=0.1/0.2/0.5`、`D=0.05/0.1/0.2`という論文値は設定内に参照値として残していますが、既定では別のUMAP尺度へ直接移しません。共通UMAPのmedian nearest-neighbor distanceと、比較対象全モデルのmedian UMAP speedから、全モデルに一つだけの`dt`と`D`を計算します。

- deterministic 1 step: median近傍距離の5%
- diffusive RMS 1 step: median近傍距離の15%
- 較正済み`dt`と`D`: `common/sde_calibration.json`

文字どおりの論文値を使う実験では、独自configで`sde.parameterization`を`paper_literal`にし、`dt`と`D`を明示してください。モデルごとに異なる値を使う機能は意図的に設けていません。

## インストール

既存の`scdiffusion`環境を使います。DynamoのAPI互換性を固定するため、Dynamo 1.4.1と、scVelo 0.2.5とも両立するMatplotlib 3.7.5を追加します。

```bash
conda activate scdiffusion
python -m pip install -r work/20260821_noeqThermo/requirements.txt
```

## 実行

### 数式unit test

```bash
conda run -n scdiffusion python work/20260821_noeqThermo/tests/test_landscape.py
```

### 合成データsmoke

Dynamoから図とLAPまでを検証します。

```bash
conda run -n scdiffusion python work/20260821_noeqThermo/tests/smoke_synthetic.py
```

学習済みcheckpointのfixtureを使い、モデル復元、gene velocity、固定UMAP、scVelo投影、Dynamo、simulationまでを一括検証します。

```bash
conda run -n scdiffusion python work/20260821_noeqThermo/tests/smoke_model_pipeline.py
```

### 実runのsmoke

```bash
python work/20260821_noeqThermo/scripts/run_analysis.py \
  --mode smoke \
  --run-dir work/20260803_ODE_hill_exp/runs/standard_hybrid_single__hill_after_linear/<run-id> \
  --device cpu
```

### full解析

`--run-dir`は複数回指定できます。`--runs-root`を使うと、配下の`standard_hybrid_single/hill_after_linear` runをすべて検出します。

```bash
python work/20260821_noeqThermo/scripts/run_analysis.py \
  --mode full \
  --runs-root work/20260803_ODE_hill_exp/runs/standard_hybrid_single__hill_after_linear \
  --device cuda
```

重要設定は[`configs/smoke.json`](configs/smoke.json)と[`configs/full.json`](configs/full.json)に集約されています。fullは論文のcell-cycle実装に近い400 trajectory、100万step、10万step burn-in、100×100 gridです。PNAS HSCの200万stepより軽く、vectorized batch評価とcheckpointを使う実用的な最終設定です。

## 出力

全モデルが一つの`common/erythropoietic_fixed_umap.*`を共有し、各モデルは`models/<model>/`に分離されます。

```text
outputs/<analysis>/
├── analysis_manifest.json
├── model_comparison_summary.csv
├── common/
│   ├── erythropoietic_fixed_umap.csv
│   ├── erythropoietic_fixed_umap.npz
│   ├── embedding_metadata.json
│   └── sde_calibration.json
└── models/<model>/
    ├── observed_umap_velocity.{csv,npz}
    ├── dynamo_vecfld_umap.npz
    ├── fixed_points_umap.csv
    ├── curl_umap_cells.csv
    ├── num_tra.csv / p_tra.csv / pot_U.csv
    ├── mean_Fx.csv / mean_Fy.csv / Xgrid.csv / Ygrid.csv
    ├── landscape_flux_arrays.npz
    ├── least_action_paths.csv
    ├── model_manifest.json
    ├── 01_umap_vector_field.png
    ├── 02_topography_fixed_points.png
    ├── 03_curl_umap.png
    ├── 04_steady_state_probability.png
    ├── 05_potential_landscape.png
    ├── 06_mean_force.png
    ├── 07_gradient_force.png
    ├── 08_probability_flux.png
    ├── 09_landscape_flux_overlay.png
    └── 10_least_action_paths.png（有効なLAPが得られた場合のみ）
```

著者Pythonコードに合わせ、CSVの`num_tra`、`p_tra`、`pot_U`、`mean_Fx/Fy`は`[x_bin, y_bin]`です。図とNPZの`*_plot`は、著者MATLABコードのtranspose後と同じ`[y_bin, x_bin]`です。

## LAP endpoint

`celltype`に十分な細胞数を持つdistinct stateが2種類以上ある場合だけ実行します。既定では頻度上位のbiological labelから代表細胞（UMAP centroid最近傍）を選び、ordered pairを計算します。発生方向をmetadataから決められないため、方向を捏造せず、両方向を別pathとして扱います。特定labelを使う場合は`lap.endpoint_labels`へ明示してください。

## 既知の制約

- UMAP上のlandscapeは、元の1024次元力学を非線形埋め込みへ投影し、さらにSparseVFCで連続化した解析です。元空間のpotentialではありません。
- `P_ss`の未訪問binは0、raw `pot_U`は`+inf`のまま保存します。gradientと図だけ、有限potential最大値+2でcapします。NaN/Infを黙って埋めたraw結果は保存しません。
- probability fluxはconstant isotropic diffusionを仮定します。state-dependent diffusionは対象外です。
- LAPはmetadataにdistinct Erythropoietic stateがない場合、またはDynamo最適化が全pairで失敗した場合は図10を作りません。失敗理由は`least_action_metadata.json`へ保存します。
- MFPT、transition matrix、loop-flux decompositionは、endpoint方向とtime scaleを正当化できないため自動生成しません。
