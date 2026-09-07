# 公式コード監査メモ

確認日: 2026-08-21

## 固定したupstream

| 対象 | commit/tag |
|---|---|
| `Zhu-1998/celldevelopment` | `cd28f32d70f410ad516713e526a71a2f63e673df` |
| `Zhu-1998/cellcycle` | `31fa16497347739e1b8c8f664fe35940ed631dec` |
| `aristoteleo/dynamo-release` | tag `v1.4.1`, commit `2293afb931cad46475199e2b31ac2f63a30b02bf` |

巨大datasetや論文生成物はcopyしていません。公式repositoryを読み、関数呼び出しと数式を本解析へ接続しています。

## 直接利用したDynamo API

| 本解析 | upstream |
|---|---|
| 連続UMAP場 | `dynamo/vectorfield/VectorField.py::VectorField`; `dyn.vf.VectorField(adata, basis="umap")` |
| 任意座標での場評価 | `dynamo/vectorfield/utils.py::vector_field_function`; 論文scriptの`con_K(x, X_ctrl, beta).dot(C)`と同じ公開関数 |
| topography/fixed point | `dynamo/vectorfield/topography.py::topography`; `dyn.vf.topography(..., basis="umap")` |
| fixed-point分類 | `FixedPoints.py`; `-1=stable`, `0=saddle`, `1=unstable` |
| 2D curl | `dynamo/vectorfield/vector_calculus.py::curl`; `dyn.vf.curl(..., basis="umap")` |
| LAP | `dynamo/prediction/least_action_path.py::least_action`; `dyn.pd.least_action(..., basis="umap")` |

Dynamoを模倣した`VecFld_umap`は作っていません。実際の`dyn.vf.VectorField`が生成した辞書を使います。保存時だけ、pickle依存を避けるためportableなarray/scalarをNPZ/JSONへ抽出します。

## PNAS公式コードとの対応

### `landscape-flux/human_hematopoiesis/hsc_landscape_multi.py`

- 32–52行: `VecFld_umap`とGaussian kernel analytical function
- 69–75行: `dt=0.2`、200万step、burn-in 10万、200 grid
- 105行: `D=0.2`
- 128–183行: Euler–Maruyama、反射境界、burn-in後の`num_tra/total_F`集計
- 187–194行: `P_ss`、`U=-log(P_ss)`、mean force、meshgrid
- 196–204行: 9種類のCSV

本解析の`landscape.py`は、この処理をtrajectory batchへvectorizeし、全履歴arrayを廃止しました。数理手順と9種類の名前は維持しています。

### `landscape-flux/mouse_retina/mouse_landscape_multi.py`

- 16–17行: PCA/UMAP上の`dyn.vf.VectorField`
- 29–47行: analytical vector-field function
- 50–83行: Latin hypercube sampling
- 96–99行: 500万step、burn-in 100万、100 grid
- 101–155行: trajectoryごとの集計
- 174–211行: 400 trajectoryの並列実行・集約・CSV保存

本解析の初期点は同じLatin hypercubeです。400 trajectoryはfull設定で維持し、trajectoryごとの一時CSVではなく一つのcheckpointへ集約します。

### `notebook/Fig 3,4B-G.ipynb` / `Fig S3.ipynb`

- `dyn.vf.VectorField(adata, basis='umap')`
- `dyn.vf.topography(adata, n=100, basis='umap')`
- cell annotation、streamline、番号付きfixed pointを重ねたtopography
- `dyn.pd.least_action(... basis="umap", adj_key="umap_distances", EM_steps=2)`
- pathを黒線、actionを連続色で重ねるLAP図

図01/02/10はこのvisual grammarを使います。論文と異なるcelltype名や固定点labelは作りません。

### `landscape-flux/plot_landscape_path.m`

- 15行: raw potentialのtranspose
- 18–55行: jet/turbo 3D surface、黒いmesh、UMAP軸
- 59–92行: landscape上のLAP overlay

図05は3D surface、jet、黒mesh、`view([30,45])`相当を採用しています。

## Advanced Science公式コードとの対応

### `cellcycle_path_landscape_multi.py`

- 43–55行: `VectorField`、`topography`、speed/curl/divergence/acceleration/curvature
- 59–77行: analytical field function
- 81–107行: Latin hypercube
- 119–121行: 100万step、burn-in 10万、100 grid
- 124–177行: Euler–Maruyamaと反射境界
- 197–234行: 400 trajectory、`P_ss`、potential、mean force、CSV

full設定はこの計算量を基準にしています。speed等の元gene-space解析は20260817側に既にあるため、本解析ではcoreの2D curl/topography/landscape/fluxを優先しました。

### `landscape/plot_landscape_flux_force.m`

- 36–103行: potential surfaceとmesh
- 134–147行: `Jx=mean_Fx.*P-D*GPx`, `Jy=mean_Fy.*P-D*GPy`、normalized flux quiver
- 163–194行: normalized mean force
- 197–233行: `F_gradient=-D*gradient(U)`

図06–09と保存arrayはこの定義をそのまま使います。probability fluxとDynamo curlは異なる量なので別々に保存・描画します。

## 意図的な修正・差分

1. 公式mouse/cell-cycle Pythonにはy下限の反射で`x_lim[0]`を参照する箇所があります。本解析は各axis固有の下限・上限を使います。
2. 1 stepでdomain幅以上overshootしても確実に反射できるtriangular reflectionを使います。
3. raw `pot_U`の未訪問binは`+inf`のまま保存します。MATLABは描画前に非finiteを有限値へ置換してcapするため、本解析もgradient/描画用arrayだけを適応的にcapします。
4. 論文の`dt`/`D`を別UMAPへ直移植せず、全モデル共通UMAP尺度から一組だけ較正します。参照値と較正式をmetadataへ保存します。
5. 全trajectory×全timeの巨大arrayやtrajectory別CSVは作らず、十分統計量を逐次更新します。RNGを含むcheckpointで再開可能です。
6. 著者コードのraw grid arrayは`[x,y]`、MATLAB plotはtranspose後の`[y,x]`です。両方を名前で区別して保存します。

## 今回実装しない解析

MFPT、transition matrix、loop-flux decompositionは、Erythropoietic metadataから一方向の発生順序とtime scaleを仮定しないと解釈が不安定です。LAP自体はbiological labelが2種類以上あるときだけ実行し、方向を捏造せずordered pairとして保存します。
