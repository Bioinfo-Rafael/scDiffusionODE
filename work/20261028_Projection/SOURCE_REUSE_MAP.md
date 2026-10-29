# Source reuse audit

独自コードは原著者のsolverを置換しない。公式APIが動かなければ例外で停止する。

| 計算 | 元論文 | 公式repository | 実際に呼ぶ公式関数 | 今回の呼出し元 | 境界 | version / commit |
|---|---|---|---|---|---|---|
| D20 ODE復元 | 本repositoryの既存学習条件 | [scDiffusionODE](https://github.com/Bioinfo-Rafael/scDiffusionODE) | 既存 `load_model.load_trained_model('lora',...)` → `checkpoint.resolve/read/check_ids`, `data.read_data/mask_for`, `model.build_field` | `common/velocity_adapter.py:load_d20` | 検証・重み復元は既存ローダー。今回追加するのはNumPy入出力・固定tでのODE呼出し | 実行時checkpoint/source hashesを保存 |
| kNN graph | [ddHodge論文](https://www.nature.com/articles/s41467-025-67782-6) | [ddHodge.jl](https://github.com/kazumits/ddHodge.jl) | `ddHodge.kNNGraph(X,k)` | `Hodge/run_ddhodge.jl` | HDF5 flat入出力とconnectivity検査のみ追加 | 0.7.1 / `d5d6b09431c0ccbb16e0b63414a5038511c19b85` |
| potential / Hodge workflow | 同上 | 同上 | `ddHodge.ddHodgeWorkflow(g,X,V;...)`, `ddh.u` | `Hodge/run_ddhodge.jl` ← `Hodge/ddhodge_bridge.py:run_ddhodge` ← `Hodge/projector.py:fit_scalar` | graph Laplacian/potential/Hessian/Jacobian solverは全て公式。既存d0,wによる結果残差検査のみ追加 | 同上、依存はJulia Manifest |
| CellRank用neighbors | [Scanpy](https://doi.org/10.1186/s13059-017-1382-0) | [scanpy](https://github.com/scverse/scanpy) | `scanpy.pp.neighbors(use_rep='X')` | `CellRank/cellrank_core.py:velocity_kernel` | 入力は元発現、scratch AnnDataで既存属性を保護 | 1.10.4 |
| velocity similarity / P | [CellRank 2論文](https://www.nature.com/articles/s41592-024-02303-9) | [cellrank](https://github.com/scverse/cellrank) | `VelocityKernel(...)`, `.compute_transition_matrix()` | `CellRank/cellrank_core.py:velocity_kernel/transition_matrix` ← `CellRank/projector.py:fit_scalar` | similarity、softmax、Pは公式。形状/順序・行和/疎性検査のみ追加。kernel混合なし | 2.0.7 / v2.0.7 commit `ce69c5f5c637c838193a11160c22fdd3cd0007db` |
| terminal state | 同上 | 同上 | `GPCCA(kernel)`, `.set_terminal_states(dict, allow_overlap=False)` | `CellRank/cellrank_core.py:terminal_estimator` | ラベル→cell ID変換のみ追加。自動macrostate推定は使わない | 同上 |
| absorption probability / time | 同上、具体的なconditional time式は採用tagのsourceを参照 | 同上 | `GPCCA.compute_fate_probabilities(keys=None)`, `.compute_absorption_times(keys=None)` | `CellRank/cellrank_core.py:absorption_times` | 線形solveとconditional timeは公式。到達性と結果残差検査、選択列抽出のみ追加。独自MFPT solverなし | 同上 |
| τ正規化 | 今回の定義。原論文標準pseudotimeとは呼ばない | なし | なし | `CellRank/projector.py:fit_scalar` | 有限・確率threshold内のTを反転min–max。元Tは保持 | `projection_v1`, 実行時source hash |
| 微分可能な連続拡張 | 原論文にはない追加処理 | [scikit-learn](https://github.com/scikit-learn/scikit-learn), [SciPy](https://github.com/scipy/scipy) | `PCA.fit`, `scipy.linalg.solve`, `scipy.spatial.distance.cdist`, `cKDTree` | `common/scalar_extension.py:ScalarExtension` | PCA+Gaussian RBF ridge、解析微分chain rule、holdout・support診断は本実装。離散potentialsやPを再推定しない | scikit-learn 1.5.2 / SciPy 1.14.1 |
| 等値面normal retraction / 接線射影 | ユーザー指定式。両原論文にない追加処理 | なし | NumPy演算のみ | `common/levelset_projection.py:ProjectedField` | yを求めてF(y)を再評価。最近点の保証なし。解析平面・円で検証 | `projection_v1`, 実行時source hash |
| 保存/再読込 | なし | [AnnData](https://github.com/scverse/anndata) | `AnnData.write_h5ad`, `scipy.sparse.save_npz/load_npz` | `common/persistence.py` | JSON/NPZ/hash/元モデル照合は本実装 | anndata 0.11.4、他はrequirements |

## 実際に確認した公式source

- [ddHodge.jl/src/ddHodge.jl](https://github.com/kazumits/ddHodge.jl/blob/d5d6b09431c0ccbb16e0b63414a5038511c19b85/src/ddHodge.jl): `graphgrad`, `calcwt`, `kNNGraph`, `ddHodgeWorkflow`。Pythonへ移植していない。
- [CellRank VelocityKernel](https://github.com/scverse/cellrank/blob/v2.0.7/src/cellrank/kernels/_velocity_kernel.py): deterministic branchは `_vdata`、stochasticはmomentを使う。今回はdeterministicだけ。
- [CellRank fate mixin](https://github.com/scverse/cellrank/blob/v2.0.7/src/cellrank/estimators/mixins/_fate_probabilities.py): `keys`を絞るとterminal集合が変わることを確認。全terminalを保持して結果列を選ぶ。
- [CellRank absorption helper](https://github.com/scverse/cellrank/blob/v2.0.7/src/cellrank/_utils/_utils.py): `_calculate_absorption_time_moments`, `_calculate_lineage_absorption_time_means`。これらprivate helperを直接呼ばず、GPCCA公開APIを経由する。
- [CellRank linear solver](https://github.com/scverse/cellrank/blob/v2.0.7/src/cellrank/_utils/_linear_solver.py): `direct`の密化とgmresの経路を確認。独自solverへの代替なし。

論文との符号不一致はREADMEのHodge節に記載した。原論文の生物学的結果を今回のD20へ移植したと主張しない。

## Surface divergence追加分（2026-10-28）

| 機能 | 再利用元と追加処理 | 境界 |
|---|---|---|
| 対象5クラス | `data_preparation/20261007/benchmark/config.py:ERYTHROID_CELLTYPES`をimport | スカラー母集団・解析対象とも5クラス（2026-10-29変更）。全trainingの保存スカラーは再利用しない |
| 元Fとその微分 | `ODE/ode_20260609_mathmlp.py`の実際のLoRAField.forwardを固定tで呼ぶ | 状態依存係数を含む。`work/20260817_vector_field_analysis/vector_field_adapter.py`のHill解析Jacobianは採用しない |
| JVP/VJP | [PyTorch 2.7 jvp](https://docs.pytorch.org/docs/2.7/generated/torch.func.jvp.html)、[vjp](https://docs.pytorch.org/docs/2.7/generated/torch.func.vjp.html)、vmap | 8/17のAD利用例を参照し、固定D20への薄い接続を追加。jacrev/hessianは小Dの独立検証だけ |
| スカラー・射影 | 既存`HodgeProjector`、`CellRankProjector`、`ScalarExtension.gradient`、`surface_points`、`projected_field(y)` | 公式スカラーsolver・retraction・gradientを再実装しない |
| 連続Hessian | 既存Gaussian RBFの解析2階微分をPCA空間に追加 | 元離散Hodge/CellRankのHessianではない |
| Surface divergence・遺伝子座標寄与 | `analysis/surface_divergence.py`、`analysis/gene_contributions.py` | ユーザー指定式の追加解析。原論文や公式packageの既成解析とは扱わない |
| 候補GRNの感度 | training manifestのhash検証済み候補辺だけに元FのJVPを評価 | 候補辺選択・局所感度であり、GRN再推定・因果推論ではない |

10/13の`d20_adapter.py`は実測細胞・CellUNetに結び付くため入口にせず、10/27の既存ODE-onlyローダーを継続利用する。平面、非ゼロ曲率球面、非対称J、実装元LoRAFieldの人工重みで独立微分との照合を行った。実D20 checkpointでのAD検証はローカルでは未実施。

## 環境分離（2026-10-29）

`CellRank/subprocess_bridge.py`は親でX/F(X)をNPZへ保存し、指定した別Pythonから`CellRank/worker.py`を起動する。子も同じ既存`cellrank_core.py`を呼び、公式計算の式・solver・terminal定義を変更しない。親と子で異なるAnnDataバージョンを使えるよう、境界はNPZ/JSON/CSRのみ。checkpointやモデルPython objectを子へ渡さない。`CELLRANK_PYTHON`未指定の既存APIは同一プロセス呼出しを維持する。検証では親Python 3.9/Torch 2.5.1/CellRankなし、子Python 3.11/CellRank 2.0.7/Torchなしを使う。
