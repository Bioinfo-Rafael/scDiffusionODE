# `20260817_vector_field_analysis`監査結果

## モデル

- `exp_config.json`が`model_family=standard_hybrid_single`、`ode_type=hill_after_linear`のrunだけを受理します。
- runの`manifest.json`/config/globからcheckpointとsampleを解決します。
- `build_diffusion(config)`と`load_model(config, genes, diffusion, checkpoint, device)`を使い、Hybrid wrapper全体を復元します。
- vector fieldとして使うのは`model.ode_model`だけです。diffusion timestep、ML branch、Hybrid出力は使いません。

本解析はこれらのprivate helperと`TrainedODEVectorField`をimportして再利用します。

## cell / gene / input

- 元h5adを`scanpy.read_h5ad(data_path)`で読み直します。
- `Superclass == "Erythropoietic"`を最初に位置filterし、その後だけ必要に応じて固定seedでsubsampleします。
- ODE入力はsubsetの`adata.X`をdense float32にしたものです。別layerやPCA値へ置換しません。
- gene名は`adata.var["gene_name"]`があればそれを使い、なければ`var_names`を使います。
- checkpoint復元に渡したgene順とh5ad gene順、cell ID順、`X.shape`、`V.shape`を検証します。

## velocityとUMAP投影

- gene-space velocityはrow-vector batchに対する`model.ode_model(x, None)`です。
- 既存Erythropoietic UMAPはPCA (`arpack`, 50) → neighbors (`15`, `40 PCs`) → UMAPを一度だけ計算します。
- `layers["X"] = adata.X`、`layers["velocity_ode"] = V(x)`を設定します。
- `scv.tl.velocity_graph(..., vkey="velocity_ode", xkey="X", backend="loky")`と`scv.tl.velocity_embedding(..., basis="umap")`でUMAP velocityへ投影します。

本解析はこの処理をモデルごとに同じcell/neighbor graph/UMAP座標上で行います。UMAP fittingは全モデルを通じて一回だけです。座標、cell順、gene順はSHA-256を記録し、各モデルのDynamo fit前に一致を確認します。

## 既存plot convention

既存実装は`work/20260609_Hybrid5x3/viz/plot_velocity_umap.py`のscVelo互換shim、stream/arrow helper、Erythropoietic lineage paletteを再利用します。本解析はvelocity projectionの互換shimを同じ経路から読み、landscape/topography図は論文公式notebook/MATLABのvisual grammarへ合わせます。
