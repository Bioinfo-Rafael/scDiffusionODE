# Hodge entry point

`projector.py:HodgeProjector.fit_scalar()` を読むと、F(X)評価 → Julia公式workflow → `obs['projection_hodge_raw']` → 連続近似 → `obs['projection_hodge']` の順序が分かる。

`run_ddhodge.jl` は `kNNGraph` と `ddHodgeWorkflow` を直接呼ぶ。Python側の `ddhodge_bridge.py` はsubprocess/HDF5 I/Oだけを担う。`ddh.u` を正規化・反転しない。`F_gradient=-grad(U)`、加法定数は不定。論文Methods式(2)の表示と公式コードの符号差は [共通README](../README.md) に記載した。

Julia **1.10.10**、ddHodge **0.7.1**、commit **d5d6b09431c0ccbb16e0b63414a5038511c19b85**。`julia_env/Manifest.toml` は実際のPkg resolverで生成し、ddHodgeと直接依存をpinしたもの。初期化はrepository rootから:

```bash
julia --startup-file=no --project=work/20261028_Projection/Hodge/julia_env -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

PythonはN×D、JuliaはD×N。flat C-order→Julia reshapeを利用する。DとNを区別するテストがある。非連結graph、dimension不正、公式solverのstationarity異常、非有限出力、Julia欠如、timeoutは明示エラー。別Hodge solverへ切り替えない。`ssa=false` は公式に存在する明示オプションであり、既定はtrue。

原論文: https://www.nature.com/articles/s41467-025-67782-6

公式: https://github.com/kazumits/ddHodge.jl

任意発現点での連続U(x)、PCA、等値面retraction、接線射影は本実装の追加であり、ddHodgeがその正しさを保証するものではない。実行・保存・リソース・検証状況は [共通README](../README.md) と [VALIDATION](../VALIDATION.md) を参照する。
