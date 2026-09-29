# Stage1 CellUNetを固定時刻のベクトル場として解析

実細胞のh5ad.Xに対して、`V_t(x) = CellUNet(x, t)`をそのまま評価します。
既定の条件は **t=100, 900, 1000**。入力にノイズを加えず、出力からxを引かず、
時刻の再スケーリングや999への置換もしません。学習時刻は0〜999のため、
t=1000は範囲外の評価として図・manifestに明記します。

このモデルはSTART_X予測器です。本解析はその出力を速度として定義した
人工的な固定時刻の場の解析であり、DDPMのreverse driftや生物学的速度の推定とは異なります。
さらにlandscape・curl・fluxは、scVelo投影後にDynamoで再構成した**2次元UMAP場**の量です。

## 再利用と入力

- Stage1復元: `20260916_x0predict_hybrid_additive` / `20260915_x0predict`の既存loader・factory。
- 非平衡解析: `20260821_noeqThermo/src/noeqthermo`をimport。投影・Dynamo・SDE・描画のコピーはありません。
- 新規コードは実行入口とCellUNetのvelocity adapterのみ。
- 既存設定どおり`Superclass == Erythropoietic`を抽出します。
- 同一h5ad、細胞順、遺伝子順、UMAPを3条件で共有します。
- sampling済みファイルは不要です。生成`.npz`を探索しません。
- final 30,000stepのStage1 EMA、START_X設定、state、データSHA256、遺伝子順を検証します。

checkpoint未指定時は20260916の`configs/base.json`に記録された元のStage1を使います。
h5ad未指定時はcheckpoint metadataの`data_dir`を使います。
`--data`は同じh5adを別の場所へ移した場合に使用できます（SHA256一致が必要）。

## remoteでpull・実行

```bash
cd /home/suzuki/Projects/scDiffusion-github
git switch feat/20260916-x0predict-hybrid-additive
git pull --ff-only origin feat/20260916-x0predict-hybrid-additive
conda activate scdiffusion
python -m pip install -r work/20260821_noeqThermo/requirements.txt
```

入力確認のみ（出力を作りません）:

```bash
python work/20260929noneqThermo/scripts/run_analysis.py \
  --mode smoke --device cuda --dry-run
```

3時刻の短い解析・作図:

```bash
python work/20260929noneqThermo/scripts/run_analysis.py \
  --mode smoke --device cuda \
  --output-dir work/20260929noneqThermo/outputs/stage1_smoke
```

3時刻のfull解析・作図:

```bash
python work/20260929noneqThermo/scripts/run_analysis.py \
  --mode full --device cuda \
  --output-dir work/20260929noneqThermo/outputs/stage1_full
```

fullは**各条件で400 trajectory × 1,000,000 step**です。CUDAはCellUNet評価に使われ、
その後のscVelo/Dynamo/SDE全体がGPU化されるわけではありません。

特定の20260916 campaign内のcanonical Stage1を使う場合は、各コマンドに次を追加します:

```bash
--campaign work/20260916_x0predict_hybrid_additive/runs/additive_<campaign-id>
```

元の20260915 Stage1を直接指定する場合は`--checkpoint /path/to/ema_0.9999_030000.pt`。
`--campaign`と`--checkpoint`は同時指定できません。
`--timesteps 100 900 1000`でも明示できます（未指定時もこの3条件）。

## 設定・出力・再開

`configs/smoke.json`と`configs/full.json`は既存20260821設定への参照、時刻、batch sizeを持つ
小さな入口設定です。`--config`で別の入口JSONを指定できます。
その`analysis_config`はこのディレクトリを基準に解決する完全な解析JSONへのパスです。

```text
outputs/<実行名>/
  analysis_manifest.json
  resume_identity.json
  model_comparison_summary.csv
  common/
    erythropoietic_fixed_umap.{csv,npz}
    embedding_metadata.json
    sde_calibration.json
    plot_scales.json
  models/
    t_0100/
    t_0900/
    t_1000/
      observed_gene_velocity.npz  # velocity, genes, cell_ids
      observed_umap_velocity.{csv,npz}
      model_manifest.json
      simulation_identity.json
      simulation/
      landscape_flux_arrays.npz
      01_umap_vector_field.png ... 09_landscape_flux_overlay.png
      10_least_action_paths.png   # 有効なLAPが得られた場合のみ
```

3条件でSDEのdt・D・領域・grid・seedを共有します。図は共通の軸範囲、curl・確率・potentialの
色範囲を使い、potential図の高さ範囲も揃えます。矢印は既存の描画規則どおり方向を正規化して
表示するため、長さで速度の大小を比較しないでください。図は全条件のsimulation終了後に作成します。

同じコマンド・同じ出力先で再実行するとSDE checkpointから再開します。
入力、時刻、設定、device指定、コードが変わった場合は既存出力への再開を拒否します。
再計算したUMAP・投影速度・連続場が変わった場合も拒否します。変更後は新しい出力先を指定してください。
smokeとfullには別の出力先を使用します。

## 検証

```bash
python work/20260929noneqThermo/tests/test_adapter.py
python work/20260821_noeqThermo/tests/test_landscape.py
python work/20260929noneqThermo/tests/smoke_pipeline.py
python work/20260821_noeqThermo/tests/smoke_model_pipeline.py
```

新規smokeは合成h5adと小さなCellUNetのStage1形式checkpointを作り、実CLIを通して
3条件の作図・raw forward一致・座標/色範囲共有・再開・条件変更時の拒否を検証します。
これは科学的結果を検証する学習済み実checkpointのfull runではありません。
検証成果物は`outputs/validation_*`に保存します（Git対象外）。
