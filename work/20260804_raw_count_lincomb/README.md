# 2026-08-04 raw-count LinComb comparison

## すぐ実行するコマンド

すべてrepository rootから実行します。最終学習データ名は全箇所で
`Embryonic_raw_count.h5ad`です。

```bash
# 0. 初回データ作成 → 6条件のtrain → sample → UMAPを一括実行
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --prepare-data --batch-id 20260804_full_30000

# 1. Drive download → raw判定 → filter → HVG選定 → raw-count H5AD保存だけ
conda run -n scdiffusion python \
  work/20260804_raw_count_lincomb/scripts/create_raw_count_data.py

# download済み原本を使う場合
conda run -n scdiffusion python \
  work/20260804_raw_count_lincomb/scripts/create_raw_count_data.py --skip-download

# 2. 6条件を指定順でtrain → sample → UMAP
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --batch-id 20260804_full_30000

# 3. familyの3 ODE
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --family standard_hybrid_lincomb_softmax --batch-id 20260804_hybrid

# 4. ODEの2 family
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --ode hill_after_softplus --batch-id 20260804_hill

# 5. 1 experiment
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --experiment ode_only_lincomb_softmax__exp --batch-id 20260804_one

# 6. 複数experiment（実行順はcanonical順）
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --experiment ode_only_lincomb_softmax__softplus standard_hybrid_lincomb_softmax__exp \
  --batch-id 20260804_subset

# 7. 全smoke / 特定条件smoke
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py --smoke
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --smoke --experiment standard_hybrid_lincomb_softmax__exp

# 8. resume
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --resume auto --batch-id 20260804_full_30000

# 9. samplingのみ / UMAPのみ
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --sample-only --batch-id 20260804_full_30000
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --umap-only --batch-id 20260804_full_30000

# 10. data作成または6条件の実行予定だけを表示
conda run -n scdiffusion python \
  work/20260804_raw_count_lincomb/scripts/create_raw_count_data.py --dry-run
conda run -n scdiffusion python work/20260804_raw_count_lincomb/scripts/launch.py \
  --prepare-data --dry-run --batch-id 20260804_full_30000

# 11. 完了したODE-only 3条件の20260802方式all_models_facets.png
python -u work/20260804_raw_count_lincomb/integrated_analysis/run_all_models_facets.py

# 選択されるrun/sampleだけ確認（UMAPは実行しない）
python -u work/20260804_raw_count_lincomb/integrated_analysis/run_all_models_facets.py \
  --dry-run
```

`--prepare-data`はデータ作成が成功した場合だけ学習へ進みます。download原本または最終
H5ADを上書きしません。download原本だけが既に存在する場合はそれを再利用します。最終H5AD
作成済みの再実行・resumeでは`--prepare-data`を外してください。使用中のPython環境に`gdown`がない場合は、download前に
同じ環境へ`gdown==5.2.0`を自動導入します。

## 目的とデータ作成

従来のnormalized/log/scaled胚発生データではなく、raw countを直接diffusion modelへ
渡した場合に、softplus、Hill飽和、exp生成項とODE-only/Standard Hybridの差がどう
現れるかを比較します。

添付Notebookから次を維持します。

- Google Drive ID `1cProQIHN_auFCowchTcW9-wd1PEPSDjF`と`gdown`
- download原本名`Embryonic_raw_data.h5ad`
- obs第2列→`celltype`、var第3列→`gene_name`
- `min_genes=10`, `min_cells=3`, HVG 1024、gzip H5AD
- HVG flavorはNotebookと同じScanpy default

最終ファイル名だけは指定に従い、次へ統一します。

```text
work/20260804_raw_count_lincomb/data/Embryonic_raw_count.h5ad
```

処理順は、raw候補audit → filter → rawの独立copy → HVG用独立copy → HVG用copyだけ
`normalize_total(1e4)`/`log1p` → HVG index選択 → 保持したraw copyを同indexでsubset
→ raw/integer/finite再検査 → gzip保存 → read-back検査です。最終`.X`にはnormalize、
log1p、scale、z-scoreを一切行いません。原本`.X`が非負・finite・整数相当でなければ
停止します。layerを調査して明示的に使う場合だけ`--raw-source layer:NAME`を指定し、
`adata.raw`を推測では使いません。

`data/metadata/data_creation.json`へfilter前後/HVG後shape、値域、sparse性、全条件、
gene-order SHAを、`gene_order.csv`へ1024 geneの順序を保存します。データ作成段階では
PCA/UMAP/分布図を作りません。

## ODEとmodel family

全ODEは8 component、raw timestep embedding、`[x,time]→256→256→8`係数MLP、
temperature 1のsoftmaxです。

- `softplus`: `softplus(W_k x+b_k)-delta*x`。`ConfigurableLinCombField`と
  `build_lincomb_field_0707`をそのまま再利用します。
- `hill_after_softplus`: `z=softplus(W_kx+b_k)`,
  `V*z^2/(K^2+z^2)-delta*x`。20260803の`hill_after_linear`はこの数式と完全一致する
  ためclassを再利用し、今回のexperiment名だけ明確化しました。
- `exp`: `exp(W_kx+b_k)-delta*x`。20260803 classとconfig guard `[-20,20]`を再利用します。

off-mask penaltyは全8 componentの`W`だけ、全要素mean×5、total loss外側係数1です。
bias/K/V/delta、LinComb softmax、Hybrid、CellUnetには掛けません。

ODE-onlyはODE softmax加重和だけです。Standard Hybridは既存
`UnifiedODEMLHybrid`の通常係数`1-t/(T-1)`でODEと既存`Cell_Unet`を混合します。
LinComb内の8係数とbranch間Hybrid係数は別物です。TS/tau80は使いません。

|順|experiment|family|ODE|
|---:|---|---|---|
|1|`ode_only_lincomb_softmax__softplus`|ODE-only|softplus|
|2|`ode_only_lincomb_softmax__hill_after_softplus`|ODE-only|Hill|
|3|`ode_only_lincomb_softmax__exp`|ODE-only|exp|
|4|`standard_hybrid_lincomb_softmax__softplus`|Standard Hybrid|softplus|
|5|`standard_hybrid_lincomb_softmax__hill_after_softplus`|Standard Hybrid|Hill|
|6|`standard_hybrid_lincomb_softmax__exp`|Standard Hybrid|exp|

## 学習条件と再利用

20260803から、30,000 steps、LR 1e-4、batch 128、microbatch -1、AdamW weight decay
1e-4、diffusion 1000/linear/epsilon-MSE、uniform sampler、EMA .9999、seed 1234、
save 5000、sampling 3000/batch 50/DDPM/clip falseを継承します。入力dataだけが
`Embryonic_raw_count.h5ad`です。既存`load_data(..., train_vae=True,
preprocess=False, layer=None)`は`.X`をfloat32 tensor化するだけでnormalize/log/scale
しません。

20260803の`common.py`, `train.py`, `sample.py`を動的adapterで再利用し、root、6条件の
identity validation、新model factoryだけを新suiteへ差し替えます。raw/optimizer/EMA
三点一致resume、immutable requested config、segment保存、strict EMA restoreは同一です。
過去ファイルはコピーも変更もしません。

## SamplingとUMAP

samplingはEMA checkpointから既存diffusion設定でstrict restoreし、run内の新しい
`samples/`へ保存します。既存sampleは上書きしません。

UMAPだけを作り、loss/parameter/branch等の追加解析は行いません。全条件でseed 1234、
real 3000 cellsの同じindex、generated 3000、neighbors 15、PCA最大50、n_pcs 40です。
real/generatedを結合した独立AnnData上で計算し、学習H5ADを変更しません。可視化でも
normalize/log/clipを加えずraw-count model spaceをそのまま使うため、generated負値も
0へclipされません。負値数、real index、PCA/neighbors/seedをUMAP metadataへ保存し、
320 dpi PNGへexperiment、step、raw-count trainingを明記します。

ODE-only 3条件の統合facetは、`work/20260802/integrated_analysis`と同じ共有実装
`run_all_models_facets_0707.py`を呼び出します。各モデルでsampling完了runのうち
checkpoint stepが最大のもの（同stepならsample更新時刻が新しいもの）を選ぶため、expの
retry runも自動選択されます。real 50,000、generated最大3,000、PCA 50、neighbors 15、
min_dist 0.5、seed 0で、各panelを独立UMAPとして計算します。出力は
`integrated_analysis/outputs/<YYYYMMDD_HHMMSS>/all_models_facets.png`です。

## run構造、smoke、数値上の注意

```text
runs/<experiment>/<batch-id>/
  train/checkpoints/segment_NNN/<model>/
  samples/
  umap/<checkpoint>_<datetime>/
  logs/ commands/ manifest.json exp_config.json
```

smokeはsynthetic integer countでdata round-trip、HVG専用copy、最終raw/integer/finite、
1024 gene、6 model construction、raw-count forward/backward、softmax`B×8`・非負・和1、
W penalty、ODE-only/Hybrid構造、standard係数を検査します。さらに各条件を4 stepだけ意図的に
中断し、step 2のraw/optimizer/EMA三点からstep 4へresume、strict EMA sampling、raw空間
UMAP PNG生成までを通します。本データの30,000-step学習はsmokeでは起動しません。

raw countは既存normalized空間より値域が大きいため、特にexpのpreactivation clamp到達
やODE出力scale増大が起こり得ます。入力を黙ってlog/clipせず、forward/loss/gradientの
NaN/Infは停止・manifest記録します。既存guard外の飽和は既知のモデル挙動です。

既存から主に再利用するものは`ConfigurableLinCombField`, `build_lincomb_field_0707`,
`LinCombOnlyDenoiser`, `UnifiedODEMLHybrid`, `Cell_Unet`, 20260803
`HillAfterLinearField`/`ExpField`、TrainLoop、diffusion、data loader、strict samplerです。
