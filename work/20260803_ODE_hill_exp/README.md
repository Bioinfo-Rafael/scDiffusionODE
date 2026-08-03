# 2026-08-03 ODE Hill / RACIPE / exp comparison

## すぐ実行するためのコマンド

以下はすべて repository root (`scDiffusionODE/`) から実行します。`BATCH_ID`
は同じ一連のrunを再開・sampling・解析するときに固定してください。

```bash
# 1. 12条件を指定順で train -> sample -> analysis
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --batch-id 20260803_full_30000

# 長時間実行を切り離す場合（同じ処理をbackgroundで開始）
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --batch-id 20260803_full_30000 --background

# 2. model familyを1つ指定し、その3 ODEをcanonical順で実行
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --family standard_hybrid_lincomb --batch-id 20260803_standard

# 3. 1実験だけ実行
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --experiment ode_only_lincomb__exp --batch-id 20260803_one

# 4. 複数実験を指定（実行順は常にcanonical 12順）
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --experiment ode_only_lincomb__exp standard_hybrid_single__racipe \
  --batch-id 20260803_subset

# 5. 全12条件のsmoke testだけ
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --smoke --batch-id 20260803_smoke

# 6. 特定条件だけsmoke test
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --smoke --experiment standard_hybrid_single__exp --batch-id 20260803_smoke_one

# 7. 中断runをraw checkpoint + optimizer + EMAから安全にresume
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --resume auto --batch-id 20260803_full_30000

# 8. samplingだけ
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --sample-only --batch-id 20260803_full_30000

# 9. 解析・可視化だけ
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --analysis-only --batch-id 20260803_full_30000

# 10. run directoryと実行予定commandを表示するだけ
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --dry-run --batch-id 20260803_full_30000
```

## 背景と比較目的

既存のRNA velocity型LinCombは、各expertの生成項を
`softplus(W_k x + b_k)`、減衰項を正の係数による `-delta*x` として
表します。本suiteはその生成機構だけを、飽和型の
`hill_after_linear`、edgeごとの活性化・抑制を表現する`racipe`、非飽和型の
`exp`へ置き換え、式の違いがdiffusion denoising、branch balance、生成分布に
どう現れるかを比較します。

同時に、ODEだけで十分か、CellUnetとの通常Hybridが必要か、`T_s`近傍を
緩やかに切り替えるtau80 Hybridが有効か、8 expertのLinCombがsingle ODEより
有効かを比較します。比較軸以外は0802が継承していた学習条件へ揃えています。

既存コードは変更しません。時間埋め込み、CellUnet、Standard/TS Hybrid係数、
diffusion、TrainLoop、edge-mask loaderなどはimportし、新しいODE、factory adapter、
launcher、resume修正、restore/plot wrapperだけをこのdirectoryに実装します。

## ODEの定義

以下で `i` は出力/target遺伝子、`j` は入力/source遺伝子、`k` はLinComb
expertです。single版では`k`を除きます。`n=2`は固定configです。

### `hill_after_linear`

```text
z[k,i](x) = softplus(sum_j W[k,i,j] x[j] + b[k,i])
h[k,i](x) = z[k,i]^n / (K[k,i]^n + z[k,i]^n)
P[k,i](x) = V[k,i] h[k,i](x)
f[k,i](x) = P[k,i](x) - delta[i] x[i]
```

`K`と`V`はexpert・targetごと、`delta`は識別性と比較可能性のためexpert間で
共有するtargetごとの正値です。LinComb出力は既存と同じ8係数で
`sum_k a[k](x,t) f[k](x)`とします。

### `racipe`

```text
u[j]            = softplus(x[j] / s)
h[k,j](u[j])    = u[j]^n / (K[k,j]^n + u[j]^n)
lambda[k,i,j]   = exp(r[k,i,j])
H^S[k,i,j]      = 1 + (lambda[k,i,j] - 1) h[k,j]
P[k,i](x)       = G[k,i] product_j H^S[k,i,j]
f[k,i](x)       = P[k,i](x) - delta[i] x[i]
```

`r>0`は活性化、`r<0`は抑制、`r=0`は中立で、対応factorはforward値として
厳密に1です。`K`は無制限な`K[k,i,j]`ではなく`K[k,j]`、すなわち同じ
component/regulatorのthresholdを全targetで共有します。これによりK parameterを
`8*d*d`から`8*d`へ減らし、edge強度`r`との識別不能性とメモリ増加を抑えます。
`G`はexpert・targetごと、`delta`はexpert間共有です。

productは数学的に等価なlog-spaceの
`logaddexp(log(1-h), log(h)+r)`を合計して計算します。`r=0`の値補正は
gradientを保持したままlog-factorを厳密に0へします。

### `exp`

```text
f[k,i](x) = exp(sum_j W[k,i,j] x[j] + b[k,i]) - delta[i] x[i]
```

既存LinCombの`W`初期化、bias、共有delta、係数MLP、mask penalty interfaceを
保ち、生成項のsoftplusだけをexpへ変更します。古いsingle GeneODEにあった
追加の`1/sqrt(d)` preactivation scaleは依頼式にないため追加せず、今回の直接の
比較元である最新LinCombの`W[k,i,j]x[j]`規約を使います。

## parameterizationと初期化

正値parameterはすべて次で作ります。

```text
positive(raw) = softplus(raw) + positive_epsilon
```

`K/V/G`の物理初期値は1.0になるようinverse-softplusでraw値を設定します。
`delta`のraw初期値は既存と同じ0.1です。`W`はおおむね
`Normal(0, 1/sqrt(d))`、`r`は全edgeを中立factor 1から始めるため0、biasは0です。実際のshape、共有方法、guard値は各runの
`exp_config.json`と`model_info.json`へ保存されます。

## TSV edge priorとoff-mask penalty

penalty対象は次だけです。

| ODE | penalty対象 | 対象外 |
|---|---|---|
| hill_after_linear | `W` | K, V, delta, bias, gate, Hybrid, CellUnet |
| racipe | `r=log(lambda)` | lambda自体, K, G, delta, gate, Hybrid, CellUnet |
| exp | `W` | bias, delta, gate, Hybrid, CellUnet |

既存helperはTSVを`mask[source,target]`にしますが、最新LinCombの式は
`W[target,source] x[source]`です。過去コードにはここが転置不整合のまま残って
います。本suiteは既存helperをimportした後、新規adapter内だけでmaskを転置し、
TSVの`source -> target`と依頼式を一致させます。過去の挙動を黙って変更したの
ではなく、新suite固有の意図的な差であり、toy edge testで固定します。

強度と正規化は既存どおりです。

```text
base = mean(abs((1 - mask_target_source) * edge_parameter))
regularization = 5.0 * base
total_loss = diffusion_loss + 1.0 * regularization
```

meanはon/off edge数ではなく、singleなら`d*d`、LinCombなら`8*d*d`全要素です。
hard maskではなくsoft penaltyなので、off-mask parameterもforwardへ参加します。

## 12実験と実行順

| # | experiment | model family | ODE |
|---:|---|---|---|
| 1 | `ode_only_lincomb__hill_after_linear` | ODE-only LinComb | hill_after_linear |
| 2 | `ode_only_lincomb__racipe` | ODE-only LinComb | racipe |
| 3 | `ode_only_lincomb__exp` | ODE-only LinComb | exp |
| 4 | `standard_hybrid_lincomb__hill_after_linear` | Standard Hybrid + LinComb | hill_after_linear |
| 5 | `standard_hybrid_lincomb__racipe` | Standard Hybrid + LinComb | racipe |
| 6 | `standard_hybrid_lincomb__exp` | Standard Hybrid + LinComb | exp |
| 7 | `ts_soft_tau80_hybrid_lincomb__hill_after_linear` | TS_soft_tau80 Hybrid + LinComb | hill_after_linear |
| 8 | `ts_soft_tau80_hybrid_lincomb__racipe` | TS_soft_tau80 Hybrid + LinComb | racipe |
| 9 | `ts_soft_tau80_hybrid_lincomb__exp` | TS_soft_tau80 Hybrid + LinComb | exp |
| 10 | `standard_hybrid_single__hill_after_linear` | Standard Hybrid + single ODE | hill_after_linear |
| 11 | `standard_hybrid_single__racipe` | Standard Hybrid + single ODE | racipe |
| 12 | `standard_hybrid_single__exp` | Standard Hybrid + single ODE | exp |

選択実行でも、この表の相対順序は変えません。

## model family

LinCombは既存と同じ8 expert、sinusoidal raw-timestep embedding、
`[x,time_emb] -> 256 -> 256 -> 8` MLP、温度1のsoftmax係数を使います。
ODE-onlyはCellUnetもHybrid係数も持ちません。singleはexpert軸、係数MLP、8個の
混合係数を一切持ちません。

本実験のCellUnet幅は既存defaultの`[2000,1000,500,500]`です。統合smokeだけは
checkpoint容量と所要時間を抑えるため同じCellUnet classを`[32,24,16,16]`で動かし、
別のunit testでdefault CellUnetそのもののforward shape/finite性も検証します。

Standard Hybridはimportした既存classで次を使います。

```text
r(t) = 1 - t/(T-1)
out  = r(t) ODE + (1-r(t)) CellUnet
```

`reverse_coef=false`なので`t=0`でODE重み1、`t=999`でCellUnet重み1です。

TS_soft_tau80も同じ既存classを使い、Standard schedulerを次でoverrideします。

```text
w_ode(t) = sigmoid((T_s-t)/80)
out      = w_ode(t) ODE + (1-w_ode(t)) CellUnet
```

raw timestepを正規化せず使用します。同一data/scheduleの既存実績は
`lambda_max=47.37206891051204`, `T_s=616`で、`w_ode(616)=0.5`です。
新suiteは過去suiteのcacheへ書かず、自分の`cache/`だけを使います。

## configと20260802から引き継いだ値

`configs/base.json`へ共通値、12個のJSONへ差分だけを置きます。代表値は
diffusion 1000/linear/epsilon-MSE、uniform sampler、LR `1e-4`、AdamW、weight
decay `1e-4`、batch 128、microbatch -1、EMA 0.9999、seed 1234、checkpoint
5000、sampling 3000 cells/batch 50/DDPM/clip falseです。唯一の予算変更は依頼に
従った100,000から30,000 stepへの変更です。

値ごとの採用元・旧値・今回値・一致判定は`configs/source_comparison.json`、
機械検査は次です。

```bash
python work/20260803_ODE_hill_exp/scripts/verify_config_sources.py
```

各runにはmerge後かつpath/`T_s`解決後の全設定を`exp_config.json`として保存します。

## output directory

```text
work/20260803_ODE_hill_exp/
  cache/
  configs/
  models/
  scripts/
  tests/
  runs/<experiment>/<batch-id>/
    exp_config.json
    model_info.json
    manifest.json
    commands/
    logs/
    train/checkpoints/segment_NNN/<model-name>/
      modelNNNNNN.pt
      ema_0.9999_NNNNNN.pt
      optNNNNNN.pt
      loss_details.csv
    samples/
      samples.npz
      sample_manifest.json
    analysis/
    viz/
  smoke_results/<batch-id>/
  validation/
```

既存runやcheckpointは上書きしません。同じexperiment/batch-idの完了stageは
manifestで識別し、明示的なresume/sample-only/analysis-onlyだけを実行します。

## checkpointとresume

既存0707/0802 pipelineはmanifestのEMAを`--resume auto`へ渡しますが、既存
TrainLoopのstep parserは`modelNNNNNN.pt`しか解釈せず、optimizerをstep 0として
扱う問題があります。本suiteは用途を分離して記録します。

- `raw_checkpoint_path`: resume用`modelNNNNNN.pt`
- `ema_checkpoint_path` / `checkpoint_path`: sampling・analysis用EMA
- `optimizer_checkpoint_path`: 同stepのoptimizer
- `checkpoint_step`: 解析済み整数step

resumeはraw/optimizer/EMA三点のstep一致を確認し、新しい`segment_NNN`へ継続
checkpointを書きます。総stepはresume前を含む30,000です。失敗時はmanifestへ
exception、終了時刻、最後に確認できた三点を保存します。

## sampling

samplingはrunの解決済みconfigとEMA checkpointから同型modelをstrict restoreし、
既存と同じancestral DDPMで3,000 cellsを生成します。NPZ keyは`cell_gen`です。
既存sampleがある場合はtimestamp付き別名へ保存し、黙って上書きしません。

## metricsと可視化

解析はplot前にmodel output、branch output、sample、parameterのfiniteを検査し、
NaN/Infを`nan_to_num`で隠しません。共通でloss、regularization、real/generated
比較、UMAP、noiseに対する相関、内積、正規化MSE、output normを保存します。

| family | 作るもの | 作らないもの |
|---|---|---|
| ODE-only LinComb | 8係数、component寄与、effective係数、ODE parameter | CellUnet norm、Hybrid係数 |
| Standard Hybrid LinComb | 上記 + ODE/Cell norm・比率、Standard係数曲線 | なし |
| TS_tau80 Hybrid LinComb | 上記 + 正しいTS係数曲線・T_s | なし |
| Standard Hybrid single | ODE parameter、ODE/Cell norm・比率、Standard係数 | LinComb係数、expert/component図 |

ODE別には次を作ります。

- hill: W on/off分布、top edge、K/V/delta分布、component要約
- racipe: rとlambdaを別々に表示、activation/suppression/neutral割合、rのon/off
  比較、top edge、K/G/delta。図中にも「penalty target = r, not lambda」と記載
- exp: W on/off分布、top edge、bias、delta、component要約

大規模行列の全要素heatmapは作らず、分布、on/off比較、上位edge、component別
要約を使います。全plot title/metadataへODE、family、checkpoint、生成日時を記録
します。

## 数値安定化

`racipe`はlog-space productに加え、`log(P)`をconfigの
`[-30,20]`へclampします。これによりfloat32 overflow/underflowを防ぎますが、範囲
外ではproduction値とgradientが飽和します。target chunk size 16は式を変えずに
一時memoryを抑える設定です。

`exp`はpreactivationをconfigの`[-20,20]`へclampしてからexpを取ります。範囲内は
依頼式と同一、範囲外は値とgradientが飽和します。guardはrun configに残るため、
黙った挙動変更ではありません。初期forward/backwardのfiniteは全12条件でsmoke
検証します。

## smoke test

smokeは小さなsynthetic AnnDataとtoy directed edgeをsuite内に作り、各12構成で
次を実行します。

- import、construction、pipelineと同じshape
- finite forward/backward/gradient
- Wまたはrだけへのoff-mask penaltyと全要素mean
- racipe `r=0 -> factor=1`、lambdaを0へ縮小しないこと
- source→target edge向き
- LinCombとsingleのparameter/structure差
- Standard/TS_tau80係数の既存classとの完全一致
- train、raw checkpoint/optimizer/EMA save、resume、strict reload
- checkpoint sampling
- loss、model I/O、branch、gate、component、ODE parameter、UMAPの最小plot
- 保護対象の既存差分が増えていないこと

結果は`smoke_results/<batch-id>/`にJSONと最小artifactを保存します。

## 既存コードからimportする主なclass・関数

- `guided_diffusion.cell_model.Cell_Unet`
- `ODE.ode_20260609_hybrid5x3.UnifiedODEMLHybrid`
- `ODE.ode_20260707_lincomb.LinCombOnlyDenoiser`
- `ODE.ode_20260609_mathmlp.build_edge_mask`, `_TimeEmb`, `_mlp`, `_prep_t`
- `guided_diffusion.train_util.TrainLoop`
- `guided_diffusion.cell_datasets_loader.load_data`
- `guided_diffusion.script_util.create_model_and_diffusion`
- `guided_diffusion.resample.create_named_schedule_sampler`
- `ODE.ode_20260609_mathmlp.load_hybrid_state_dict`
- `work/20260707_lincomb/utils/regime_time.py`の`T_s`推定関数

正確なimport一覧は実装内と各runの`model_info.json`にも保存します。

## 既知の制限と途中失敗時の確認

- RACIPEはdenseな`r[k,i,j]`と全regulator productを持つため、3 ODE中で最も計算量が
  大きく、特にCPUでは遅くなります。batch sizeは比較条件なので無断変更しません。
- mask transposeは生物学的方向と依頼式を整合させる新suite固有差です。既存
  LinCombのbyte-for-byte再現ではありません。
- exp/racipe guard外では上記の飽和が起きます。guard到達率を解析へ保存します。
- Apple Siliconでは既存`dist_util`のdefaultはCPUです。launcherは実際に選択した
  deviceをconfig/manifestへ記録しますが、学習条件自体は変更しません。

途中失敗時はまず`runs/<experiment>/<batch-id>/manifest.json`、`logs/`、
`raw_checkpoint_path`、`optimizer_checkpoint_path`を確認し、上記`--resume auto`を
使います。既存保護状態は次で再検査できます。

```bash
python work/20260803_ODE_hill_exp/scripts/verify_protected.py --write-report
```

baselineは`protected_baseline.json`、検査結果は
`validation/protected_diff_report.json`です。作業開始前から変更されていた
`work/20260801/load_embryonic_h5ad.ipynb`は、開始時SHA-256が変わっていないことを
別途確認し、本suiteの変更として扱いません。
