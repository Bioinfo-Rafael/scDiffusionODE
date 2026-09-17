# 20260916: x0 prediction + additive Hybrid + independent-target PCA OT

**現在の既定値（2026-09-17更新）:** Stage2は10k、OT targetは8192、
`pca_ot.gradient_mode="envelope"`です。既存campaignへは
`--target-size 8192 --sinkhorn-gradient-mode envelope`を明示して切り替えます。
指定しないresumeは保存済みのtarget数・勾配方式を維持します。
実行順は全4reconstruction、その後全4OTで、各条件のtrain→sample→analysis→UMAPを
終えてから次条件へ進みます。元の32768/full-autograd結果は保持します。

## Background

従来の `r*ODE + (1-r)*CellUNet` では、ODEの比重を増やすほどCellUNetの
denoising出力そのものが弱まり、生成分布が中心にcollapseする可能性があります。
今回は **`CellUNet + r*ODE`** とし、CellUNetを全stepで100%残します。
CellUNetに学習済みmanifold / attractorへ戻す働き、ODEにそのtrajectoryを
別方向へずらすresidual driftを担わせる、という仮説を検証します。
これは役割についての仮説であり、生物学的に正しい方向を保証するものではありません。

20260915のx0-prediction CellUNetの単体UMAP coverageが良かった、という
ユーザー提供の観察に基づいてSTART_Xに固定します。このMacには指定学習結果が
存在しないため、その観察をここで再測定したという意味ではありません。

実装は独立した `work/20260916_x0predict_hybrid_additive/` に限定しています。
既存実験・checkpoint・`guided_diffusion`・ODE定義は変更しません。
Stage1を新規学習するコマンドはありません。

## 監査結果・再利用する実装

| 対象 | 確認した実装 / 再利用方法 |
|---|---|
| 20260915 START_X | `work/20260915_x0predict/common.py`, `models/__init__.py`, `training/objectives.py` |
| 従来Hybrid | `ODE/ode_20260421_regODEMLratio.py::ODE_ML_Hybrid` は係数付きblend。`ODE/ode_20260609_hybrid5x3.py::UnifiedODEMLHybrid` と20260913の`Hybrid500Mixin`が最近の実装の基底 |
| Single ODE | `work/20260830/models/ode_fields_20260830.py` の4クラスを20260915 factory経由で再利用 |
| Reconstruction | 20260915 `training_loss` → 既存 `GaussianDiffusion.training_losses` |
| OT | `work/20260913_2step/losses/sinkhorn.py::entropic_ot` / `sinkhorn_divergence` |
| Sampling | 20260915 `sampling/trajectory.py`、native `diffusion.p_sample` |
| 評価 | 20260915 `analysis/diagnostics.py`, `distributions.py`, `evaluation_ot.py`, `sliced_wasserstein.py`, `umaps.py`, `plotting.py`, `runner.py` |
| Recovery | 20260915のbundle形式・hash検証を継承。新suiteで上限・OT方式ごとの復元を実装。campaignの排他lockは再利用 |

`reuse.py` は元ファイルを新スイートのprivate module namespaceへ読み込みます。
相対importが新しいpath confinement・checkpoint restore・additive modelへ解決するため、
sampling・UMAP・plottingの大きなコピーが不要です。旧moduleのグローバル変数を
monkeypatchせず、旧ファイルも書き換えません。テストで両namespaceの独立を確認します。

既存評価のOT/Sliced Wasserstein/UMAPの数値定義は維持します。
今回追加するcoverage指標は後述の別名で記録し、過去のcoverage数値と同一とは扱いません。

## Experimental matrix

| Condition | ODEクラス | Objective |
|---|---|---|
| `centered_hill_reconst` | `CenteredSignedHill20260830` | START_X reconstruction + soft |
| `centered_hill_ot` | 同上 | independent-target PCA OT + soft |
| `shifted_hill_reconst` | `ShiftedHillRho20260830` | reconstruction + soft |
| `shifted_hill_ot` | 同上 | PCA OT + soft |
| `hill_after_linear_reconst` | `HillAfterLinear20260830` | reconstruction + soft |
| `hill_after_linear_ot` | 同上 | PCA OT + soft |
| `softplus_reconst` | `SimpleSoftplus20260830` | reconstruction + soft |
| `softplus_ot` | 同上 | PCA OT + soft |

4クラスともK=1の明示的パラメータモデルです。expert軸、`coeff_net`、NN gate、
時刻依存expert mixtureはありません。既存のshape/class assertionを継承します。
centered/shiftedはregulator遺伝子jへの直接和で、数値guardを含め元コードをそのまま使います。
softplusはsoftmaxではありません。

## Stage1 checkpoint・共通条件

全条件で次の同一final EMAを使います。

```text
/home/suzuki/Projects/scDiffusion-github/work/20260915_x0predict/runs/x0predict_20260915T131558Z_6802c20b/stage1_cellunet/20260915T131601Z_1085706d/checkpoints/ema_0.9999_030000.pt
```

起動前にstrict state load、START_X設定、30,000 stepのfinal EMA、gene ordering、
dataset/edge/checkpoint SHA256を検証します。sidecarがある場合はそのSHA256も照合します。
新campaign内に同じstateのself-contained checkpointを保存し、元checkpointは読み取り専用です。
元ファイルと新ファイルはmetadataが違うためfile hashは異なりますが、CellUNet state hashは同じです。

Stage2のCellUNetは `requires_grad=False`、常時evalです。親モデルの`train()`でも
evalを保ち、optimizerに入らないこと、raw/EMAのstate hashが不変であることを検証します。

| 設定 | 既定値 |
|---|---|
| Stage2 updates / condition | 10,000（2026-09-17変更。Stage1は既存30,000step EMA） |
| Source batch | 128、shuffle、drop_last |
| Optimizer | AdamW、lr `1e-4`、weight decay `1e-4` |
| EMA / seed | `0.9999` / 1234 |
| Diffusion | linear、1000 steps、`predict_xstart=True`、START_X / MSE |
| Rescaling / respacing | 両方なし。original timestepがそのままモデルに入ることをassert |
| 前処理 | checkpointと同じデータの`X`、float32、gene order維持。追加normalize/log/scaleなし |
| Source index RNG | 1234。モデル初期化の乱数消費から分離 |
| Target index RNG | 1235。sourceと別のRNG |
| Sampling | seed 1234、3000 cells、batch 50、native ancestral DDPM、clipなし |

`lr_anneal_steps`は元の30,000を維持します。10kへの変更は終了stepの短縮であり、
途中までのlearning-rate scheduleを変えません。完了済み30k runの10k checkpointと
新たに10kで止めたrunを同じscheduleのprefixとして比較できます。

データとedgeの既定パスはStage1 metadataから継承します。元実験の既定値は
`/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad` と
`/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv` です。
移設した同一ファイルだけ `--data` / `--edge-tsv` で指定できます。hashが違えば拒否します。

source loaderはindex追跡のため新設しています。元loaderと同じX・batch数・shuffle/
drop_lastを使い、行indexとtensorの対応を保証します。shuffle乱数を独立させたため、
旧loaderとのbatch順のbitwise一致は主張しません。新8条件間のsource順は一致します。

## Additive Hybrid / r(t)

`models.py::AdditiveHybrid500._outputs` の内部で合成します。

```text
r(t) = max(0, 1-t/500),  raw t ∈ {0,...,999}
Hybrid(x_t,t) = CellUNet(x_t,t) + r(t)*ODE(x_t)

t=999..501: r=0
t=500:     r=0
t=250:     r=0.5
t=0:       r=1
```

CellUNetに `(1-r)` は掛けません。`Δτ` を追加せず、samplerの後処理で加算することも
ありません。raw ODE-form出力をx0 predictionへのresidualとして直接使います。
新classは比較・回帰テスト用に `model.mode="blend"` も保持し、従来class自体は未変更です。

## Reconstruction experiment

```text
t ~ Uniform{0,...,999}
x_t = diffusion.q_sample(x0,t,noise)
L_reconst = diffusion.training_losses(Hybrid,x0,t)["loss"].mean()
          = mean((Hybrid(x_t,t)-x0)^2)
L_total = L_reconst + L_soft
```

元のx0へのpoint-wise reconstruction baselineです。通常のuniform timestepを維持し、
t>=500でODEにdata-gradientがなくなることも維持します。soft penaltyは有効です。
既存のfloat64 Gaussian noise生成とfloat32モデルを継承します。

## OT experiment / independent targets

同じx0をOT targetに含めると、その近傍に輸送されやすくreconstructionに近づく可能性が
あります。今回はsourceとは別の乱数で実データtargetを選び、**sourceの元の行indexを
targetから完全に除外**します。対応づけを崩し、別のreal cell群との分布一致を促す仮説です。
重複した発現を持つ別cellなどもあり、非ゼロの移動や生物学的に正しい移動方向が必ず
生まれると保証するものではありません。

- Source batchは128のまま、low-noise `t ~ Uniform{0,...,49}`。
- `prediction=Hybrid(x_t,t)` はx0そのもの。epsilon→x0変換はありません。
- Targetは既定 **8192 distinct cells**。source batchと同じtraining populationから選びます。
- `DisjointTargets.sample` は毎step、unique件数と `source∩target=∅` をassertします。
- 当初の32768はtarget distributionのsampling sparsityを減らすためでした。
  ユーザー指定により8192へ短縮しました。source128は維持します。
- `target_refresh_interval=10`。zero-based step 0/10/20/...で全targetをrefreshします。
- 中間stepでsourceがcached targetに入った場合は、そのslotだけ現在のsourceと残りtargetを
  除外した集合からwithout replacementで補充します。それ以外のtargetは維持します。
- training populationが `target_size + batch_size` 未満なら明示的に失敗します。
  targetを黙って減らしたり、同じcellを重複させて件数を埋めたりしません。

### 固定PCA

training data全体に、deterministicな一度の`IncrementalPCA` fitを行います。
既定50次元、whiteningなし、fit batch4096。最終chunkが小さい場合は前chunkと結合します。
これは大きなXの追加dense copyを避けるための方式です。次元数は黙って切り下げません。

一度だけ `real_pca = (real_X - mean) @ components.T` をfloat32 memmapへprecomputeし、
全OT条件で同じcacheを共有します。training中のtarget取得は行indexによる読出しのみです。
`FixedPCA` はmean/componentsをtorch bufferとして持ち、prediction側の式は
`(prediction.float()-mean) @ components.T`。**detachは一切せずODEへbackpropagate**します。
PCA cacheはdataset hash、gene-order hash、transform/hashで検証します。
可視化用PCA/UMAPは従来定義のままであり、このtraining OT用PCAとは別です。

### OTの数学的定義とtarget自己行列を作らない理由

従来のOTはexact unregularized OTではなく、log-domain **debiased Sinkhorn divergence**です。

```text
OTε(P,Q) = min_π <π,C> + ε KL(π || a⊗b)
Sε(P,Q)  = OTε(P,Q) - 0.5 OTε(P,P) - 0.5 OTε(Q,Q)
```

従来実装をそのまま32768 targetへ適用すると最後の項が32768²となります。
しかし今回Qは固定real PCA tensorで、モデルのどのparameterにも依存しません。
そこでtrainingでは**target自己項だけを省略**します。

```text
L_PCA_OT = OTε(P,Q) - 0.5 OTε(P,P)
P = FixedPCA(Hybrid(x_t,t))
Q = cached_real_pca[disjoint_target_indices]
∇model L_PCA_OT = ∇model Sε(P,Q)
L_total = L_PCA_OT + L_soft
```

target-only定数の省略はmodel-gradientを変えません。prediction側自己項は残します。
`full_autograd`では既存solverによる完全なSεとの勾配の厳密一致を検証しています。
`envelope`では収束した輸送計画から勾配を計算します。有限反復を微分する旧方式と
完全一致する保証はなく、その違いを以下で説明します。ログのscalarは完全なdivergenceではありません。
`sinkhorn_divergence_up_to_target_constant` と明記し、targetが変わるstep間や旧実験との
絶対値比較には使用しません。target-only項の省略は学習勾配についての等価性であり、
divergenceの数値を計算したという意味ではありません。

現在のcost matrixは **128×8192** と **128×128** のみです。float64のcross cost
本体は8 MiBです（他の作業領域は別途必要）。巨大なtarget自己行列は計算しません。
KeOps等の新しい依存はありません。

### Envelope勾配

`training/objectives.py::converged_entropic_ot`は2方式を持ちます。

- `full_autograd`: 10反復ごとのcheckpointingで全反復を微分する従来方式。
- `envelope`: 同じcost・epsilon・収束判定でdual変数を`torch.no_grad()`内で求め、
  `π=exp(-C/epsilon+u+v)`を固定して`sum(π*dC)`だけをbackpropagateします。

prediction→PCA→costのグラフは維持します。predictionをdetachすることはありません。
lossのforward値には従来どおりKL正則化も含め、単なる輸送距離だけを表示しません。
自己OTはpredictionが両方の引数に入るため、両側からの勾配を含みます。
有限の許容誤差で止めたplanは近似解なので、envelope勾配もその精度に依存します。
`epsilon=0.1`、`tolerance=1e-5`、float64、固定PCA50、target除外とrefresh10は維持します。
厳しい許容誤差でのfinite-difference testと全4ODEへのbackwardを検証しています。

参照: [GeomLossの収束点での明示勾配の説明](https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_kernel_truncation.html)。

PCA costは `mean((p-q)^2)`、すなわち50次元ならsquared distance / 50です。
epsilon 0.1、uniform marginals、float64、absolute marginal tolerance `1e-5`、
初回上限200 iterations、未収束時は200ずつ最大16000まで増やします。
最終上限でも非収束を有限lossとして採用せず、その条件を失敗として
記録し、独立条件を続行します。full-size GPUの速度/peak memoryは未測定です。

評価時は従来のgene-space・bounded subsample（最大128）による**完全な**Sinkhorn
divergenceを使います。既存のepsilon-scaling retryと欠測記録も維持します。

### Soft regularization

両目的で `L_soft = 5 * mean(abs((1-mask)*parameter))` を維持します。
parameterはsimple/Hill-after-linearでW、centeredでalpha、shiftedでrhoです。
外側`ode_reg_lambda=1`。maskをforwardへ強制適用せず、ratio/velocity/kinetic/manifold/
trajectory OTの追加lossはありません。

## Sampling / evaluation / collapse diagnostics

同一seed・checkpointの**CellUNet-only baselineをcampaignごとに一度**sampling/解析し、
全8条件で共有します。各Hybridの結果にはこの共通baselineを比較対象として明記します。
native START_X DDPM、従来のsnapshot/UMAP/分布評価を再利用します。

`BranchRecorder` はsampling中、全raw timestep **999..0** について、per-cell gene-wise
Pearson（既存`_sample_corr`）、CellUNet/ODE/r*ODEのL2 norm、r、CellUNet weightを保存します。
snapshot保存のための重複forwardは二重集計しません。t=501/500/499も含まれます。
trainingでもstep 0とlog intervalごとに同じ指標を収集し、記録対象updates全体を
raw timestepごとにまとめます（単一checkpointの診断とは区別してください）。

gene-wise varianceと平均variance、prediction centroidからのRMS距離を全sampling timestepで
保存します。state snapshotではさらに平均centroid距離と平均kNN距離を計算します。
raw predictionの診断とupdate後のstateの診断はファイルを分けています。

UMAP、true-x0 correlation/MSE/cosine、branch比較、diversity、gene-space Sinkhorn、SW2は
従来定義です。旧pipelineのErythropoietic reference selectionを維持します。
既存対象モジュールに再利用できるcoverage関数はなかったため、別名の追加指標
`added_real_knn_radius_coverage` を導入します：reference各cellのreal 5th-neighbor距離を半径とし、
その半径内にgenerated cellが存在するreferenceの割合です。既存評価値をこの指標へ置換しません。

従来のpost-hoc ODE dynamics解析（diffusion終了後Euler100 steps、dt0.001）も既存samplerに
付随して残します。これはtraining lossにもadditive Hybridにも使われず、main比較表は
**reverse_step=1000のdiffusion endpoint**を使います。

## Outputs

```text
runs/additive_<UTC>_<RUNID>/
  campaign.json, source_sha256.json, configs/*.json
  stage1/ema_0.9999_030000.pt      # 同じStage1 state、新campaign metadata
  canonical_stage1.json           # 元checkpointとコピーのhash/state provenance
  pca.json, pca/<attempt>/         # transform.npz, real_pca.npy, completed.json
  <condition>/<attempt>/
    metadata.json, effective_config.json
    losses.csv                    # UTC timestamp付き。OT primaryはtarget-only定数を除いた値
    timing.jsonl                  # 各起動の最初の3更新をdata/forward/backward/updateに分けて計測
    index_audit.jsonl             # zero-based step、source IDs、target SHA、refresh/repair数、overlap=0
    training_branch_metrics.csv/png, training_gene_variance.npz
    checkpoints/                  # raw/EMA/optimizer/complete bundle
    completed.json or failed.json
  steps/<step>/<attempt>/output.log
  invocation_<UTC>_<RUNID>.json

results/additive_<UTC>_<RUNID>/
  cellunet_only/<attempt>/         # 共通baseline
  <condition>/<attempt>/
    sampling_metadata.json, snapshot_metadata.csv
    sample_state.npy, pred_xstart.npy, model_x0.npy, cellunet_raw.npy
    ode_raw.npy                    # Hybridのみ
    sampling_branch_metrics.csv/png, sampling_gene_variance.npz
    analyze/<attempt>/
      true_x0_metrics.csv, cellunet_vs_ode_metrics.csv, norm_metrics.csv
      trajectory_diversity.csv, sinkhorn_to_real.csv, sliced_wasserstein_snapshots.csv
      collapse_coverage.csv, snapshot_gene_variance.npz
      figures/<attempt>/*.png
    embed/<attempt>/               # 独立UMAP、Hybridはterminal joint UMAPも
  comparisons/<attempt>/terminal_comparison.csv/png
```

過去の結果を上書きしないexclusive creation、output confinement、campaign lockを使用します。
完了stepは再利用し、失敗した条件があっても独立条件は続行します。数値解析とUMAPも独立です。
PCAが中断した場合は新attemptでfitし直し、完成したcacheだけを共有します。

resumeはraw/EMA/optimizer/update countを復元し、source/target/diffusion RNG streamは
seedから再開します。旧suiteと同じく、途中再開と無中断runのbitwise一致は保証しません。
再開直後のtargetは新規生成し、source除外を再検証します。全provenanceをmetadataに残します。

## Run commands（リモート）

### OTの反復上限と再開

PCA OTは200反復を初回上限とし、未収束の場合、同じprediction・targetで上限を
400 → 600 → ... → 16000へ200ずつ増やし、dual変数を保持して続行します。
勾配の反復履歴は`full_autograd`のときだけ保持します。
既存solverのcost関数を再利用し、固定epsilonの更新式・目的関数を同じまま
新suiteの`converged_entropic_ot`に継続処理を実装しています。
200回までの計算を400回枠でやり直すことはなく、追加で200回だけ進めます。
旧方式で上限16000まで全再試行すると合計648000反復でしたが、継続方式は合計16000反復です。
epsilon=0.1、marginal tolerance=1e-5、cost、目的関数は維持します。
10回ごとに収束を判定し、収束すれば早期終了します。16000でも未収束なら停止します。
非有限値などの例外でも停止します。Envelopeでもforwardの収束計算は必要です。
累積反復数・上限延長履歴・`restarts=0`をlosses.csvのsinkhorn欄へ記録します。
stdoutは既定で50stepごとの進捗・秒/step・残り時間・cross/selfの反復数を出します。
毎200反復の通知は`pca_ot.verbose=true`の場合のみです。反復履歴は常にCSVへ保存します。
各起動の最初の3更新ではCUDAを区間境界で同期し、`[timing]`と`timing.jsonl`に
data・forward（ODE/CellUNet/PCA/OTを含む）・backward・updateを記録します。
通常更新には追加の区間同期を入れません。forwardだけでOTとODEの寄与を区別はできません。
旧campaignのimmutable configに2000が保存されていても、実行時の初回上限は200です。
再試行上限の既定値16000は修正版コードで適用され、run metadataのgit commitと
source SHA、および各lossのsolver情報で追跡できます。

修正版をpull後、`--resume-campaign additive_20260916T034022Z_fcc39362`
で同じcampaignを再開できます。Stage2の実行上限は既定10,000、
`--training-steps 10000`でも明示できます。元の設定が30kでも書き換えず、
10k以下で最新の完全なraw/EMA/optimizer checkpoint bundleから再開します。
30kまで完了済みのreconstructionは10kのbundleを使い、更新せずに
実行上限10kのmetadataを付けた新checkpointとして保存します。
10kのbundleがなければ、それ以前の最新bundleから10kまで学習します。
変更された上限のstep markerは`<condition>_s10000_train/sample/analyze/...`となり、
30kの解析を10kの結果として再利用しません。共通CellUNet baselineは再利用します。
10kの比較表・UMAPなどは新attemptに保存し、30k結果を保持します。
bundleがなければその条件をstep 0から開始します。元の失敗ログは保持します。
再開時のsource照合は`audit/resume_compatibility.json`で既知の修正前後SHA256を
照合し、確認済みのSinkhorn・10k上限・OT方式切替・計測・launcher修正だけを許可します。
それ以外のsource変更は引き続き拒否します。campaign作成時のsource snapshotと
immutable configは書き換えず、新しいrun/checkpointの`source_migration`に
変更前後のSHA256と適用policyを保存します。
実際のリモートPCAデータで16000以内に収束するかは未検証です。

高速化調査、CPU比較結果、未採用のアルゴリズム候補は
[`audit/PERFORMANCE_REVIEW.md`](audit/PERFORMANCE_REVIEW.md)を参照してください。
今回のEnvelope採用・8192への切替・測定結果は
[`audit/ENVELOPE_8192.md`](audit/ENVELOPE_8192.md)に記録しています。

既存campaignを8192/envelopeへ切り替える場合、変更前のraw/EMA/AdamW stateを
保持し、`ot_transitions`に切替前後の設定と最初の新方式update番号を保存します。
同方式のcheckpointがすでにあればその続きを優先します。旧方式が10k完了済みなら、
10k未満の最新bundleから続きを実行します。旧10kをenvelopeで学習したと付け替えません。
切替前に旧方式で学習した履歴を持つため、結果は最初から8192/envelopeで学習した実験とは異なります。

OTのmarkerは`<condition>_s10000_n8192_envelope_train/sample/analyze/...`になります
（元から10kのcampaignでは`s10000`部分がありません）。比較表は新方式の解析を参照し、
target数・gradient mode・切替履歴も併記します。reconstructionの既存10k解析は再利用します。

旧job停止→pull→8192/envelopeで再開→ログ監視を一括実行する例：

```bash
(
  set -eo pipefail
  cd /home/suzuki/Projects/scDiffusion-github
  git fetch origin feat/20260916-x0predict-hybrid-additive
  git show origin/feat/20260916-x0predict-hybrid-additive:work/20260916_x0predict_hybrid_additive/scripts/stop_campaign.py \
    | python3 - --campaign additive_20260916T034022Z_fcc39362
  git switch feat/20260916-x0predict-hybrid-additive
  git pull --ff-only origin feat/20260916-x0predict-hybrid-additive
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scdiffusion
  LOG_DIR=$(mktemp -d work/20260916_x0predict_hybrid_additive/runs/launch_XXXXXXXX)
  touch "$LOG_DIR/output.log"
  nohup env PYTHONUNBUFFERED=1 PYTHON="$(command -v python)" \
    bash work/20260916_x0predict_hybrid_additive/scripts/run_all.sh \
    --resume-campaign additive_20260916T034022Z_fcc39362 \
    --all-eight --training-steps 10000 --target-size 8192 --sinkhorn-gradient-mode envelope \
    --gpu 0 --device cuda --analysis-device cpu \
    > "$LOG_DIR/output.log" 2>&1 < /dev/null &
  echo "$!" > "$LOG_DIR/pid"
  echo "PID: $(cat "$LOG_DIR/pid")"
  echo "LOG: $LOG_DIR/output.log"
  tail -n 100 -F "$LOG_DIR/output.log"
)
```

停止scriptはLinux `/proc`で同じユーザー・campaignに属するlauncher/workerを特定し、
launcherを一時停止して次の条件への遷移を防いでから、その子孫も含めてSIGTERMを送ります。
PIDの開始時刻も照合し、PID再利用による別processの停止を避けます。
停止確認が取れない場合は非ゼロ終了し、上記コマンドは新jobを起動しません。
停止だけならpull済み環境で以下を使えます（ML用conda環境は不要）。

```bash
python3 work/20260916_x0predict_hybrid_additive/scripts/stop_campaign.py \
  --campaign additive_20260916T034022Z_fcc39362
```

### 1. Pull・環境activate

```bash
cd /home/suzuki/Projects/scDiffusion-github
git fetch origin
git switch feat/20260916-x0predict-hybrid-additive
git pull --ff-only origin feat/20260916-x0predict-hybrid-additive
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scdiffusion
```

### 2. 実checkpointの事前検証（学習しない）

```bash
bash work/20260916_x0predict_hybrid_additive/scripts/run_all.sh --preflight-only
```

指定Stage1 checkpointをstrict loadし、入力ファイルSHA256を検証します。
`--dry-run` はplan表示のみであり、実checkpointのload検証は行いません。

### 3. 1条件 / 全8条件

```bash
bash work/20260916_x0predict_hybrid_additive/scripts/run_all.sh \
  --condition centered_hill_ot --gpu 0 --device cuda

bash work/20260916_x0predict_hybrid_additive/scripts/run_all.sh \
  --all-eight --gpu 0 --device cuda --analysis-device cpu
```

それぞれ新campaignになります。`--gpu 1`なら物理GPU1を子processにも引き継ぎます。
CUDA visibilityはtorchをimportする前に設定します。CellUNet-only解析はどちらでも含みます。
PCA次元/target件数/refresh間隔はconfig化され、新campaignでは
`--pca-dimension 50 --target-size 8192 --target-refresh-interval 10 --sinkhorn-gradient-mode envelope`
で明示もできます。

### 4. Pullから全8条件のbackground実行まで

```bash
(
  set -e
  cd /home/suzuki/Projects/scDiffusion-github
  git fetch origin
  git switch feat/20260916-x0predict-hybrid-additive
  git pull --ff-only origin feat/20260916-x0predict-hybrid-additive
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scdiffusion

  bash work/20260916_x0predict_hybrid_additive/scripts/run_all.sh --preflight-only
  mkdir -p work/20260916_x0predict_hybrid_additive/runs
  LOG_DIR=$(mktemp -d work/20260916_x0predict_hybrid_additive/runs/launch_XXXXXXXX)
  nohup env PYTHONUNBUFFERED=1 PYTHON="$(command -v python)" \
    bash work/20260916_x0predict_hybrid_additive/scripts/run_all.sh \
      --all-eight --gpu 0 --device cuda --analysis-device cpu \
    > "$LOG_DIR/output.log" 2>&1 < /dev/null &
  PID=$!
  echo "$PID" > "$LOG_DIR/pid"
  echo "PID: $PID"
  echo "LOG: $LOG_DIR/output.log"
  tail -f "$LOG_DIR/output.log"
)
```

`mktemp` が `XXXXXXXX` をランダム文字列に置き換えます。手動置換不要です。
`tail`はCtrl-Cで終了できます。nohupで起動したworkerは継続します。

### 5. Resume

ログに表示されたcampaign名を使います。保存済みcheckpoint/data/configは変更せず、
実行上限だけを短縮できます。10kと30kの評価markerは分かれます。

```bash
bash work/20260916_x0predict_hybrid_additive/scripts/run_all.sh \
  --resume-campaign additive_<UTC>_<RUNID> --all-eight --gpu 0 --device cuda
```

## Validation / limitations

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s work/20260916_x0predict_hybrid_additive/tests -v
git diff --check
```

Tests cover全8config、Stage1形式load/freeze、additive境界、従来blend一致、32768件の
source exclusion/repair/refresh、固定PCA、全4ODEへのdata-gradient、完全Sinkhornとの
gradient一致、実32768 targetのrectangular solve/backward、sampling全1000 timestep、
PCA共有、2-step interruption/resume、数値評価・小規模UMAP・PNG、条件失敗時の続行、
旧tracked fileのbyte-level不変です。初回17件のログは `audit/tests.txt` にあります。
2026-09-17は20件が成功し、上限短縮時のEMA再利用・元config/checkpointの不変・
10k/30k markerの分離・時刻/区間計測も確認しました。

**指定された実Stage1 checkpointはこのMacに存在しないため、その実ファイルのload検証は
未実施です。** 同形式の合成checkpointでは成功しており、リモートでは上記preflightが
必須確認になります。実データでの本学習、full-size GPU benchmark、生成分布の改善判定は
実施していません。ユーザー提供の20260915 coverage観察も、この実装作業で再測定していません。
