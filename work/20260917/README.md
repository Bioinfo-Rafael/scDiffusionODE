# 20260917: 5 ODE models × 2 losses

`20260916_x0predict_hybrid_additive`のStage2を、ODE構造と学習目的だけ変えて比較する。
元ファイルは変更せず、コード・cache・結果はこのディレクトリ以下へ保存する。
この環境ではLinux指定パスが存在しないため、同じrepoの
`/Users/cls-lab/Git/scDiffusionODE/work/20260917`に実装した。
Linuxではrepo内の同じ相対パスで実行できる。

## コード確認と再利用

|対象|確認した実装|今回の利用|
|---|---|---|
|2段階学習・固定CellUNet|`work/20260916_x0predict_hybrid_additive/training/runner.py`、`work/20260915_x0predict/models/__init__.py`|既存30k final EMA、freeze、AdamW、EMA更新を利用|
|係数・合成|9/16 `models.py::AdditiveHybrid500`、9/13 `models/hybrid500.py::Hybrid500Mixin`|継承し、baseline/Hybrid CellUNetの内訳を追加|
|Reconstruction|9/16 `training/objectives.py` → 9/15 `training/objectives.py` → `GaussianDiffusion.training_losses`|同じSTART_X MSE呼び出し、同じfloat64 noise|
|LowRank / MatSum / LoRA|`work/20260609_MathMLPHybrid/cell_train_20260609.py`、`ODE/ode_20260609_mathmlp.py::{LowRankField,MatSumField,LoRAField}`|**元クラスを直接import**。推測再実装なし|
|GRN|6/9 `build_edge_mask`、`work/20260830/models/factory.py::build_target_source_mask_20260830`|既存3モデルは元loader＋転置。A/Bは同じTSV列・filterをedge listへ変換|
|Dataset|9/16 `data.py::source_batches`、9/15 `common.py::load_real`|元のX・gene order・shuffle・drop_lastをそのまま利用|
|Sampling・評価|9/15 `sampling/trajectory.py`、`analysis/*.py`、9/16の拡張diagnostics/distributions|private namespaceでロードし、計算を再利用。元moduleをmonkeypatchしない|

`2-step`は**Step1のCellUNet学習→Step2の固定CellUNet＋ODE学習**という2段階を意味する。
1 timestep内に別々のCellUNetを2回学習させる構造ではない。Step1は全条件で同一checkpointを使い、再学習しない。

## 10条件とlossの対応

|model_type|既存/新規|aux_loss|
|---|---|---|
|`lowrank`|6/9 LowRankField|`reconstruction`, `knn`|
|`matsum`|6/9 MatSumField|`reconstruction`, `knn`|
|`lora`|6/9 LoRAField|`reconstruction`, `knn`|
|`direct_message`|新規 DirectMessageODE|`reconstruction`, `knn`|
|`multihop_graph_filter`|新規 MultiHopGraphFilterODE|`reconstruction`, `knn`|

**原実装には別個のreconstruction auxiliaryと共通diffusion lossはない。**
9/16のR条件はSTART_X MSEそのもの＋soft制約で、OT条件はMSEをOTで置き換えている。
今回も同じ置換規約を採用する。`aux_loss`は依頼に合わせたconfig名。

```text
R: L = mean(GaussianDiffusion.training_losses(H, x0, t)["loss"]) + L_soft
K: L = lambda_knn * (L_pos + lambda_neg * L_neg) + L_soft
```

K条件ではMSEを計算も加算もしない。Stage1の学習済みdiffusionとsamplingは保持する。
KのODE学習は実データanchorから直接行うため、Hybrid出力を通るRとは入力と勾配経路も異なる。
全条件ともtは同じuniform 0..999、source batch順とdiffusion乱数streamも合わせる。
negativeの乱数は専用Generatorへ分離する。

`L_soft = ode_reg_lambda * off_mask_lambda * field.off_mask_penalty("l1")`。
6/9のfieldは係数なしのpenaltyを返すため、9/16と同じ外側係数5を明示的に掛ける。
ratio lossは無効。LowRankの既存subsample（最大8 cell）も保持する。
A/BはGRN外の独立edge parameterが存在しないため、off-mask penaltyは構造上0。
Bの複数hop依存を新しいoff-mask edgeとして罰することはしない。

## START_Xと係数schedule

raw diffusion timestepを`t`、同じ固定CellUNetを`C`、ODE RHSを`F`とする。

```text
p(t) = clamp((500-t)/500, 0, 1)
H(x_t,t) = (1-p(t))*C(x_t,t) + p(t)*C(x_t,t) + p(t)*F(x_t,t)
         = C(x_t,t) + p(t)*F(x_t,t)
             ↑ baseline経路       ↑ Hybrid側のC/ODEはともにp
```

`t=500: p=0`, `t=250: p=0.5`, `t=0: p=1`。
`alpha_ODE=alpha_HybridCellUNet=p`。最初のbaseline経路は`1-p`。
固定Cを共用するため、合成されたCellUNetの総係数は1。これは元のadditive forwardと数値的に同じ。
枝の診断に`baseline_weight`, `hybrid_cell_weight`, それぞれのcontributionを追加する。
元評価の`ml_weight`/`cell_weight`は合計係数1を示す。

Hybridのraw出力をx0 predictionとしてnative DDPMへ渡す。epsilon変換やsampling後のODE加算はしない。
`t>=500`のODE寄与は0で、通常のHybrid forwardはその行のODE計算を省く。
診断時はraw ODEも評価する。LowRankの動的penalty cacheはforwardごとに初期化し、古いbatchの勾配を再利用しない。

## GRNとA/Bの実装

`.tsv`の`from`がsource、`to`がtarget。datasetの実際のgene順で辞書を作り、
`j=gene_to_column[from]`, `i=gene_to_column[to]`として対応づける。
重複gene名・欠損GRN名・一致edgeゼロはエラー。重複edgeは1本にまとめ、self-edgeは入力にあれば保持する。
片端でもdatasetにないedgeは除外するが、件数と不在geneを`gene_mapping.json`へ必ず保存する。
datasetにのみあるgeneの列は削除しない。

6/9 loaderは`mask[source,target]`を返すが、6/9 fieldの実際の行列演算は`W[target,source]`。
**fieldクラスはそのまま、maskだけ9/16の元factoryと同じく転置**して渡す。
A/Bはdense maskを作らず、同じ読み込み規約を`src/grn.py`でedge listへ変換する。

### A: DirectMessageODE

```text
m_(j→i) = phi([x_i, x_j, t])
dx_i/dτ = -softplus(gamma_i)*x_i + Σ_(j→i) W_ij*m_(j→i)
```

1個のshared `phi`（Linear–SiLU–Linear–SiLU–Linear、hidden=64）を全edgeに適用する。
gene embeddingなし。GRN edgeごとのWとgeneごとのgammaを学習する。
W初期値は`1/sqrt(G)`、raw gamma初期値は6/9と同じ0.1。
`[B,edge_chunk,3]`入力からmessageを計算し、`index_add`でtargetへ集約する。
edge chunkは中間テンソルの単回サイズを抑えるが、backward用の全edge activation保存量は残る。

### B: MultiHopGraphFilterODE

```text
A_ij = 1 if j→i is retained
P_ij = A_ij / max(in_degree(i), 1)
h1 = P*x; h2 = P*h1; h3 = P*h2
dx_i/dτ = f_theta([x_i, h1_i, h2_i, h3_i, t])
```

Pは**target行ごとの入次数正規化**。符号・重み列は元loader同様使用せず、入力はbinary directed GRN。
self-loopや逆方向edgeの追加なし。入次数0の行は0。
Pは固定で、1個のshared MLP（hidden=64）を全geneへ適用する。
COO index/valueから`sparse_coo_tensor`を作り、`torch.sparse.mm(P, x.T)`を逐次3回行う。
P²/P³やdense G×Gを生成しない。checkpointにはCOOの構成bufferを保存するので既存EMAも利用できる。

既存LowRank/MatSum/LoRAは元のdense parameter／soft制約計算を保持する。
A/Bに対するsparse制約をこれら既存3モデルへ勝手に適用しない。

## kNN graph lossとcache

前処理は元のtraining XのEuclidean kNN（self除外、k=15）。追加PCA/scaleなし。
`NearestNeighbors(algorithm="kd_tree")`を明示し、chunk単位でqueryする。
N×N距離行列を作らず、学習中はcacheのedgeのみ読む。高次元でのexact KD-treeは遅い可能性があり、
前処理ではdense Xとtreeのメモリが必要。実データでの時間・メモリは未測定。

```text
sigma_i = max(kth_neighbor_distance_i, 1e-12)
p_ij = exp(-d_ij²/sigma_i²)
y_i = EulerSolve(F, x_i, t, t+delta_t)
q_ij = 1/(1 + a * ||y_i-x_j||^(2b))
L_pos = -mean(p_ij * log(q_ij + epsilon))
L_neg = -mean(log(1-q_ij + epsilon))
L_knn = L_pos + lambda_neg * L_neg
```

`x_j`は常に元の実データ。anchorだけを微分可能なEuler積分で進める。
全5モデルでsolver/intervalを統一。既定は1 step、`delta_t=0.001`。
ODEの時刻入力は既存fieldに合わせraw diffusion t、積分各stepでは`t+s*delta_t/integration_steps`。
Hybrid RHS加算自体にはこのdelta_tを掛けない。既存post-hoc解析のEuler100 steps×0.001も別途保持する。
`b<1`かつ完全一致点の勾配発散だけを避けるため、距離²の下限をfloat32の最小正規数にする。

positiveは各anchorのcached k本全部、negativeは各anchor5本。
全cell pairをlossへ使わず、selfと全positiveを除いた補集合から一様・復元抽出する。
禁止IDのsorted rankを使うため、N要素の候補配列やrejection loopは不要。
negative同士の重複は許す。各edgeのtarget expression読み出しもchunk化する。

`cache/knn/<SHA256>/`には`cell_ids.npy`, `indices.npy`, `weights.npy`, `distances.npy`, `sigma.npy`,
`completed.json`を保存。identityはdataset SHA256、gene順hash、shape、k、距離・sigma定義・row順。
読み込み時に全cacheファイルのchecksumを検証し、read-only memmapで利用する。
完成markerとprocess lockで不完全cache／同時書き込みを防ぐ。不完全cacheは自動上書きしない。
同じgraphは全K条件で共有し、epoch/batchごとの再計算はしない。
Rと`lambda_knn=0`ではgraph準備・参照・積分・negative抽出をすべてskipする。
lambda=0でも既存soft制約は残すため、LowRankではその動的Wを得るRHS評価だけを行う。

## 共通configと出力

`configs/base.json`は9/16の実際のdefaultを抽出して保存したもの。
`configs/<model>_<loss>.json`は10通りのselector。launcherが共通overrideをmergeし、
campaign内に全10条件の解決済みconfigを保存する。

|config|default / 意味|
|---|---|
|`stage1_checkpoint`|9/16が参照する30,000-step final START_X EMA|
|`data_dir`, `edge_tsv_path`|元Linuxデータパス。同一内容の移設はCLIで指定可能|
|`split`|`all_cells_no_validation`：元実装にはhold-out splitなし。全条件で全cellを使用|
|`seed`, `source_seed`|1234 / 1234|
|`batch_size`, `total_steps`, `epochs`|128 / 10000 / null。元はupdate数指定。epochs指定時はfloor(N/batch)×epochs|
|`lr`, `weight_decay`, `lr_anneal_steps`|AdamW、1e-4、1e-4、30000。元update後LR更新を維持|
|`ema_rate`, `log_interval`, `save_interval`|0.9999 / 50 / 1000|
|`diffusion_steps`, `noise_schedule`|1000 / linear、START_X、respacingなし、clipなし|
|`cell_unet_hidden_num`|[2000,1000,500,500]、全条件で固定eval、optimizer対象外|
|`rank`, `K`, `time_dim`, `field_hidden`|16 / 8 / 64 / 256（6/9 default）|
|`field_dropout`, `use_decay`, `lowrank_penalty_subsample`|0 / true / 8|
|`graph_hidden`, `edge_chunk_size`|64 / 4096、A/BのMLP幅とAのedge chunk|
|`off_mask_lambda`, `ode_reg_lambda`, `ode_reg_norm`|5 / 1 / l1|
|`knn.k`, `knn.num_negatives`|15 / 5|
|`knn.delta_t`, `knn.solver`, `knn.integration_steps`|0.001 / euler / 1|
|`knn.lambda_knn`, `knn.lambda_neg`, `knn.a`, `knn.b`|すべて1|
|`knn.epsilon`, `knn.seed`|1e-8 / 1236|
|`knn.query_batch_size`, `knn.edge_chunk_size`|1024 / 1024|
|`num_samples`, `sample_batch_size`|3000 / 50|
|`post_ode_steps`, `post_ode_dt`, `post_ode_integrator`|100 / .001 / euler（既存post-hocのみ）|
|`analysis_cells`, `analysis_ot_cells`, `analysis_batch_size`, `analysis_pairs`|2000 / 128 / 128 / 100000|
|`evaluation.*`, `ot.*`|元SW2（256投影、2048点、seed4321）・完全Sinkhorn評価を保持|

全設定は`configs/base.json`参照。既存と同じ前処理・gene順・source shuffle/drop_lastを使用。
checkpoint/data/GRN hashが異なる場合は停止する。Stage1 stateはstrict loadし、保存時にもraw/EMA両方の不変性を検証。
`epochs`を指定しなければ最後のepochは端数になり得る。`epochs.csv`にcomplete/partialを明記し、
`mean_epoch_seconds`は完了したepochだけ、`mean_observed_epoch_seconds`は端数も含む。

```text
results/comparison_<id>/
  campaign.json, source_sha256.json, configs/*.json
  stage1_ema.pt                 # 評価時の共通CellUNet baseline
  <condition>/training/
    effective_config.json, metadata.json, gene_mapping.json, knn_cache.json
    losses.csv, epochs.csv, performance.json, checkpoints/, completed.json
  <condition>/sampling/<id>/
    *.npy, sampling_metadata.json, sampling_branch_metrics.csv
    analyze/<id>/*.csv, embed/<id>/*.csv
  summaries/<id>/comparison.csv, all_metrics.csv
  <condition>_<action>.log, finished.json
```

training time・各epoch時間・GPU allocated/reserved peak bytes・総parameter数／学習parameter数を保存。
GRN edge数・5回平均のODE RHS時間も保存。RHS時間は学習後、eval/no_grad、t=0、同じbatch size。
GPUは同期して計測。CPU実行のGPU値は0でなくnull。
kNNは前処理時間・cache hit・cache bytes・loss forwardの総時間／平均時間を記録する。
cache hit時の新規前処理時間は0。元のgraph構築時間と今回の検証・loadを含む準備時間は別項目に保存する。
kNN backward時間はtraining timeに含むが、kNN loss個別タイマーには含めない。

既存のtrue-x0 Pearson/MSE/cosine/norm、CellUNet対ODE、分布diversity、Sinkhorn、SW2、
coverage、gene variance、post-ODE収束、UMAPを保持。
比較CSVは**diffusion endpoint reverse_step=1000**を使い、post-ODE endpointと混同しない。
全評価CSVは`all_metrics.csv`にもまとめる。未生成値をゼロで埋めない。

## 実行

repoルートで既存scdiffusion環境を使用する（新しい依存packageなし）。

```bash
cd /home/suzuki/Projects/scDiffusion-github
conda activate scdiffusion

# 全10条件：各条件のtrain → sample → analyze/plot → UMAP、共通baselineは一度
python -m work.20260917.launcher

# 1条件（各モデル・loss名は上の表）
python -m work.20260917.launcher --model-type direct_message --aux-loss knn

# 全10条件の学習のみ
bash work/20260917/scripts/run_all.sh --train-only

# checkpoint・データがなくてもconfigと未実行の比較CSVを生成
python -m work.20260917.launcher --dry-run

# 同一ファイルを移設した場合
python -m work.20260917.launcher --stage1-checkpoint /path/to/ema_0.9999_030000.pt \
  --data /path/to/Embryonic.h5ad --edge-tsv /path/to/tf_target_edges.tsv

# 全条件共通の短いrun（sampling量なども共通override可）
python -m work.20260917.launcher --training-steps 8 --train-only
python -m work.20260917.launcher --epochs 2 --train-only
python -m work.20260917.launcher --overrides work/20260917/configs/example_overrides.json --train-only

# 後から学習済みcheckpointをsampling・評価
python -m work.20260917.cli sample --checkpoint <このsuite内のEMA.pt> --device cuda
python -m work.20260917.cli analyze --trajectory <TRAJECTORY_DIR> --device cpu
python -m work.20260917.cli embed --trajectory <TRAJECTORY_DIR> --device cpu
python -m work.20260917.cli plot --input <ANALYSIS_DIR>

# 再集計。launcherも各条件の終了後に自動実行
python -m work.20260917.cli summarize --campaign <CAMPAIGNの絶対パス>

# CPU smoke tests
python -m unittest work.20260917.tests.test_smoke -v
```

launcherは新campaignを作り、既存結果を上書きしない。独立条件の失敗後も他条件を継続し、
`finished.json`と非ゼロ終了コードで失敗を報告する。途中checkpointはraw/EMA/optimizerを保存するが、
このsuiteに途中学習resumeのCLIはない。再実行は新campaignのstep0からとなる。

## 検証状況

`tests/smoke_test.log`にCPU結果（9 tests passed）。小規模の12 cells×7 genesで全10条件を2 epoch（各8 updates）実行。
5モデルのforward/backward、Aのphi勾配、Bの演算dispatchによるdense G×G非生成、
cache loadとepoch間非再計算、negative除外、loss式、lambda=0 skip、NaN/Infなし、
固定CellUNet不変、checkpoint strict復元を検証。native 1000-step DDPM＋100-step Eulerも実行。
R/K間の初期state・乱数stream一致、既存評価関数から比較CSVまでの接続も検証した。

`tests/local_data_audit.json`はこのMacにある実データとGRNのmapping確認であり、学習結果ではない。
指定Stage1 checkpointがこの環境にないため、実データの10条件本学習・GPU速度／peak memory・
最終品質比較は未実行。synthetic smokeの数値を品質評価として扱わない。
