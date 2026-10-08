# ODE optimizer変更実験（20261008）

CellUnetはAdamW、ODEだけSGDに変更し、consistency係数λを0へ近づけたときにCellUnet予測がλ=0へ近づくか検証する。実装・設定・出力はすべてこのディレクトリ内に限定する。`work/20260830/`、`guided_diffusion/`、`ODE/`は読み取り/importのみ。

## 条件とloss

\[
L=L_{\mathrm{diffusion}}+\lambda_{\mathrm{ODE}}L_{\mathrm{soft}}+\lambda L_{\mathrm{consistency}},\qquad
L_{\mathrm{consistency}}=\mathbb E[\|\mathrm{CellUnet}(x_t,t)-\mathrm{ODE}(x_t,t)\|^2_{\mathrm{mean\ over\ genes}}].
\]

| config | λ | CellUnet | ODE |
|---|---:|---|---|
| `configs/lambda1.json` | 1 | AdamW | SGD |
| `configs/lambda0p1.json` | 0.1 | AdamW | SGD |
| `configs/lambda0p001.json` | 0.001 | AdamW | SGD |
| `configs/lambda0.json` | 0 | AdamW | SGD |

4条件のみ。共通設定は`configs/base.json`。ODEは`hill_after_linear`、K=1、gateなし。モデルfactory・初期化・diffusion・lossは20260830をimportする。soft constraintは内部`off_mask_lambda=5`、外側`ode_reg_lambda=1`、L1を維持する。CSVには内部係数前・内部係数後・最終寄与を別々に保存する。

- CellUnet: AdamW、初期LR `1e-4`、既存PyTorch AdamWの既定betas/eps。
- ODE: SGD、初期LR `1e-4`、momentum `0`、Nesterovなし。
- **既存`work/20260830/configs/base.json`のweight decayは`1e-4`。両optimizerで同値を継承する。** AdamWはdecoupled decay、SGDはgradientに加えるdecayである。momentum=0なのでSGD更新は `-lr*(gradient + weight_decay*parameter)`。gradient normはweight decay追加前、update normはweight decayを含む実際の変更量。
- 全条件、初期seed `1234`、30,000更新、batch=128、microbatch=-1、EMA=0.9999、FP32。更新k（1始まり）に使うLRは `1e-4*(1-(k-1)/total_steps)`。最後の更新後、保存される次回LRは0。
- `--set total_steps=N`は両optimizerのLR減衰期間もNへ揃える。直接resolved configを編集する場合は`total_steps`と`lr_anneal_steps`を一致させる。
- λ=0ではconsistencyのCellUnet勾配は0で、CellUnetへのloss勾配はdiffusionのみ。ODEにはsoft constraintとSGD weight decayの更新が残る。CellUnet側には従来のAdamW weight decayが残る。
- AdamWの適応的正規化により、λに対する更新量の比例性は一般には成立しない。一方、SGDでもODEのsoft constraint/weight decayはλで縮まらず、CellUnetは引き続きAdamWである。この実験でλ→0の挙動を観測するが、単調な収束は前提にしない。

学習データは従来の前処理済み`Embryonic.h5ad`のX、gene_name順、celltypeラベルをそのまま使用する。正規化・log1p・scaleの再適用はない。shuffle/drop_lastを維持し、新4条件では独立した同一seedのデータ用Generatorを使う。**過去runのDataLoaderの乱数消費順を再現するものではない。** 4条件間の初期パラメータ・データ順序を一致させ、各batchのindex hashも記録する。

## 過去結果との比較

20260830の現在のbase設定は100,000 step / 100,000-step線形LR decayであり、30,000-stepの実験も混在している。今回の標準は30,000 step / 30,000-step線形decay。100,000-step設定の途中30,000 checkpointは今回の30,000完了checkpointと同じLR履歴ではない。学習長・減衰期間・checkpointのraw/EMAを揃えない比較をoptimizerだけの効果と解釈しない。`summary/condition_summary.csv`と`comparison_protocol.json`に今回のtraining length/LR horizonと過去比較上の注意を保存する。

## 実装とresume

`training/train_loop.py`は既存`TrainLoop20260830.forward_backward()`、loss assembly、EMAを再利用し、初期化・step・checkpoint管理をこのディレクトリ内で実装する。

CellUnet/ODEパラメータは重複なし・全パラメータを網羅することをassertする。FP32では既存`MixedPrecisionTrainer.master_params`はmodel parameterそのもの。`MixedPrecisionTrainer.optimize()`を一度呼び、その中でAdamW/SGDを各一度stepする。EMAは両方のstep後に一度更新する。

**FP16は明示的に非対応。** 元wrapperには`convert_to_fp16()`がなく、既存FP16 master tensorは枝を跨いでflattenする。これを位置で分割せず、`use_fp16=true`を拒否する。今回の既存既定FP32実験には影響しない。単一process/GPU用。microbatchによるmeanの加算係数の変更を避け、既存既定のfull batchを使用する。

checkpointの数字は完了済みoptimizer更新回数。`model000000.pt`は真の初期状態。従来base loopの更新後step=0保存とは区別する。

- `modelNNNNNN.pt`: raw state dict（既存解析互換）
- `ema_0.9999_NNNNNN.pt`: EMA state dict（既定sampling用）
- `trainerNNNNNN.pt`: model、両optimizer、EMA、完了step、config、Python/NumPy/Torch/CUDA RNG、shuffle permutation/cursor/Generator、MPT scale
- `checkpoints/latest.json`: atomicに保存完了した再開点

`--resume`は最新の完全なtrainer checkpointを復元し、保存点より新しいCSV行を除去して続行する。SGD(momentum=0)のstateは空dictが正常で、param_groupsのLR/weight_decay/momentum等も復元する。旧共通AdamW checkpointやEMAだけからのresumeは非対応。configとデータ/GRN hashが変わるresumeを拒否する。再開後も最初に指定したLR horizonを維持する。

## Samplingと共有UMAP

従来と同じ1000-step ancestral sampler、clip_denoised=false、3000 cells、batch=50。CellUnet eval出力のみを使用する。ODE forwardが呼ばれないことを検査する。モデル構築・checkpoint load後にsampling seedをresetするので、4条件の同じcell indexには同じ初期Gaussian noiseと同じsampling乱数系列が割り当てられる。同一device・batch size・PyTorch環境を使う。

`s`は**完了済みreverse diffusion update回数**であり、forward timestep `t`、training stepとは別物。

| s | 保存状態 |
|---:|---|
| 0 | 初期Gaussian noise（更新前） |
| 200/400/600/800 | 一つのtrajectoryの途中状態 |
| 1000 | t=0のreverse updateを含む全1000回更新後の最終sample |

各chunkで`p_sample_loop_progressive()`を一度だけ走らせ、上記6状態をコピーする。新しいsampling式は実装しない。`samples/step_0000.npz`〜`step_1000.npz`には`cell_gen`、cell_index、seed、checkpoint、reverse_updates、step_definitionを保存。`sampling.json`にcheckpoint hashと乱数プロトコルを保存する。

UMAPは`Superclass == "Erythropoietic"`（`superclass`表記も対応）の**Real全細胞**と**全4条件×6状態のGenerated全細胞**をconcatし、一度だけfitする。20260911・20260913_2step・20260915_x0predictと同じReal選択であり、3000件への間引きは行わない。既存run configの`umap_real_cells`は互換性のため残すが、作図では使用しない。該当ラベルがない場合は全細胞へのfallbackをせずエラーにする。20260830の`hematopoietic_viz/core.py::compute_common_umap`と同じPCA(arpack,50)、neighbors(15,40 PCs)、Scanpy UMAP設定を継承する。小データでは次元/neighbor数を上限に合わせる。neighbors乱数seedも明示する。正規化・log・scaleなし。

`s=0`の全条件の配列一致、gene順序、sampling seed/batch/device、cell index対応を検査する。joint UMAPの座標は同じembedding内でのみ比較可能であり、距離を発現空間の定量誤差と同一視しない。

`umap/coordinates.npz`に全座標、Realの元index/細胞名、Generated index、全図共通の軸範囲を保存。24枚の各条件/step図と以下を再samplingなしで再描画できる。

`umap/umap_comparison_600_1000_all_conditions.png`

4行（λ=1,0.1,0.001,0）×2列（s=600,1000）。共通座標、共通軸範囲、統一Real/Generated凡例。`embedding.json`で一回fitの出所と座標hashを検証する。

## 出力

`runs/<batch-id>/<condition>/`以下:

- `exp_config.json`, `input_fingerprints.json`, `model_info.json`
- `checkpoints/segment_000/model/training_metrics.csv`: 毎更新の実測loss、Cell/ODE gradient norm、Cell/ODE実update norm、ODE出力のcell平均L2 norm、λ、両LR、data index hash
- 同階層`loss_components_20260830.csv`: 既存loss loader互換の全loss情報
- `samples/step_*.npz`, `sampling.json`
- `analysis/figures/01_*.png`〜`12_*.png`: 20260830の関数を直接再利用。図06は元のaliasも保持
- `analysis/checkpoint_csv/`: checkpointからのloss target/Cell-ODE指標、post-hoc gradient（実測更新量とは区別）

`runs/<batch-id>/summary/`以下:

- training_measurements.csv、parameter_update_norm.png、gradient_norm.png、loss_curves.png、ode_output_norm.png
- condition_summary.csv/.png、comparison_protocol.json（4条件用）
- fixed_prediction_inputs.npz、predictions_lambda*.npz、prediction_difference_vs_lambda0.csv/.png（同じx_t/t、eval、最終EMAで比較）
- parameters/: 初期raw＋保存された共通EMA checkpoint列、4条件行のparameter distribution PNG。W全体/mask有無、b、raw/effective K/V/delta、CellUnet全体/weight/biasを継承。統計CSVとhistogram bin/count CSVも保存

## リモートでの実行

既存`scdiffusion`環境（Torch、Scanpy、umap-learn、NumPy/Pandas、Matplotlib/Seaborn、blobfile等）を使う。ソースを`~/Projects/scDiffusion-github/work/20261008_ode_optimizer/`に配置する。ローカルでGPU学習は実行しない。

まずCPU smoke test（12 cells/4 genes、各条件3 training updates、samplingだけは本番と同じ1000 reverse updates）。可視化まで含み、本格学習ではない。

```bash
set -o pipefail
cd ~/Projects/scDiffusion-github
conda activate scdiffusion
export PYTHONDONTWRITEBYTECODE=1
SUITE=work/20261008_ode_optimizer
# 別checkoutへ初めて配置したときのみ、その既存ファイルを監査baselineとして記録。
# すでに存在するbaselineは上書きしない。
python -B "$SUITE/scripts/verify_protected.py" --record-remote-baseline
python -B "$SUITE/tests/smoke.py" 2>&1 | tee "$SUITE/validation/smoke.log"
python -B "$SUITE/scripts/verify_protected.py"
```

通常の回帰テストのみは `python -B -m unittest discover -s "$SUITE/tests" -p 'test_*.py' -v`。完全smokeの画像は`validation/smoke/campaign/umap/`に残す。実データは一切使わない。

本番4条件の学習→sampling→解析→全UMAP→比較PNGを一括実行:

```bash
cd ~/Projects/scDiffusion-github
conda activate scdiffusion
export CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
SUITE=work/20261008_ode_optimizer
BATCH=ode-sgd-30000
DATA=/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad
GRN=/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv
python -B "$SUITE/scripts/launch.py" --batch-id "$BATCH" --stage all \
  --set device=cuda --set total_steps=30000 \
  --set "data_dir=$DATA" --set "edge_tsv_path=$GRN"
```

データ/GRNを`scDiffusion-github`側へ配置している場合はDATA/GRNだけ実パスへ変更する。launcherは各条件を同じGPUで順番に実行し、失敗時は非ゼロ終了する。本番学習を自動でバックグラウンド起動はしない。

```bash
# 一切書き込まず、4条件のresolved config/コマンドを表示
python -B "$SUITE/scripts/launch.py" --batch-id "$BATCH" --dry-run
# 4条件を最新checkpointから再開。完了条件は学習を追加しない
python -B "$SUITE/scripts/launch.py" --batch-id "$BATCH" --resume
# 単一条件の学習のみ
python -B "$SUITE/scripts/launch.py" --batch-id "$BATCH" --condition lambda0p1 --stage train
# 個別stage: train / sample / analysis / umap / comparison
python -B "$SUITE/scripts/launch.py" --batch-id "$BATCH" --stage sample
python -B "$SUITE/scripts/launch.py" --batch-id "$BATCH" --stage analysis
python -B "$SUITE/scripts/launch.py" --batch-id "$BATCH" --stage umap
python -B "$SUITE/scripts/launch.py" --batch-id "$BATCH" --stage comparison
# 保存座標から再描画のみ（再samplingもUMAP再fitもなし）
python -B "$SUITE/scripts/plot_umap.py" --campaign "$SUITE/runs/$BATCH" --redraw-only
python -B "$SUITE/scripts/plot_comparison.py" --campaign "$SUITE/runs/$BATCH"
```

単一条件`--stage all`は学習・sampling・per-run解析まで実行する。全4条件sampleが存在するときのみ共有UMAPへ進む。4条件比較を作るには全条件を実行する。configを変えた実験は新しいbatch-idを使う。既存samplingと異なるcheckpointを使う再samplingは`sample.py --force`、異なるsampleからUMAPを作るときは`plot_umap.py --refit`を明示する。

## 検証状況

実行結果は`validation/IMPLEMENTATION_REPORT.md`を参照。静的検査と実際に実行したテストを分けて記録する。

## 保存済みsamplingからUMAPだけ全て作り直す

```bash
python -B work/20261008_ode_optimizer/scripts/replot_umaps.py
# 特定campaignを指定する場合:
python -B work/20261008_ode_optimizer/scripts/replot_umaps.py \
  --campaign work/20261008_ode_optimizer/runs/<batch-id>
```

指定なしでは、4条件×6状態が全て保存されている最新samplingのcampaignを選ぶ。ログ冒頭に選択先を表示する。`plot_umap.py --refit`→`plot_comparison.py`のみ実行し、旧座標・24枚の各step PNG・4×2比較PNGを置き換える。学習、sampling、checkpoint解析、全リポジトリbaseline検査は実行しない。`embedding.json`に選択列、全選択細胞数、celltype別件数、Generated件数を保存し、凡例にもReal/Generatedの件数を表示する。Generatedは保存済み件数（標準3000）であり、再作図で細胞数を増やすことはない。

Realの選択と表現は202609*を参考にしているが、今回保存したGeneratedの表現は指定どおりreverse update後のstate。20260913等の`pred_xstart`とは区別する。今回の既存NPZを別表現とみなして描画することはしない。全4条件×6状態の共通座標系は維持する。

軽量検証（学習・samplingなし）:

```bash
python -B -m unittest discover -s work/20261008_ode_optimizer/tests -p 'test_umap_reference.py' -v
```

## X_START予測で4条件を新規実行

`--set predict_xstart=true`を指定する。`GaussianDiffusion.model_mean_type`は`START_X`、diffusion MSEのtargetはnoise εではなく前処理済みのclean X（x_start）となる。samplingも同じconfigでSTART_Xとして出力を解釈し、checkpoint解析もx_startを正解として評価する。モデル構造、初期化、soft constraint、AdamW/SGD設定、λの4条件、step数はそのまま。

consistencyは引き続きCellUnet出力とODE出力のMSEであり、このrunではCellUnetのX_START予測とODE出力を合わせる正則化になる。旧ε予測のMSEとはtarget・尺度が異なるため、loss値を同一尺度の改善として比較しない。

```bash
python -B work/20261008_ode_optimizer/scripts/launch.py \
  --batch-id ode-sgd-xstart-NEW_ID --stage all \
  --set predict_xstart=true --set total_steps=30000 --set device=cuda
```

旧ε予測runからresumeせず、新しいbatch-idで最初から学習する。既存configの既定値falseは旧run互換のため維持する。model_info.json、sampling.json、analysis metadata/summary CSVにprediction_targetを記録する。UMAPはErythropoietic全Real cellsと保存済みGenerated全件を使い、全4条件×6状態の共通座標系で24図＋4×2比較図を作る。

専用のCPU小規模検証（実データ/本格学習なし）:

```bash
python -B -m unittest discover -s work/20261008_ode_optimizer/tests -p 'test_xstart.py' -v
```
