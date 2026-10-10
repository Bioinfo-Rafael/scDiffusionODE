# Mouse Gastrulation model comparison — 20261010

既存モデル・loss・DDPM samplingを直接importする比較環境です。初回 **17条件**と拡張 **26条件**（計43 config）を用意しました。本学習・実データsampling・VeloEV実評価は自動実行していません。

- 全34 work directoryの調査: [audit/WORK_DIRECTORY_AUDIT.md](audit/WORK_DIRECTORY_AUDIT.md)
- モデルの式・class・状態: [MODEL_INVENTORY.md](MODEL_INVENTORY.md)、[CSV](MODEL_INVENTORY.csv)、[registry.json](registry.json)
- 学習法・歴史的既定値・変更点: [TRAINING_METHOD_INVENTORY.md](TRAINING_METHOD_INVENTORY.md)
- 直接importと新規adapterの区分: [SOURCE_REUSE_MAP.md](SOURCE_REUSE_MAP.md)
- 検証結果・未実施工程: [audit/VALIDATION_REPORT.md](audit/VALIDATION_REPORT.md)

## 比較条件

| Group | Conditions | Training | Sampling output |
|---|---|---|---|
| A01 | CellUNet START_X | 30000; 10/09 final EMAを検証して再利用可能 | CellUNet |
| B01–04 | simple_softplus / hill_after_linear / centered_signed_hill / shifted_hill_rho | ODE-only START_X MSE + source soft; 30000 | ODE |
| C01–04 | simple/hill × blend/additive | joint START_X MSE + soft; 30000 | Hybrid |
| D01–04 | hill; lambda 0/.001/.1/1 | joint START_X + original consistency + soft; 30000 | **CellUNetのみ** |
| E01–04 | simple/hill × blend/additive | shared A01 final EMA; C freeze; ODE 10000 | Hybrid |

Pilot: cells=89267、HVG=1024、normalized spliced linear X、1000-step linear diffusion、B128、AdamW lr1e-4/WD1e-4、EMA.9999、seed1234。Native ancestral DDPM、clip=False、nw=.5、生成3000細胞。Blend `rF+(1-r)C`、Additive `C+rF`、`r=max(0,1-t/500)`。

Eの**停止10000 updates**と**LR anneal horizon 30000**は別設定です。9/16のpost-update zero-based annealingを保持します。Dは8/30のEPSILONからSTART_Xへ変更、8/30の100kから30kへ変更、10/08のODE SGDからAdamWへ変更します。ソースのsoft constraint内部係数5は保持し、June系の「外側で5倍」と混同しません。

Jointと2-stepは初期値・累積更新数が違います。freezeだけを比較する追加対照 `X_stage1_hill_additive_freeze_true/false` は同じA01 final EMAから開始します。全E条件はrun-root内の `shared_stage1.json` に同じcheckpoint/hashを要求します。

## データ・GRN

`prepare` は `work/20261009_newBenchmark/runs/data/training.h5ad` があれば、shape、metadata、SHA256、cell/gene順序、linear-scale/HVG provenance、元入力hashを検証して再利用します。なければ **既存10/09のprepare関数**をimportし、出力だけ本ディレクトリ内へ変更して作成します。

元入力は `data_preparation/20261007/data/MouseGastrulation.h5ad`。既存prepareは53801遺伝子全体でS/Uが独立1e4正規化済み、X==spliced、log/scaleなし、erythroid9815を検証します。HVGはコピーへlog1p→Seurat dispersion→tieを元位置で解消→元順序で1024。学習Xを再log/scale/normalizeしません。

共通 `runs/data/manifest.json` が全条件の入力です。cell/gene IDリストとhashをcheckpoint/exportにも保存します。

GRNは `external_data/*.tsv` をすべて調べ、既定はMouse TSVです。exact gene IDを優先し、次に一意なcase-sensitive `gene_name` で対応します。曖昧記号は除外し、case変換・orthology推定は行いません。`grn_audit.json` に各TSVのedge数・対応率・gene coverage、`mouse_grn_gene_ids.tsv` に対応後edgeを出力します。**edgeが0ならODE構築・学習を拒否**します。ローカルに実データがないため、実データ対応率は未測定です。

8/30/Juneは `[target,source]`、古いGeneODEは `[source,target]`。旧loaderの戻り値を境界で一度だけ転置します。4/21 GeneODE constructorのsub-gene indexとfull indexのズレはadapterでmaskだけ置換し、forward・parameter layoutを変更しません。

## Velocityとbenchmark

clean X、t=49で利用可能な `ode_raw`、`cellunet_raw`、`hybrid_raw` を別H5ADに保存します。Dには混合出力がないためhybrid_rawを作りません。モデルのprediction target、branchの学習上の役割、field定義、expression scale、checkpoint/hash、IDs、proxy仮定をsidecarとAnnData.unsに記録します。

`START_X prediction` と生物学的 `ds/dt` は同義ではありません。共通benchmarkが要求する `definition=ds_dt` は **raw outputをvelocityとみなす実験上の仮定**として明示します。符号反転、X差し引き、q_sample、Euler積分はexportに加えません。状態/時刻に依存するfieldも自律ODEと区別します。

評価は既存 `data_preparation/20261007/benchmark/run.py --evaluation 3fold --prediction ...`。normalized参照を使う**full-trained**方式です。89,267細胞で一度学習し、erythroid9815を3つのtest fold（3272/3272/3271）へ抽出・評価します。3回再学習するCVではありません。参照の元入力hashと学習元入力hashも照合します。

既存VeloEVがCBDir/ICVCoh/CTO/TSCを計算します。各fold値と公式mean/std（ddof=0）を残します。`20261009_reproduce` のscVelo stochasticは **official-raw/per-fold** で、normalized/full-trainedとは別protocolです。集計図もprotocol/profile/reference hash別に分け、独自総合スコアを作りません。

## Remote Linux commands

学習には既存 `scdiffusion` 環境を使用してください。benchmarkは既存入口が専用Pythonへsubprocessで移ります。VeloEVの依存を学習環境へinstall/updateしません。ローカルテストには `runs/.venv-test` を隔離して使用しました（Git対象外）。本コードはPython 3.9の既存scdiffusionでもimportを確認しています。

```bash
cd /home/suzuki/Projects/scDiffusion-github
conda activate scdiffusion
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
```

事前検証と一覧（本学習なし）:

```bash
python -m work.20261010_model_comparison.cli preflight
python -m work.20261010_model_comparison.cli list
python -m work.20261010_model_comparison.cli list --extended
python -m work.20261010_model_comparison.cli launch --all --dry-run
# pytestが利用可能な学習互換のテスト環境で:
python -m pytest work/20261010_model_comparison/tests -q \
  --basetemp=work/20261010_model_comparison/runs/pytest \
  -o cache_dir=work/20261010_model_comparison/runs/pytest-cache
```

`preflight` はprotected file、config、入力/環境の存在を報告します。`false` の入力があればremoteに配置してからprepareを実行します。初期ローカル変更のhashも保護対象です。remoteがclean mainで異なる場合、後述「ソース保護の移送」を参照してください。

入力作成とbenchmark参照作成（初回のみ。参照作成は学習ではありませんが実データ/RAMを使用します）:

```bash
python -m work.20261010_model_comparison.cli prepare
python data_preparation/20261007/benchmark/prepare.py \
  --evaluation 3fold --profile normalized
```

参照が既に完成していれば2行目は不要です。既存参照へprepareを再実行すると上書きを拒否します。専用環境がない場合だけ [共通benchmark README](../../data_preparation/20261007/benchmark/README.md) の環境セットアップを使います。

A01を再利用または学習、単一条件を学習:

```bash
python -m work.20261010_model_comparison.cli train \
  --condition A01_cellunet_startx --device cuda
python -m work.20261010_model_comparison.cli train \
  --condition C02_hill_after_linear_blend --device cuda
python -m work.20261010_model_comparison.cli train \
  --condition E04_hill_after_linear_additive --device cuda
```

A01に互換な既存30k final EMAがあれば再利用します。`--no-reuse` は新規学習を指定します。既存runのcheckpointを上書きするオプションではありません。再学習する場合は新しい `--run-root work/20261010_model_comparison/runs/repeat1` を指定してください。同じrun-rootを後続コマンドにも渡します。

単一条件のsampling・velocity・benchmark・可視化:

```bash
python -m work.20261010_model_comparison.cli sample \
  --condition C02_hill_after_linear_blend --device cuda
python -m work.20261010_model_comparison.cli export_velocity \
  --condition C02_hill_after_linear_blend --device cuda
python -m work.20261010_model_comparison.cli evaluate \
  --condition C02_hill_after_linear_blend
python -m work.20261010_model_comparison.cli visualize \
  --condition C02_hill_after_linear_blend
```

`--field ode_raw` 等で特定出力だけ評価できます。`evaluate --prepare-reference` は参照が未作成なら作成します。`visualize --scope training|samples|velocity` は独立実行できます。velocity図は専用評価環境で公式fold geometryとpostprocess pickleの投影を使います。generative図は8/30由来のjoint UMAP、9/15のdiversity/Sliced Wassersteinを利用し、元遺伝子空間の分散・分布・collapse診断を出力します。UMAP距離は定量誤差として扱いません。

17条件の順次実行（**このコマンドはGPU本学習を開始します。今回自動実行していません**）:

```bash
python -m work.20261010_model_comparison.cli launch --all --device cuda
# 学習だけ:
python -m work.20261010_model_comparison.cli launch --all --device cuda --stages train
# 学習済みから評価だけ:
python -m work.20261010_model_comparison.cli launch --all --stages evaluate
```

A→B→C→D→Eの順です。完成済みcheckpoint/sample/prediction/評価はhash・config照合後に再利用します。未完了runは `--resume` を明示します。失敗したbenchmarkディレクトリは成功として再利用せず、保持して別labelで再評価できます。

checkpointからresume（`--stop-after` は**絶対更新数**。horizonは変更しません）:

```bash
python -m work.20261010_model_comparison.cli train \
  --condition C02_hill_after_linear_blend --device cuda --resume
python -m work.20261010_model_comparison.cli train \
  --condition C02_hill_after_linear_blend --device cuda \
  --resume work/20261010_model_comparison/runs/default/C02_hill_after_linear_blend/checkpoints/model010000.pt
```

同一runで過去checkpointへ巻き戻して既存checkpoint名を再作成する用途は拒否します。通常はlatestを使用してください。Stage2はCのparameter/buffer hashをcheckpoint保存時と復元時に検証します。学習config・gene/cell順序・data/source hash不一致は停止します。古い9/15のbare state_dictだけを新データに適合するものと見なしません。

既存predictionの3-fold評価だけ再実行:

```bash
python -m work.20261010_model_comparison.cli evaluate \
  --condition C02_hill_after_linear_blend \
  --prediction work/20261010_model_comparison/runs/default/C02_hill_after_linear_blend/predictions/ode_raw_t49.h5ad \
  --genes work/20261010_model_comparison/runs/default/C02_hill_after_linear_blend/predictions/gene_ids.txt \
  --label ode_raw_t49_retry
```

比較CSV・図の再作成、scVelo結果の追加:

```bash
python -m work.20261010_model_comparison.cli summarize
python -m work.20261010_model_comparison.cli summarize \
  --extra-result work/20261009_reproduce/runs/scvelo_stochastic/evaluation
```

結果は `runs/default/comparison/{comparison.csv,status.json,metrics_*.png,heatmap_*.png,scatter_*.png}`。未実行条件はstatusに残し、値を捏造しません。field別のnorm/nonfinite/near-zero診断はpredictionsとvelocity図のディレクトリに残ります。

追加モデル:

```bash
python -m work.20261010_model_comparison.cli launch \
  --condition X_lowrank_additive --dry-run
python -m work.20261010_model_comparison.cli launch \
  --condition X_lowrank_additive --device cuda
# 同じStage1からfreeze有無だけを変更:
python -m work.20261010_model_comparison.cli train \
  --condition X_stage1_hill_additive_freeze_true --device cuda
python -m work.20261010_model_comparison.cli train \
  --condition X_stage1_hill_additive_freeze_false --device cuda
```

個別入口 `python work/20261010_model_comparison/scripts/train.py ...` 等も同じCLIです。追加config JSONを `--condition path/to/config.json` で指定できます。数理構造を新規定義せず、registryの実装済みfield/compositionを選びます。

## Artifact layout / limits

```text
runs/data/                       # shared data manifest, mapped GRN and coverage
runs/default/shared_stage1.json  # immutable final-EMA identity for every E/control
runs/default/<condition>/
  effective_config.json, provenance.json
  checkpoints/latest.json, modelNNNNNN.pt
  logs/                         # losses, gradients, branch norms, failures
  samples/generated.h5ad, metadata.json
  predictions/<branch>_t49.h5ad, .json, *_diagnostics.json, gene_ids.txt
  metrics/<branch>_t49/          # official per-fold and mean/std tables
  figures/{training,generative,velocity}/
```

- 1 process/GPU per condition。DDP/FP16を新設していません。旧スクリプトのunqualified importsはscoped bindingで解消し、既存module bindingを復元します。
- PCA OT・trajectory occupation OT・kNN置換loss・learnable-forward・旧dual optimizerはsourceを確認しregistryへ登録済みですが、新データのcache/別process用adapterは未実装です。近似MSEへ置換していません。
- historical FactorMLP/MoE等は独立したsource・式・必要なmask/state adapterをinventoryへ記録しています。June版と同一classだとは扱いません。
- 実データHVG/GRN対応率、GPU性能、実データsampleの安定性、4指標の実値はremote実行まで未確認です。
- 旧A01 checkpointを再利用する場合、過去に記録されなかった勾配履歴は復元できません。新規A01学習ではsource loopへ観測hookだけ追加して記録します。

## ソース保護の移送

`preflight` の保護baselineはこの作業開始時のローカル状態です。既存ユーザー変更7ファイルも含みます。remoteのclean mainにsuiteだけ移送した場合は、差分を確認してremote用baselineを新規ファイルへ保存し、`verify_sources.py --baseline ...` で前後を比較してください。初期audit証跡を書き換える必要はありません。実行時checkpointにはremoteで実際にimportしたsource hashが保存されます。

```bash
python work/20261010_model_comparison/scripts/verify_sources.py \
  --capture work/20261010_model_comparison/runs/remote_protected.json
# テスト/実装確認後:
python work/20261010_model_comparison/scripts/verify_sources.py \
  --baseline work/20261010_model_comparison/runs/remote_protected.json
```
