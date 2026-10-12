# Mouse Gastrulation model comparison — 20261010

既存モデル・loss・DDPM samplingを直接importする比較環境です。初回 **17条件**と拡張 **26条件**（計43 config）を用意しました。本学習・実データsampling・VeloEV実評価は自動実行していません。

- 全34 work directoryの調査: [audit/WORK_DIRECTORY_AUDIT.md](audit/WORK_DIRECTORY_AUDIT.md)
- モデルの式・class・状態: [MODEL_INVENTORY.md](MODEL_INVENTORY.md)、[CSV](MODEL_INVENTORY.csv)、[registry.json](registry.json)
- 学習法・歴史的既定値・変更点: [TRAINING_METHOD_INVENTORY.md](TRAINING_METHOD_INVENTORY.md)
- 直接importと新規adapterの区分: [SOURCE_REUSE_MAP.md](SOURCE_REUSE_MAP.md)
- 検証結果・未実施工程: [audit/VALIDATION_REPORT.md](audit/VALIDATION_REPORT.md)

## 比較条件

この17条件で調べるのは、**CellUNetによる発現データの生成に、ODEをどう組み込むと結果が変わるか**です。A〜Eは組み込み方、末尾の数字はODEの種類や合成方法などの違いを表します。

まず、ここでいう「学習」はvelocityの正解値に合わせる学習ではありません。全17条件で、ノイズを加えた発現量 `x_t` を入力し、元の発現量 `x_0` を予測します。これが **START_X** です。生成性能の評価とは別に、学習後の各出力をvelocityと仮定してCBDirなども評価します。

### A〜Eで何を変えるか

以下では **U = CellUNetの出力、F = ODEの出力、H = 両者を合成した出力**と書きます。Uは条件グループCと区別するための記号です。

| グループ | 比較したいこと | 学習するもの・初期状態 | 発現量の予測と細胞生成に使う出力 |
|---|---|---|---|
| **A：CellUNet単独** | ODEを使わない基準 | CellUNetを30,000更新。互換な既存の学習済みモデルがあれば再利用 | U |
| **B：ODE単独** | CellUNetをODEに置き換えるとどうなるか。ODEの式による違いはあるか | ODEをランダム初期値から30,000更新。CellUNetは存在しない | F |
| **C：両方を同時学習** | CellUNetとODEを最初から組み合わせるとどうなるか | 両方をランダム初期値から同時に30,000更新 | H |
| **D：出力を近づける補助損失** | CellUNetの出力をODEの出力に近づける制約の強さで、結果がどう変わるか | 両方をランダム初期値から30,000更新。発現量予測の誤差はUに、出力一致の誤差はUとFの両方にかかる | **Uのみ。Fを混ぜて生成する条件ではない** |
| **E：学習済みCellUNetにODEを追加** | 既に学習したCellUNetを固定し、ODEだけ追加学習するとどうなるか | A01の学習済みCellUNetを全E条件で共有・固定。ODEをランダム初期値から10,000更新 | H |

「固定（freeze）」は重みを更新しないこと、「同時学習（joint）」は両方の重みを更新することです。1更新は128細胞のミニバッチによるoptimizerの1ステップで、1 epochではありません。

### 17条件の対応表

| 条件ID | ODEの種類 | 組み合わせ方・変更する値 |
|---|---|---|
| A01 | なし | CellUNet単独 |
| B01 | simple_softplus | ODE単独 |
| B02 | hill_after_linear | ODE単独 |
| B03 | centered_signed_hill | ODE単独 |
| B04 | shifted_hill_rho | ODE単独 |
| C01 | simple_softplus | 同時学習・Blend |
| C02 | hill_after_linear | 同時学習・Blend |
| C03 | simple_softplus | 同時学習・Additive |
| C04 | hill_after_linear | 同時学習・Additive |
| D01 | hill_after_linear | 出力一致の係数 λ = 0 |
| D02 | hill_after_linear | 出力一致の係数 λ = 0.001 |
| D03 | hill_after_linear | 出力一致の係数 λ = 0.1 |
| D04 | hill_after_linear | 出力一致の係数 λ = 1 |
| E01 | simple_softplus | CellUNet固定・Blend |
| E02 | hill_after_linear | CellUNet固定・Blend |
| E03 | simple_softplus | CellUNet固定・Additive |
| E04 | hill_after_linear | CellUNet固定・Additive |

ODE名は式の違いです。`simple_softplus` は遺伝子入力を線形結合してsoftplusを通す生成項、`hill_after_linear` はその後にHill型の飽和を入れる生成項を使います。`centered_signed_hill` と `shifted_hill_rho` は遺伝子間の各作用を異なる符号付きHill型応答で表します。いずれも発現量に比例する減衰項を持ちます。厳密な式は [MODEL_INVENTORY.md](MODEL_INVENTORY.md) を参照してください。

### BlendとAdditiveの違い

- **Blend：`H = r F + (1-r) U`**。CellUNetとODEを重み付きで混ぜます。ODEの比重が大きいほどCellUNetの比重は小さくなります。
- **Additive：`H = U + r F`**。CellUNetをそのまま残し、ODEの出力を加えます。

どちらも `r = max(0, 1-t/500)` です。`t`は拡散のノイズ時刻（0〜999）で、生物学的な時間ではありません。`t >= 500`ではODEの寄与が0になります。velocity評価の`t=49`では`r=0.902`なので、Blendは`0.902 F + 0.098 U`、Additiveは`U + 0.902 F`です。

### Dのλは何に掛かるか

Dの損失は概略として **`MSE(U(x_t,t), x_0) + λ MSE(U(x_t,t), F(x_t)) + GRN制約`** です。λが大きいほど、CellUNetとODEの出力の一致を強く求めます。ODEは固定教師ではなく、両方が更新されます。

**D01のλ=0では、ODEに発現量復元やCellUNetとの一致を学ばせる損失がかかりません。** ODEにはGRN制約とoptimizerのweight decayが残ります。そのため、D01の`ode_raw`を「発現量の復元を学習したODE」と解釈してはいけません。

### 図の1行は「学習条件 × 取り出した出力」

例えば `C02_hill_after_linear_blend / ode_raw_t49` は、**C02として同時学習したモデルから、ODE側の出力だけを取り出して評価した行**です。B02のようにODEだけを単独学習したモデルではありません。

| 図のラベル | 評価に渡すもの | 対象グループ |
|---|---|---|
| `cellunet_raw_t49` | CellUNet単体の出力 U(X,49) | A、C、D、E |
| `ode_raw_t49` | ODE単体の出力 F(X) | B、C、D、E |
| `hybrid_raw_t49` | 合成後の出力 H(X,49) | C、E |

ここでXはノイズを加えていない前処理済みの実細胞の発現量です。`raw`は出力からXを引いたり積分したりせず、そのまま使うという意味です。**17学習条件から37行の評価結果**が生じます（A:1、B:4、C:12、D:8、E:12）。Eの`cellunet_raw`は全条件で同じ固定A01を取り出すので、同じ評価設定なら同じ結果になります。

### どの条件同士を比べるか

- **ODEの式の違い**：B01〜B04。Hybrid内ならC01対C02、C03対C04など。
- **混ぜ方の違い**：C01対C03、C02対C04。Eも同じ対応。
- **出力一致の制約の強さ**：D01〜D04の同じ出力同士。
- **初めから同時学習か、学習済みCellUNetへの追加学習か**：C01対E01など、同じ末尾番号同士。ただし初期値と学習量も違うので、freezeだけの効果とは言えません。

freezeの有無だけを比較する追加条件は `X_stage1_hill_additive_freeze_true/false` です。どちらも同じA01から開始します。追加26条件は`--extended`で選ぶもので、通常の`--all`が実行する17条件には含まれません。

### 共通設定と更新数の注意

全条件で89,267細胞・1,024 HVG、normalized spliced linear X、1000-step linear diffusion、batch size 128、AdamW（lr=1e-4、weight decay=1e-4）、EMA=0.9999、seed=1234を使います。ODEを持つ条件では、既知GRNにない結合へ罰則を課すsoft constraintを使います（結合を強制的に0へ固定するhard maskではありません）。生成は既存のancestral DDPM、clip=False、nw=0.5、3,000細胞です。

EはA01の30,000更新に加え、ODEを10,000更新します。学習率を減らすスケジュールの基準は30,000更新のままなので、「10,000更新で学習率を0まで減らす」設定ではありません。全E条件は`shared_stage1.json`で同じA01 checkpoint/hashを使うことを検証します。

歴史的実装からの変更として、Dは8/30のEPSILON予測をSTART_X予測に、100,000更新を30,000更新に変更しています。また10/08のODE SGDではなくAdamWを使います。ソースのsoft constraint内部係数5は保持し、June系の「外側で5倍」と区別します。

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

## Harmony補正したjoint UMAPを別名で作成

全17条件の既存`figures/generative/joint_umap.h5ad`を読み、保存済みの共同PCA（通常50成分）へHarmonyを適用します。batchは`sampling_origin`（Real / Generated）です。条件ごとに補正し、補正後PCAの先頭40成分・近傍15・seed=1234で近傍グラフを作成し、UMAPを計算します（min_dist=0.5、spread=1.0）。全モデルを一つの共通座標へ統合する処理ではありません。

Harmonyは`harmonypy==0.0.10`を使用し、theta=2、lambda=1、sigma=0.1、最大10反復です。完全な設定・ライブラリversion・入力hashは出力JSONに保存します。[ScanpyのHarmony説明](https://scanpy.readthedocs.io/en/stable/generated/scanpy.external.pp.harmony_integrate.html)と同様に、PCAと近傍グラフ作成の間に補正を入れます。

元の`joint_umap.png`、`joint_umap.h5ad`、`umap_metadata.json`は変更しません。同じディレクトリに次の3ファイルを作成します。

- `joint_umap_harmony.png`
- `joint_umap_harmony.h5ad`
- `umap_harmony_metadata.json`

Real/Generatedの差そのものを補正するため、補正後に重なることだけで生成精度が高いとは判断できません。元の図と併せて解釈してください。発現量、学習済みモデル、velocity benchmarkの値は変更しません。

remoteで既存のscdiffusion環境を有効にしてから、補正用の隔離環境で実行します。GPUは不要です。

```bash
cd /home/suzuki/Projects/scDiffusion-github
git pull --ff-only origin main
conda activate scdiffusion
python -m venv --system-site-packages work/20261010_model_comparison/runs/.venv-harmony
work/20261010_model_comparison/runs/.venv-harmony/bin/python -m pip install --no-deps harmonypy==0.0.10
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
nohup work/20261010_model_comparison/runs/.venv-harmony/bin/python \
  work/20261010_model_comparison/scripts/replot_harmony.py \
  --run-root work/20261010_model_comparison/runs/retry_20261012_052933 \
  > work/20261010_model_comparison/runs/retry_20261012_052933/harmony.log 2>&1 < /dev/null &
```

全17条件の入力が揃っているかを先に確認し、足りなければ欠けた条件名を表示して停止します。生成や学習を自動でやり直すことはありません。

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
