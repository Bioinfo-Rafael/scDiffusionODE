# 20260911: terminal-step post-hoc analysis

既存の最終EMAを復元し、1000 reverse updatesの途中にあるclean predictionを解析する。
対象は20260803の12条件と20260816の4条件。新規実装・出力はこのディレクトリ内だけ。
学習・既存モデル定義・元runのmanifest更新は行わない。

固定値は **T=1000、M=20、S=100、N=20**。各モデル2000 trajectories。
`configs/models.json`と`src/settings.py`に設定をまとめ、production commandはこの値を検証する。
小規模fixtureのみN/Sを縮小してテストする。

## 再利用した実装

`<suite>`は`work/20260803_ODE_hill_exp`または`work/20260816`。
通常のimport名`common`/`models`の衝突を避けるため、モデルごとにfresh processで動かす。

| 目的 | reused source | function/class | modification |
| --- | --- | --- | --- |
| 保存configの読込・検証 | `<suite>/scripts/common.py` | `read_json`, `validate_config`, `validate_run_dir` | unchanged |
| configのbase/条件merge | 同上 | `load_experiment_config` | unchanged、復元fixtureで使用。本番は解決済み`exp_config.json`を読む |
| checkpoint選択 | `<suite>/scripts/sample.py` | `_choose_checkpoint` | unchanged。追加adapterでcompleted・30,000-step・EMAを検証 |
| checkpoint fallback | `<suite>/scripts/common.py` | `latest_checkpoint_bundle`, `discover_checkpoint_bundles`, `checkpoint_step`, `ema_rates` | unchanged。bundle探索は既存selector経由 |
| ordered gene loading/device | `<suite>/scripts/sample.py` | `_genes`, `_select_device` | unchanged |
| diffusion復元 | `<suite>/viz/analysis_helpers.py` | `build_diffusion` | unchanged |
| EMA/model復元 | 同上 | `load_model` | unchanged、strict loadingとfinite検査 |
| model構築 | `<suite>/models/factory.py` | `build_model_from_config` | unchanged、`load_model`経由 |
| state読込・wrapper除去 | `guided_diffusion/dist_util.py`, `ODE/ode_20260609_mathmlp.py` | `load_state_dict`, `clean_state_dict` | unchanged、`load_model`経由 |
| diffusion構築/timestep map | `guided_diffusion/script_util.py`, `respace.py` | `create_model_and_diffusion`, `SpacedDiffusion` | unchanged、`build_diffusion`経由 |
| reverse trajectory | `guided_diffusion/gaussian_diffusion.py` | `p_sample_loop_progressive` | unchanged、generator出力をmemmapへ保存 |
| Erythropoietic selection | `work/20260830/hematopoietic_viz/core.py` | `select_hematopoietic_subset` | unchanged、`superclasses=("Erythropoietic",)`を明示 |
| data結合・gene alignment | 同上 | `build_sampling_anndata`, `gene_names_from_adata`, `assert_gene_alignment`, `gene_order_hash` | unchanged |
| PCA/neighbors/UMAP | 同上 | `compute_common_umap` | unchanged、ndarray adapterのみ |
| plotting | `src/umap_adapter.py` | `_plot_embedding` | 小さいscatter表示adapter。既存plottingはtitle/filename固定のため直接reuse不可。origin/celltypeの意味は維持 |
| breakpoint | [arXiv:2608.14067v1](https://arxiv.org/html/2608.14067v1) | Eq. 7, Eq. 8, Algorithm 1, Appendix D | 数式から実装。公式コードは確認範囲で未発見 |

20260830の`runner.py`/`plotting.py`、20260817 vector-field analysisも調査した。
20260817のvector-field/velocity fitは今回不要。20260830のpackage全体のimportは
runnerを読み込むため、必要な`core.py`のみ薄い`importlib` adapterで読む。

## 論文対応と差分

正式なタイトル・著者・検索先と確認限界は[PAPER_PROVENANCE.md](PAPER_PROVENANCE.md)。
公式upstreamコードのコピー/importはなく、URL/commit/licenseはmetadataではnull。

`pred_xstart.reshape(N,S,M,G)`について、S方向の平均を取り、さらにgene方向の平均を
取るのがEq. 7のscore。結果は`[20,20]`。norm、MSE、ground truthは使用しない。
Eq. 8はoriginal diffusion timeで`i < breakpoint`と`i >= breakpoint`に分けた
**独立した2直線**の最小二乗。連続性制約は加えない。保存順はt降順なので、boundary
自身は先頭側segmentに含まれる。候補はsnapshot **2..17**の16点。

各groupの最小SSE点を求め、Algorithm 1に対応して20個のmedianをglobal breakpointとする。
medianが7.5などになる場合は丸めず保存する。reverse_step/diffusion_tは線形補間し、
`is_saved_snapshot=false`を記録する。global median自体には単一fitのSSEがないため
`SSE=null`とし、SSEは各groupのCSVに保存する。同率最小は保存順で最初の候補を採る。

論文のNはconditional forecastingのpilot inputs。本実装はunconditional scRNAなので、
**N=20 independent groups × S=100 independent trajectories**というreplicate grouping。
このscoreから生物学的品質の最適性が保証されるわけではなく、Stage 1を目視して判断する。
論文の20-step DDIMに対して、今回は既存ancestral samplerの1000 updatesを全て実行する。

## Trajectoryの意味・20時点

- `pred_xstart`: そのupdateで推定したclean prediction。breakpointとUMAPの入力。
- `sample`: そのupdateが完了した後のreverse-chain state。debug用に別arrayへ保存。
- `reverse_step`: 完了したdenoising update数（1-based progress）。
- `diffusion_t`: そのupdateに渡されたoriginal diffusion timestep（このrepoでは0-based）。

既存generatorは`range(start_time)[::-1]`を走査し、`p_sample`の**後**にyieldする。
従って50回目のyieldはt=950の処理後、1000回目はt=0の処理後。
`diffusion_t`は保存後stateのラベルではなく、今回処理したmodel timestepを指す。

| snapshot_index | reverse_step | diffusion_t |
| ---: | ---: | ---: |
| 0 | 50 | 950 |
| 1 | 100 | 900 |
| 2 | 150 | 850 |
| 3 | 200 | 800 |
| 4 | 250 | 750 |
| 5 | 300 | 700 |
| 6 | 350 | 650 |
| 7 | 400 | 600 |
| 8 | 450 | 550 |
| 9 | 500 | 500 |
| 10 | 550 | 450 |
| 11 | 600 | 400 |
| 12 | 650 | 350 |
| 13 | 700 | 300 |
| 14 | 750 | 250 |
| 15 | 800 | 200 |
| 16 | 850 | 150 |
| 17 | 900 | 100 |
| 18 | 950 | 50 |
| 19 | 1000 | 0 |

全1000 updates・20 snapshotsの完了を検証する。不完全なgeneratorやNaN/Infは失敗扱い。
最終sampleと最終pred_xstartのallcloseおよび最大絶対差をmetadataに記録する。
`t=0`では確率的noise項がなく、posterior meanはclean predictionに一致する。
CPU小規模テストで最終保存sampleと通常`p_sample_loop`のfinal outputの**完全一致**を確認した。

memmapの`pred_xstart.npy`/`sample_state.npy`はfloat32、shape `[2000,20,G]`。
row IDは`n*S+j`で、batch境界に関係なく連続順序を維持する。
1024 genesの場合は各array約156.25 MiB、2つで約312.5 MiB/model。
`--no-save-states`はrun_one_modelで利用可能。全trajectoryをPython listに蓄積しない。

## Stage 1 / Stage 2

**Stage 1: 20 snapshots → 20 independent UMAP fits。** 各fitはreal Erythropoietic
全細胞と、そのsnapshotの2000 clean predictionsのみ。各回で新しいAnnDataを作り、
既存関数がPCA(arpack,50)、neighbors(15,40 PCs)、UMAP(seed)を実行する。
小さいfixtureでは既存関数のdimension capが働く。新規normalize/log1p/scaleはない。
20図間の座標は比較対象ではない。contact sheetは既存PNGを並べただけ。

**Stage 2: user-selected snapshots only → one joint UMAP fit。** 独立commandでのみ
実行する。snapshot指定なしではargument error。real referenceは1回だけ入れ、指定
snapshotごとのgenerated cellsを連結する。generatedはsnapshot/stepで色分けし、
cell typeは常に`Generated (unconditional)`。生物学的cell typeを推定・付与しない。

```bash
python work/20260911/scripts/plot_selected_joint_umap.py \
  --result-dir <MODEL_RESULT_DIR> --snapshots 7,8,9,10,11
# 同じ選択をreverse updatesで指定する場合:
python work/20260911/scripts/plot_selected_joint_umap.py \
  --result-dir <MODEL_RESULT_DIR> --reverse-steps 400,450,500,550,600
```

Stage 2はStage 1目視後にユーザーが選んだ場合に実行する。`run_all`/`run_one_model`/
Stage 1 smokeはStage 2を呼ばない。異なるselectionは別subdirectoryに保存する。

## Run選択・実checkpoint復元確認

既存`scdiffusion`環境を使用する（PyTorch/Scanpy/NumPy/Pandas/Matplotlib/Pillowなど）。
以下のcommandはrepository rootで実行する。

```bash
conda activate scdiffusion
python work/20260911/scripts/discover_runs.py \
  --output work/20260911/results/run_inventory.json
python work/20260911/scripts/run_all.py --dry-run
python work/20260911/scripts/run_all.py --inspect-only
python work/20260911/scripts/run_all.py --restore-only --device cpu
```

`models.json`に16条件を列挙済み。`run_dir: null`はcompletedかつ最終30,000-step EMAの
candidateが**1個だけ**の場合に限り選択する。0個または複数なら候補を表示して停止し、
一部モデルだけを勝手にsamplingしない。複数のcompleted runsがある場合は、対象entryの
`run_dir`と必要に応じて`checkpoint`を明示する。未学習条件を除外するなら明示的に
`enabled: false`にする。`checkpoint`だけの指定は認めない。

```json
{
  "suite": "20260816",
  "experiment": "linear_centered_signed_hill",
  "enabled": true,
  "run_dir": "work/20260816/runs/linear_centered_signed_hill/<batch-id>",
  "checkpoint": null
}
```

既存selectorはmanifestのEMAを優先し、記載がなければcomplete checkpoint bundleを探索する。
manifestの保存先が旧環境を指す場合、勝手にpathを推測せず`checkpoint`を明示する。
既存run layoutは`<suite>/runs/<experiment>/<batch-id>`で、checkpointはそのrunの
`train/checkpoints/`以下が必須。raw model、途中step、completedでないrunは拒否する。
最終EMAだけ移設したrunはmanifestに実在するEMA pathを用意するかmappingで明示する。

データの移設時は`--data-path /absolute/Embryonic.h5ad`と
`--edge-path /absolute/tf_target_edges.tsv`を明示できる。元configは書き換えず、
effective configを結果に記録する。gene順序・元のtraining representationを維持すること。
全モデルで同じ移設先でない場合は`run_one_model`で個別指定する。

## GPU Stage 1実行

1モデル:

```bash
python work/20260911/scripts/run_one_model.py \
  --suite 20260816 \
  --run-dir work/20260816/runs/linear_centered_signed_hill/<batch-id> \
  --device cuda --batch-size 50
```

seedは元configを引き継ぐ。結果はtimestamp付きの新規directoryへ保存し、上書きしない。
異なるbatch sizeでは乱数消費順が変わるため、再現時はseedだけでなくbatch sizeも揃える。
既存scheduleを引き継ぎ、1000-step identity mapを検証する。DDIM/conditional設定は拒否する。

リモートへ取得して全モデルをbackground実行:

```bash
git fetch origin
git switch --track origin/feat/20260911-posthoc-terminal-step
# すでに同名のlocal branchがある場合は git switch feat/20260911-posthoc-terminal-step
git pull --ff-only origin feat/20260911-posthoc-terminal-step
conda activate scdiffusion
python work/20260911/scripts/discover_runs.py \
  --output work/20260911/results/run_inventory.json
# 複数candidateがあるentryはconfigs/models.jsonでmappingを設定する。
python work/20260911/scripts/run_all.py --restore-only --device cpu
mkdir -p work/20260911/results/logs
nohup python -u work/20260911/scripts/run_all.py --device cuda --batch-size 50 \
  > "work/20260911/results/logs/stage1_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
echo $!
```

`run_all`は全mapping/pathのpreflight後に、1モデルずつ別processでStage 1を実行する。
GPUを選ぶ場合は起動時に`CUDA_VISIBLE_DEVICES`を設定する。失敗時は停止し、後続modelは起動しない。
本実装作業ではこの重いcommandは実行していない。

## 出力と再現情報

```text
results/<suite>/<experiment>/<source-batch>_<UTC-timestamp>/
  metadata.json
  trajectory/
    pred_xstart.npy
    sample_state.npy
    snapshot_table.csv
    sampling_metadata.json
  breakpoint/
    score_curves.csv
    per_group_breakpoints.csv
    global_breakpoint.json
    breakpoint_fit_group_00.png ...
    breakpoint_summary.png
  umap_independent/
    snapshot_00_step_0050_t950.png ... snapshot_19_step_1000_t000.png
    snapshot_00_step_0050_t950.csv ...
    umap_independent_contact_sheet.png
    metadata.json
  umap_selected_joint/snapshots_<explicit selection>/  # 別command実行時だけ
```

metadataにはsource suite/experiment/run/config、元/effective config、EMA path/step/rate/hash、
data/edge path、edge hash、gene順序全体とhash、T/M/S/N、batch size、seed、device、時点表、
ソースcommit、Python/NumPy/PyTorch version、paper provenanceを保存する。各UMAPにも
実際のPCA/neighbors/UMAP設定・real選択・snapshot・生成細胞数を記録する。

`results/.gitignore`で結果全体を除外する。既存checkpoint/dataもrepoのignore policyに従う。
途中失敗のtrajectoryは`status=failed`として解析を拒否する。sampling成功後に解析だけ
失敗した場合は、未作成のstageを個別commandで実行できる。失敗して残ったstage directoryは
内容を確認して別名へ移す必要がある（自動上書き/resumeは行わない）。

```bash
python work/20260911/scripts/analyze_breakpoint.py --result-dir <MODEL_RESULT_DIR>
python work/20260911/scripts/plot_independent_umaps.py --result-dir <MODEL_RESULT_DIR>
```

## 作成ファイルと役割

| ファイル | 役割 |
| --- | --- |
| `README.md`, `PAPER_PROVENANCE.md`, `TEST_RESULTS.md` | 使用方法、reuse/論文対応、調査記録、検証結果 |
| `configs/models.json` | 16条件と明示的run/checkpoint mapping、固定parameter |
| `src/__init__.py` | post-hoc package |
| `src/settings.py` | T/M/S/N dataclassとpaper provenance |
| `src/source_imports.py` | source suite/import衝突の検出、薄いfile import adapter |
| `src/model_runs.py` | completed EMA選択、候補一覧、既存restore呼出し |
| `src/artifacts.py` | 結果JSON/CSV、出力先制約、保存trajectory検証 |
| `src/trajectory.py` | 20時点選択、既存generator→memmap、完了/finite検査 |
| `src/breakpoint.py` | score、2直線fit、median、CSV/JSON/図 |
| `src/umap_adapter.py` | 既存data/UMAP関数へのadapter、独立/joint制御と表示 |
| `scripts/_bootstrap.py` | direct script import path設定 |
| `scripts/discover_runs.py` | read-only候補一覧、suiteごとにsubprocess |
| `scripts/run_one_model.py` | 1モデルのinspect/restore/Stage 1 |
| `scripts/run_all.py` | 全mapping preflight、モデルごとのfresh process |
| `scripts/analyze_breakpoint.py` | 保存済みclean predictionのbreakpoint解析 |
| `scripts/plot_independent_umaps.py` | Stage 1、20 independent fits |
| `scripts/plot_selected_joint_umap.py` | Stage 2、明示選択だけのjoint fit |
| `tests/test_posthoc.py` | snapshot/grouping/fit/UMAP/restore unit tests |
| `tests/test_cli.py` | CLIとcoordinator失敗・実行順のテスト |
| `tests/restore_fixture.py` | 両suiteの実class・合成EMA・strict復元・source非変更検証 |
| `tests/smoke_pipeline.py` | 合成dataで20回の実Scanpy fitと出力smoke |
| `ruff.toml` | この新規codeだけのPython 3.9対応lint/format設定 |
| `results/.gitignore` | raw結果、図、テスト生成物の除外 |

## テスト

```bash
python -m unittest discover -s work/20260911/tests -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  python work/20260911/tests/smoke_pipeline.py \
  --output work/20260911/results/smoke_new
```

`smoke_new`は未作成directoryを指定する。学習済み実checkpointがローカルにないため、
今回の復元確認は**実source classと合成EMA fixture**によるもの。実checkpointの検証は
リモートの`--restore-only`で行える。詳細は[TEST_RESULTS.md](TEST_RESULTS.md)。
