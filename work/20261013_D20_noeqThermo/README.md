# D20 ODE の非平衡熱力学解析

既存D20 checkpointの **ODE branch / clean real X / `ode_raw_t49`** を、既存の `20260821_noeqThermo` に接続する。新規学習・生成細胞の作成は行わない。標準はEMA。`paper_literal`, **D=0.2, dt=0.2** をSmoke/Fullで共通にする。

## 調査した実装と確認結果

参照元はローカルcheckoutの以下のコード。GitHubの指定URLはこのセッションのWeb取得ではアクセスできなかったため、ネットワーク同期せずローカル実装を調査した。

- `../20261010_model_comparison/common.py`, `adapters/{checkpoint,data,model,velocity,objectives,source_compatibility}.py`
- `../20261010_model_comparison/configs/{base.json,model_expansion/D20_lora_consistency_lambda1.json}`
- `../../ODE/ode_20260609_mathmlp.py` の `LoRAField`
- `../20260821_noeqThermo/src/noeqthermo/{pipeline,config,dynamo_bridge,landscape,plotting}.py`

| 確認項目 | 実装上の定義 |
|---|---|
| 対象run | `work/20261010_model_comparison/runs/model_expansion_20261012/D20_lora_consistency_lambda1` |
| 最終checkpoint | `checkpoints/model030000.pt` が通常の命名。`latest.json` を既存 `checkpoint.resolve` で解決し、bundleの `step == effective_config.total_steps` を検証 |
| latest形式 | `{"checkpoint":".../model030000.pt","sha256":"...","step":30000}`。既存loaderは旧絶対パスが消失した場合、同じrunのcheckpoints内の同名ファイルだけを探索しhashを検証 |
| 復元 | `checkpoint.load` → `model.build` → 既存 `LoRAField`。strict state_dict、config一致、cell/gene IDと順序、data/GRN/source hashを検証 |
| raw / EMA | bundleの `raw` / `ema`。CLI `--weights ema` が既定。EMAがなければ停止し、rawへ暗黙に切り替えない |
| D20 | family=`consistency`, ode=`lora`, lambda=1, hybrid=`none`。標準LoRAはK=8、rank=16、time_dim=64、hidden=256、decay有効 |
| 学習目的 | START_X CellUNet損失 + consistency lambda1 + GRN soft penalty。`external_5` でoff-mask係数5を外側で適用。推論時にはlossを計算しない |
| velocity | 既存exportと同じ `model.ode_model(x, t)`。`t` は `(batch,1)` のlong整数49。1000-step diffusionの元のindexで、実時間49や0.049への変換ではない。LoRAはtime-conditioned |
| 入力 | 実細胞の `training.h5ad.X`。scale=`spliced_independent_normalize_total_1e4`。q_sample、引き算、ODE積分、出力再スケールなし |
| 元データ | config上は `data_preparation/20261007/data/MouseGastrulation.h5ad`、89,267細胞・53,801遺伝子。splicedを独立にnormalize_total(1e4)、log1pコピーでSeurat HVGを選び、モデルへ渡すXは線形scaleの1,024 HVG |
| ID順序 | manifestのmetadataとtraining h5adの `obs_names` / `var_names`、checkpointを完全一致で検証。`var.gene_name` は別名なのでモデルのgene IDに使わない |
| export再利用 | `run/predictions/ode_raw_t49.h5ad` と同名JSONを自動検出（EMA）。既存 `velocity.definition` をimportし、branch、時刻、weights、checkpoint/data/IDのhash、IDリスト・順序、本体hashを検証。再利用時もcheckpointの復元と互換性検証を実行するがvelocityを重複推論しない |

**ローカルで未確認のもの:** 対象runのcheckpoint、effective_config、実training h5ad、対象velocity exportは2026-10-13時点のローカルcheckoutに存在しなかった。実際のID一覧、実ファイルのhash、全annotation別の細胞数、学習済みD20の数値結果はリモート上で検証する。小さいLoRA fixtureによるローカルテストと、学習済みD20の検証は区別する。

## 実行

リモート上の既存scDiffusionODE環境に、`../20260821_noeqThermo/requirements.txt` のDynamo 1.4.1 / matplotlib 3.7.5を用意する。scVelo 0.2.5等の既存環境を引き継ぐ。スクリプトは依存関係のインストール、Git操作、SSHを実行しない。

以下は**ユーザーがリモート上で実行するコマンド**。repository rootを作業ディレクトリにする。

```bash
# checkpoint・data・annotationと先頭実細胞の推論を確認（熱力学計算なし）
bash work/20261013_D20_noeqThermo/scripts/run_remote.sh --check-only --device cuda

# 短時間のパイプライン確認
bash work/20261013_D20_noeqThermo/scripts/run_remote.sh --mode smoke --device cuda

# 長時間本解析
bash work/20261013_D20_noeqThermo/scripts/run_remote.sh --mode full --device cuda
```

同じコマンド・同じ出力先でSDE checkpointからresumeする。SmokeとFull、rawとEMAは別ディレクトリになる。設定、入力、weights、code、主要package version、deviceが変わったら新しい出力先を指定する。埋め込み・投影velocity・連続場の検証用grid hashも再開時に照合する。

任意の場所のcheckpointやmanifestを指定できる:

```bash
python work/20261013_D20_noeqThermo/scripts/run_analysis.py \
  --mode smoke \
  --run-dir work/20261010_model_comparison/runs/model_expansion_20261012/D20_lora_consistency_lambda1 \
  --data-dir work/20261010_model_comparison/runs/data \
  --checkpoint /path/to/model030000.pt \
  --weights ema --device cpu \
  --output-dir work/20261013_D20_noeqThermo/outputs/smoke_custom
```

`--run-dir` は `effective_config.json` を持つ既存run。既存loaderが互換性監査を書き込むため、従来どおり `20261010_model_comparison/` 配下を使う。`--data-dir` は既存 `manifest.json` のあるディレクトリ。manifest内のtraining/GRNパスは実環境で存在する必要がある。パス移設時もhashやIDを書き換えて検証を回避しない。`--velocity-export` はh5adと隣接JSONの明示指定。不一致exportは黙って無視せず停止する。raw指定では自動検出のEMA exportを使わず、rawで計算する。

## 細胞選択とLAP

既定は `selection.mode="prefer_erythropoietic"`。`Superclass == Erythropoietic` が実データにあれば優先する。なければ全細胞へfallbackし、別annotationからErythropoieticを推測しない。全データの利用可能なカテゴリと件数はcheck-onlyのJSON、実行ログ、model manifestに保存する。適用した選択と標本化seedは `common/embedding_metadata.json` に保存する。Smokeでは選択後最大300細胞、Fullでは上限なし。

ローカルの既存調査資料 `data_preparation/20261007/reports/five_celltype_counts.csv` には、Blood progenitors 1=623、Blood progenitors 2=2460、Erythroid1=2929、Erythroid2=1106、Erythroid3=2697（計9815）の記録がある。これは今回のh5adを読み込んだ結果ではなく、Superclassとの自動対応にも使用しない。`celltype` と `stage` の存在は既存data loaderが検証する。

全細胞を明示する場合は設定の `selection.mode="all"`。任意ラベルに限定する場合は `mode="filter"`, `obs_key`, `values` を指定する。存在しないkey、空の選択は停止する。例（ユーザーがラベルの意味と対象を確認して指定する）:

```json
"selection": {
  "mode": "filter",
  "obs_key": "celltype",
  "values": ["Erythroid1", "Erythroid2", "Erythroid3"],
  "celltype_key": "celltype"
}
```

LAPは既定で実データの `celltype` ラベルから、選択後 `minimum_cells_per_state`（Smoke 5 / Full 20）を満たす状態を、件数の多い順に `max_states` 件まで選ぶ。2状態未満なら理由を記録してskipする。方向は推定せず、各状態ペアをA→B / B→Aの両方向で計算する。確認済みのラベルだけを解析する場合は `lap.endpoint_labels` に2つ以上指定する。Smokeは最大2方向、Fullは最大12方向、`max_pairs` は偶数。LAPの一方向でも失敗したら全体を成功と表示せず、metadataを残して非zero終了する。

## 数値設定と出力

| 設定 | Smoke | Full |
|---|---:|---:|
| trajectories | 16 | 400 |
| steps | 600 | 1,000,000 |
| burn_in | 100 | 100,000 |
| grid_size | 32 | 100 |
| seed | 1234 | 1234 |
| D / dt | 0.2 / 0.2 | 0.2 / 0.2 |
| checkpoint interval | 200 | 10,000 |

境界は既存の反射境界。増分は `sqrt(2*D*dt)*N(0,I)`。`calibrate_common_sde` は既存関数名だが、`paper_literal` 分岐は指定のD/dtをそのまま返し、近傍距離等は診断情報のみ。UMAPはPCA最大50、近傍15、n_pcs最大40、min_dist=0.5。モデル入力に追加のlog変換は行わない。

`landscape.py` は変更せずimportする。既存定義のPは格子binの確率質量（総和1）、forceはbin内の軌跡平均を用いたflux。連続場のgrid値も別途保存する。未訪問binの生Uは+inf、mean forceはNaNのまま保持し、勾配・表示用Uだけ既存のcapを使う。J/PはP>0で計算。離散微分ではchain ruleが厳密でないため、force分解残差を診断値として保存する。Smoke通過は定常分布への収束を意味しない。

`outputs/<mode>_<weights>/models/D20_lora_consistency_lambda1/` に:

- `observed_gene_velocity.npz`: velocity、cell_ids、genes（checkpointのgene ID）
- `observed_umap_velocity.npz/.csv`: 座標、UMAP velocity、cell ID
- `dynamo_vecfld_umap.npz`, `dynamo_vecfld_metadata.json`, `fixed_points_umap.csv`, `curl_umap_cells.csv`
- `landscape_flux_arrays.npz`: P、U、mean/gradient/rotational force、flux、curl、訪問mask、grid。既存CSV中間配列も保存
- `simulation/`: SDE checkpointとRNG状態
- `least_action_paths.csv`, `least_action_metadata.json`: 経路・action・timeと成功/skip/失敗理由
- PNG 01–09（velocity、topography、curl、P、3D U、mean force、gradient force、flux、overlay）。LAP成功時のみ10を追加
- `model_manifest.json`: checkpoint path/hash、raw/EMA、時刻、scale、データhash、config/hash、ID hash、実際の細胞選択、各計算の診断

出力rootには `analysis_manifest.json`, `resume_identity.json`, `numerical_validation.json`。`common/erythropoietic_fixed_umap.*` という既存ファイル名は互換性のため維持するが、**実際の解析対象はmetadataのselectionに従う**。全細胞fallbackをErythropoieticと表示する図タイトルは使わない。

## 既存コードの変更と互換性

モデル・loss・data loader・velocity exporter・熱力学数式は変更していない。以下の追加引数/設定は省略時に旧動作を使う。

| ファイル | 変更理由 |
|---|---|
| `noeqthermo/config.py` | 明示的な細胞選択modeを追加。旧configはErythropoietic制限を維持 |
| `noeqthermo/pipeline.py` | 検証済みAnnData受け渡し、選択の記録、var_names ID指定、velocity providerへのcell ID受け渡し、Superclass列のないデータ対応 |
| `noeqthermo/dynamo_bridge.py` | 明示的LAP endpointを要求するoptionと往復ペアを保つoption |
| `noeqthermo/plotting.py` | field titleの任意指定。旧既定タイトルは維持 |
| `model_comparison/audit/model_expansion_source_compatibility.json` | 上記4ファイルのみ、変更前hash→変更後hashの厳密な互換性ペアを追加 |

互換性ファイルに汎用ignore、hash検証無効化、任意hash受入は追加していない。未知のsource変更は従来どおり停止する。他の未コミット変更がcheckpointのsourceと異なる場合も停止する。

## ローカル検証

```bash
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -m pytest \
  work/20261013_D20_noeqThermo/tests \
  work/20260821_noeqThermo/tests/test_landscape.py -q
```

テストは60細胞×8遺伝子の人工fixtureと元のLoRA実装の保存済みstateを使い、optimizer更新を一切行わない。raw/EMA差、final checkpoint復元、CellUNet/Hybrid未使用、t49直接出力一致、cell/gene順序、非finite/不正source/config/data/hashの拒否、export再利用、UMAP投影、Dynamo、SDE、10図、往復LAP、resumeを検証する。実D20 checkpointの読み込みは、上記のとおりローカルでは未実施。

2026-10-13 JSTの実行結果: **18 passed**（31.79秒）。上記テストに、SDEの40-step checkpoint後に人工中断し、再開後のcount/force累積配列が連続実行と完全一致するテストを含む。旧 `tests/smoke_model_pipeline.py` も60細胞×52遺伝子でpassed。D20 fixtureのend-to-endテストはSmokeのSDE設定（16 trajectories / 600 steps / D=dt=0.2）を使い、明示した2状態の往復LAPと10 PNGを生成した。Pythonコンパイル、shell構文、git diffの空白チェックも通過。

検証環境は `/tmp/d20-noeq-venv`（既存scdiffusion環境を参照する一時venv、Dynamo 1.4.1、matplotlib 3.7.5）。ログは `outputs/local_tests.log`, `outputs/legacy_smoke.log`。人工fixtureの図・数値データは `outputs/local_test_fixtures/test_end_to_end_and_resume0/output/` にあり、学習済みD20の結果ではない。outputsはGit対象外。

再現性のためDynamoをscVelo投影より先にimportし、乱数seedも固定する。Dynamo初回importがmatplotlib設定を変えると、scVeloのquiverベースのautoscaleが初回と再開で異なり得るため。この順序を固定した状態で投影velocityと連続場の厳密hash一致を確認した。
