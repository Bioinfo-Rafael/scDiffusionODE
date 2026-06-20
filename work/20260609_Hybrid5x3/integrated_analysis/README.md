# integrated_analysis — per-model / integrated UMAP 解析

`work/20260609_Hybrid5x3/runs/` 配下の **20 構成（5 ODE branch × 4 variant）の generated サンプル**を使い、
**2 系統**の UMAP 可視化を作る独立した追加解析。

```
geneode / lowrank / lincomb / matsum / lora   ×   none / ratio_reg / scale_model_x / scale_model_ml_emb
```

## 方針（重要）

- **既存 pipeline / training / sampling / visualization のコードは一切変更しない**。
  このディレクトリ（`integrated_analysis/`）の中だけで完結する。
- 既存コードの import すらしない（**torch 不要**）。各 run の成果物ファイル
  （`samples.npz` / `exp_config.json` / checkpoint の存在）だけを読む。
- training / sampling は再実行しない。`runs/` 内は読むだけ（削除・上書きしない）。
- どの run（どの timestamp dir）を参照したかを **必ず JSON / CSV に残す**（再現性）。

## 2 系統の違い

| | A) per-model 個別比較 | B) integrated 統合比較 |
|---|---|---|
| 目的 | 各モデルの generated が real のどこに乗るかを**モデル別に**見る | 全モデルの generated を**まとめて**real と比較する |
| real | **各モデル 50000**（共通サブサンプル） | **全部（無制限、~150k）** |
| generated | **そのモデルだけ 3000** | **各モデル 500**（20 で合計 ~10000） |
| UMAP | **モデルごとに独立**に計算（real+そのモデルのみ） | **1 つ**の UMAP に real + 全 gen |
| 出力 | `per_model/<label>.png` ×20 + `per_model/all_models_facets.png` | `integrated/` に overlay 3 種 + 横並び summary |

> per-model は real を 20 回 UMAP するので**重い**。integrated は real ~150k の 1 回が重い。

## ファイル

| ファイル | 役割 |
|---|---|
| `integrated_umap_utils.py` | helper（run 探索・選択 / real+gen 読込 / PCA→neighbors→UMAP / 配色 / 描画 / metadata）。notebook・CLI 共有 |
| `run_integrated_umap.py` | 一括 CLI（`--dry_run` あり） |
| `tune_integrated_umap.ipynb` | 調整用 notebook。cell 1 で per-model / integrated 設定を切替 |
| `outputs/` | 出力（`<timestamp>/` ・dry-run は `<timestamp>_dryrun/`）。git 管理外 |

## 使い方（CLI）

```bash
cd /home/suzuki/Projects/scDiffusion-github/work/20260609_Hybrid5x3   # runs/ のある場所
# （ローカルは .../scDiffusion/work/20260609_Hybrid5x3）

# 探索だけ（read-only。selected/skipped と使う予定の run・config preview）
python integrated_analysis/run_integrated_umap.py \
  --runs_root runs --output_root integrated_analysis/outputs --run_suffix ALL100k --dry_run

# 本番（指定デフォルト）
python integrated_analysis/run_integrated_umap.py \
  --runs_root runs --output_root integrated_analysis/outputs --run_suffix ALL100k \
  --per_model_real_cells 50000 --per_model_gen_cells 3000 \
  --integrated_real_cells 0 --integrated_gen_per_model 500 --seed 0

# 片方だけ作る
python integrated_analysis/run_integrated_umap.py ... --only per_model
python integrated_analysis/run_integrated_umap.py ... --only integrated
```

主な引数（`--help` で全部）:

| 引数 | 既定 | 意味 |
|---|---|---|
| `--runs_root` | `runs` | runs ディレクトリ |
| `--output_root` | `integrated_analysis/outputs` | 出力の親 |
| `--run_suffix` | `ALL100k` | この suffix を含む timestamp dir を**優先**（無ければ最新） |
| `--data_dir` | Embryonic.h5ad | real data。`local_paths.resolve_path` でローカル解決 |
| `--per_model_real_cells` | 50000 | A) real 件数 |
| `--per_model_gen_cells` | 3000 | A) generated 件数（モデルごと） |
| `--integrated_real_cells` | 0 | B) real 件数（**0 = 全部**） |
| `--integrated_gen_per_model` | 500 | B) generated 件数（モデルごと） |
| `--only` | （両方） | `per_model` / `integrated` 片方だけ |
| `--n_pcs`/`--n_neighbors`/`--min_dist` | 50/15/0.5 | UMAP |
| `--annotation_priority` | Superclass,Subclass,ClassAnn,celltype,final_annotation | real の色付け列の優先順位 |
| `--seed` | 0 | 乱数シード |
| `--dry_run` | off | 探索だけ |

出力先: `integrated_analysis/outputs/<YYYYMMDD_HHMMSS>/`

## 使い方（notebook）

`tune_integrated_umap.ipynb` を **`integrated_analysis/` をカレントにして**開き、上から実行。

- cell 1 に **per-model 用**（`PER_MODEL_REAL_CELLS=50000` / `PER_MODEL_GEN_CELLS=3000`）と
  **integrated 用**（`INTEGRATED_REAL_CELLS=0` / `INTEGRATED_GEN_PER_MODEL=500`）の設定。
- `LIGHT_MODE = True`（既定）で軽く回る（per_model_real=4000 / integrated_real=20000）。フルは `False`。
- cell 5 = A) per-model、cell 6 = B) integrated。

## 出力物

`outputs/<timestamp>/` に:

**A) per_model/**
| ファイル | 内容 |
|---|---|
| `<config_label>.png`（×20） | real を **Superclass 色**、そのモデルの generated を **赤で固定**して上に重ねる。凡例（Superclass + Generated）を図外に |
| `all_models_facets.png` | 上記 20 枚を 1 枚に（各 panel は独立 UMAP・real 50000 + gen 3000）。figure 全体に Superclass+Generated の凡例 1 つ |

**B) integrated/**
| ファイル | 内容 |
|---|---|
| `integrated_real_annotation.png` | real を annotation 色（Superclass）、generated を濃い灰で重ねる（凡例つき） |
| `integrated_origin.png` | Real（灰）vs Generated（赤）の二値（凡例つき） |
| `integrated_branch_marker_variant_color.png` | generated を **branch=marker・variant=color** で分解表示。`config_label.split("__",1)` で branch（元モデル: geneode/lowrank/lincomb/matsum/lora → marker）と variant（正則化: none/ratio_reg/scale_model_x/scale_model_ml_emb → color）に分解。branch 凡例（marker）と variant 凡例（color）の 2 つ |
| `integrated_summary_side_by_side.png` | 上 3 種を横に並べた 3-panel summary（real_annotation / origin / branch_marker_variant） |

> 注: 「generated を config_label で 20 色塗り分ける」図（`integrated_by_model.png`）は廃止。代わりに branch×variant 分解版を使う。

**metadata / config（出力 dir 直下）**
| ファイル | 内容 |
|---|---|
| `selected_runs_config.json` | **採用した run の完全な記録**（下記） |
| `selected_runs.csv` | 同内容の表 |
| `skipped_runs.csv` | skip した構成と理由 |
| `run_config.json` | この実行のパラメタ全部 |
| `color_map_real_annotation.csv` | real annotation（Superclass）→ 色 |
| `variant_color_map.csv` | variant（正則化）→ 色（branch_marker_variant 用） |
| `branch_marker_map.csv` | branch（元モデル）→ marker（branch_marker_variant 用） |
| `model_color_map.csv` | config_label → 色（参考。by_model 廃止につき現在は未使用） |
| `per_model_counts.csv` / `integrated_gen_counts.csv` | 使用 cell 数 |

## config_label の分解（branch × variant）

`config_label = "<branch>__<variant>"`。`integrated_branch_marker_variant_color.png` ではこれを分解して可視化:

- `ode_branch = config_label.split("__", 1)[0]`（元モデル: geneode / lowrank / lincomb / matsum / lora）→ **marker**
- `variant   = config_label.split("__", 1)[1]`（正則化: none / ratio_reg / scale_model_x / scale_model_ml_emb）→ **color**

例: `lowrank__scale_model_ml_emb` → branch=`lowrank`, variant=`scale_model_ml_emb` /
`lora__ratio_reg` → branch=`lora`, variant=`ratio_reg`。
対応は `branch_marker_map.csv`（branch→marker）と `variant_color_map.csv`（variant→color）に保存。

## `selected_runs_config.json` の意味

「どのモデルのどのディレクトリを参照したか」を完全に残すファイル。`models.<config_label>` に:

```json
{
  "created_at": "...", "runs_root": "...", "run_suffix": "ALL100k",
  "data_dir": "...", "data_dir_resolved": "...", "annotation_col": "Superclass",
  "selection_rule": "latest timestamp dir (preferring run_suffix) with samples.npz, exp_config.json, checkpoint",
  "models": {
    "geneode__none": {
      "run_dir": "<abs>", "run_dir_rel": "runs/geneode__none/<ts>",
      "sample_path": "<abs>", "exp_config_path": "<abs>",
      "checkpoint_path": "<abs>", "checkpoint_step": 100000, "checkpoint_kind": "model",
      "loss_path": "<abs|null>", "pipeline_command_path": "<abs|null>",
      "passed_over": [{"timestamp_dir": "<newer-but-incomplete>", "reason": "missing samples.npz"}]
    }
  },
  "skipped": { "<label>": {"reason": "missing samples.npz"} }
}
```

- 各パスは **絶対パスと相対パス（`*_rel`）の両方**を残す。
- `passed_over`: より新しいが条件を満たさず飛ばした timestamp dir。

### run 選択ルール

`runs/<config_label>/` の timestamp dir を**新しい順**に見て、**最初に下記すべてを満たす run**を採用:

1. `sample/*/SampledData/*.npz`
2. `train/*/exp_config.json`
3. `train/*/checkpoints/hybrid5x3/model*.pt`（step 最大を優先）、無ければ `ema_0.9999_*.pt`（step 最大）

- `--run_suffix` 指定時はその suffix を**含む** dir を優先（無ければ最新）。
- 条件を満たす run が無い config は `skipped_runs.csv` / JSON に理由付きで記録。
- 新しいが不完全な run（TEST・途中失敗）は飛ばして次の完全な run を採用。

## real data の annotation

色付け用 annotation 列は自動選択（優先順位 `Superclass` → `Subclass` → `ClassAnn` → `celltype` → `final_annotation`）。
使った列名は `selected_runs_config.json` の `annotation_col`、色対応は `color_map_real_annotation.csv` に保存。

## よくある失敗と対処

| 症状 | 原因 / 対処 |
|---|---|
| ある config が `skipped`（`missing samples.npz`） | sample がまだ無い / 失敗。`runs/<label>/<ts>/sample/*/SampledData/samples.npz` を確認 |
| `missing exp_config.json` | train が途中で落ちた。`train/*/exp_config.json` を確認 |
| `missing checkpoint` | `train/*/checkpoints/hybrid5x3/` に `model*.pt`/`ema_0.9999_*.pt` が無い |
| `FileNotFoundError: real data h5ad` | `--data_dir` が解決できない。`local_paths` が無い環境では絶対パスで指定 |
| UMAP が重い / 時間がかかる | per-model は 20 回 UMAP するため特に重い。`--per_model_real_cells` を下げる、`--only integrated` で片方だけ、notebook は `LIGHT_MODE=True` |
| generated が `'cell_gen' が無い` で skip | `samples.npz` に `cell_gen` キーがあるか確認 |
| 遺伝子次元不一致で skip | generated の遺伝子数が real（既定 1024）と違う。同じ前処理の real h5ad を `--data_dir` に |

## 注意

- 出力は `outputs/` に画像と metadata だけ。巨大な中間 AnnData は保存しない。
- `outputs/` `logs/` `__pycache__/` `.ipynb_checkpoints/` は `.gitignore` 済み。
