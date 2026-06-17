# Hybrid 5 枝 × hybrid mode 一括実行（20260609）

5 ODE 枝 × hybrid mode を同条件で一括学習・比較する。

```
ODE 枝 (5)        : geneode | lowrank | lincomb | matsum | lora
hybrid mode (4)   : none（通常） | ratio_reg（比正則化） | scale_model（方向×scalar scale）
                    ＋ normed_learned_scale（deprecated・互換のため温存）
baseline (2)      : baseline_cellunet | baseline_geneode_blend
```

- **旧 17 マトリクス**（§3 `run_experiments_5x3.py` / §8 `test_all_models.sh`）= 5×{ratio_reg, normed_learned_scale, none}
  + baseline 2 = 17（不変）。
- **新 20 マトリクス**（§9 `run_all_pipelines.sh`）= 5×{none, ratio_reg, scale_model+x, scale_model+ml_emb}（scale_model 含む）。

`r = 1 − t/(T−1)`。**`train_util.py` / `ode_20260421`（GeneODE）は無編集**、`Cell_Unet.forward` も無変更
（`forward_with_features` を追加しただけ）。`ode_20260609_*` は scale_model / `decompose_W` 追加で拡張。
`GeneODE` と `build_math_field` を import 再利用。

---

## 1. ファイル

| ファイル | 役割 |
|---|---|
| `../../ODE/ode_20260609_hybrid5x3.py` | 統合 `UnifiedODEMLHybrid`（`ode_model(x,t)`・**4 mode**: none/ratio_reg/scale_model/normed_learned_scale）+ `build_ode_branch` + `build_denoiser` |
| `cell_train_5x3.py` | 引数で枝・mode を選ぶ統合 train。`exp_config.json` 出力（scale_model 系 `--scale_model_type/--scale_input_source/--ode_input_source/--scale_hidden/--scale_eps` も保存） |
| `cell_sample_5x3.py` | `exp_config.json` から再構築してサンプリング（scale_* も復元） |
| `run_experiments_5x3.py` | **17 matrix の launcher**（dry-run / only / skip-existing / gpu / summary） |
| `run_all_5x3.sh` | launcher を起動する薄い .sh |
| `run_pipeline_5x3.py` | **単一 config を cell_train→cell_sample→全 viz で一貫実行**（最小テスト用 → §7） |
| `run_paths.py` | **出力構造ヘルパ**（3 dir 共有）。`{work}/runs/{model}/{YYYYMMDD_HHMMSS}/{train,sample,viz}` を採番/逆算 |
| `viz/` | 可視化スイート（役割別: `plot_loss` / `plot_params` / `eval_model_io` / `plot_velocity_umap`）。`viz/README.md` 参照 |
| `smoke_test_5x3.py` | 17 config の construction/forward/backward・t 伝播・hook 検証（torch のみ・高速） |
| `test_all_models.sh` | **全 17 モデルを train→sample→viz で一括 e2e テスト**（旧マトリクス: 5×3 + baseline2。§8） |
| `run_all_pipelines.sh` | **5 ODE × 4 hybrid変種 = 20 を一括 e2e テスト**（新マトリクス: none/ratio_reg/scale_model×2）。手順は `PIPELINE_TEST.md` |
| `PIPELINE_TEST.md` | 上記 20 構成の実行手順 + **1 モデルだけ全パイプライン実行する方法** |
| `../../ODE/ode_20260609_scalemodel.py` | `scale_model` mode 用 `SimpleScalarScaleModel`（方向×scalar scale）+ `build_scale_model` |
| `test_scale_model_forward.py` | scale_model 追加の軽量 forward test（実データ不要・13 項目） |

> ODE モデル本体（`ode_20260609_hybrid5x3.py` ほか 2 つ）の解説は `../../ODE/README_20260609.md`。
> hybrid_norm_mode は `none` / `ratio_reg` / `scale_model`（+deprecated `normed_learned_scale`）。
> 全構成テスト・単一モデル実行の手順は **`PIPELINE_TEST.md`**。

---

## 2. 統合の肝（5 枝 × 4 mode を 1 経路で）

`UnifiedODEMLHybrid.forward` は ODE 枝に **t を渡す**:
```
ode_out = self.ode_model(x, t)   # GeneODE は t 無視、4 fields は t 使用 → 全 5 枝で正しく動く
ml_out  = self.ml_model(x, t, y)
```
これで「field を hybrid に差すと t=0 で時刻依存が消える」問題を回避（GeneODE 専用だった
`ODE_ML_HybridNorm` の制約を解消）。mode は **4 つ**（`scale_model` を追加）:

| mode | 出力 | `ode_model._cached_ratio_reg` |
|---|---|---|
| none | `r·ode+(1−r)·ml` | None |
| ratio_reg | `r·ode+(1−r)·ml` | training 時 ratio penalty |
| **scale_model** | `scale·(r·ode_unit+(1−r)·ml_unit)`、`scale=scale_model(scale_in,t)∈(B,1)` | None（ratio penalty なし） |
| normed_learned_scale | `exp(log_scale)·(r·ode_unit+(1−r)·ml_unit)` | None（**deprecated**・温存） |

- **scale_model**（新規）: ODE/ML 出力を L2 正規化して**方向**を作り、`scale_model` が予測する **scalar scale (B,1)**
  を全 gene に掛ける。`scale_in` は `--scale_input_source ∈ {x（遺伝子ベクトル）, ml_emb（Cell_Unet 中間表現）}`。
  `--scale_model_type simple` のとき `SimpleScalarScaleModel` を生成。`Cell_Unet.forward_with_features()` で
  `ml_out` と `ml_emb` を 1 回で取得（二重 forward 回避）。`--ode_input_source` 既定 `none`（将来用 `x_ml_emb` は
  未対応 branch で `RuntimeError`）。
- `normed_learned_scale` は共有 scalar `exp(log_scale)`（deprecated）。
- `self.ode_model = branch` なので `train_util` の off_mask_penalty hook が **無変更で発火**。
  plain Cell_Unet は ode_model を持たないので不発火（baseline）。
- off-mask 構造罰則（SoftReg）は全 mode 直交で共通に効く。

---

## 3. 実験 matrix（17、launcher が定義）

- 15 = {geneode,lowrank,lincomb,matsum,lora} × {ratio_reg,normed_learned_scale,none}
  （`SoftReg=True`, `ode_reg_lambda=5`）。exp_id 例 `lowrank__ratio_reg`。
- `baseline_cellunet`: `ode_branch=plain`（Cell_Unet のみ、ODE/hook なし、reg なし）。
- `baseline_geneode_blend`: `ode_branch=geneode, mode=none, SoftReg=False, ode_reg_lambda=0`
  （GeneODE+Cell_Unet を素朴に内分比 blend、penalty なし）。

**共通条件（全 17 統一）**: data_dir / edge_tsv_path / batch128 / microbatch −1 / lr 1e-4 /
weight_decay / lr_anneal_steps 30000 / uniform / diffusion 1000 / seed 1234。
出力は `runs/{exp_id}/{YYYYMMDD_HHMMSS}/train/` に分離（17 実験は 1 つの batch 時刻を共有）、model_name は `hybrid5x3_{exp_id}`。

---

## 4. dry-run / 実行

```bash
cd work/20260609_Hybrid5x3

# dry-run（17 コマンド生成・出力path一意・必須ファイル存在を検証、実行しない）
bash run_all_5x3.sh --dry-run
#   or: python run_experiments_5x3.py --dry-run

# 全 17 実行（GPU 0）
bash run_all_5x3.sh --gpu 0

# 一部だけ / 既存スキップ
python run_experiments_5x3.py --only lowrank__ratio_reg,baseline_cellunet
python run_experiments_5x3.py --skip-existing
```

ログ・サマリ:
- 各実験 stdout/stderr → `runs/{exp_id}/{YYYYMMDD_HHMMSS}/train/train.log`（先頭に COMMAND / GIT / START）。
- `experiments/hybrid5x3_{TS}/summary.csv` と `summary.jsonl`:
  exp_id / ode_branch / hybrid_norm_mode / model_name / output_dir / exit_code /
  start / end / duration / git_hash / trained_model_path / log / command。

---

## 5. テスト

- ローカル（torch 不在）: 全 `.py` を `py_compile`、`.sh` を `bash -n`、`--dry-run` で matrix 検証。
- 学習マシン: `python smoke_test_5x3.py`（17 config の forward/backward、field の t 伝播、
  ratio cache 出し分け、log_scale grad、hook 発火/plain 不発火）。
- `python test_scale_model_forward.py`（scale_model の軽量 forward test・実データ不要。`forward_with_features`、
  scale=(B,1) 正値、ratio_reg/none/scale_model の 3 mode、x_ml_emb 未対応エラー等を検証）。

---

## 6. 注意

- `train_util.py` は無変更（hook は全 hybrid 枝で発火、plain で不発火）。`Cell_Unet.forward` も無変更。
- checkpoint: normed の log_scale（top-level param）は保存。sample は exp_config.json で同型再構築。
- **scale_model**: `scale_model.*`（`SimpleScalarScaleModel` の MLP param）が checkpoint に増える。復元は
  exp_config.json の scale_* で同型再構築すれば `scale_model.*` も一致ロード（ratio_reg/none の checkpoint には
  `scale_model.*` は無く、復元時も生成されない）。
- 実行環境のパス（`sys.path` / `data_dir` / `edge_tsv_path`）は `/home/suzuki/...`（学習マシン）。
  別マシンでは launcher 内 `COMMON` と各 py 冒頭の `sys.path` を書換える。
- baseline_geneode_blend は matrix の `geneode__none`（SoftReg=True 付き）とは別物
  （こちらは penalty 完全オフの素朴版）。

---

## 7. cell_train → cell_sample → viz 一貫テスト（`run_pipeline_5x3.py`）

17 matrix とは別に、**単一 config を train→sample→全 viz で一気通貫**する最小テスト。
出力構造は **`runs/{ode_branch}__{hybrid_norm_mode}/{YYYYMMDD_HHMMSS}/{train,sample,viz}/`**（記述的 model 名・
日付 dir は **時刻 HHMMSS まで**含むので実行ごとにユニーク＝同日の再実行も完全分離。work 直下を散らかさないため runs/ 配下）。

```bash
cd work/20260609_Hybrid5x3

# 最小テスト（lora / ratio_reg。lr_anneal_steps は 2 以上にすること）
python run_pipeline_5x3.py --ode_branch lora --hybrid_norm_mode ratio_reg \
  --lr_anneal_steps 4 --batch_size 8 --diffusion_steps 1000 \
  --num_samples 16 --max_cells 120 --analyze_max_cells 100 --num_t_points 4

# コマンドだけ確認: --dry-run / viz を省く: --skip_viz / 一部だけ: --skip velocity,analyze
# branch ∈ {geneode, lowrank, lincomb, matsum, lora, plain}、mode ∈ {ratio_reg, normed_learned_scale, none}
# run 識別名: --name <名前> → 出力が runs/{model}/{YYYYMMDD_HHMMSS}_<名前>/ になる（同条件の振り分けに便利）
```

**実行条件の記録**: `exp_config.json`（train dir 内）に **モデル/正則化条件に加え学習ハイパラ全部**を保存
（`run_name` / `batch_size` / `lr` / `lr_anneal_steps` / `weight_decay` / `ema_rate` / `save_interval` / `log_interval` /
`schedule_sampler` / `model_name` ＋ `command`=叩いた cell_train コマンドそのまま ＋ `all_args`=全引数スナップショット）。
pipeline 経由なら **`runs/{model}/{date[_name]}/pipeline_command.txt`** に pipeline コマンドも残る。
（sample 復元は従来キーだけ参照するので追加は無害・後方互換。）

**3 段の流れ**（pipeline が学習ログから model_path / `exp_config.json` / loss_details.csv を grep して次段へ渡す）:
1. `cell_train_5x3.py`（`--save_interval 1` 固定）→ `train/.../checkpoints/<name>/model00000{0..N}.pt` + `ema_*` + `loss_details.csv` + `exp_config.json`
2. `cell_sample_5x3.py`（`exp_config.json` で再構築）→ `sample/.../*.npz`
3. `viz/run_all_viz.py` が役割別 4 スクリプトを呼び、`viz/{loss,params,eval_io,velocity}/` に分離出力:
   - `plot_loss.py` → `viz/loss/`、`plot_params.py` → `viz/params/<ts>_viz/`、
     `eval_model_io.py` → `viz/eval_io/<ts>_analyze/`、`plot_velocity_umap.py` → `viz/velocity/`

**主な出力図**:
- `viz/loss/`: `loss_curves.png` / `loss_curves_simple.png`（**lr_anneal_steps≥2 で非退化**）
- `viz/params/<ts>_viz/`: `W_hist_t*.png` / `W_heat_t*.png`（**フル解像度・枠線なし**）/ `W_abs_vs_t.png` /
  `gamma_decay_hist.png` / `param_dist_W.png` / `param_dist_misc.png`（**学習ステップ横軸**、lora は W0+Δ_k×on/off）/
  `W_meanabs_t_vs_step.png` / `W_offmask_t_vs_step.png`（**t×学習ステップ**）/ `scale_vs_step.png`（normed のみ）
- `viz/eval_io/<ts>_analyze/`: `corr/`(corr_dim_vs_t, corr_scatter_t*) /
  `alignment_mse/`(alignment, mse_norm, norm_ratio) / `timestep_metrics.csv` / `analysis_config.json`
- `viz/velocity/`: `umap_analysis.png`(**real vs gen・凡例なし**) +
  `velocity_by_superclass/<Superclass>/` の velocity stream/arrow（scvelo compat shim でローカルでも生成可）

**3 dir を一括**（このディレクトリ含む 3 実験を最小設定で）:
```bash
bash work/run_all_20260609_pipelines.sh
# 最小サイズ上書き: STEPS=4 MAXCELLS=120 AMAX=100 NT=4 bash work/run_all_20260609_pipelines.sh
```

**注意**:
- `--lr_anneal_steps` は **2 以上**必須。1 だと loss 曲線が描けず、param 分布 / W t×step も checkpoint 1 点になる
  （pipeline は `save_interval=1` 固定なので **step 数 = checkpoint 数**）。
- `--data_dir` / `--edge_tsv_path` は未指定/`/home/suzuki/...` でも `local_paths.resolve_path` でローカル解決
  （リモート default・ローカル fallback）。`--diffusion_steps` は train/sample/viz で揃える（≥20、既定 1000）。
- velocity はローカル env（numpy≥1.24 / pandas≥2.0 + scvelo0.2.5）でも `plot_velocity_umap.py` 冒頭の
  **version-guarded 互換 shim** で stream/arrow が生成される（remote の互換 env では shim は no-op）。詳細は `viz/README.md`。

---

## 8. 全モデル一括 e2e テスト（`test_all_models.sh`）

このディレクトリの **全 17 モデル**（5 ODE 枝 × 3 mode = 15 + baseline 2）を train→sample→viz で
順に回し、OK/FAIL サマリを出す。各 config は `run_pipeline_5x3.py` を呼ぶ（出力は `runs/{model}/{YYYYMMDD_HHMMSS}/`）。

```bash
cd work/20260609_Hybrid5x3
export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"   # conda env python を最優先
DRY=1 bash test_all_models.sh                 # 17 コマンドだけ表示（dry-run）
bash test_all_models.sh                        # 全 17 を最小設定で実行
STEPS=4 MAXCELLS=120 bash test_all_models.sh   # サイズ上書き
```

- `smoke_test_5x3.py`（torch のみ・高速）は構築/forward/backward を 17 config 検証。`test_all_models.sh` は
  実データで train→sample→viz まで通す e2e 版。HybridNorm/MathMLP にも同名の `test_all_models.sh` がある
  （それぞれ 3 / 4 モデル）。
- `DRY=1` は各 pipeline に `--dry-run` を渡すだけ（ディレクトリは作らない）。

---

## 9. 新マトリクス: 5 ODE × 4 hybrid変種 = 20（scale_model 含む）クイックスタート

`hybrid_norm_mode` は **`none`（通常） / `ratio_reg`（比正則化） / `scale_model`**（+deprecated
`normed_learned_scale`）。`scale_model` は ODE/ML 出力を L2 正規化して方向を作り、`scale_model(scale_in,t)`
が予測する **scalar scale (B,1)** を掛ける mode（scale 入力は遺伝子ベクトル `x` か Cell_Unet 中間表現 `ml_emb`）。

> 旧 §8 `test_all_models.sh` は別マトリクス（5×3 + baseline2 = 17、`normed_learned_scale` 含む / scale_model 無し）。
> 本節（scale_model を含む 20 構成）の **完全な手順・結果確認は [`PIPELINE_TEST.md`](PIPELINE_TEST.md)**。

### 前提（毎回）
```bash
export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"   # conda env python を最優先（SMA venv が PATH を隠す）
cd work/20260609_Hybrid5x3
```
※ **必ず `bash` で実行**（対話シェルは zsh。zsh だと文字列フラグが word-split されず argparse が落ちる）。
※ `--data_dir` / `--edge_tsv_path` 未指定でも `local_paths.resolve_path` でローカル解決。

### 全 20 構成
```bash
bash run_all_pipelines.sh                       # 全20を最小設定で train→sample→viz
DRY=1 bash run_all_pipelines.sh                 # コマンドだけ表示（dir 作らない）
STEPS=5 MAXCELLS=300 SKIP=velocity bash run_all_pipelines.sh   # サイズ/skip 上書き
```
env: `STEPS`(=lr_anneal_steps,≥2) / `BS` `SBS` / `DIFF`(=diffusion_steps) / `NSAMP` / `MAXCELLS` `AMAX` `NT` /
`BRANCHES`(枝を限定) / `SKIP`(viz一部省略) / `DRY`。末尾に OK/FAIL サマリ。

### どれか 1 モデルだけ全パイプライン実行
**(a) その ODE 枝の 4 変種すべて:**
```bash
BRANCHES="lora" bash run_all_pipelines.sh
# → lora__none / lora__ratio_reg / lora__scale_model_x / lora__scale_model_ml_emb
```
**(b) 特定の 1 (モデル, 方法) だけ — `run_pipeline_5x3.py` を直接:**
```bash
# scale_model（ml_emb 入力）
python run_pipeline_5x3.py --ode_branch lora --hybrid_norm_mode scale_model \
  --scale_model_type simple --scale_input_source ml_emb \
  --lr_anneal_steps 3 --batch_size 8 --diffusion_steps 1000 --num_samples 8

# 通常 blend / 比正則化（scale 引数は不要）
python run_pipeline_5x3.py --ode_branch matsum --hybrid_norm_mode none      --lr_anneal_steps 3 --num_samples 8
python run_pipeline_5x3.py --ode_branch geneode --hybrid_norm_mode ratio_reg --lr_anneal_steps 3 --num_samples 8
```
- `--ode_branch ∈ {geneode, lowrank, lincomb, matsum, lora, plain}`
- `scale_model` のときだけ `--scale_model_type simple --scale_input_source {x|ml_emb}` を付ける
  （`none`/`ratio_reg` は既定 `--scale_model_type none` でOK）。
- `--dry-run`（確認のみ）/ `--skip_viz` / `--skip velocity,eval_io` も可。

### 出力
`runs/{branch}__{none|ratio_reg|scale_model_x|scale_model_ml_emb}/{YYYYMMDD_HHMMSS}/{train,sample,viz}/`
（`runs/` は .gitignore 済み。scale_model は checkpoint に `scale_model.*` が入り `viz/params/.../param_dist_misc.png` に出る）。

---

## 10. 旧 `work/20260215_embryonic` 条件の再現（lambda 振り）

旧実験（`VaryLambda20260224.sh → train_20260123.sh → cell_train_soft20260123_hvg1024.py`）の正体は
**`GeneODE(soft=True)` + `Cell_Unet` を `ODE_ML_Hybrid` で線形内分比 blend したもの**:
`out = r·ode_out + (1−r)·ml_out`（`r = 1 − t/(T−1)`、**penalty なし**）。
これは新フレームワークの **`--ode_branch geneode --hybrid_norm_mode none`** に一致する。
`ode_reg_lambda`(5/50/500) / `SoftReg` / `ode_reg_norm='l1'` は **TrainLoop レベルの ODE 正則化**で、
`cell_train_5x3.py` でも同じく TrainLoop にそのまま渡される（=意味が完全一致）。lambda を振っていただけ。

### キーワード対応表

| 旧（`train_20260123.sh` / VaryLambda） | 新キーワード | 値 |
|---|---|---|
| GeneODE + Cell_Unet | `--ode_branch` | **geneode** |
| ODE_ML_Hybrid（線形 blend・penalty なし） | `--hybrid_norm_mode` | **none** |
| `--SoftReg True` | `--SoftReg` | **True** |
| `--ode_reg_lambda 5/50/500`（振っていた条件） | `--ode_reg_lambda` | **5 / 50 / 500** |
| `DATA=Embryonic.h5ad` / edge tsv | `--data_dir` / `--edge_tsv_path` | いずれも既定でOK（未指定で `resolve_path` がローカル解決） |
| `BATCH_SIZE=128` | `--batch_size` | **128** |
| `DIFFUSION_STEPS=1000` | `--diffusion_steps` | **1000** |
| `LR_ANNEAL_STEPS=100000` | `--lr_anneal_steps` | **100000** |
| `NUM_SAMPLES=10000` | `--num_samples` | **10000** |
| `SAMPLE_BATCH_SIZE=50` | `--sample_batch_size` | **50** |

既定で**そのまま一致**（パイプラインは露出しないが `cell_train_5x3.py` の既定が旧と同値）:
`lr=1e-4` / `ema_rate=0.9999` / `weight_decay=0.0001` / `ode_reg_norm='l1'`。変更不要。

### ⚠️ 注意点
- **`run_pipeline_5x3.py` は内部で `--save_interval 1 --log_interval 1` を固定**（短時間テスト前提）。
  100000 step だと checkpoint 10 万個になる（旧は `save_interval=5000 / log_interval=1000`）。
  **本番長時間 run を忠実に再現するなら下記 (B) で `cell_train_5x3.py` を直接叩く**。

### (A) パイプラインで（短め検証向け・lambda 3 条件）
```bash
export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"
cd work/20260609_Hybrid5x3
for L in 5 50 500; do
  python run_pipeline_5x3.py \
    --ode_branch geneode --hybrid_norm_mode none \
    --SoftReg True --ode_reg_lambda $L --name lambda$L \
    --batch_size 128 --diffusion_steps 1000 --lr_anneal_steps 100000 \
    --num_samples 10000 --sample_batch_size 50
done
# ※ --name で日付 dir 末尾に _{name} が付く → runs/geneode__none/<日時>_lambda5/ のように lambda 別に分離
```

### (B) 忠実再現（`save_interval=5000` を効かせる・train/sample を直接実行）
```bash
export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"
cd work/20260609_Hybrid5x3

python cell_train_5x3.py \
  --ode_branch geneode --hybrid_norm_mode none \
  --SoftReg True --ode_reg_lambda 5 --ode_reg_norm l1 \
  --batch_size 128 --lr 1e-4 --diffusion_steps 1000 \
  --lr_anneal_steps 100000 --save_interval 5000 --log_interval 1000 \
  --output_dir runs/geneode__none/train
# → 出力の TRAINED_MODEL_PATH= と EXP_CONFIG= を控える

python cell_sample_5x3.py \
  --model_path <TRAINED_MODEL_PATH> --exp_config <EXP_CONFIG> \
  --num_samples 10000 --batch_size 50 --diffusion_steps 1000 \
  --output_dir runs/geneode__none/sample
```
lambda は `--ode_reg_lambda 5 / 50 / 500` で振る。ローカルは CPU なので 100000 step は重い（リモート CUDA 推奨）。
