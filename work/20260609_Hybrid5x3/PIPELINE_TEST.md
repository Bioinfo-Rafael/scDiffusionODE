# パイプライン一括テスト（5 ODE × 4 hybrid変種 = 20）

`train → sample → viz(loss/params/eval_io/velocity)` を全構成で一気通貫テストする手順。
実体は **`run_all_pipelines.sh`**（各構成で `run_pipeline_5x3.py` を呼ぶだけ）。

## マトリクス（20 構成）

| | 説明 | 渡すフラグ |
|---|---|---|
| `none` | 通常の内分比 blend（penalty なし） | `--hybrid_norm_mode none` |
| `ratio_reg` | 比の正則化（log-norm-ratio penalty） | `--hybrid_norm_mode ratio_reg` |
| `scale_model_x` | scale_model mode・scale 入力 = 遺伝子ベクトル `x` | `--hybrid_norm_mode scale_model --scale_model_type simple --scale_input_source x` |
| `scale_model_ml_emb` | scale_model mode・scale 入力 = Cell_Unet 中間表現 `ml_emb` | `--hybrid_norm_mode scale_model --scale_model_type simple --scale_input_source ml_emb` |

× ODE 枝 5（`geneode / lowrank / lincomb / matsum / lora`）= **20**。

> scale_model は ODE/ML 出力を L2 正規化して方向を作り、`scale_model(scale_in, t)` が予測した
> **scalar scale (B,1)** を掛ける mode。詳細は `../../ODE/ode_20260609_scalemodel.py` と
> `../../ODE/ode_20260609_hybrid5x3.py` の `UnifiedODEMLHybrid._forward_scale_model`。

## 0. 前提（重要）

```bash
# conda env scdiffusion の python を最優先にする（SMA venv が PATH 上の python を隠すため）
export PATH="$HOME/miniconda3/envs/scdiffusion/bin:$PATH"
cd work/20260609_Hybrid5x3
```

- **必ず `bash` で実行**（このマシンの対話シェルは zsh。zsh だと文字列フラグが
  word-split されず argparse が `unrecognized arguments` で落ちる）。
- `--data_dir` / `--edge_tsv_path` 未指定でも `local_paths.resolve_path` でローカル解決される。

## 1. 全 20 構成を実行

```bash
bash run_all_pipelines.sh
```

env で上書き可（既定は最小設定）:

| 変数 | 既定 | 意味 |
|---|---|---|
| `STEPS` | 3 | `--lr_anneal_steps`（>=2 で loss 曲線/複数 checkpoint） |
| `BS` / `SBS` | 8 / 8 | train / sample の batch |
| `DIFF` | 1000 | `--diffusion_steps`（train/sample/viz で揃える。>=20） |
| `NSAMP` | 8 | 生成サンプル数 |
| `MAXCELLS` / `AMAX` | 300 / 150 | velocity / eval の cell 数 |
| `NT` | 4 | eval の t 点数 |
| `BRANCHES` | 5 枝全部 | 実行する ODE 枝（空白区切り） |
| `SKIP` | (なし) | viz 一部を飛ばす（例 `velocity`） |
| `DRY` | (なし) | `DRY=1` でコマンド表示のみ（dir も作らない） |

例:
```bash
DRY=1 bash run_all_pipelines.sh                       # 20 コマンドだけ確認
STEPS=5 MAXCELLS=300 bash run_all_pipelines.sh        # 少し大きめ
SKIP=velocity bash run_all_pipelines.sh               # velocity を飛ばして高速化
```

末尾に `OK / FAIL` サマリが出る。

## 2. どれか 1 モデルだけ全パイプライン実行したいとき

### (a) その ODE 枝の 4 変種すべて
```bash
BRANCHES="lora" bash run_all_pipelines.sh
# → lora__none / lora__ratio_reg / lora__scale_model_x / lora__scale_model_ml_emb
```

### (b) 特定の 1 (モデル, 方法) だけ — `run_pipeline_5x3.py` を直接
```bash
# 例: lora × scale_model(ml_emb)
python run_pipeline_5x3.py \
  --ode_branch lora --hybrid_norm_mode scale_model \
  --scale_model_type simple --scale_input_source ml_emb \
  --lr_anneal_steps 3 --batch_size 8 --diffusion_steps 1000 \
  --num_samples 8 --sample_batch_size 8 \
  --max_cells 300 --analyze_max_cells 150 --num_t_points 4

# 例: geneode × ratio_reg（既存方法）
python run_pipeline_5x3.py --ode_branch geneode --hybrid_norm_mode ratio_reg \
  --lr_anneal_steps 3 --batch_size 8 --diffusion_steps 1000 --num_samples 8

# 例: 通常 blend
python run_pipeline_5x3.py --ode_branch matsum --hybrid_norm_mode none \
  --lr_anneal_steps 3 --num_samples 8
```

- `--ode_branch ∈ {geneode, lowrank, lincomb, matsum, lora, plain}`
- `--hybrid_norm_mode ∈ {none, ratio_reg, scale_model, normed_learned_scale(deprecated)}`
- `scale_model` のときだけ `--scale_model_type simple --scale_input_source {x|ml_emb}` を付ける。
- `--dry-run` でコマンド確認、`--skip_viz` で viz 省略、`--skip velocity,eval_io` で一部のみ。

## 3. 出力

```
work/20260609_Hybrid5x3/runs/{model}/{YYYYMMDD_HHMMSS}/
  train/  {ts}_train/checkpoints/<name>/model*.pt, ema_*, loss_details.csv, exp_config.json
  sample/ {ts}_sample/SampledData/*.npz
  viz/    loss/  params/<ts>_viz/  eval_io/<ts>_analyze/  velocity/
```

- `{model}` は `{branch}__{none|ratio_reg|scale_model_x|scale_model_ml_emb}`
  （scale_model は入力ソース別に dir 分離）。
- `runs/` は `.gitignore` 済み（生成物は commit されない）。

## 4. 結果確認の例

```bash
# 全構成の OK/FAIL（run_all_pipelines.sh の末尾サマリでも分かる）
ls -d runs/*/                                   # 構成一覧
ls runs/lora__scale_model_ml_emb/*/viz/params/*_viz/   # 図一覧

# scale_model 構成は checkpoint に scale_model.* が入り、param_dist_misc.png に出る
python -c "import torch,glob;ck=sorted(glob.glob('runs/lora__scale_model_ml_emb/*/train/*_train/checkpoints/*/model0000*.pt'))[-1];sd=torch.load(ck,map_location='cpu',weights_only=False);print([k for k in sd if k.startswith('scale_model.')])"
```
