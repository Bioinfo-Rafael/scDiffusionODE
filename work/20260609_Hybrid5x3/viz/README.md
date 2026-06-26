# 20260609 可視化スイート（viz/）

20260609 の新モデル（UnifiedODEMLHybrid / MathML_Hybrid / ODE_ML_HybridNorm、
ode_branch ∈ {geneode, lowrank, lincomb, matsum, lora, plain}）用の作図一式。
**役割別にファイルを分割**（loss / パラメタ・W / 入出力評価 / velocity UMAP）。

## ファイル（役割別）

| ファイル | 役割 / 出力 | run_all_viz の出力先 |
|---|---|---|
| `_restore.py` | 共有: config 正規化 + `build_denoiser` で復元 + checkpoint ロード + velocity 計算 + **`find_viz_checkpoints`**（init+EMA 選択）+ `extract_param_groups` | — |
| `model_introspection.py` | 共有: **effective W(x,t) 取得 + W metric**（off_on_ratio/sparsity/hoyer/effective_rank/top1pct）+ `decompose_W` + **branch-specific の step 横軸作図** | — |
| `plot_loss.py` | **loss 曲線**（loss_curves / loss_curves_simple。CSV のみ依存） | `{out}/loss/` |
| `plot_params.py` | **2 階層**: ① raw パラメタ分布（param_dist_W/misc・step 軸）② **effective operator W**（metric CSV + heatmap + hist + sparsity + top edges）+ branch-specific | `{out}/params/<ts>_viz/` |
| `eval_model_io.py` | **入出力評価**（ノイズ方向 metrics weighted/unweighted + 次元方向相関）。corr/ と alignment_mse/ に分離 | `{out}/eval_io/<ts>_analyze/` |
| `plot_velocity_umap.py` | **Superclass 別 velocity UMAP**（stream/arrow、配色2通り）+ **real vs gen UMAP**（`umap_analysis.png`・凡例なし） | `{out}/velocity/` |
| `plot_lincomb_a_embedding.py` | **単体（LinComb 専用）: 係数 a(x,t) の K 次元 cell embedding**（raw signed `a_k` の PCA/UMAP + 補助指標 abs/top_abs_k/entropy + 係数統計）。run_all_viz には未統合 | —（単体。`{run_dir}/viz/lincomb_a_embedding/<ts>_a_embedding/`） |
| `run_all_viz.py` | 上 4 つを 1 checkpoint に対し順次実行（`--skip loss,params,eval_io,velocity`。`--heatmap_*` を plot_params へ forward） | — |

3 dir の checkpoint は config json（`exp_config`/`field_config`/`hybrid_config`）を渡せば全て復元可能。
他 2 dir（HybridNormModes / MathMLPHybrid）には同名の 1 行 shim（runpy 委譲）あり。
出力構造は上位ランナーが決める **`{work}/runs/{model}/{YYYYMMDD_HHMMSS}/viz/{role}/`**（`work/20260609_Hybrid5x3/run_paths.py`）。
単体実行で `--output_dir` 未指定なら各スクリプトが `--model_path` から `{base}/viz/{role}` を逆算する。

## モデル結合の要点

- 復元は `_restore.build_model`（`ODE/ode_20260609_hybrid5x3.py: build_denoiser` 経由、5 枝 + plain、
  scale_model 系 config も読込み。旧 config に無ければ default で従来どおり復元）。
- **checkpoint 選択** `_restore.find_viz_checkpoints`: 1 列目 `model000000`(raw init) + EMA 最大 `max_ema_points` 点
  （`np.linspace(0, N, max_ema_points)` 近傍。N=lr_anneal_steps→total_steps→ema 最大 step）。dedup・不足でも落ちない。
  `save_interval=1` + `lr_anneal_steps≥2` で複数 checkpoint。

### plot_params は 2 階層

**① raw parameter visualization**（state_dict そのもの。学習が動いているか）
- `param_dist_W.png`: 静的 W 系を **per-k × on/off mask** で分割（geneode `W`、lincomb `expertW{k}`、
  matsum `A{k}`、lora `W0`+`Delta{k}`）。lowrank は静的 W 無し → W 行は出ない。**モデル間比較用ではない**（同一モデルの step 変化用）。
- `param_dist_misc.png`: gamma / b(expert_b) / coeff_net / U_net / V_net / time_emb / cellunet / **scale_model**（scale_model 使用時のみ）/ other。
  **gamma は misc**（W には入れない）。専用 `gamma_decay` 図は**出さない**。
- `scale_vs_step.png`: normed モデルの `log_scale`→`exp` を step に対して折れ線。

**② effective operator visualization**（`model.ode_model.compute_W(x,t)` の有効作用素。**モデル間比較はこちら**。`model_introspection.py`）
- `effective_W_metrics.csv`: 全モデル共通 metric（`mean_abs_all/on/off`・`off_on_ratio`・`off_mass_fraction`・
  `density_abs_gt_*`・`fraction_abs_lt_*`・`hoyer_sparsity`・`effective_rank`・`top1pct_mass_fraction`・`W_IS_EXACT`）。
- heatmap（y=diffusion t, x=checkpoint）: `W_off_on_ratio` / `W_off_mass_fraction` / `W_effective_rank` /
  `W_density_abs_gt_{eps}` / `W_meanabs` / `W_offmask` の `_t_vs_step.png`。
- `W_hist_by_step_t{0,T//2,T-1}.png`（代表 t で all/on/off を checkpoint 別に）、`W_sparsity_vs_step.png`、
  `top_W_edges.csv`（|W| 上位エッジ、gene 名）、`W_abs_vs_t.png`。
- lincomb の `compute_W` は **proxy**（`W_IS_EXACT=False`）→ タイトル/CSV/ログに `[PROXY]`。
- **full W_heat（n×n、ほぼノイズ）は既定 off**。`--plot_full_W_heat` 指定時のみ `W_heat_t{t}.png`。
- **heatmap だけ高粒度**: heatmap 用に別途 `find_viz_checkpoints(max_ema_points=--heatmap_max_ema_points 既定10)` ＋
  `--heatmap_n_t_grid`(既定20) で計算し `effective_W_metrics_heatmap.csv` を別名保存（通常の
  `effective_W_metrics.csv` / hist / sparsity / top_edges は `--max_ema_points`(既定5)・`--n_t_grid`(既定16) のまま）。

### branch-specific（`model_introspection`、**checkpoint step 横軸**）
全 checkpoint を build して描く（`--skip_branch_specific` で抑制）。各図に対応 CSV を保存。
- coeff: `{lincomb,matsum,lora}_coeff_vs_step.png`（signed `a_k` の t 方向 mean±std を line+band。互換名 `*_coeff_t_vs_k.png` も）。
- lincomb: `expert_contribution_vs_step`（`|a_k|·‖softplus(W_k x+b_k)‖`）/ `expertW_k_sparsity`(density>eps)。
- matsum: `Ak_pairwise_corr`（**Pearson** 相関・下三角・対角非表示・off-diag スケール。互換名 `*_cosine.png`）/ `Ak_sparsity`(density>eps)。
- lora: `delta_component_norm_vs_step`（per-k `‖a_kΔ_k‖_F` + `‖W0‖_F` 破線）/ `Delta_k_metrics`(Δ_k+W0 の density>eps)。
- lowrank: `singular_values_vs_t`（凡例外）/ `UV_mean_vs_t`（`mean(U)/mean(V)/mean(W=UVᵀ)`、norm でなく mean）。
- geneode（`decompose_W` 無）/ plain は branch-specific を skip。

### LinComb 係数 a の cell embedding（`plot_lincomb_a_embedding.py`・単体）
学習済み **LinComb** run（`ode_branch=lincomb`）専用。`V=Σ_k a_k(x,t)·softplus(W_k x+b_k)−decay(x)` の
係数 `a = ode_model._coeffs(x,t)` を `(B,K)` で抽出し、**K 次元 a-space の cell embedding** として可視化する。
主解析は **raw signed `a_k`**（softmax なし。`compute_W` は LinComb では proxy なので使わない）。
既存の学習/モデル/forward は不変、復元は `_restore.build_model`。**run_all_viz には未統合**（単体実行。
`ROLE="lincomb_a_embedding"` 定義済みで後から統合可）。LinComb 以外を渡すと `model_type` を見て分かりやすく停止。
- 入力 `--run_dir`: config を `train/**/exp_config.json`→`**/exp_config.json`、checkpoint を
  `train/**/checkpoints/**/{ema_*,model*}.pt`→`**/{ema_*,model*}.pt` の順で自動検出（**ema 優先・step 最大**、
  `--config`/`--model_path` で上書き）。`--t_values`(既定 `0,499,999`) / `--max_cells`(既定 50000、≤0 で全 cell) /
  `--use_noisy_xt`(既定 false=clean x0 を a(x0,t) に。true は `q_sample` の x_t) / `--color_cols`(既定 `Superclass,celltype,final_annotation`、存在する列のみ)。
- 出力 `{run_dir}/viz/lincomb_a_embedding/<ts>_a_embedding/`: t ごとに `lincomb_a_values_t{t}.csv`
  （`a0..aK-1` / `abs_a0..K-1` / `top_abs_k` / `top_abs_value` / `a_l2_norm` / `a_abs_sum` / `abs_a_entropy`）+
  `lincomb_a_space_t{t}.h5ad` + UMAP/PCA 図（`top_abs_k`=離散、entropy・l2・各 `a_k`/`abs_a_k`=連続、`--color_cols` の annotation）。
  全 t 集計 `lincomb_a_summary_by_t.csv`（per k: mean/std a・mean/std |a|・top fraction）+ `lincomb_a_mean_abs_by_t.png` +
  `lincomb_top_expert_fraction_by_t.png` + `summary.json`。**PCA/UMAP 失敗時も CSV/h5ad は必ず残し図だけ skip**（K<3 等は defensive に skip）。

### その他
- velocity = `model.ode_model(x, velocity_t)`（GeneODE は t 無視 / fields は t 使用）。`--velocity_t` 既定 0。
  scvelo は**元 velocity_by_Superclass.py の方式を踏襲**（`layers["velocity_ode"]`・`layers["X"]`、`velocity_graph(...,n_jobs=32)`→stream/arrow）。
- 次元方向相関（eval_model_io）= 各サンプルで `pearson(out, ε)`。ε は `diffusion.q_sample` の付加ノイズ。
  corr_scatter は等間隔 5 点 t、corr_dim_vs_t は 100 間隔。出力は `corr/` と `alignment_mse/` に分離。
- plain baseline（ode_model 無）は W/effective/branch/velocity 解析を自動スキップ（loss / eval_io の hybrid 出力のみ）。

### plot_params の主な CLI（既定は安全側）
- 選択: `--max_ema_points`(5) / `--heatmap_max_ema_points`(10) / `--n_t_grid`(16) / `--heatmap_n_t_grid`(20) / `--max_viz_cells`(4)
- effective W: `--top_edges`(100) / `--sparsity_eps`("1e-4,1e-3,1e-2") / `--max_svd_dim`(512) / `--plot_full_W_heat`(off)
- skip: `--skip_param_dist` / `--skip_W` / `--skip_effective_W_metrics` / `--skip_branch_specific`
- 後方互換: 旧 `--skip_W_t_step` は `--skip_effective_W_metrics` にマップ（旧 `--max_ckpts` は legacy fallback 用に残置）。

## 使い方

4 種一括（→ {out}/{loss,params,eval_io,velocity}/）:
```bash
python run_all_viz.py \
  --model_path <ckpt.pt> --config <exp_config.json> \
  --sample_path <samples.npz> --loss_path <.../loss_details.csv> \
  --data_dir <Embryonic.h5ad> --edge_tsv_path <tf_target_edges.tsv> \
  --output_dir <out> [--skip loss,velocity] [--max_cells 0]   # max_cells 既定 0=制限なし（全 cell）。重い時のみ正の値で間引く
```

単体実行（`--output_dir` 省略時は `--model_path` から `{base}/viz/{role}` に保存）:
```bash
python plot_params.py        --model_path <ckpt.pt> --config <exp_config.json> --data_dir <h5ad>
python eval_model_io.py      --model_path <ckpt.pt> --config <exp_config.json> --sample_path <npz> --data_dir <h5ad>
python plot_velocity_umap.py --model_path <ckpt.pt> --config <exp_config.json> --data_dir <h5ad>
python plot_loss.py          --loss_path <loss_details.csv> --output_dir <dir>
# LinComb 専用・単体（config/checkpoint は run_dir から自動検出。出力は {run_dir}/viz/lincomb_a_embedding/<ts>_a_embedding/）:
python plot_lincomb_a_embedding.py --run_dir <runs/lincomb__*/<ts>> --max_cells 50000 --t_values 0,499,999 \
  --color_cols Superclass,celltype,final_annotation
```

train→sample→viz 一気通貫（単一 config、上位ディレクトリ。出力 `runs/{model}/{YYYYMMDD_HHMMSS}/...`）:
```bash
cd work/20260609_Hybrid5x3
python run_pipeline_5x3.py --ode_branch lora --hybrid_norm_mode ratio_reg \
  --lr_anneal_steps 4 --batch_size 8 --diffusion_steps 1000 --num_samples 16
# --max_cells 既定 0=制限なし（velocity/UMAP に全 cell）。重くて困る時だけ --max_cells 5000 等
```

## 注意

- data/edge は未指定/`/home/suzuki/...` でも `local_paths.resolve_path` でローカル解決（PYTHONPATH 不要）。
- `--diffusion_steps` は 20 以上（linear schedule 制約）。train/sample/viz で揃える（既定 1000）。
- scvelo は `n_jobs=32`（元コード踏襲・remote 確認済み）。ローカル env（numpy≥1.24 / pandas≥2.0 + scvelo 0.2.5）は
  `velocity_graph`(ragged np.array) と `set_legend`(cat.categories setter) が非互換で落ちる。`plot_velocity_umap.py`
  冒頭の **version-guarded 互換 shim**（`_apply_scvelo_compat_shims`）で両者を回避し、ローカルでも stream/arrow を生成。
  remote の互換 env（numpy<1.24 / pandas<2.0）では shim は **no-op**。
- **`--max_cells` 既定 0 = 制限なし（全 cell で velocity/UMAP）**。real データを最初にサブサンプルする箇所
  （`plot_velocity_umap.py`）と real-vs-gen UMAP の両方に効く。scvelo は cell 数で重くなるので、重い時のみ
  正の値（例 `--max_cells 5000`）で間引く。
- ローカルは CPU。CUDA は dev() で自動利用（リモート）。
