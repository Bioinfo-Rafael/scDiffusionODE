# 20260609 可視化スイート（viz/）

20260609 の新モデル（UnifiedODEMLHybrid / MathML_Hybrid / ODE_ML_HybridNorm、
ode_branch ∈ {geneode, lowrank, lincomb, matsum, lora, plain}）用の作図一式。
**役割別にファイルを分割**（loss / パラメタ・W / 入出力評価 / velocity UMAP）。

## ファイル（役割別）

| ファイル | 役割 / 出力 | run_all_viz の出力先 |
|---|---|---|
| `_restore.py` | 共有: config 正規化 + `build_denoiser` で復元 + checkpoint ロード + velocity 計算 | — |
| `plot_loss.py` | **loss 曲線**（loss_curves / loss_curves_simple。CSV のみ依存） | `{out}/loss/` |
| `plot_params.py` | **W 可視化**(hist/heat/W_abs_vs_t/gamma) + **パラメタ分布(学習ステップ軸)** + **t×step の W ヒートマップ** | `{out}/params/<ts>_viz/` |
| `eval_model_io.py` | **入出力評価**（ノイズ方向 metrics weighted/unweighted + 次元方向相関）。corr/ と alignment_mse/ に分離 | `{out}/eval_io/<ts>_analyze/` |
| `plot_velocity_umap.py` | **Superclass 別 velocity UMAP**（stream/arrow、配色2通り）+ **real vs gen UMAP**（`umap_analysis.png`・凡例なし） | `{out}/velocity/` |
| `run_all_viz.py` | 上 4 つを 1 checkpoint に対し順次実行（`--skip loss,params,eval_io,velocity`） | — |

3 dir の checkpoint は config json（`exp_config`/`field_config`/`hybrid_config`）を渡せば全て復元可能。
他 2 dir（HybridNormModes / MathMLPHybrid）には同名の 1 行 shim（runpy 委譲）あり。
出力構造は上位ランナーが決める **`{work}/runs/{model}/{YYYYMMDD_HHMMSS}/viz/{role}/`**（`work/20260609_Hybrid5x3/run_paths.py`）。
単体実行で `--output_dir` 未指定なら各スクリプトが `--model_path` から `{base}/viz/{role}` を逆算する。

## モデル結合の要点

- 復元は `_restore.build_model`（`ODE/ode_20260609_hybrid5x3.py: build_denoiser` 経由、5 枝 + plain）。
- velocity = `model.ode_model(x, velocity_t)`（GeneODE は t 無視 / fields は t 使用）。`--velocity_t` 既定 0。
- scvelo は**元 velocity_by_Superclass.py の方式を踏襲**: velocity を `layers["velocity_ode"]`、`layers["X"]=X.copy()` を
  作って `velocity_graph(vkey="velocity_ode", xkey="X", backend="loky", n_jobs=32)` → embedding → stream/arrow/grid。
- W 可視化は field の `compute_W(x,t)`、GeneODE は静的 `W` を使用。lincomb は proxy（`W_IS_EXACT=False`）。
  W ヒートマップ `W_heat_t{t}.png` は **フル解像度 (n×n)・枠線なし**（方眼紙塗り）。
- **パラメタ分布（学習ステップ横軸）**: `find_checkpoints` が model_path の dir 内の `model*.pt` を step 順に集め、
  `extract_param_groups` が branch 別に学習パラメタを抽出 → ヒストグラム格子（行=param群, 列=step）。
  - `param_dist_W.png`: 静的 W 系を **per-k × on/off mask** で分割（geneode `W`、lincomb `expertW{k}`、
    matsum `A{k}`、lora `W0`+`Delta{k}`）。lowrank は静的 W 無し → W 行は出ず U_net/V_net 等は misc へ。
  - `param_dist_misc.png`: gamma / b(expert_b) / coeff_net / U_net / V_net / time_emb / cellunet / other。
  - `scale_vs_step.png`: normed モデルの `log_scale`→`exp` を step に対して折れ線。
  - 複数 checkpoint が必要（`save_interval=1` + `lr_anneal_steps≥2`）。`--max_ckpts`(既定8) で間引き。
- **t×学習ステップ の W ヒートマップ**: `W_meanabs_t_vs_step.png` と `W_offmask_t_vs_step.png`
  （y=diffusion t, x=training step, 値=mean|W| / off-mask mean|W|）。`--n_t_grid`(既定16) / `--max_ckpts`。
- CLI 追加: `--skip_param_dist --skip_W_t_step --max_ckpts --n_t_grid`。
- 次元方向相関（eval_model_io）= 各サンプルで `pearson(out(1024), ε(1024))`。ε は `diffusion.q_sample` の付加ノイズ。
  corr_scatter は等間隔 5 点 t（T=1000→0,249,499,749,999）、corr_dim_vs_t は 100 間隔（0..900,999）。
  出力は `corr/`（corr_*）と `alignment_mse/`（alignment/mse/norm_ratio）のサブディレクトリに分離。
- plain baseline（ode_model 無）は W/velocity/branch 解析を自動スキップ（loss / eval_io の hybrid 出力のみ）。

## 使い方

4 種一括（→ {out}/{loss,params,eval_io,velocity}/）:
```bash
python run_all_viz.py \
  --model_path <ckpt.pt> --config <exp_config.json> \
  --sample_path <samples.npz> --loss_path <.../loss_details.csv> \
  --data_dir <Embryonic.h5ad> --edge_tsv_path <tf_target_edges.tsv> \
  --output_dir <out> [--skip loss,velocity] [--max_cells 800]
```

単体実行（`--output_dir` 省略時は `--model_path` から `{base}/viz/{role}` に保存）:
```bash
python plot_params.py        --model_path <ckpt.pt> --config <exp_config.json> --data_dir <h5ad>
python eval_model_io.py      --model_path <ckpt.pt> --config <exp_config.json> --sample_path <npz> --data_dir <h5ad>
python plot_velocity_umap.py --model_path <ckpt.pt> --config <exp_config.json> --data_dir <h5ad>
python plot_loss.py          --loss_path <loss_details.csv> --output_dir <dir>
```

train→sample→viz 一気通貫（単一 config、上位ディレクトリ。出力 `runs/{model}/{YYYYMMDD_HHMMSS}/...`）:
```bash
cd work/20260609_Hybrid5x3
python run_pipeline_5x3.py --ode_branch lora --hybrid_norm_mode ratio_reg \
  --lr_anneal_steps 4 --batch_size 8 --diffusion_steps 1000 --num_samples 16 --max_cells 800
```

## 注意

- data/edge は未指定/`/home/suzuki/...` でも `local_paths.resolve_path` でローカル解決（PYTHONPATH 不要）。
- `--diffusion_steps` は 20 以上（linear schedule 制約）。train/sample/viz で揃える（既定 1000）。
- scvelo は `n_jobs=32`（元コード踏襲・remote 確認済み）。ローカル env（numpy≥1.24 / pandas≥2.0 + scvelo 0.2.5）は
  `velocity_graph`(ragged np.array) と `set_legend`(cat.categories setter) が非互換で落ちる。`plot_velocity_umap.py`
  冒頭の **version-guarded 互換 shim**（`_apply_scvelo_compat_shims`）で両者を回避し、ローカルでも stream/arrow を生成。
  remote の互換 env（numpy<1.24 / pandas<2.0）では shim は **no-op**。重い場合は `--max_cells` で間引き。
- ローカルは CPU。CUDA は dev() で自動利用（リモート）。
