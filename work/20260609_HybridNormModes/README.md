# ODE-ML Hybrid 出力統合 3 モード比較（20260609）

`ODE_ML_Hybrid` の ODE/ML 出力統合方法を **3 モードから選べる**ようにし、同条件で比較学習する実験セット。

```
ratio_reg            out = r·ode + (1−r)·ml,  log-norm-ratio penalty を loss に加算（現行互換）
normed_learned_scale ode/ml を L2 正規化 → 共有 scalar scale 倍 → blend。ratio penalty なし
none                 out = r·ode + (1−r)·ml,  penalty なし（baseline）
```

`r = 1 − t/(T−1)`（既存スケジューラ）。3 モードは **排他**（`--hybrid_norm_mode` で 1 つ選択）。

---

## 0. 既存ファイルは無変更

以下は **一切変更していない**:

- `guided_diffusion/train_util.py`（TrainLoop / 正則化 hook）
- `ODE/ode_20260421_regODEMLratio.py`（`GeneODE` / 既存 `ODE_ML_Hybrid`）
- `work/20260421_RegODEMLratio/*`

3 モード対応は **新規ファイル** `ODE/ode_20260609_hybridnorm.py` の新クラス `ODE_ML_HybridNorm` に閉じ込め、
`GeneODE` は既存から import 再利用する。

---

## 1. ファイル一覧

| ファイル | 役割 |
|---|---|
| `../../ODE/ode_20260609_hybridnorm.py` | **新クラス** `ODE_ML_HybridNorm`（3 mode）。`GeneODE` は ode_20260421 から import |
| `cell_train_20260609.py` | 学習（`cell_train_20260421.py` ベース、新 hybrid + CLI + `hybrid_config.json` 出力） |
| `cell_sample_20260609.py` | サンプリング（新 hybrid 再構築、`hybrid_config.json` から mode 復元） |
| `train_hybridnorm_20260609.sh` | train→sample 本体（`--hybrid_norm_mode` で分岐） |
| `train_{ratio_reg,normed,none}_20260609.sh` | 3 mode の thin launcher（mode と出力名のみ差替、他引数 seed=1234 統一） |
| `run_pipeline_hybridnorm.py` | **単一 mode を cell_train→cell_sample→全 viz で一貫実行**（最小テスト用 → §10） |
| `plot_loss.py` / `plot_params.py` / `eval_model_io.py` / `plot_velocity_umap.py` / `run_all_viz.py` | viz への 1 行 shim（`../20260609_Hybrid5x3/viz/` の同名スクリプトへ runpy 委譲） |
| `smoke_test_20260609.py` | 3 mode の単体テスト（torch のみ、h5ad 不要） |
| `test_all_models.sh` | **全 3 mode を train→sample→viz で一括 e2e テスト**（`run_pipeline_hybridnorm.py` を loop） |

> ODE モデル本体（`ode_20260609_hybridnorm.py`）の解説は `../../ODE/README_20260609.md`。
> 可視化は 3 dir 共有の `../20260609_Hybrid5x3/viz/` を使う（このディレクトリの `plot_*.py` 等は 1 行 shim）。

---

## 2. 最小差分の仕組み（train_util / GeneODE 無変更で 3 mode）

既存 `GeneODE.off_mask_penalty()` は `_cached_ratio_reg is None` のとき **base のみ**返す。
新 hybrid が forward で mode 別に `ode_model._cached_ratio_reg` を出し分けるだけで、
`train_util` の hook（`model.ode_model.off_mask_penalty(norm)` を `ode_reg_lambda` 倍して loss に加算）
を **そのまま**使える:

| mode | `_cached_ratio_reg`（training） | `off_mask_penalty` の返り | SoftReg=True 時の loss 追加分 |
|---|---|---|---|
| ratio_reg | ratio penalty | base + ratio_reg_weight·ratio | off-mask + ratio |
| normed_learned_scale | None | base | off-mask のみ |
| none | None | base | off-mask のみ |

- ratio penalty の総重みは現行どおり `ode_reg_lambda · ratio_reg_weight · ratio_penalty`。
- eval 時は全 mode で `_cached_ratio_reg=None`（現行挙動と一致）。
- `SoftReg=False` のときは hook 自体が発火しない（既存と同じ）。

---

## 3. normed_learned_scale の scale

```
ode_unit = ode_out / (||ode_out||_2 + eps)
ml_unit  = ml_out  / (||ml_out||_2  + eps)
out = exp(log_scale) * (r·ode_unit + (1−r)·ml_unit)
```

- `log_scale` は **共有 scalar 1 個**の `nn.Parameter`。`scale = exp(log_scale)` で常に正・滑らか。
  init=1.0 → `log_scale=0`。`hybrid.parameters()` に入るので optimizer が学習する。
- **scale param は normed_learned_scale のときだけ生成**。ratio_reg/none は現行と同じ param 集合
  → 既存 checkpoint と互換。
- 含意: 共有 scalar は「正規化後の全体 velocity 倍率」を学習する。ODE/ML の**配分は r(t) のみ**。
  個別調整したくなったら `ode/ml_log_scale` の 2 個に割るのは 1 行の拡張。

---

## 4. CLI（cell_train / cell_sample 共通）

| 引数 | default | 備考 |
|---|---|---|
| `--hybrid_norm_mode` | ratio_reg | ratio_reg / normed_learned_scale / none（不正値は hybrid 側で ValueError） |
| `--hybrid_scale_init` | 1.0 | normed の scale 初期値 |
| `--hybrid_scale_eps` | 1e-8 | normed の正規化 & scale eps |
| `--ratio_reg_weight` / `--ratio_reg_target` | 1.0 / 1.0 | 既存維持。ratio_reg mode でのみ実効 |
| `--SoftReg` / `--ode_reg_lambda` / `--ode_reg_norm` | True / 5(sh) / l1 | 既存どおり off-mask penalty を制御 |

学習時に出力 dir へ `hybrid_config.json`（mode/scale_init/scale_eps 等）を書き出す。
サンプリングは `--hybrid_config <path>` で復元（**normed の `log_scale` を正しく用意するために必須**）。

---

## 5. 実行（3 モード比較）

```bash
cd work/20260609_HybridNormModes
bash train_ratio_reg_20260609.sh
bash train_normed_20260609.sh
bash train_none_20260609.sh
```

各 launcher は `train_hybridnorm_20260609.sh --hybrid_norm_mode <mode>` を呼ぶだけ。
出力は `{TIMESTAMP}_{mode}/` に mode 別分離。数値引数・seed=1234 は 3 本で統一。

---

## 6. checkpoint 互換性

- ratio_reg / none: param 集合が現行と同一 → 既存 checkpoint も互換。
- normed_learned_scale: `log_scale`（1 scalar）が増える。新規学習で保存。サンプリングは
  **同 mode で再構築**すれば param 一致でロード可（`load_state_dict(strict=False)` + missing/unexpected をログ）。
- 旧 checkpoint の normed resume は非対応（必須にしない）。`train_util` の loading は無変更。

---

## 7. テスト

学習マシンで（torch のみ）:

```bash
python smoke_test_20260609.py
```

3 mode の forward/backward、`log_scale` の grad と parameters 包含、ratio cache の出し分け、
`off_mask_penalty` 返り値（hook シミュレート）、SoftReg=False 時の hook 不発火、DDP wrapper 経路、
不正 mode の ValueError を検証。ローカルは torch 不在のため `py_compile` まで実施済み。

---

## 8. デバッグログ（train_util 非干渉）

`ODE_ML_HybridNorm.forward` が training 時に `self._cached_hybrid_stats`
（`hybrid_norm_mode / ratio_penalty / ode_norm_mean / ml_norm_mean / scale`）を保持する（副作用なし）。
`loss_details.csv` は既存どおり `reg_value / reg_weighted / total_loss`。CSV 拡張が必要なら
将来 train_util に数行追加する案もあるが、本実験では非干渉を優先。

---

## 9. 注意

- 実行環境のパス（`sys.path` / `DATA_DIR` / `EDGE_TSV_PATH`）は過去スクリプトと同じ
  `/home/suzuki/Projects/scDiffusion`（学習マシン前提）。別マシンでは書き換える。
- サンプリング/学習で同じ h5ad（同じ `adata.var["gene_name"]`）を使うこと。

---

## 10. cell_train → cell_sample → viz 一貫テスト（`run_pipeline_hybridnorm.py`）

3 mode 比較（§5 の `train_*.sh` は train→sample のみ）とは別に、**1 mode を train→sample→全 viz で
一気通貫**する最小テスト。出力構造は **`runs/hybridnorm__{mode}/{YYYYMMDD_HHMMSS}/{train,sample,viz}/`**
（日付 dir は時刻 HHMMSS まで含むので実行ごとにユニーク＝同日再実行も完全分離。work 直下を散らかさないため runs/ 配下）。

```bash
cd work/20260609_HybridNormModes

# 最小テスト（normed_learned_scale。lr_anneal_steps は 2 以上にすること）
python run_pipeline_hybridnorm.py --hybrid_norm_mode normed_learned_scale \
  --lr_anneal_steps 4 --batch_size 8 --diffusion_steps 1000 \
  --num_samples 16 --max_cells 120 --analyze_max_cells 100 --num_t_points 4

# --dry-run でコマンドのみ / --skip_viz で viz 省略
# mode ∈ {ratio_reg, normed_learned_scale, none}（model は GeneODE 固定）
```

**3 段の流れ**（pipeline が学習ログから model_path / `hybrid_config.json` / loss_details.csv を grep）:
1. `cell_train_20260609.py`（`--save_interval 1` 固定）→ `model00000{0..N}.pt` + `ema_*` + `loss_details.csv` + `hybrid_config.json`
2. `cell_sample_20260609.py`（`--hybrid_config` で再構築、**normed は `log_scale` 復元に必須**）→ `*.npz`
3. `viz/run_all_viz.py` が役割別 4 スクリプトを `viz/{loss,params,eval_io,velocity}/` に分離出力。

**主な出力図**:
- `viz/loss/`: `loss_curves.png`（**lr_anneal_steps≥2 で非退化**）
- `viz/params/<ts>_viz/`: `W_hist_t*` / `W_heat_t*`（**フル解像度・枠線なし**）/ `W_abs_vs_t` / `gamma_decay_hist` /
  `param_dist_W.png`（GeneODE の **W_on/W_off**）/ `param_dist_misc.png`（gamma/b/cellunet）/
  `W_meanabs_t_vs_step` / `W_offmask_t_vs_step`（**t×学習ステップ**）/ `scale_vs_step.png`（**normed のみ**）
- `viz/eval_io/<ts>_analyze/`: `corr/` / `alignment_mse/` / `.csv` / `.json`
- `viz/velocity/`: `umap_analysis.png`(**real vs gen・凡例なし**) + `velocity_by_superclass/` の stream/arrow（compat shim でローカルでも生成）

**3 dir を一括**: `bash work/run_all_20260609_pipelines.sh`（STEPS=4 既定で本 dir も含め 3 実験を最小実行）。

**注意**:
- `--lr_anneal_steps` は **2 以上**必須（pipeline は `save_interval=1` 固定 → step 数 = checkpoint 数）。
  1 だと loss 曲線が描けず param 分布 / W t×step も 1 点。
- `--data_dir` / `--edge_tsv_path` は未指定/`/home/suzuki/...` でも `local_paths.resolve_path` でローカル解決。
- velocity はローカル env では `velocity_graph` で落ちて UMAP fallback（traceback がログに出る）。
  リモート互換 env では stream 図。詳細は `../20260609_Hybrid5x3/viz/README.md`。
