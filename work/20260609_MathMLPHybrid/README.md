# Math-MLP Hybrid Denoising Models (20260609)

scDiffusion の denoising model を「数理構造 + MLP」の hybrid にして、4 系統を
**同条件で比較学習**するための実験セット。各 field は

```
V(x,t) = softplus(W(x,t) x + b) − softplus(γ) ⊙ x
```

の形を取り、`W(x,t)` を MLP でパラメタ化する。`MathML_Hybrid` で `Cell_Unet` と
t-scheduler で blend し、既存の `TrainLoop` にそのまま渡す。

> **最重要要件は W(x,t) への正則化**（既存 `GeneODE.off_mask_penalty` の発想を流用）。

---

## 0. 設計の大原則（既存実験を壊さない）

以下は **一切変更していない**（後方互換・既存実験保護）:

- `guided_diffusion/train_util.py`（TrainLoop）
- `guided_diffusion/gaussian_diffusion.py`（diffusion）
- `guided_diffusion/cell_datasets_loader.py`（data loader）
- `ODE/ode_20260421_regODEMLratio.py`（既存 GeneODE / ODE_ML_Hybrid）
- 既存の `visualization_analysis.py`

新規追加は **新ファイルのみ**。TrainLoop の正則化 hook
（`model.ode_model.soft` と `off_mask_penalty(norm)` を呼ぶ部分）を
**duck-type でそのまま発火**させることで、コア改変ゼロを実現している。

---

## 1. ファイル一覧

| ファイル | 役割 |
|---|---|
| `../../ODE/ode_20260609_mathmlp.py` | **コアモジュール**。4 field + 基底 + hybrid wrapper + factory + mask helper |
| `cell_train_20260609.py` | 学習 entrypoint（`cell_train_20260421.py` ベース） |
| `cell_sample_20260609.py` | サンプリング（`cell_sample_20260421.py` ベース） |
| `train_mathmlp_20260609.sh` | train→sample→W可視化の本体パイプライン（`--model_type` で分岐） |
| `train_{lowrank,lincomb,matsum,lora}_20260609.sh` | 4 系統の thin launcher（本体に `--model_type` を渡すだけ） |
| `run_pipeline_mathmlp.py` | **単一 model_type を cell_train→cell_sample→全 viz で一貫実行**（最小テスト用 → §11） |
| `visualize_W_20260609.py` | **legacy** 単体 W(x,t)/γ 可視化。現在は共有 `viz/plot_params.py`（上位互換）に置換済み（train_mathmlp_20260609.sh も plot_params shim を使用） |
| `plot_loss.py` / `plot_params.py` / `eval_model_io.py` / `plot_velocity_umap.py` / `run_all_viz.py` | viz への 1 行 shim（`../20260609_Hybrid5x3/viz/` の同名スクリプトへ runpy 委譲） |
| `smoke_test_20260609.py` | GPU マシン用 smoke test（4 field 系統 × torch のみ、h5ad/tsv 不要） |
| `test_all_models.sh` | **全 4 field 系統を train→sample→viz で一括 e2e テスト**（`run_pipeline_mathmlp.py` を loop） |

> ODE モデル本体（`ode_20260609_mathmlp.py`）の解説は `../../ODE/README_20260609.md`。
> 可視化は 3 dir 共有の `../20260609_Hybrid5x3/viz/` を使う（このディレクトリの `plot_*.py` 等は 1 行 shim）。

---

## 2. 4 系統のモデル（`ODE/ode_20260609_mathmlp.py`）

すべて `MathMLPField` を基底に持ち、forward は **batched W (B×d×d) を materialize しない**
（associativity を利用）。d は遺伝子数（HVG なら 1024）。

### `LowRankField`  —  W(x,t) = U(x,t) V_low(x,t)^T
- `U_net`, `V_net`（MLP）が `(x,t)` から `(B,d,r)` を出力。
- forward: `Wx = U (V^T x)` = `einsum('bdr,bd->br') → einsum('bdr,br->bd')`。W 非構築。
- 正則化: 静的因子が無いため、**batch を subsample（既定 8 件）して W を限定構築**し off-mask 罰則。
  → forward 内で `_cached_W_sub`（≤ n_sub·d·d、grad 付き）にキャッシュ。

### `LinCombField`  —  V = Σ_k a_k(x,t) softplus(W_k x + b_k)
- `coeff_net`（MLP）が `a_k(x,t)` を `(B,K)` 出力。`expert_W (K,d,d)` / `expert_b (K,d)` は静的。
- forward: `F = softplus(einsum('kij,bj->bki', W, x)+b)` → `V = einsum('bk,bkd->bd', a, F)`。
- 正則化: 静的 `expert_W` の off-mask（x,t 非依存・安価）。
- 注: softplus の和構造なので単一の `W(x,t)` を持たない。`compute_W` は実効線型化 `Σ_k a_k W_k` を返す（可視化の近似）。

### `MatSumField`  —  W(x,t) = Σ_k a_k(x,t) A_k
- `coeff_net` が `(B,K)`、`A (K,d,d)` は静的、`b (d,)`。
- forward: `Wx = Σ_k a_k (A_k x)` = `einsum('kij,bj->bki') → einsum('bk,bkd->bd')`。W 非構築。
- 正則化: 静的 `A` の off-mask（安価）。

### `LoRAField`  —  W(x,t) = W_0 + Σ_k a_k(x,t) U_k V_k^T
- `W0 (d,d)`, `U (K,d,r)`, `V (K,d,r)`, `b (d,)` は静的、`coeff_net` が `(B,K)`。
- forward: `Wx = W0 x + Σ_k a_k U_k(V_k^T x)`。W 非構築。
- 正則化: 静的 `W0` off-mask + `Σ_k off-mask(U_k V_k^T)`（Δ_k は K 枚だけ materialize、batch 無し）。

### 共通 interface（GeneODE と duck-type 互換）

```python
forward(x, t=None) -> (B, d)            # V(x,t)。t 省略/NaN は t=0 扱い（GeneODE 互換）
compute_W(x, t=None) -> (B, d, d)       # 可視化用（少数サンプルで呼ぶこと）
off_mask_penalty(norm="l1") -> scalar   # train_util の hook が呼ぶ
get_model_info() -> dict                # {type,d,rank,K,use_mask,soft,use_decay,w_is_exact}
属性: soft(bool), mask(buffer (d,d) or None),
      ratio_reg_weight/target/eps, _cached_ratio_reg,
      gamma(Parameter (d,)), use_decay(bool),
      enable_offmask_cache(bool), W_IS_EXACT(class attr)
```

**t は省略可能**: `forward(x, t=None)`。`t=None` または NaN sentinel のときは `t=0` 扱い。
これにより `field(x)`（GeneODE 形式の呼び出し）でも落ちない。`t` を渡せば time embedding
（`timestep_embedding` の sinusoidal）に流れる。既存 `GeneODE.forward(x, t=None)` も
互換のため `t` を受け取る（時刻非依存なので無視）よう変更済み（`ode_model(x)` は不変）。

---

## 3. GeneODE 風の減衰項（符号付き出力）

出力を符号付きにするため、GeneODE と同じ減衰項を全モデルに追加:

```
V(x,t) = softplus(W(x,t) x + b) − softplus(γ) ⊙ x
```

- `self.gamma = nn.Parameter(torch.ones(d) * 0.1)` — GeneODE と同じ初期値・生パラメタ。
- `_decay(x) = F.softplus(self.gamma) * x` — **使用時に softplus で非負化**
  （GeneODE の `gamma = softplus(self.gamma)` と完全一致）。
- `--use_decay`（既定 `True`）で on/off。`False` なら純粋な `softplus(Wx+b)`（正値のみ）。

**γ は W とは独立な加法項**なので、`off_mask_penalty` / `compute_W`（W 可視化）には影響しない。
`gamma` は常に shape `(d,)` で **K に依存しない**。可視化は `field.gamma` を**名前で参照**するため、
K を変えても呼び出しがズレない。

---

## 4. 正則化（off-mask penalty）

TrainLoop の既存 hook が `model.ode_model.off_mask_penalty(ode_reg_norm)` を呼び、
`loss += ode_reg_lambda * penalty` する（**train_util は無変更**）。

- `off_mask_penalty(norm)` は `_offmask_base(norm) + ratio_reg_weight * _cached_ratio_reg`
  を返す（GeneODE と同形。ratio 項は `MathML_Hybrid.forward` が `_cached_ratio_reg` にキャッシュ）。
- `_offmask_base`:
  - lincomb / matsum / lora → 静的パラメタの off-mask（x,t 非依存）
  - lowrank → forward 内 subsample で作った `_cached_W_sub` の off-mask
- **mask 両対応**: `--use_mask_reg True` で off-mask 罰則（`tf_target_edges.tsv` 由来の
  `(d,d)` edge mask）、`False` で W 全体に L1/L2 罰則。

### LowRank の罰則の脆さへの防御

LowRank は `W(x,t)` が完全に動的なため、`off_mask_penalty()` は forward 内で作った
`_cached_W_sub`（subsample W）を読む。既存 TrainLoop は **microbatch ループ内で
forward → off_mask_penalty を毎回ペアで呼ぶ**ので、最後の microbatch だけ使われる事故は無い。
さらに以下の防御を入れている:

- `off_mask_penalty()` は `_cached_W_sub is None` なら **0 を返す**（forward 前/eval で安全）。
- `forward` は **`training and soft and enable_offmask_cache` のときだけ** cache を作り、
  それ以外（eval / sampling / `ode_reg_lambda=0`）では `None` にして **stale cache を残さない**。
- `cell_train` が `field.enable_offmask_cache = (soft and ode_reg_lambda>0)` を設定するので、
  **`ode_reg_lambda=0` のとき subsample W を作る無駄が無い**。
- 罰則項の device は `field.gamma.device` 基準（全 field が持つ Parameter）で安全。

### メモリ見積り（d=1024, float32, 4MB/枚）

| 項目 | サイズ | 可否 |
|---|---|---|
| batched W (B=128) | 512MB(+grad で 1GB超) | **forward では作らない** |
| lowrank subsample W (8件) | ~32MB | OK |
| 静的 expert_W / A (K=8) | ~32MB | OK（params） |
| lora Δ_k materialize (K=8) | ~32MB | OK（罰則時のみ、batch 無し） |

---

## 5. 学習の実行

4 系統を **同条件**（seed=1234 を Python 側で固定、数値パラメタは shell で統一）で比較:

```bash
cd work/20260609_MathMLPHybrid
bash train_lowrank_20260609.sh
bash train_lincomb_20260609.sh
bash train_matsum_20260609.sh
bash train_lora_20260609.sh
```

各 launcher は `train_mathmlp_20260609.sh --model_type <type>` を呼ぶだけ。
出力は `work/20260609_MathMLPHybrid/{TIMESTAMP}_{model_type}/` に **model_type ごと分離**。

### パラメタ上書き例
```bash
bash train_lowrank_20260609.sh --rank 8 --use_mask_reg True --use_decay True --ode_reg_lambda 5
```

### 学習 argparse の主な追加項目（`cell_train_20260609.py`）

| 引数 | 既定 | 説明 |
|---|---|---|
| `--model_type` | `lowrank` | `lowrank` / `lincomb` / `matsum` / `lora` |
| `--rank` | 16 | low-rank / LoRA の r |
| `--K` | 8 | lincomb / matsum / lora の成分数 |
| `--use_mask_reg` | True | off-mask 罰則 (True) / W 全体罰則 (False) |
| `--use_decay` | True | GeneODE 風の `−softplus(γ)x` を加える |
| `--time_dim` `--field_hidden` `--field_dropout` | 64 / 256 / 0.0 | MLP の構成 |
| `--lowrank_penalty_subsample` | 8 | lowrank 罰則の subsample 件数 |

既存の `--ode_reg_lambda` `--ode_reg_norm` `--ratio_reg_weight` `--ratio_reg_target`
`--edge_tsv_path` `--SoftReg` を流用。

---

## 6. checkpoint / サンプリング

- **hyperparameter は state_dict に入らない**ため、学習時に
  `{TIMESTAMP}_train/field_config.json` を書き出す（model_type / rank / K / use_mask_reg /
  use_decay 等）。
- `cell_sample_20260609.py --field_config <path>` で config から field を**同型再構築**し、
  `load_hybrid_state_dict(...)` でロード。
- train.sh は学習ログから `FIELD_CONFIG=...` を grep して自動で sample / viz に渡す。

### 誤 checkpoint 防止（strict=False の危険性対策）

`load_hybrid_state_dict`（`ode_20260609_mathmlp.py`）が安全に読み込む:

1. `clean_state_dict` で EMA / `model` / `state_dict` ネスト / `module.` prefix を剥がす。
2. **missing / unexpected / shape-mismatch を全部ログ出力**（key 名つき）。
3. `ode_model.*` / `ml_model.*`（=実体パラメタ）に missing か shape-mismatch があれば
   **critical** とみなす（model_type / rank / K / dim ズレを検出）。
4. `--strict_load True`（既定）なら critical で **RuntimeError で中断**。
   `--strict_load False` で警告のみ続行（部分ロード）。
5. shape-mismatch の項目は除外して読む（PyTorch が拒否するため）。

→ **間違った checkpoint を読んでも黙って動く事故を防ぐ**。4 モデル比較で特に重要。
可視化スクリプトは `strict=False`（部分ロード許容）だが、全 key を必ずログ出力する。

```bash
python cell_sample_20260609.py \
    --model_path <ema_*.pt> \
    --field_config <.../field_config.json> \
    --data_dir <h5ad> --edge_tsv_path <tsv>
```

---

## 7. W(x,t) 可視化（`visualize_W_20260609.py`）

既存 viz を触らず別ファイルで追加。`compute_W` を少数サンプル（N_VIS=4）で呼んで出力:

- `hist_t{t}.png` — W 全要素 / on-mask / off-mask の重ね描き（t ごと）
- `heat_t{t}.png` — 平均 W のヒートマップ（d>256 なら行・列を sampling した部分行列）
- `W_abs_vs_t.png` — t ごとの `|W|` 平均（全体 / on / off）の推移
- `gamma_decay_hist.png` — `softplus(γ)`（実効減衰率 ≥0）の分布（`field.gamma` を名前参照、K 非依存）
- `W_stats.json` — 数値サマリ + `get_model_info()`

t は `{0, T//2, T-1}` を既定で評価。lowrank / lora / lincomb など W を明示的に持たない
モデルも `compute_W` が少数サンプルだけ materialize する（≤ 数十MB）。

> **⚠ lincomb の W は厳密ではない（proxy）**
> `LinCombField` は `V = Σ_k a_k(x,t) softplus(W_k x + b_k)` で、これは一般に
> 単一の `softplus(W_eff x + b_eff)` と **等価でない**。
> `compute_W` が返すのは preactivation の係数付き線型近似 `Σ_k a_k W_k`（**proxy**）であって
> 厳密な `W(x,t)` ではない。`field.W_IS_EXACT == False` で識別でき、可視化では
> タイトルに `[PROXY: Σ a_k W_k, not exact W]` を付け、コンソールに警告を出し、
> `W_stats.json` に `"w_is_exact": false` を記録する。
> **他モデル（lowrank/matsum/lora は厳密 W）との横比較では解釈に注意。**
> なお off-mask 正則化は proxy ではなく静的 `expert_W` に厳密に掛かる。

```bash
python visualize_W_20260609.py \
    --model_path <ema_*.pt> --field_config <.../field_config.json> \
    --data_dir <h5ad> --edge_tsv_path <tsv> --output_dir <dir>
```

---

## 8. テスト

GPU/学習マシンで（torch のみ必要、h5ad/tsv 不要・mask はランダム代用）:

```bash
python smoke_test_20260609.py
```

確認項目: import / forward shape / γ shape が `(d,)` で K 非依存 / 符号付き出力 /
hybrid の duck-type `ode_model` / `off_mask_penalty`(l1,l2,mask有無) / reg の backprop /
`compute_W` shape / checkpoint round-trip / forward 時に batched W を確保しないこと。

ローカルでは torch 不在のため、`py_compile`（全ファイル）と
W-free forward の einsum 恒等式（純 Python で lowrank/matsum/lora を検証）まで実施済み。

---

## 9. shape / t のセマンティクス（確認済み）

diffusion → model の呼び出しは `model(x_t, self._scale_timesteps(t).unsqueeze(1), **kwargs)`:

| 項目 | 値 |
|---|---|
| `x` | `(B, d)`（`(B,1,d)` ではない。d=遺伝子数） |
| `t`（model へ） | `(B, 1)` の **float**（`scale_timesteps`。`rescale_timesteps=False` 既定なら 0..T-1 の値） |
| `Cell_Unet.forward(x, t, y)` 出力 | `(B, d)`（内部で `timestep_embedding(t)`） |
| `MathMLPField.forward(x, t)` 出力 | `(B, d)`（`_prep_t` で `(B,1)→(B,)`、`timestep_embedding` で sinusoidal 埋め込み） |
| `MathML_Hybrid.forward` 出力 | `(B, d)`（blend 前に field と ML の shape 一致を `assert`） |
| denoising target の意味 | `predict_xstart=False` 既定 → **ε 予測**（符号付き。だから decay 項で符号付き出力にしている） |

**t は生の整数をそのまま MLP に入れていない**: `_TimeEmb` が `timestep_embedding`(sinusoidal)
で埋め込んでから線形層に通すので、整数 t による学習不安定は避けられている。
`MathML_Hybrid.forward` は `field_out.shape == ml_out.shape` を `assert` して blend する。

---

## 10. 注意・既知の論点

- **softplus の符号**: `--use_decay True`（既定）で `−softplus(γ)x` により符号付き。
  ε 予測（`predict_xstart=False`）との相性のため、減衰項を有効にしておくことを推奨。
- **実行環境のパス**: スクリプトは `/home/suzuki/Projects/scDiffusion` を
  `sys.path` / `DATA_DIR` / `EDGE_TSV_PATH` に**ハードコード**（学習マシン前提）。
  別マシンで動かす場合は各スクリプト冒頭の `sys.path.insert(...)` と train.sh の
  `DATA_DIR` / `EDGE_TSV_PATH` を書き換えること。
- **gene_list の一致**: サンプリング/可視化は学習時と同じ h5ad（同じ `adata.var["gene_name"]`）が必須。

---

## 11. cell_train → cell_sample → viz 一貫テスト（`run_pipeline_mathmlp.py`）

§5 の `train_*.sh`（train→sample→W 単体可視化）とは別に、**1 model_type を train→sample→全 viz
（役割別 `viz/{plot_loss,plot_params,eval_model_io,plot_velocity_umap}`）で一気通貫**する最小テスト。
出力構造は **`runs/mathmlp__{model_type}/{YYYYMMDD_HHMMSS}/{train,sample,viz}/`**（日付 dir は時刻 HHMMSS まで含み実行ごとにユニーク・work 直下を散らかさないため runs/ 配下）。

```bash
cd work/20260609_MathMLPHybrid

# 最小テスト（lowrank。lr_anneal_steps は 2 以上にすること）
python run_pipeline_mathmlp.py --model_type lowrank \
  --lr_anneal_steps 4 --batch_size 8 --diffusion_steps 1000 \
  --num_samples 16 --max_cells 120 --analyze_max_cells 100 --num_t_points 4

# --dry-run でコマンドのみ / --skip_viz で viz 省略 / --rank --K で field 構成を変更
# model_type ∈ {lowrank, lincomb, matsum, lora}
```

**3 段の流れ**（pipeline が学習ログから model_path / `field_config.json` / loss_details.csv を grep）:
1. `cell_train_20260609.py`（`--save_interval 1` 固定）→ `model00000{0..N}.pt` + `ema_*` + `loss_details.csv` + `field_config.json`
2. `cell_sample_20260609.py`（`--field_config` で同型再構築、`load_hybrid_state_dict` で安全ロード）→ `*.npz`
3. `viz/run_all_viz.py` が役割別 4 スクリプトを `viz/{loss,params,eval_io,velocity}/` に分離出力。

**主な出力図**（§7 の `visualize_W_20260609.py` 単体とは別の統合 viz）:
- `viz/loss/`: `loss_curves.png`（**lr_anneal_steps≥2 で非退化**）
- `viz/params/<ts>_viz/`: `W_hist_t*` / `W_heat_t*`（**フル解像度・枠線なし**）/ `W_abs_vs_t` / `gamma_decay_hist` /
  `param_dist_misc.png`（**学習ステップ横軸**。lowrank は静的 W 無し → `U_net`/`V_net`/`gamma`/`b`/`cellunet`）。
  lincomb/matsum/lora は `param_dist_W.png` に **W を per-k×on/off**（expertW{k} / A{k} / W0+Δ{k}）/
  `W_meanabs_t_vs_step` / `W_offmask_t_vs_step`（**t×学習ステップ**。lowrank は動的 W を `compute_W` で評価）
- `viz/eval_io/<ts>_analyze/`: `corr/` / `alignment_mse/` / `.csv` / `.json`
- `viz/velocity/`: `umap_analysis.png`(**real vs gen・凡例なし**) + `velocity_by_superclass/` の stream/arrow（compat shim でローカルでも生成）
- ※ lincomb の W は proxy（`W_IS_EXACT=False`、§7 の注記参照）

**3 dir を一括**: `bash work/run_all_20260609_pipelines.sh`（STEPS=4 既定で本 dir も含め 3 実験を最小実行）。

**注意**:
- `--lr_anneal_steps` は **2 以上**必須（pipeline は `save_interval=1` 固定 → step 数 = checkpoint 数）。
  1 だと loss 曲線が描けず param 分布 / W t×step も 1 点。
- `--data_dir` / `--edge_tsv_path` は未指定/`/home/suzuki/...` でも `local_paths.resolve_path` でローカル解決。
- velocity はローカル env では `velocity_graph` で落ちて UMAP fallback（traceback がログに出る）。
  リモート互換 env では stream 図。詳細は `../20260609_Hybrid5x3/viz/README.md`。
