# Implementation report — 2026-09-21

## Scope / files

作業場所: `/Users/cls-lab/Git/scDiffusionODE`。
本番実行場所: `/home/suzuki/Projects/scDiffusion-github`（リモート）。
作成したファイル:

| File | Role |
|---|---|
| `analyze.py` | CLI、native sampling、snapshot 保存、解析 orchestration、実行後の不変性検証 |
| `model_io.py` | canonical discovery、checkpoint/config/hash 検証、直接 model restore、元 X 読込 |
| `geometry.py` | scaled anchor、lazy local SVD、tangent/normal projection、距離と cosine |
| `diagnostics.py` | SWD、mode assignment と TV/JS、balanced mixing、11 figures |
| `tests/test_geometry.py` | 合成数学テストと native sampler/provenance/end-to-end テスト |
| `README.md` | 既存実装監査、数式、文献、provenance、CLI、リモート実行手順 |
| `IMPLEMENTATION_REPORT.md` | この監査記録 |
| `.gitignore` | この解析の results / logs を version control から除外 |

既存 `guided_diffusion/` と `work/20260915_x0predict/` には変更していない。
作業開始時から存在した他の work ディレクトリの変更は本変更に含めない。

## Existing functions used directly

- `guided_diffusion.cell_model.Cell_Unet` コンストラクタと forward。
- `guided_diffusion.script_util.create_gaussian_diffusion`。
- factory から返る native diffusion の `q_sample`, `p_mean_variance`, `p_sample`。
- PyTorch `load(weights_only=True)`, `load_state_dict(strict=True)`, `eval`, `no_grad`。
- scikit-learn `PCA.fit`, `NearestNeighbors.fit/kneighbors`、NumPy `linalg.svd`。

Stage1 wrapper や historical restore function は import していない。
その保存形式を読んで、直接 Cell_Unet を同じ input_dim / hidden_num で構築する。
`SpacedDiffusion` は既存 factory 内の依存で、1000-step identity timestep map を assert する。

## New mathematical / diagnostic code

START_X Tweedie score、centered PCA の scaled manifold 座標変換、kNN local PCA、
orthogonal tangent/normal projection、local affine normal residual、model drift / noise / total の
差分と normal cosine、sorted-projection sliced W1、majority-vote real-defined mode assignment、
TV/JS divergence、balanced joint kNN real-neighbor fraction。

sampling equation、CellUNet architecture、diffusion schedule は再実装していない。
checkpoint state hash と gene-order hash の byte serialization は既存実装と互換になるよう
小さく記述した。論文公式コードをコピー/import した箇所は **ない**。
論文の small-noise / local PCA / exposure mismatch の概念と今回独自の診断は README で区別した。

## Dependency chain

```text
analyze -> model_io -> existing Cell_Unet / create_gaussian_diffusion
                       -> existing SpacedDiffusion -> GaussianDiffusion -> nn/losses
analyze -> geometry -> numpy/sklearn
analyze -> diagnostics -> geometry/numpy/pandas/sklearn/matplotlib
```

モデル復元のために過去 work の wrapper chain、ODE、VAE、学習 runner を読み込まない。
詳細は README の依存図を参照。

## Parameter / checkpoint immutability

全 parameter を `requires_grad_(False)` とし、全 module が eval であることを検証する。
全 inference は `torch.no_grad()`。optimizer の作成、backward、step、EMA update はない。
checkpoint は読込専用、結果は新しい output directory へ書く。

解析前後の checkpoint SHA256 と parameter/buffer SHA256 は
`checkpoint_provenance.json` の以下の値に記録する:

- `checkpoint_sha256_before` / `checkpoint_sha256_after`
- `model_hash_before` / `model_hash_after`
- `unchanged_assertion_passed`
- `all_parameters_frozen` / `all_modules_eval`

一致は必須 assert。例外時も finally で検証・記録する。
本番 checkpoint の before/after 値は **未取得**（本番ファイルはリモート）。
架空の SHA256 を記入しない。合成 checkpoint で before/after 一致を検証済み。

## Executed validation

実行 interpreter:
`/Users/cls-lab/miniconda3/envs/scdiffusion/bin/python`。
Python 3.9.23、PyTorch 2.5.1、NumPy 1.24.0、SciPy 1.10.0、scikit-learn 1.2.0、
anndata 0.8.0、pandas 2.0.0、matplotlib 3.7.0。

```text
python -m unittest discover -s work/20260921/tests -v
Ran 19 tests in 3.889s
OK
```

確認した内容:

- xy plane の tangent/normal、orthogonal rotation invariance。
- v=v_T+v_N と直交性、既知 normal offset の距離、cosine、ゼロ vector の未定義比。
- 非ゼロ PCA mean を含む scaled coordinate / anchor / plane distance。
- lazy cache と不正な d の拒否。
- Gaussian oracle の START_X prediction から得る score が既知の `-x_t` と一致。
- SWD の同一分布・順序交換・既知 translation、等標本数制約。
- mode majority vote、TV/JS、分離した generated-only island、重複座標の self exclusion。
- valid final-EMA 形式の strict restore、checkpoint 改変、dataset 改変、non-final EMA、
  不正 prediction config、gene 順序変更の拒否。
- 複数 valid campaign で停止、invalid 候補があっても唯一の valid 候補を選択。
- dry-run が dataset 読込・model restore/inference をしないこと。
- 小さな合成 end-to-end: real 24 cells / 4 genes、generated 8 cells、sampling batch 4、
  PCA3、d=1,2、t=999,300,1,0、**全1000 native DDPM steps**。
  11 PNG、CSV、provenance 完成、hash 不変性、高ノイズ CSV 分離を検証。
- 同じ乱数初期化・model construction・batch 順で、診断なしの native p_sample のみを
  1000回呼び出した結果と **最終 sample が bitwise 完全一致**。
  保存した t=999 input が初期 Gaussian と一致し、t=0 pred_xstart が最終出力と一致。

合成 fixture は random weights であり、実学習済み checkpoint と同じ**保存形式**を作るだけ。
学習はしていない。科学的なモデル評価結果には使わない。
合成出力は TemporaryDirectory 内で作成・検証後に削除しており、本番 run directory ではない。

ローカルの実 campaign discovery も実行した:

```text
analyze.py --dry-run
Found 0 valid Stage1 campaigns.
searched .../scDiffusionODE/work/20260915_x0predict/runs
exit code 1
```

本番 checkpoint 不在に対する期待どおりの停止であり、推測した checkpoint や別 dataset への
fallback は行わなかった。

## Not executed / remaining runtime evidence

**実データでの解析は未実行**。本番 run directory はまだない。
本番 dataset SHA256、gene 順序、annotation 列、実 final EMA、
GPU 実行、3000 cells の数値と図、実 checkpoint の before/after hash は
リモート実行時に初めて確認される。
実装だけから manifold/exposure/coverage に関する実験上の結論を報告しない。

リモート実行は README の unit tests → dry-run → full analysis の順。
`metadata.json` の status=completed と checkpoint_provenance の unchanged assertion を確認する。
候補が複数あれば明示的な `--campaign` が必要。データを移動した場合は同一 SHA256 の
`--data-path` を指定する。sampler の学習条件を回避する override は設けていない。
