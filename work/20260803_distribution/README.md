# 2026-08-03 real/generated expression distributions

## すぐ実行するコマンド

repository rootで、学習時と同じenvironmentを有効化して実行します。

```bash
# 1. 6モデル: checkpoint/sample確認、必要時だけsampling、集計、作図
conda run -n scdiffusion python work/20260803_distribution/analyze_distribution.py \
  --batch-id 20260803_full_30000

# 2. checkpointに一致する既存sampleだけを使い、samplingせず作図
conda run -n scdiffusion python work/20260803_distribution/analyze_distribution.py \
  --batch-id 20260803_full_30000 --plot-only

# 3. familyを指定（出力は3行×5列）
conda run -n scdiffusion python work/20260803_distribution/analyze_distribution.py \
  --batch-id 20260803_full_30000 --family standard_hybrid_lincomb

# 4. ODEを指定（出力は2行×5列）
conda run -n scdiffusion python work/20260803_distribution/analyze_distribution.py \
  --batch-id 20260803_full_30000 --ode racipe

# 5. experimentを1つ指定
conda run -n scdiffusion python work/20260803_distribution/analyze_distribution.py \
  --batch-id 20260803_full_30000 \
  --experiment standard_hybrid_lincomb__exp

# 6. bin数・描画subsample数を変更
conda run -n scdiffusion python work/20260803_distribution/analyze_distribution.py \
  --batch-id 20260803_full_30000 --bins 300 --plot-subsample 500000 --seed 1234

# 7. checkpointを実験ごとに明示
conda run -n scdiffusion python work/20260803_distribution/analyze_distribution.py \
  --batch-id 20260803_full_30000 \
  --checkpoint standard_hybrid_lincomb__exp=/absolute/path/ema_0.9999_030000.pt

# 8. 読み取り専用preflightと実行予定
conda run -n scdiffusion python work/20260803_distribution/analyze_distribution.py \
  --batch-id 20260803_full_30000 --dry-run

# 9. 最新出力
find work/20260803_distribution/outputs -mindepth 2 -maxdepth 2 -type d | sort | tail -1
```

## 目的と対象

real dataだけから遺伝子を平均発現量の5群へ分け、各群に含まれる全
`cell × gene`値をflattenした分布を、学習済みgenerated sampleと比較します。
対象順は固定です。

1. `ode_only_lincomb__hill_after_linear`
2. `ode_only_lincomb__racipe`
3. `ode_only_lincomb__exp`
4. `standard_hybrid_lincomb__hill_after_linear`
5. `standard_hybrid_lincomb__racipe`
6. `standard_hybrid_lincomb__exp`

名前からrunを決めず、各`manifest.json`と`exp_config.json`のexperiment、family、
ODEを照合します。defaultは`runs/<experiment>/20260803_full_30000`です。train stageが
completedで、同stepのraw model・optimizer・EMAが揃う最新checkpointだけを使います。

## Sampleの再利用

既存sampleはsidecar manifestに記録されたcheckpoint絶対パス、sample shape、seed、
diffusion stepsが選択checkpoint/configと一致する場合だけ再利用します。`--plot-only`
は一致するsampleがなければ停止します。通常実行で不足する場合は既存model factory、
diffusion、DDPM/DDIM設定からstrict restoreしてsamplingし、新しい解析runの
`generated_samples/`へ保存します。元runのmanifest、checkpoint、sampleは変更も
上書きもしません。

6モデルの`data_dir`, layer, sampling数、seed、diffusion steps/schedule、DDPM/DDIM、
clip設定が一致しない場合は解析を中断します。

## Real dataと比較空間

各runの保存済みconfigからAnnDataと`ts_layer`を解決します。今回のtraining loaderは
`train_vae=True, preprocess=False`であり、modelはAnnData `.X`（layer指定時はその
layer）を直接学習しています。したがってrealとgeneratedはこの保存表現空間で比較
します。正式なinverse transformはなく、推測による逆変換、log変換、標準化、clipは
追加しません。

backed AnnDataをrow chunkで読み、shape、gene数、`var['gene_name']`の一意な順序、
NaN/Infを検査します。sparse/denseと値域はmetadataへ保存します。

## 5遺伝子group

全real cellにおける各geneの平均をfloat64で計算し、平均降順、同値なら元gene index
昇順で安定に並べます。`numpy.array_split`で可能な限り均等な5群へ分けるため、全gene
は重複なく一度だけ現れます。`group_0`が最高、`group_4`が最低です。この分割を全model
で共有し、generated dataから再計算しません。

`gene_group_assignments.csv`にはgroup、gene index/name、real mean、group内順位、
全体順位を、`gene_group_summary.csv`にはgene数とreal meanのmin/max/medianを保存します。

## 分布・bin・subsampling

group内ではgeneを区別せず、cell平均やgene平均へ縮約しません。主図にはゼロを含む
全値を使います。default `--bins 200`です。同じgroupのreal＋全selected generatedを
合わせたdefault percentile `0.1,99.9`から共通bin edge/x範囲を決め、ゼロが存在すれば
必ず0を範囲へ含めます。変更は`--display-percentiles LOW,HIGH`です。

外側の値はデータから削除せず、各分布の範囲外割合をCSVへ記録します。密度は
`bin count / (範囲外も含む全描画値数 × bin幅)`なので、表示範囲外を暗黙に再正規化
しません。同じgroupではbin、x軸、y軸を全modelで共有します。

描画はdefault最大1,000,000値、real/generatedの少ない方へ揃えた固定seed sampleを
使います。`--plot-subsample 0`で全値です。mean/std/median/zero fraction/min/maxは
全値から計算します。Wasserstein/KSだけはdefault同数200,000値を使い、件数をCSVへ
保存します。

## 出力

各実行は上書きせず`outputs/YYYYMMDD/HHMMSS/`を作ります。

```text
figures/
  distribution_real_vs_generated_6x5.png/.pdf
  distribution_real_vs_generated_6x5_nonzero.png/.pdf
  real_distribution_by_expression_group.png/.pdf
  zero_fraction_by_group.png/.pdf
tables/
  gene_group_assignments.csv
  gene_group_summary.csv
  distribution_summary.csv
  zero_fraction.csv
metadata/
  run_metadata.json
  histogram_bins.json
  command.txt
generated_samples/        # 既存sampleが無い場合だけ
```

PNGは320 dpi、PDFはvectorです。subset解析では主図名の行数が`3x5`等になります。
`distribution_summary.csv`にはfamily、ODE、experiment、checkpoint/step、group、gene数、
real/generatedのvalue数、mean、std、median、zero fraction、min/max、表示範囲外割合、
Wasserstein、KS、距離計算subsample数を保存します。

## 再現性・制限

- 全乱数は`--seed`（default 1234）から決定します。
- exact medianのためgroupごとにreal flatten配列を一時的にmemoryへ載せます。1024 gene
  なら約204〜205 gene/groupで、float32配列はgroup単位です。
- percentile範囲は裾を削除する処理ではなく表示範囲です。範囲外割合を必ず記録します。
- zero-only groupでは非ゼロ補助図を0 densityとして表示します。
- NaN/Infは黙って除去・置換せず停止します。
- サーバーrunがないmachineでは本番図を生成できません。`--dry-run`もrun/config/
  checkpointの実在を検査します。

検証は次です。

```bash
conda run -n scdiffusion python -m unittest -v \
  work/20260803_distribution/tests/test_distribution.py
python work/20260803_distribution/verify_protected.py
```
