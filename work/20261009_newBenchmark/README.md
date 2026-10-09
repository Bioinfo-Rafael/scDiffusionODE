# Mouse Gastrulation: CellUNet単独 Stage 1

旧`work/1009_newBenchmark`を`work/20261009_newBenchmark`へ移動しました。既存ファイル・生成結果を保持しています。
このworkの`evaluate.py`は従来の一括評価を維持するため、共通入口に`--evaluation full`を明示します。
共通benchmark自体の既定値は3-foldです。既存予測を3-foldで評価する場合は
[`benchmark/README.md`](../../data_preparation/20261007/benchmark/README.md)の手順で別名の結果を作成してください。

### Remote: pull後に旧run成果物を移す

Git更新後、旧ディレクトリに残った実験成果物を新しい場所へ移します。既存ファイルは上書きせず、同一内容の重複だけを整理します。
内容が異なるファイルが同じ相対パスにある場合は旧側に残し、競合パスを表示します。

```bash
cd /home/suzuki/Projects/scDiffusion-github
git pull --ff-only origin main
python3 - <<'PY'
from pathlib import Path
import hashlib
import shutil
import sys

old = Path("work/1009_newBenchmark")
new = Path("work/20261009_newBenchmark")
conflicts = []

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def move_contents(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for item in list(src.iterdir()):
        target = dst / item.name
        if not target.exists():
            shutil.move(str(item), str(target))
        elif item.is_dir() and target.is_dir():
            move_contents(item, target)
        elif item.is_file() and target.is_file() and digest(item) == digest(target):
            item.unlink()
        else:
            conflicts.append(str(item))
    if not any(src.iterdir()):
        src.rmdir()

if old.exists():
    move_contents(old, new)
if conflicts:
    print("Different files were left at the old path; review before removing them:")
    print("\n".join(conflicts))
    sys.exit(2)
print(f"Migrated old run artifacts into {new}")
PY
```

移行後に旧パスの競合ファイルが表示された場合、旧ディレクトリは残ります。内容を確認してから手動整理してください。

`work/20260915_x0predict` の **START_X予測のStage 1** を、新しいMouseGastrulationの全89,267細胞・1,024 HVGで学習する。Stage 2、ODE、Hybrid、GRN/TSV、soft constraintは使わない。長時間の本学習は自動実行していない。

この実験では**計算上の仮定**として、CellUNetの生出力を $ds/dt$ と定義する。全細胞/Erythroidでその場を可視化し、既存VeloEVのCBDir・ICVCoh・CTO・TSCを計算する。出力の予測対象がx_startであることは、この評価を止める条件にしない。他の実験ではHybridやODEのみから別の $V(x)$ を定めてもよく、各実験が実際の計算式・入力・時間条件を明記する。

## 採用した既存コードと設定

| 調査元 | 確認内容・今回の採否 |
| --- | --- |
| `work/20260913_2step/{common.py,models/,training/,sampling/}` | Stage 1はCellUNet単独、`predict_xstart=False`＝epsilon予測。モデルfactoryのStage 1分岐、AdamW、EMAを直接再利用。 |
| `work/20260915_x0predict/{common.py,configs/base.json,training/objectives.py,training/runner.py,sampling/trajectory.py}` | **今回の採用条件**。Stage 1は `predict_xstart=True`＝x_start予測。設定・loss関数を直接import。サンプリングは同じ `diffusion.p_sample` を1,000回呼ぶ。 |
| `work/20260916_x0predict_hybrid_additive/{common.py,reuse.py}` | 9/15のStage 1を継承。独自のStage 2や短縮実行設定は採用しない。 |
| `work/20260917/{configs/base.json,common.py,reuse.py}` | 9/15のStage 1 EMA 30,000-step checkpointを参照する。10,000-stepの後段学習設定は今回のStage 1設定ではない。 |
| `work/20260803_ODE_hill_exp/configs/base.json` | 9/13→9/15が継承する30,000-step等の学習既定値。 |
| `guided_diffusion/{cell_model.py,script_util.py,gaussian_diffusion.py,respace.py,resample.py}` | Cell_Unet、START_X MSE、noise schedule、q_sample、ancestral p_sampleを変更せず利用。 |
| `work/20260830/hematopoietic_viz/{core.py,plotting.py}` | UMAP、scVelo graph→embedding→stream/arrow/gridを直接再利用。古いscVelo互換処理も元helperを利用。 |
| `work/20260215_embryonic/20260330_Analyze-20260224Lamda5/embed_velocity.py` | 可視化APIを確認。モデル出力の生物学的意味は継承しない。 |
| `work/20260830/`, `work/20261008_ode_optimizer/` | ODE consistency/別optimizer条件を確認。今回の学習設定には使わない。 |
| `data_preparation/20261007/{prepare_mouse_gastrulation.py,README.md,benchmark/}` | 入力の正規化・ID・benchmark契約を確認。benchmark本体の前処理・評価式・保存先は変更しない。 |

実効設定は `runs/effective_config.json`、元コードSHA256はそこおよびcheckpointに保存する。設定は既存Stage 1から必要な項目だけ取得し、不要なGRNパス等は持ち込まない。

| 項目 | 採用値 |
| --- | --- |
| architecture | `Cell_Unet(input_dim=1024, hidden_num=[2000,1000,500,500])`。元と同じLayerNorm/SiLU/skip/time embedding。dropout引数既定0.1だが、元のforwardでは末端dropoutは呼ばれず、ResidualBlock内dropoutは0。 |
| diffusion | 1,000 steps、linear beta schedule（1e-4→0.02）、respacingなし、時刻rescaleなし |
| prediction / loss | START_X / MSE、uniform timestep、learn_sigma=False、use_kl=False |
| optimizer | AdamW、lr=1e-4、weight_decay=1e-4、既定betas/eps |
| 学習 | 30,000 updates、batch=128、float32 parameters、学習noiseは元と同じfloat64、AMPなし |
| LR | 元と同じ各更新の後に `lr=1e-4*(1-index/30000)`（indexは0始まり） |
| EMA / seed | 0.9999 / 1234 |
| 保存 / log | 1,000 stepsごとcheckpoint、50 stepsごと標準出力、loss CSVは毎step |
| sampling | EMA、3,000細胞、batch=50、Gaussian初期値、ancestral p_sample、clip_denoised=False、DDIMなし |

`p_sample` の既定 `nw=0.5` は `exp(0.5 * log_variance)` という**標準偏差**の係数であり、「ノイズ分散を0.5倍」という意味ではない。元と同じFIXED_LARGE分散を使う。最終サンプルの負値をclipする追加処理はしない。

## 出力と評価用velocityの定義

$x_0$ は全遺伝子で独立1e4正規化済みのsplicedからHVG列を取った線形発現量、$k\in\{0,\ldots,999\}$ は拡散index。元実装は

$$x_k=\sqrt{\bar\alpha_k}x_0+\sqrt{1-\bar\alpha_k}\,\epsilon,\quad \epsilon\sim N(0,I)$$

に対して、$f_\theta(x_k,k)=\widehat{x_0}$ を返し、$\mathbb{E}_{x_0,k,\epsilon}\|f_\theta(x_k,k)-x_0\|^2$ の遺伝子平均・batch平均を最小化する。**epsilonを出力する9/13の条件とは異なる。**

このworkで採用する実験仮説と計算式は、正規化済み線形spliced発現 $x$ と固定した拡散条件 $k=49$ に対し

$$V(x)=f_\theta(x,49),\qquad ds/dt:=V(s)$$

である。`export_velocity.py`は**ノイズを加えず**に実細胞の学習用Xをそのまま入力し、EMA checkpointのCellUNet生出力を `layers['velocity']` へ保存する。$-x$、epsilon/x_start変換、正規化、時間割り算は加えない。`k=49`はモデルへの拡散条件であり、上式のモデル時間$t$そのものではない。`time_direction='forward'`と`time_unit='model time (arbitrary unit)'`はこの仮説に従う宣言で、胚日への校正や時刻の推定を含まない。モデル時刻列は作らず、既存VeloEVのvelocity pseudotime fallbackを利用する。

Stage 1の学習損失がx_startのdenoising MSEであることと、評価用に出力を$ds/dt$と**定義して扱うこと**は別の記述である。VeloEVの指標はこの仮説で定めた場と既知の細胞遷移の整合性を数値化する。生物学的機構の証明を評価実行の条件にはしない。

2026-10-09の初回実行で作成された `predictions/cellunet_field.h5ad`、`x_start.npy`、`displacement.npy` と `figures/{all,erythroid}/` は、旧式 $f_\theta(q(x,49),49)-x$ の成果物である。今回の修正版は `predictions/cellunet_direct_t49.h5ad` と `figures/{all,erythroid}_direct_t49/` に別保存し、旧ファイルを上書きしない。旧 `metrics/benchmark_status.json` の `not_applicable` はcheckpointや予測を検査せずに出した固定statusであり、新しいVeloEV結果ではない。

## データとID

入力は `data_preparation/20261007/data/MouseGastrulation.h5ad`。形状89,267×53,801、5 celltypeのErythroid 9,815細胞、正規化provenance、非負・有限、全遺伝子でのS/U行和1e4（ゼロ行は保持）、`X==spliced`、一意なcell/gene IDsを検証する。

HVGは全細胞のXの**一時コピー**にだけlog1pし、既存benchmarkと同じScanpy Seuratのnormalized dispersionを計算する。学習側は正確に1,024列を要求するため、finiteな`dispersions_norm`の降順、同率なら元列順で上位1,024を選び、**保存順は元列順**に戻す。Scanpyの閾値選択が同率で1,024を超える場合との違いを意図的に明記する。benchmark側のHVG選択は変更しない。

学習用Xは元Xの列抽出だけ。再正規化・log1p・scaleは行わない。X以外の大きいlayers・旧embedding・rawは学習用h5adへ複製しない。S/Uは元データに残る。`gene_mapping.csv`に元のvar情報、gene ID、`gene_name`、元列番号、dispersionを保存し、`gene_ids.txt`とmetadataのcell/gene ID一覧で対応を保持する。gene symbolへの変換・suffix除去・IDの自動一意化はしない。元データ作成scriptがvar_namesをgene symbolとして保存していても、それをそのままIDと扱い、Ensembl IDを捏造しない。

学習loaderは `preprocess=False` を必須とし、元 `load_data(train_vae=True, preprocess=False, layer=None)` と同じfloat32・shuffle・drop_last・collateを採用する。元loaderの全行dense化を避け、行単位で読み出してミニバッチだけdense化する薄いadapterを追加した。dedicated shuffle RNG・epoch cursorと、Torch/NumPy/Python/CUDA RNG、optimizer、EMA、stepをcheckpointへ保存し、resumeで復元する。元runnerがStage 1 resumeを拒否するため、この保存・復元部分は今回の追加機能。

## 実行

既存学習環境を利用する。確認済みローカル環境はPython 3.9 / Torch 2.5.1 / Scanpy 1.9.3 / scVelo 0.2.5。学習環境へのinstall/updateは行っていない。リポジトリの場所はscriptから解決し、`config.json`内の相対パスはリポジトリroot基準。

```bash
cd /home/suzuki/Projects/scDiffusion-github
# 使用する既存学習環境をactivateしてから実行
python work/20261009_newBenchmark/prepare_data.py
python work/20261009_newBenchmark/train.py --device cuda
python work/20261009_newBenchmark/sample.py --device cuda
python work/20261009_newBenchmark/visualize.py --scope samples
python work/20261009_newBenchmark/export_velocity.py --device cuda
python work/20261009_newBenchmark/visualize.py --scope all
python work/20261009_newBenchmark/visualize.py --scope erythroid
python work/20261009_newBenchmark/evaluate.py
```

CPUは `--device cpu`、自動選択は `--device auto`（CUDAがあればCUDA、それ以外CPU）。MPSは元のfloat64 noise条件に対応しないため選択肢に含めない。`sample.py` / `export_velocity.py`には `--checkpoint /absolute/path/model030000.pt` を指定できる。省略時は `runs/checkpoints/latest.json` を使用。各工程に `--config /absolute/path/config.json` を指定できる。

```bash
# 短い実データsmoke: 同じarchitecture/loss/diffusionで2更新だけ
python work/20261009_newBenchmark/train.py --device cpu --steps 2
# そこから30,000更新へ継続（--stepsは追加数でなく絶対到達step）
python work/20261009_newBenchmark/train.py --device cuda --resume
# 特定checkpointから継続する場合
python work/20261009_newBenchmark/train.py --device cuda --resume /absolute/path/model002000.pt
# 生成smoke（1000 diffusion stepsは維持）
python work/20261009_newBenchmark/sample.py --device cpu --count 4
```

smoke生成後に3,000生成へ変える場合やcheckpointを変える場合は、既存samples/fieldを上書きしないため、以前の成果物を別途退避するか、別の `runs` を使う。入力・checkpoint・設定・IDの一致を検証できる同じ成果物だけ再利用する。不完全な生成物は成功扱いしない。異なるresume branchを同じrunsへ書く場合、既存stepのcheckpointは上書き拒否するので、別runsを使う。checkpointは信頼できるこのworkflowの出力だけを指定する。

一括実行は次の通り。**未学習なら本学習を開始するコマンド**なので、実行時間を確保した上で使う。

```bash
PYTHON_BIN=python STAGE1_DEVICE=cuda bash work/20261009_newBenchmark/run_all.sh
```

完了済みcheckpoint/同一生成物を再利用し、途中checkpointなら学習をresumeする。既存の学習・samplingを再実行せず、今回の直接場のexport、全細胞/Erythroidの図、VeloEVだけを個別に実行することもできる。

```bash
python work/20261009_newBenchmark/export_velocity.py --device cuda
python work/20261009_newBenchmark/visualize.py --scope all
python work/20261009_newBenchmark/visualize.py --scope erythroid
python work/20261009_newBenchmark/evaluate.py
```

初回のVeloEV referenceがなければ`evaluate.py`が既存`benchmark/prepare.py`を呼ぶ。専用`.venv-eval`とVeloEV submoduleが未準備なら、既存benchmark READMEのセットアップを先に行う。評価が成功すればexit 0。失敗時はこのworkの新しいbenchmark log/statusと、既存benchmarkの`results/<method>/run.log`/`metadata.json`に原因を残す。失敗済みmethod名の結果は既存runnerの上書き拒否を尊重し、再試行時は`--method`に新しい名前を指定する。

`uv python install 3.12.14` が `No download found` で止まる場合は、まず `uv --version` を確認する。CPython 3.12.14の配布情報はuv 0.12.5以降に含まれる。古いuvを使っている場合、[uv公式インストーラー](https://docs.astral.sh/uv/getting-started/installation/)で新しい実行ファイルをユーザー領域へ入れ、絶対パスで呼ぶ。`setup_env.sh`は評価専用Python 3.12.14を必須とするため、空の`PYTHON_BIN`を渡してもscdiffusionのPython 3.9では代用できない。

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
"$HOME/.local/bin/uv" python install 3.12.14
PYTHON_BIN="$("$HOME/.local/bin/uv" python find 3.12.14)" bash data_preparation/20261007/benchmark/setup_env.sh
python work/20261009_newBenchmark/evaluate.py
```

この再実行は既存の予測h5adと可視化を再生成せず、未作成のreferenceとVeloEV評価のみ進める。失敗の詳細は`runs/logs/cellunet_direct_t49_benchmark.log`を見る。

remoteで `scdiffusion` 環境を明示的にactivateした後、次の一括コマンドで `origin/main` をfast-forward pullし、入力を確認してバックグラウンド実行できる。base環境のPythonには`anndata`等がない場合があるため、`RUN_PYTHON`はactivate後に取得する。実行中のPID、全ログ、最終exit codeをこのworkの`runs/logs/`に残す。旧実行のexit code 2や`not_applicable`ファイルは今回の指標計算結果ではない。

```bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scdiffusion
cd /home/suzuki/Projects/scDiffusion-github
git pull --ff-only origin main
test -f data_preparation/20261007/data/MouseGastrulation.h5ad
RUN_PYTHON="$(command -v python)"
test -n "$RUN_PYTHON"
mkdir -p work/20261009_newBenchmark/runs/logs
nohup env PYTHON_BIN="$RUN_PYTHON" STAGE1_DEVICE=cuda bash -c '
  bash work/20261009_newBenchmark/run_all.sh
  result=$?
  printf "%s\n" "$result" > work/20261009_newBenchmark/runs/logs/remote_exit_code.txt
  exit "$result"
' > work/20261009_newBenchmark/runs/logs/remote_pipeline.log 2>&1 < /dev/null &
printf "%s\n" "$!" > work/20261009_newBenchmark/runs/logs/remote_pipeline.pid
```

状態確認: `cat work/20261009_newBenchmark/runs/logs/remote_pipeline.pid`、`tail -f work/20261009_newBenchmark/runs/logs/remote_pipeline.log`、終了後 `cat work/20261009_newBenchmark/runs/logs/remote_exit_code.txt`。GPUがない場合は `STAGE1_DEVICE=cpu` とする。最初の学習開始時は30,000 updatesを実行する。

すでに30,000-step checkpointとsamplingがあるremoteでは、次の3工程だけで修正版の結果を追加できる。初回の`remote_exit_code.txt`は旧判定の履歴なので、修正版は別ログに残す。

```bash
cd /home/suzuki/Projects/scDiffusion-github
git pull --ff-only origin main
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate scdiffusion
nohup bash -c '
  python work/20261009_newBenchmark/export_velocity.py --device cuda &&
  python work/20261009_newBenchmark/visualize.py --scope all &&
  python work/20261009_newBenchmark/visualize.py --scope erythroid &&
  python work/20261009_newBenchmark/evaluate.py
  result=$?
  printf "%s\n" "$result" > work/20261009_newBenchmark/runs/logs/direct_t49_exit_code.txt
  exit "$result"
' > work/20261009_newBenchmark/runs/logs/direct_t49_pipeline.log 2>&1 < /dev/null &
printf "%s\n" "$!" > work/20261009_newBenchmark/runs/logs/direct_t49_pipeline.pid
```

`tail -f work/20261009_newBenchmark/runs/logs/direct_t49_pipeline.log`で進捗を確認する。VeloEV実行中の詳細ログはstatus JSON内に記録された既存benchmarkの`run.log`に保存される。専用`.venv-eval`やvendor submoduleがない場合は、`benchmark/README.md`に従って先にセットアップする。


評価referenceだけ作る必要がある場合は既存専用venvを構築済みであることを確認して、次を実行できる。

```bash
python work/20261009_newBenchmark/evaluate.py --prepare-reference
```

これは既存 `data_preparation/20261007/benchmark/prepare.py` をsubprocessで呼び、そこが専用 `.venv-eval/bin/python` へ移る。既存referenceの上書き拒否も維持する。通常の`evaluate.py`もreferenceがなければ自動準備し、`run.py`でVeloEVを実行する。専用venv/固定VeloEVのセットアップ仕様は既存benchmark READMEを参照。

## 可視化と成果物

全細胞とErythroidはそれぞれ**独立に**PCA→近傍→UMAPを再計算する。学習と同じ線形Xをそのまま使い、旧UMAP・benchmark用geometryを再利用しない。HVG subset後の再正規化も行わない。実／生成比較は全89,267実細胞と3,000生成細胞のjoint UMAPで、生成細胞にcelltypeは与えない。実細胞celltype図も同じ座標で保存する。

可視化は元helperの50 PCs、40 neighbor PCsを使い、近傍数はこのworkのconfigで30を明示する。細胞数/遺伝子数が小さいsyntheticでは元helperと同様に利用可能な数へ減らし、実効値をmetadataへ記録。scVelo graphは線形Xと**生のCellUNet出力**のcosine graph、全モデル遺伝子、既定のapproxなし・sqrt transformなし。scVeloのgraph/embeddingは内部でcosine相関やベクトル中心化を行うため、描画上のUMAP矢印の数値は生の1024次元出力そのものではない。生出力の正本は予測h5adの`velocity`層。細胞subsamplingやvelocity graphのapproxは使わない。UMAP用近傍探索は元のScanpy自動選択を継承し、大規模入力ではNN-descent近似探索になり得ることをmetadataに記録する（scVelo graphのapproxとは別）。scVelo velocity graphの`n_jobs`はconfigで変更可能で、既定は32。32並列ではRAM使用量も増えるため、リモートの空きメモリを確認する。

```text
work/20261009_newBenchmark/
  common.py                 元Stage 1 adapter、設定・ID・checkpoint検証
  config.json               データ/可視化条件（学習条件は元Stage 1から取得）
  prepare_data.py            HVG学習データ
  train.py                  学習・resume・loss CSV/PNG
  sample.py                 3,000細胞ancestral sampling
  export_velocity.py        V(x)=CellUNet(x,t=49)の生出力をvelocityとしてexport
  visualize.py              samples/all/erythroidの独立再実行
  evaluate.py               既存VeloEVのreference準備・4指標実行
  run_all.sh
  tests/test_pipeline.py
  runs/
    data/{training.h5ad,gene_mapping.csv,gene_ids.txt,metadata.json}
    checkpoints/{modelXXXXXX.pt,latest.json}
    samples/{generated.h5ad,metadata.json}
    predictions/{cellunet_direct_t49.h5ad,cellunet_direct_t49.npy,cellunet_direct_t49_metadata.json}
    figures/loss.png
    figures/samples/{real_vs_generated.png,real_celltype.png,embedding.h5ad,completed.json}
    figures/{all,erythroid}_direct_t49/{celltype.png,stage.png,field_stream.png,field_arrow.png,field_grid.png,embedding.h5ad,completed.json}
    metrics/{cellunet_direct_t49_benchmark_status.json,cellunet_direct_t49_metrics.csv}
    logs/{loss_*.csv,cellunet_direct_t49_benchmark.log,failure_*.log,tests_direct.log,...}
```

VeloEVの正本は既存の`data_preparation/20261007/data/benchmark/results/<method>/`へ保存される。work内の`metrics/cellunet_direct_t49_metrics.csv`は同じCSVのコピーで、status JSONに正本・入力・checkpointへの参照を残す。benchmark専用UMAPは既存referenceのgeometryを使い、このworkの可視化UMAPと混同しない。モデル時刻を渡さないので、既存benchmark側のvelocity pseudotime fallbackを使う。

メモリ: HVG準備時は元のsparse AnnDataと一時XコピーがRAMに必要。学習はミニバッチだけdense化。field exportはfloat32のmemmapを1枚利用（約366 MB）し、h5adにも保存する。scVeloは全89,267×1,024の発現/velocityを内部dense化し複数コピーを持つため、可視化は数GB以上のRAMを要する。全細胞での時間・ピークRAMはローカルで未測定。十分なメモリのあるリモート環境で実行する。

## 検証

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MPLBACKEND=Agg \
  python -m unittest discover -s work/20261009_newBenchmark/tests -v
```

syntheticは80細胞×64遺伝子、うちErythroid60細胞、32 HVG。テスト内のみhidden幅を縮小し、実際のCell_Unet・START_X loss・1,000 diffusion stepsを使う。線形XとID保持、source未変更、preprocess拒否、optimizer/EMA/RNG resume、sampler、生出力の一致、3種UMAP・stream/arrow/grid、ID不一致拒否、既存benchmark CLIへの連携と結果再利用を検証する。VeloEV公式4指標の**実計算**はsynthetic mockの範囲外であり、リモートのreference/専用venvで実行する。テスト成果物は `runs/logs/synthetic_*/` に保持する。

実行結果と未実行範囲は `VALIDATION.md` に記録する。
