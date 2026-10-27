# MatSum / LoRA の細胞別係数・共有基底解析（2026-10-27）

**入口は [`run.py`](run.py)、実行順序は `analyze_model()` にあります。** 学習済みの D16（MatSum）・D20（LoRA）、CellUNet–ODE consistency λ=1 を読み、再学習せず解析します。新しいブランチは作りません。

## 結果は `index.html` から見る

PNGを個別に探す代わりに、解析出力直下の **`index.html`** をブラウザで開いてください。バックグラウンド実行なら `outputs/job_…/analysis/index.html` です。

- 最初に見る5枚：符号付き平均、⚠絶対値平均、細胞ごとに正規化した構成比、PCA、UMAP。
- モデル・拡散tを切り替えられます。分布・行列統計はcelltypeを選んで表示します。
- 各図に日本語で「何を見るか」「軸・点・色」「解釈上の注意」を添えています。
- 基底比較・28散布図・cluster分割表は詳細として折りたたんであります。
- ページ末尾の全ファイル辞典は、ファイル名・celltype・説明文で検索できます。
- B＝共有基底、C＝aを掛けた成分、W＝8成分を合成した重みです。基底間stdと細胞間stdを区別して説明します。

今後の実行は終了時（途中失敗時も結果フォルダがあれば）に自動生成します。**既存結果には再計算せず後付けできます。** 修正版をpull後、接続先で以下を実行してください。

```bash
# outputs内の既存runすべてに案内を追加（NumPy/PyTorch/Matplotlibすら不要）
python work/20261027_matsum_lora_analysis/report.py \
  work/20261027_matsum_lora_analysis/outputs --all

# 特定runだけの場合（jobディレクトリを指定してもanalysis/を解決）
python work/20261027_matsum_lora_analysis/report.py \
  work/20261027_matsum_lora_analysis/outputs/job_YYYYMMDD_HHMMSS_PID
```

生成されたパスを表示します。VS CodeのHTMLソース表示ではなく、HTMLプレビューまたはブラウザで開いてください。別PCへコピーする場合は、`index.html` 単体ではなく同じ解析runのフォルダ全体をコピーします。図へのリンクは相対パスで、外部サービスやネット接続は不要です。実行中の結果に後付けした場合は部分的な案内になるため、完了後に再度生成してください。既存PNG・CSV・NPZは変更しません。

## 接続先で実行する

VS Code Remote Tunnel の**接続先のターミナル**で、既存の `scdiffusion` 環境を使います。このディレクトリと既存リポジトリのコードを接続先にも配置してください。このスクリプト自体はトンネル接続・同期・環境変更を行いません。

```bash
cd /home/suzuki/Projects/scDiffusion-github
conda activate scdiffusion

# 1. 実checkpoint・データ・ID・source hash・元forwardとの一致を確認
python work/20261027_matsum_lora_analysis/run.py --check-only --device cuda

# 2. D16とD20の本解析（新規学習なし）
python work/20261027_matsum_lora_analysis/run.py --device cuda
```

### pullからバックグラウンド起動まで

修正版をpush済みの状態で、接続先のターミナルで実行します。

```bash
(
  set -e
  cd /home/suzuki/Projects/scDiffusion-github
  git pull --ff-only
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate scdiffusion
  bash work/20261027_matsum_lora_analysis/run_background.sh --device cuda
)
```

`run_background.sh` は確認処理も含めてnohupで起動します。確認が成功した場合だけ本解析へ進みます。起動時にPIDと `tail -f` コマンドを表示します。出力は `outputs/job_<JST日時>_<PID>/` 内の `check/` と `analysis/`、共通ログは `console.log`、終了コードは `exit_code`（0なら両段階成功）です。`tail -f` のCtrl+Cはログ表示だけを止め、ジョブは継続します。

`run_background.sh` に `--run-root`、`--data-dir`、`--models`、`--weights`、`--config` も渡せます。`--output` は起動スクリプトが管理します。人工smokeは前述の `run.py --smoke` を使います。

起動直後にNumPy/PyTorch読込の進捗を表示します。`--help` は数値・描画ライブラリを読みません。`--check-only` はMatplotlib/scikit-learn/UMAPを読みません。本解析はモデル読込後に描画ライブラリを読み、その開始・完了をログに表示します。確認途中でのCtrl+Cは、作成済みmanifestに `status=interrupted` を記録します。

既定の読込先は次です。絶対パスへの書き換えは不要で、リポジトリルートを基準に解決します。

```text
work/20261010_model_comparison/runs/model_expansion_20261012/
  D16_matsum_consistency_lambda1/
    effective_config.json
    checkpoints/latest.json → 最終checkpoint
  D20_lora_consistency_lambda1/
    effective_config.json
    checkpoints/latest.json → 最終checkpoint
work/20261010_model_comparison/runs/data/manifest.json
```

場所が異なるときだけ指定します。manifest内のtraining/GRNパスも接続先で有効である必要があります。

```bash
python work/20261027_matsum_lora_analysis/run.py \
  --run-root /path/to/model_expansion_20261012 \
  --data-dir /path/to/runs/data \
  --models matsum lora --weights ema --device cuda \
  --output work/20261027_matsum_lora_analysis/outputs/full_ema_t49
```

出力先を省略すると `outputs/full_ema_YYYYMMDD_HHMMSS/`（JST）を作ります。既存の非空ディレクトリには書き込みません。再実行時は新しい出力先を選びます。途中失敗は `manifest.json: status=failed` と `error.txt` に残り、完走を装いません。自動resumeはありません。

**`--check-only` が通るまで、本解析は実行しないでください。** source hash不一致を無視するオプションはありません。既存のレビュー済み互換規則を利用し、それ以外の変更は停止します。元run/checkpointは書き換えません。

### 短い動作確認とテスト

```bash
# 人工48細胞×16遺伝子、元のMatSum/LoRAクラス、ランダム重みで全工程
python work/20261027_matsum_lora_analysis/run.py --smoke --device cpu

# 数式・元forward・実形式checkpoint読込・改変拒否・遺伝子順位計算を検証
python -m unittest discover -s work/20261027_matsum_lora_analysis/tests -v
```

`--smoke` は学習済みD16/D20を読みません。人工データの結果と実細胞の結果を混ぜないでください。必要パッケージは [`requirements.txt`](requirements.txt)。基本は既存モデルの環境をそのまま利用します。各実行の正確なパッケージ版・解析コードhash・設定・checkpoint hashは出力manifestに記録します。CPUでは `--device cpu` を使用できます。

## 依頼された出力だけを生成する

入口 `run.py:analyze_model()` → `analyze_basis.py:run_basis_analysis()` → `analyze_coefficients.py:run_coefficient_analysis()` → `analyze_cells.py:run_cell_analysis()` の順です。数式は `numerics.py` と `spectral.py`、描画は `plots.py` にあります。

| 対象 | 残す図・数値 |
|---|---|
| a | celltype別の符号付き平均・⚠絶対値平均・細胞ごとにL1正規化した絶対値の平均の積み上げ図 |
| aの分布 | 全体・celltype別の各成分の分散CSVと符号付きhistogram、全28ペア散布図 |
| aの8次元構造 | 元の符号付きaのPCA・UMAP（celltype色）、クラスタ×celltypeの件数表 |
| 8つのB | 全重みヒートマップ、要素ごとの基底間std・std/meanのhistogram、通常ラプラシアンの固有値スペクトル |
| B間比較 | 小さいq固有値のRMSD、低固有空間の固有ベクトル積のFrobeniusノルム、行列cosine距離の8×8表 |
| C=aB | 全体・celltype別の平均ヒートマップ、細胞方向std・std/meanのhistogram、celltype別抽出細胞のスペクトル |
| 遺伝子順位 | B・C・合成Wのincoming/outgoing順位CSV（従来の第6項を維持） |

クラスタ数は既定8、seed=0の1条件です。分散・stdは母分散（ddof=0）。
`mean(abs(a))` は平均的な絶対係数の大きさであり、`abs(mean(a))` とは異なるため図に警告を付けます。
割合は各細胞で `abs(a_k)/sum(abs(a))` としてからcelltype内で平均します。
Cの要素平均は `mean(a_k) B_k`、stdは `std(a_k) abs(B_k)` です。
std/meanは符号付き平均で割り、ゼロ付近は欠損にします。閾値と有効要素数は出力JSON/CSVに記録します。

グラフは `S=abs((B+B.T)/2)` の対角を0にし、`L=diag(S.sum(axis=1))-S` を固有値分解します。
Cのラプラシアン固有値はBの固有値の `abs(a_k)` 倍なので、再分解せず厳密に計算します。
小さい方からq本の固有ベクトルを列に持つQについて `||Q_i.T @ Q_j||_F` を表示します。
これは距離ではなく類似度です。固有値RMSDは `sqrt(mean((lambda_i-lambda_j)**2))`、cosine距離は `1-cos(vec(B_i),vec(B_j))` です。
縮退固有値の途中でq本を区切ると固有空間比較が不安定になります。クラスタとcelltypeの一致は関連を示すもので、因果関係を証明しません。

正規化ラプラシアン、ズームの重複図、正規化aの追加埋め込み、stage/sample色、発現UMAP上のa、Wの追加図・分散分解、成分除去解析は生成しません。
`matrix_moments.npz` は係数平均・std・共分散・絶対値平均だけを保存します。Bは `basis/shared_matrices.npz` に1回保存し、巨大なW統計配列の重複保存を廃止しました。

## 既存結果から追加出力を削除する

修正版コードを接続先へ配置してから、解析が停止した状態で実行します。再学習・解析の再実行は不要です。

```bash
conda activate scdiffusion
# 削除予定・拡張子別容量を表示するだけ
python work/20261027_matsum_lora_analysis/prune_outputs.py \
  work/20261027_matsum_lora_analysis/outputs --all
# 既知の追加出力を削除・冗長NPZを縮小し、HTMLも更新
python work/20261027_matsum_lora_analysis/prune_outputs.py \
  work/20261027_matsum_lora_analysis/outputs --all --apply
```

`requested_outputs.py` が残す図の一覧です。削除処理は旧コードが生成した既知の追加ファイルに限定し、不明なファイル、入力データ、学習checkpointには触れません。停止済みmanifestが必要です。変更内容はrun直下の `cleanup_*.json` に残ります。NPZの縮小にはNumPyが必要です。

**Gitで既にcommitした大きなファイルは、作業ディレクトリから削除するだけでは過去のcommitから消えません。** 2.28 GiBのpush失敗への対処には、整理後の結果を使って未pushのcommitを組み直す作業が別途必要です。

`VALIDATION.md` の旧出力件数は変更前の記録です。現在の範囲はこのREADMEと `requested_outputs.py` を参照してください。
