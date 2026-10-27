# MatSum / LoRA の細胞別係数・共有基底解析（2026-10-27）

**入口は [`run.py`](run.py)、実行順序は `analyze_model()` にあります。** 学習済みの D16（MatSum）・D20（LoRA）、CellUNet–ODE consistency λ=1 を読み、再学習せず解析します。新しいブランチは作りません。

## 結果は `index.html` から見る

PNGを個別に探す代わりに、解析出力直下の **`index.html`** をブラウザで開いてください。バックグラウンド実行なら `outputs/job_…/analysis/index.html` です。

- 最初に見る6枚：符号付き平均、⚠絶対値平均、細胞ごとに正規化した構成比、PCA、UMAP、Wのcelltype間分散比η²。
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

# 数式・元forward・実形式checkpoint読込・改変拒否・寄与計算を検証
python -m unittest discover -s work/20261027_matsum_lora_analysis/tests -v
```

`--smoke` は学習済みD16/D20を読みません。人工データの結果と実細胞の結果を混ぜないでください。必要パッケージは [`requirements.txt`](requirements.txt)。基本は既存モデルの環境をそのまま利用します。各実行の正確なパッケージ版・解析コードhash・設定・checkpoint hashは出力manifestに記録します。CPUでは `--device cpu` を使用できます。

## コードを読む順番・実装計画

| 順序 | 直接呼ばれる関数 | 内容 |
|---|---|---|
| 1 | `load_model.py:load_trained_model()` | 最終D16/D20を検証・復元、細胞IDで注釈結合 |
| 2 | `load_model.py:shared_matrices()` | 8個のBとW0を取得 |
| 3 | `analyze_basis.py:run_basis_analysis()` | 共有基底のヒートマップ、基底間std/CV、グラフ、8×8比較 |
| 4 | `load_model.py:extract_coefficients()` | 元モデルの `_coeffs(x,t)` で係数取得 |
| 5 | `run.py:verify_reconstruction()` | Bと係数による復元値を元の `compute_W` / `forward` と比較 |
| 6 | `analyze_coefficients.py:run_coefficient_analysis()` | 統計・histogram・積み上げ・28散布図・PCA/UMAP・cluster |
| 7 | `analyze_cells.py:group_weight_moments()` | CとWのcelltype別平均、std、比率、分散分解 |
| 8 | `analyze_cells.py:batched_cell_effects()` | 全細胞の成分寄与・除去効果・遺伝子順位 |
| 9 | `analyze_cells.py:sampled_cell_spectra()` | celltype別抽出細胞のグラフスペクトル |

計算式は [`numerics.py`](numerics.py)、グラフ・固有分解は [`spectral.py`](spectral.py) の短い関数に置いています。`plots.py` は描画、`artifacts.py` は保存のみです。モデル種類による分岐は復元とB取得に限定し、科学計算は共通の配列で実行します。

## 解析条件

[`config.json`](config.json) が全設定です。初期値は以下です。

| 項目 | 既定値 |
|---|---|
| 重み | EMA、最終stepのみ |
| 入力 | cleanな学習用X、全89,267細胞・HVG 1,024遺伝子（実configでも検証） |
| 拡散t | 49。生物学的stageとは異なる |
| 固有値・固有ベクトル比較の本数q | 小さい順に32本、ゼロ固有値も含む |
| 細胞別グラフの抽出 | celltypeごとに最大3細胞、seed=20261027 |
| 28散布図 | celltypeごとに最大300細胞。同じ抽出集合を全ペアで利用 |
| K-means | k=8,16,34、seed=0,1,2、n_init=10 |
| silhouette | 固定seedで最大5,000細胞、元8次元で計算 |
| 分散・共分散 | ddof=0（母分散としての記述統計） |
| 比率のゼロ近傍 | `abs(mean) <= max(1e-12, 1e-6*max(abs(mean)))` を未定義扱い |

変更する場合はconfigをコピーし、`--config path/to/config.json` で渡します。`timesteps` を `[0,49,499,999]` のようにすると各tを独立解析します。共有基底解析は1回だけです。複数tは実行時間と出力容量が増えます。

元データの追加obsとUMAPは `effective_config.input` を読み、hashと細胞IDを照合して結合します。HDF5からobs・UMAPのみ取得し、元の全遺伝子行列/layersを読みません。元h5adが存在しない場合は、trainingのcelltype/stageのみで解析し、その事実をprovenanceに記録します。

## 記号とモデルの式

細胞cの係数を `a_ck = a_k(x_c,t)`、共有基底をBとします。行iは標的、列jは制御元です。

```text
MatSum: B_k = A_k                         W0 = 0
LoRA:   B_k = U_k V_k^T, rank <= 16       W0 = 学習された共有行列

C_ck = a_ck B_k
W_c  = W0 + Σ_k C_ck
f(x,t) = softplus(W(x,t)x+b) - softplus(gamma)⊙x
```

A,U,Vは入力xに依存しません。係数はx,t依存で、softmaxなし、負値あり、和1の制約なしです。Bは共有パラメータ、Cは係数付き成分行列、Wは合成行列としてファイル名も区別します。

本解析は固定t・clean Xに対する応答であり、ノイズ付きの学習入力全体の分布を再現するものではありません。Wは合成重みであり、a(x,t)の微分を含むJacobianではありません。

## 係数の統計と可視化

- 符号付き平均 `mean_c(a_ck)`、絶対値化後の平均 `mean_c(abs(a_ck))` を**両方**出します。後者の図のタイトルには **⚠ WARNING: mean after absolute value; sign discarded** を付けます。両者は一般に異なります。
- 分散・std・正/負/ゼロの割合を全体・celltype別に出します。
- 符号付き/絶対値histogramは同じ成分で共通binを使用。celltype図には全細胞分布を重ね、密度で比較します。
- 割合は `p_ck=abs(a_ck)/Σ_l abs(a_cl)` のcelltype内平均と、`mean(abs(a_ck))/Σ_l mean(abs(a_cl))` を両方出します。前者は細胞を等重み、後者はL1総量の大きい細胞を強く反映します。全ゼロ係数は除外数を記録します。
- PCAは中心化のみ、成分標準化なし。UMAPとK-meansは元8次元のユークリッド距離を使用します。表現は符号付きa、符号付きL1正規化a、絶対値L1構成比の3通りです。
- celltype、stage、存在すればsample・sequencing.batchで同じ埋め込みを色付けします。既存発現UMAPに係数を色付けした図は別ファイルです。
- cluster×celltypeは件数・cluster内割合・celltype内割合を出し、ARI・AMI・silhouette・seed間ARIを記録します。kはcelltype一致で選択しません。
- 28ペアは符号付き値で描画します。全体の散布図行列と個別28枚、抽出細胞IDを保存します。抽出は表示用で、統計は全細胞です。

## 8個の共有基底の比較

### ヒートマップと要素ごとのばらつき

全1,024×1,024を描くヒートマップと、上位遺伝子の拡大図を出します。8基底で同じ遺伝子順序・色の尺度を使います。

```text
mu_ij    = mean_k(B_kij)
sigma_ij = sqrt(mean_k((B_kij-mu_ij)^2))
ratio_ij = sigma_ij / mu_ij
```

**基底方向kで計算**し、1024²個（対角を含む）のmean・std・std/mean・std/|mean|をhistogramにします。平均ゼロ近傍はNaN、件数と閾値を記録します。負の平均ならstd/meanも負です。NaNをゼロに置換しません。極端値を黙ってclipしません。

### グラフ・固有分解

各行列Bについて、**対称化してから要素絶対値**を取ります。

```text
S = abs((B+B.T)/2);  S_ii = 0
D_ii = Σ_j S_ij
L = D-S
Lnorm = D^(-1/2) L D^(-1/2)       # 孤立点の逆平方根は0
```

LとLnormの全固有値を小さい順に保存・描画します。数値丸めで生じる微小な負値も保存値ではそのまま残します。ゼロ固有値の個数と縮退の判定には `1e-9*max(1,max(abs(lambda)))` を使用し、実際の閾値も記録します。先頭q本の固有ベクトルをQとして保存します。q境界で固有値が縮退している場合、部分空間の切り方が一意でないため `cutoff_degenerate` を記録します。

### 8×8の比較行列

| 指標 | 式 | 意味 |
|---|---|---|
| 小固有値RMSD | `sqrt(mean_r<q((lambda_i,r-lambda_j,r)^2))` | 小さいほど近い。通常/正規化Lで別々に計算 |
| 固有ベクトル積のFrobeniusノルム | `||Q_i.T Q_j||F` | 大きいほど部分空間が近い。距離とは呼ばない |
| 正規化部分空間類似度 | `||Q_i.T Q_j||F / sqrt(q)` | 0〜1、自己比較1 |
| 部分空間距離 | `sqrt(max(0,1-||Q_i.T Q_j||F²/q))` | 正規化した射影行列距離、自己比較0 |
| 符号付き行列cosine | `<vec(B_i),vec(B_j)>/(||B_i||F ||B_j||F)` | 正負を保った元行列の比較 |
| cosine距離 | `1-cosine` | 0〜2。一般に三角不等式を満たすmetricではない |

固有ベクトル比較はユーザー確認済みの解釈です。全固有ベクトルを使うと直交基底どうしの積のノルムが一定になってしまうため、先頭q本を使います。符号反転・選択部分空間内の基底回転に不変です。元行列がゼロならcosineは未定義です。

## 細胞依存の係数付き行列

全体とcelltypeごとに、固定した要素(i,j)の**細胞方向**の平均・std・std/meanを求めます。

```text
mean(C_k) = mean(a_k) B_k
std(C_k)  = std(a_k) abs(B_k)
mean(W)   = W0 + Σ_k mean(a_k) B_k
var(W_ij) = B_:ij.T Cov(a) B_:ij
```

これらは近似ではなく、全細胞から求めた係数の平均・共分散による厳密な式です。Cの平均・stdは巨大な配列を重複保存せず、Bと係数の統計を保存して再構成できるようにします。Wの平均・std・比率は行列で保存します。

同じC_kの `std/|mean|` は、B_kijが非ゼロなら全要素で同じ値です。そのhistogramが狭いことを独立した生物学的結果と解釈しません。Wでは複数基底が合成されるので要素ごとに異なります。

### 全体とcelltype内の分散

```text
total = Σ_g p_g within_g + Σ_g p_g (mean_g-grand_mean)^2
eta²  = between / total, p_g=n_g/N
```

係数aとWの要素について出します。全体分散0のeta²は未定義です。「celltype内のばらつきが小さく、celltype間で違いが大きい」は検証する仮説であり、結果を前提にしません。aについてstage別にも同じ分解を出します。これはstage内の記述的比較で、sample交絡や生物学的因果を除去したことにはなりません。

### 入力への寄与・成分除去

```text
成分行列の規模: |a_ck| ||B_k||F
活性化前の寄与: r_ck = a_ck B_k x_c;  ||r_ck||2
成分除去効果:   ||f(x_c,t)-f^(-k)(x_c,t)||2
```

成分除去では他の係数を再計算せず固定し、softplus直前の総和からr_ckだけを引きます。減衰項は差で相殺されます。寄与のノルムの和は元出力のノルムには一致しません。LoRAのW0xのノルムも別途保存します。

### 細胞別グラフ

celltypeごとにseed固定で抽出し、Wに対して共有基底と同じ変換・固有分解をします。各細胞のスペクトルとcelltypeの中央値・25〜75%区間を描きます。既定3細胞の図は探索用であり、集団推定精度を保証しません。

Cの通常Lの固有値は `abs(a_ck)*lambda(B_k)`、正規化Lはa_ck≠0ならBと同一です。これを正確な式で保存し、無意味な重複固有分解を避けます。a_ck=0ならグラフと固有値は全ゼロです。

## 重みが上位の遺伝子

自己ループを除き、incomingは行和、outgoingは列和です。

```text
incoming_i = Σ_(j≠i) abs(W_ij)
outgoing_j = Σ_(i≠j) abs(W_ij)
```

共有B、celltype別C、celltype別Wに対して出します。主順位は全細胞で各スコアを計算してから平均したものです。`abs(mean(W))` の順位も別に出し、平均による符号相殺と区別します。Wの順位はサンプル推定ではなく全細胞の正確なバッチ集計です。全遺伝子と上位30を保存し、celltype別平均スコアのstd/rangeによる変動順位も出します。これらはモデル重みの順位であり、実証された制御因子の順位ではありません。

## 出力の案内

```text
outputs/<run>/
  resolved_config.json          実行した設定
  manifest.json                 完了/失敗・コードhash・package版・コマンド
  progress.log                  どの工程まで実行したか
  matsum/ または lora/
    model_provenance.json        checkpoint・training hash・復元条件
    genes.csv                   元の遺伝子IDとgene_name等
    basis/
      shared_matrices.npz       B[8,d,d], W0[d,d], gene_ids
      across_eight_basis_statistics.npz
      B1_graph.npz ... B8_graph.npz
      *_similarity.csv/png, *_distance.csv/png, *_rmsd.csv/png
      spectral_diagnostics.csv  孤立点・ゼロ固有値・q境界の縮退
    t049/
      forward_reconstruction_check.json
      coefficients/
        cell_coefficients.csv.gz
        coefficient_statistics.csv
        signed_mean.png, mean_absolute.png, *composition*.png
        variance_decomposition*.csv
        groups/<celltype>/      符号付き/絶対値histogram、L1分布
        pairs_balanced/         28個別散布図＋全ペア一覧
        embeddings/{signed,signed_L1,absolute_L1}/
          coordinates_and_clusters.csv.gz, pca_loadings.csv
          clustering_metrics.csv, cluster_k*_*.csv/png
      cell_matrices/
        groups/<celltype>/matrix_moments.npz, mean_C_*.png, mean_W_*.png, *histogram.png
        W_variance_decomposition.npz, W_eta_squared_histogram.png
        cell_effects.csv.gz, mean_effects_by_celltype.csv
        effective_gene_rankings.csv, top30_effective_genes.csv
        component_gene_rankings.csv.gz, genes_varying_by_celltype.csv
        ratio_validity.csv
        sampled_graphs/         抽出細胞ID・スペクトル・分位点図
```

CSVのcomponentは1〜8、NumPyの軸indexは0〜7です。全体は `groups/all/`、celltype別は番号付きフォルダで保存します。各フォルダの `definition.json` に実ラベルと細胞数があります。配列の遺伝子順序は `genes.csv` / `shared_matrices.npz` と共通です。

## 計算資源・再現性・解釈

- 全細胞×8×1024×1024は保存しません。Wのバッチは既定16細胞、float64で約128 MiB、作業配列はその数倍が必要です。GPUメモリが足りなければ `matrix_batch_size` を下げます。
- 全細胞の係数・PCA/UMAP・重みスコア、およびcelltype別固有分解は計算時間を要します。行列統計と多数の図により出力はGB単位になり得ます。進捗を逐次表示します。
- celltype数や次元が増えても乱数seed、抽出ID、設定は保存します。GPU/BLAS/package版が異なる環境でbit単位一致は保証しません。TF32は無効です。
- 基底のスケールを変えて係数を逆スケールしてもWは同じになり得ます。8基底間のstdや係数の大きさはパラメータ表現依存です。MatSumの第k成分とLoRAの第k成分が対応するとは仮定しません。
- 対称化・絶対値化は方向と符号を捨て、逆方向の正負が相殺される場合があります。スペクトル一致は同じ制御ネットワークであることを証明しません。
- clusterとの一致は係数がcelltypeに関連する情報を含む根拠です。因果・独立した生物学的機序・sampleやstageの交絡排除の証明にはしません。

## 検証状況

ローカルの人工データによる検証と、接続先の学習済みデータによる実行は区別します。実checkpointはこのローカル環境に存在しません。接続先での `--check-only` と本実行は未実施です。検証結果は [`VALIDATION.md`](VALIDATION.md) に記録します。
