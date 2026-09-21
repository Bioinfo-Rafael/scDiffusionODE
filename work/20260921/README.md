# Frozen Stage1 CellUNet: manifold geometry / exposure bias / mode occupancy

ローカル実装先は `scDiffusionODE/work/20260921`、本番実行先は
`/home/suzuki/Projects/scDiffusion-github/work/20260921`。
対象は **20260915_x0predict の canonical Stage1 final EMA** のみ。
既存モデルの再学習・fine-tuning・parameter update は行わない。

## 最初に監査した既存実装

2026-09-21 に、以下の**コードと保存形式**を実際に読んで確認した。
ローカルには本番 campaign/checkpoint がないため、実 checkpoint の値やハッシュを
確認済みとはしていない。実ファイルの確認は `--dry-run` および本実行が行う。

| 確認したファイル / class / function | 確認した意味論 | 実行時の検証 |
|---|---|---|
| `work/20260915_x0predict/training/checkpoints.py`: `save_checkpoint`, `canonical_stage1`, `restore` | payload は `state_dict` と `metadata`。canonical は final EMA と parameter/buffer hash を照合 | payload のキー、final step、EMA、campaign 内の場所、SHA256、state hash を検証 |
| `work/20260915_x0predict/training/runner.py`: `register_final_stage1`, `train` | canonical record に checkpoint、cellunet、dataset、gene-order の provenance を記録 | canonical と checkpoint metadata の一致 |
| `work/20260915_x0predict/common.py`: `effective_config`, `validate_config`, `build_diffusion`, `load_real`, `state_hash`, `source_provenance` | START_X、1000 original steps、linear、MSE、clip=False、未加工 X | required effective config、campaign config と checkpoint config の一致 |
| `work/20260915_x0predict/configs/base.json`, `configs/stage1_cellunet.json`; `work/20260913_2step/common.py`, `configs/base.json`, `configs/stage1_cellunet.json` | Stage1 の config 合成経路。x0 suite が `predict_xstart=True` を適用 | 実値は保存済み `campaign/configs/stage1_cellunet.json` と metadata を使用。historical config を再合成しない |
| `work/20260915_x0predict/models/__init__.py`: `build_model` Stage1 分岐 | `Cell_Unet(input_dim=len(genes), hidden_num=config['cell_unet_hidden_num'])` | 同じ直接コンストラクタ、strict state load |
| `guided_diffusion/cell_model.py`: `Cell_Unet.__init__`, `forward` | gene 入出力、時刻 embedding、eval で決定論的 | `model.eval()`, `requires_grad_(False)`、raw output と pred_xstart の一致 |
| `guided_diffusion/script_util.py`: `create_gaussian_diffusion` | START_X / MSE / FIXED_LARGE。Stage1 は `sigma_small` を渡さず factory default=False | enum、1000 steps、identity timestep map を assert |
| `guided_diffusion/gaussian_diffusion.py`: `q_sample`, `p_mean_variance`, `p_sample` | START_X 分岐は raw output を pred_xstart とし native posterior mean を使う。`nw=0.5`、t=0 は noise 無効 | 直接呼び出し。sampler の式を今回の実装へコピーしない |
| `work/20260915_x0predict/sampling/trajectory.py`: `sample_to_disk` | standard Gaussian 初期値、native ancestral DDPM、`clip_denoised=False`、guidance なし、default `nw=0.5` | 同じ呼び出し条件を metadata に記録 |
| `guided_diffusion/cell_datasets_loader.py`: `load_data`, `custom_collate_fn` | `train_vae=True, preprocess=False, layer=None` は元の X を float32 へ変換。VAE encoding なし | h5ad 全体の SHA256、X と gene 順序。追加 preprocessing なし |
| `work/20260830/hematopoietic_viz/core.py`: `gene_names_from_adata`, `gene_order_hash` | `adata.var['gene_name']` を順序付きで使用。index、NUL、UTF-8 gene、改行を hash | 同じ serialization の小さな独立実装。順序の入れ替えは許可しない |
| `runs/*/canonical_stage1.json`, checkpoint metadata | 本番ファイルはリモートのみ | リモートで実読込。候補と invalid 理由を stdout、選択結果を provenance JSON へ |

## 1. Question

検証したいのは以下の5点である。解析コードがあらかじめ結論を決めることはない。

1. manifold から離れた state では model drift の normal component が支配的か。
2. manifold に近づいた後、tangent movement が残るか、ほぼ停止するか。
3. reverse trajectory が training input marginal `q_t` からいつ離れるか。
4. real-defined mode の occupancy が途中から偏り、最終 coverage loss に結び付くか。
5. 見かけの散布を担うのは model drift か、DDPM noise か。

主要解析は **actual deterministic model drift** の分解。
score の方向だけでは、discrete reverse update の大きさや noise の寄与を判断できない。

## 2. Existing model と immutable provenance

`work/20260915_x0predict/runs/*/canonical_stage1.json` を列挙し、
チェックを通過する Stage1 が1つなら選択する。0件または複数ならエラー終了し、
候補と `--campaign` の指定方法を表示する。mtime やファイル名から checkpoint を推測しない。
`--campaign` は campaign 名またはそのディレクトリの絶対パスを受け付ける。

検証内容:

- canonical record が指すファイルが、その campaign の `stage1_cellunet` 内にある。
- checkpoint file SHA256 が canonical record と一致する。
- metadata の objective/condition が Stage1、checkpoint_kind が EMA、
  step が effective config の total_steps、canonical status が completed。
- canonical と checkpoint の CellUNet parameter/buffer hash、gene-order hash、dataset hash が一致。
- 保存済み campaign config と checkpoint effective config が完全一致。
- 読み込む実 dataset の SHA256 と `var['gene_name']` の並びが学習時と完全一致。
- 直接依存ファイルの現在の SHA256 を保存し、training metadata にそのファイルの hash が
  ある場合は一致を要求する。元の metadata に記録がないファイルは `training: null` と明示する。

hidden sizes、遺伝子数、seed、data path、学習 total_steps、EMA rate などは
実 checkpoint の effective config が正本である。学習時の seed がなければ1234。
`--data-path` は同一ハッシュの dataset の移動にのみ使える。
本番 checkpoint を含む campaign 全体を移動した場合、canonical の絶対パスは勝手に書き換えない。

モデルは strict restore 後に `eval()` と `requires_grad_(False)`。
全 inference は `torch.no_grad()` 内。optimizer は作成しない。
`finally` で checkpoint file hash と全 parameter/buffer hash を再計算し、一致を assert する。
エラー時も可能な限り検証結果と failed status を保存する。
強制 kill / OS crash では finally は保証されないため、`metadata.status=completed` と
`checkpoint_provenance.unchanged_assertion_passed=true` の両方を完了判定に使う。

## 3. Code provenance と review burden

| Component | Source | Implementation type | Review burden |
|---|---|---|---|
| CellUNet | existing repo `guided_diffusion.cell_model` | direct import | low |
| GaussianDiffusion | existing repo `script_util` / `respace` / `gaussian_diffusion` | direct import through existing factory | low |
| q_sample / p_mean_variance / p_sample | existing repo | direct call | low |
| checkpoint restoration | inspected Stage1 serialization | direct strict PyTorch state load, new validation | medium |
| global PCA / kNN / SVD | scikit-learn / NumPy | standard library calls | low |
| local PCA neighborhood / tangent basis | mathematical definition | independent small implementation | medium |
| tangent/normal projection / scaled centers | linear algebra definition | independent small implementation | medium |
| SWD | empirical one-dimensional W1 definition | independent diagnostic implementation | medium |
| mode assignment / TV / JS / mixing | analysis-specific diagnostic | independent implementation | medium |
| snapshots / figures / CSV orchestration | this analysis | new code | medium |

**This is not a reproduction or copy of the authors' implementation.
The analysis is an independent diagnostic implementation based on the mathematical definitions described in the cited papers.**
論文の公式 GitHub code のコピー/import はない。
既存 scDiffusionODE の CellUNet / GaussianDiffusion は、そのまま直接利用する。
互換 hash serialization のみ、既存実装と同じ形式を小さく記述した。

## 4. Dependency graph

```text
analyze.py
 ├─ model_io.py
 │   ├─ guided_diffusion.cell_model.Cell_Unet
 │   │   └─ guided_diffusion.nn
 │   ├─ guided_diffusion.script_util.create_gaussian_diffusion
 │   │   └─ guided_diffusion.respace.SpacedDiffusion
 │   │       └─ guided_diffusion.gaussian_diffusion.GaussianDiffusion
 │   │           ├─ guided_diffusion.nn
 │   │           └─ guided_diffusion.losses
 │   └─ torch / anndata / scipy.sparse
 ├─ geometry.py → numpy / sklearn.neighbors
 ├─ diagnostics.py → geometry / numpy / pandas / sklearn.neighbors / matplotlib
 └─ sklearn.decomposition.PCA / threadpoolctl
```

過去 work ディレクトリから Python wrapper を import しない。
ODE、VAE、historical training runner への依存も作らない。
Stage1 campaign config/metadata は JSON として読む。
`script_util` が返す `SpacedDiffusion` は timestep map が恒等写像のものに限定する。

## 5. Mathematical definitions

### Common PCA と scaled manifold

学習時と同じ `adata.X` を float32 で使い、real 全細胞のみで
`PCA(n_components=50, whiten=False)` を fit する。
normalize_total / log1p / scale / whitening / gene filtering を追加しない。
座標とベクトルを分ける:

$$
z=C(x-\mu),\qquad v^{PCA}=Cv,\qquad CC^T=I.
$$

実装は行ベクトル表記で `(x-mu) @ C.T` と `v @ C.T`。
PCA は randomized SVD、seed 固定。fit と transform を分離し、すべての点に同じ
transform を適用する。解析側の演算と局所 SVD は float64。

**UMAP空間では距離・角度・直交性が保存されないため、tangent/normal decompositionにはUMAPを使用しない。**
PCA50 でも捨てた gene-space 成分は測れない。
また `C s(x,t)` は full score を投影したベクトルであり、一般に PCA marginal の score と
同一とは限らない。ここで扱うのは固定 PCA ambient space 内の診断である。

時刻 t では `a_t=sqrt(alpha_bar_t)` とおくと、forward process の中心は
`a_t M`。固定 PCA 原点を保持して変換するため、

$$
z_i^{(t)}=C(a_t x_i-\mu)=a_t z_i+(a_t-1)C\mu.
$$

単に `a_t*z_i` とすると PCA mean の分だけ anchor と距離がずれる。
nearest anchor は
`(z_query-(a_t-1)*C*mu)/a_t` を clean real PCA で検索することと等価。
nearest distance はその検索距離を `a_t` 倍する。uniform scaling なので近傍の順序と
tangent basis の向きは変わらない。

### Local tangent basis と projection

anchor 自身を除いた clean real の k=50 nearest neighbors を取り、

$$
\mu_i=\frac1k\sum_{j\in N_i}z_j,\qquad
Z_i=[(z_j-\mu_i)^T]_{j\in N_i}=U\Sigma V^T.
$$

上位 d=5,10,20 の right singular vectors を列に並べた `V_T` を使う。

$$
P_T=V_TV_T^T,\quad v_T=V_T(V_T^Tv),\quad v_N=v-v_T.
$$

巨大な projector を作らず `v @ V_T @ V_T.T` を計算。
query が実際に使った anchor についてのみ SVD を計算し、anchor ID で cache。
local rank と上位 d の分散説明率も CSV に保存する。
rank < d の場合は flag を付ける。残りの軸の向きは識別できず、解釈対象から除外すべきである。

### START_X score

$$
x_t=\sqrt{\bar\alpha_t}x_{clean}+\sqrt{1-\bar\alpha_t}\epsilon,
\quad \hat x_{clean}=f_\theta(x_t,t),
\quad s_\theta(x_t,t)=\frac{\sqrt{\bar\alpha_t}\hat x_{clean}-x_t}{1-\bar\alpha_t}.
$$

`GaussianDiffusion.p_mean_variance` の START_X 分岐は raw model output を
`pred_xstart` にする。clip=False、denoised_fn=None なので余計な変換はない。
コードは enum と raw output の一致を assert し、上式を **gene space で計算してから**
`C s` に写す。epsilon model の変換関数を使わない。

全ベクトルについて norm、tangent/normal norm、その比と

$$
R_T=\|v_T\|^2/\|v\|^2,\quad R_N=\|v_N\|^2/\|v\|^2
$$

を保存。ゼロベクトルの比は未定義なので CSV では NaN（空欄）。
例えば t=0 では noise norm=0 だが noise normal fraction は0ではなく未定義。
等方的 noise では、D 次元の解析空間における squared tangent fraction の期待値は d/D、
normal は (D-d)/D になる。D=50 で normal fraction が大きいことだけでは、
学習された manifold correction の証拠にならない。norm と向きの cosine を併せて読む。

### 実際の DDPM movement（主要解析）

native `p_mean_variance()` の mean と、native `p_sample()` の sample を使い、

$$
\Delta x_{model}=\mu_\theta(x_t,t)-x_t,\quad
\Delta x_{noise}=x_{t-1}-\mu_\theta(x_t,t),\quad
\Delta x_{total}=x_{t-1}-x_t.
$$

3つとも gene-space 差分を `C` で投影して同じ `V_T` で分解する。
PCA 上で total=model+noise の一致も assert する。
**sampling equation 自体は再実装していない。**
保存時刻にだけ追加で p_mean_variance を呼ぶ。eval の決定論的予測なので、p_sample 内の
同じ予測と一致し、乱数列を消費しない。入力の raw prediction の照合も行う。
通常の step は p_sample だけで進む。sampling は全1000 step。

### Manifold distance と normal correction cosine

局所平均にも同じ affine scaling を適用:

$$
\mu_i^{(t)}=a_t\mu_i+(a_t-1)C\mu,\quad
r_N=-(I-P_T)(z_t-\mu_i^{(t)}),\quad d_M=\|r_N\|.
$$

$$
\cos_{normal}=\frac{\Delta z_{model}\cdot r_N}
 {\|\Delta z_{model}\|\|r_N\|}.
$$

ここで `Delta z_model=C Delta x_model`。
`cos_normal≈1` は局所平面へ向かう normal correction と model drift が揃うことを表す。
距離は nearest-real distance ではなく **affine plane の normal residual**。
nearest-real distance は補助として保存。norm がゼロの場合の cosine は未定義。

main geometry は t=0,1,5,10,20,50,100,200。
追加指定した t>200 は `regime=reference_high_noise` として別 CSV にも保存し、
main の Figures 01–07 に混ぜない。高ノイズでは局所平面近似の解釈が弱い。
離れた点における局所平面も true global manifold distance を保証しない。

### Timestep の厳密な意味

repo の index は0始まりで `alpha_bar[0]=0.9999`。したがって、ここで
`reverse_state[t=0]` は **最後の更新に入れる、まだ小さな noise を持つ state**。
これは clean sample と区別する。`q_sample(real,t=0)` も同じ index の noisy state。
最後の `p_sample(...,t=0)` の**出力**が `final_generated.npy`。
最終 sample の mode/mixing 行は便宜的に `timestep=-1` とし、図は `final` と表示する。
これは追加の diffusion step を意味しない。final と t=0 の pred_xstart は一致する。

### Exposure bias: state marginals

$$
q_t(x)=\int q(x_t\mid x_{clean})p_{data}(x_{clean})\,dx_{clean},
\qquad p_{\theta,t}(x)=\operatorname{Law}(x_t^{reverse}).
$$

real を経験分布から復元抽出し、既存 `diffusion.q_sample()` で forward state を生成する。
reverse **更新前** state と同じ timestep で比較する。
`q(x_t|x0)` と `p_theta(x_{t-1}|x_t)` の条件付き分布間の比較ではない。

主指標は sliced **1**-Wasserstein distance。
N 個ずつの PCA 座標 q,p、単位ベクトル u_l を L=256 本使い、

$$
SWD_1(q,p)=\frac1{LN}\sum_{l=1}^{L}\sum_{i=1}^{N}
|\operatorname{sort}(q^Tu_l)_i-\operatorname{sort}(p^Tu_l)_i|.
$$

全時刻で同じ seed+3 の unit projections を共有する。
補助は centroid distance と covariance trace ratio `tr(Cov(p))/tr(Cov(q))`。
独立した2組の forward state による forward/forward SWD も有限標本の基準として保存。
これは厳密な検定や confidence interval ではない。

**SWDがどのtimestepから上昇するかを見ることで、sampling trajectoryがtraining input distributionからいつ外れ始めるかを調べる。**
noise level ごとの尺度と forward/forward 基準も併せて読む。
t=999 の差には standard Gaussian prior と有限時間 forward marginal の不一致も含まれる。
SWD は mismatch を検出するが、原因を exposure bias の累積誤差だけに特定しない。

### Mode occupancy と kNN label transfer

`Superclass`, `celltype` があればそれぞれ独立に使う。
欠損 annotation 値は `<missing>` という明示的カテゴリとして保持。
両方の列がなければ mode CSV は header のみ、図は annotation 不在を表示する。
勝手に Leiden clustering を追加しない。

real PCA だけを reference に、各 snapshot の **pred_xstart** に k=15 の多数決:

$$
\hat c(g)=\arg\max_c\sum_{j\in kNN_{real}(g)}1[y_j=c],\quad
p_{real}(c)=\#\{i:y_i=c\}/N_{real},\quad
p_{gen,t}(c)=\#\{g:\hat c(g)=c\}/N_{gen}.
$$

同票は label の辞書順で決定し、vote_fraction も保存する。
kNN label transfer は mode 所属を定める手段であり **distribution metric ではない**。
noise level が異なる raw x_t を clean real に直接割り当てない。

$$
TV(p,q)=\frac12\sum_c|p_c-q_c|,\quad
JS(p,q)=\tfrac12 KL(p\|m)+\tfrac12 KL(q\|m),\quad m=(p+q)/2.
$$

JS は natural log、単位は nats、平方根を取らない。ゼロ項は0と扱う。
label、real/generated fraction、difference、timestep を保存する。
遠く離れた点にも label は付くため、mode 比率だけで品質や coverage を断定せず mixing と併読する。

### Real/generated mixing

各 pred_xstart snapshot と final sample について、real/generated を同数
`n=min(Nreal,Ngen)` に非復元抽出し joint kNN (k=30) を検索する。
同じ seed+4 で各時刻の real subset と trajectory subset を揃える。

$$
mix(g)=\#\{\text{real neighbors of }g\}/k.
$$

query 自身は **index の一致**で除く。座標の重複があっても、返ってきた先頭を
無条件に落とす処理はしない。mean、median、全 point の値、
`fraction(mix<=0.05)` を保存する。0.05 は今回の near-zero 定義で CLI から変更可能。
無作為 mixing の有限標本基準は `n/(2n-1)`（ほぼ1/2）。
低い fraction は generated-only island の候補を示すが、有限 N・k・PCA の影響を受ける。

## 6. Papers と今回独自の diagnostic の境界

論文本文を確認した版の Section / Equation 番号だけを以下に記す。
定理の仮定を single-cell data と learned network が満たすとは主張しない。

| 論文に直接ある概念 | 確認した本文箇所 | 今回の接続 / 違い |
|---|---|---|
| small-noise score と normal bundle | Stanczuk et al., arXiv v5 §5, Theorem 5.1, Corollary 5.2, Appendix D | normal alignment を測る動機。今回の cosine は **score ではなく actual model drift** が主対象。PCA kNN tangent は今回の診断 |
| normal/tangent の multiscale discrepancy | Liu, Zhang, Li, arXiv:2505.09922v3 §3, Theorem 3.1, Eq. (10)–(12)、その後の projection と loss decomposition Eq. (13) | normal score の small-noise singularity と tangent 成分を区別する根拠。Niso-DM / Tango-DM の学習法は導入しない |
| local PCA による tangent estimation | Lim, Oberhauser, Nanda, arXiv v3 §1「Estimators from Local PCA」Eq. (1.1)、Theorem A、§5 | local covariance の上位固有空間を使う根拠。論文の radius ball に対して、こちらは固定 kNN。k,d の選択や保証は論文の再現ではない |
| training/sampling input mismatch | Ning et al., arXiv v3 §3.2–3.4、Eq. (18) と §3.4 の variance error metric | mismatch という問いの根拠。こちらは **state marginals の SWD** を使用。論文の conditional variance metric / Epsilon Scaling をコピーしない |

以下はすべて **our diagnostic choices**: global PCA50、k=50、d=5/10/20、
scaled local anchor、model-drift cosine、SWD による q_t/p_theta,t 比較、
forward/forward baseline、kNN mode assignment + TV/JS、balanced joint-kNN mixing。
引用論文がこの組合せをそのまま提案したという意味ではない。

### References（2026-09-21 確認）

1. **Jan Stanczuk, Georgios Batzolis, Teo Deveney, Carola-Bibiane Schönlieb.**
   *Your diffusion model secretly knows the dimension of the data manifold.*
   arXiv preprint (2022, revised 2023), **arXiv:2212.12611**.
   [Canonical arXiv](https://arxiv.org/abs/2212.12611),
   [確認した v5 本文](https://arxiv.org/html/2212.12611v5).
   後の出版版は *Diffusion Models Encode the Intrinsic Dimension of Data Manifolds*,
   ICML 2024, PMLR 235:46412–46440（Jan Pawel Stanczuk 表記）。
   [出版元](https://proceedings.mlr.press/v235/stanczuk24a.html)。上表の番号は arXiv v5 のもの。
2. **Zichen Liu, Wei Zhang, Tiejun Li.**
   *Improving the Euclidean Diffusion Generation of Manifold Data by Mitigating Score Function Singularity.*
   Advances in Neural Information Processing Systems 38, **NeurIPS 2025**.
   **DOI:10.52202/085713-3704**, arXiv:2505.09922.
   [Canonical proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/a0d2345b43e66fa946155c98899dc03b-Abstract-Conference.html),
   [DOI](https://doi.org/10.52202/085713-3704),
   [確認した v3 本文](https://arxiv.org/html/2505.09922v3)。上表の番号はこの arXiv 版で確認したもの。
3. **Uzu Lim, Harald Oberhauser, Vidit Nanda.**
   *Tangent Space and Dimension Estimation with the Wasserstein Distance.*
   SIAM Journal on Applied Algebra and Geometry **8(3):650–685, 2024**.
   **arXiv:2110.06357**, DOI:10.1137/22M1522711.
   [Canonical arXiv](https://arxiv.org/abs/2110.06357),
   [確認した v3 本文](https://arxiv.org/html/2110.06357v3),
   [Publisher DOI](https://doi.org/10.1137/22M1522711)。上表の番号は arXiv v3 のもの。
4. **Mang Ning, Mingxiao Li, Jianlin Su, Albert Ali Salah, Itir Onal Ertugrul.**
   *Elucidating the Exposure Bias in Diffusion Models.* **ICLR 2024**.
   **arXiv:2308.15321**.
   [Canonical arXiv](https://arxiv.org/abs/2308.15321),
   [確認した v3 本文](https://arxiv.org/html/2308.15321v3),
   [出版元](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4267d84ca2f6fbb4aa5172b76b433aca-Abstract-Conference.html)。上表の番号は arXiv v3 のもの。

## 7. CLI defaults と計算量

| CLI | Default |
|---|---|
| `--sample-count` | 3000 |
| `--sampling-batch` | 50 |
| `--seed` | Stage1 seed、なければ1234 |
| `--pca-components` | 50 |
| `--local-knn` | 50（anchor を除く） |
| `--tangent-dimensions` | 5,10,20 |
| `--mode-knn` | 15 |
| `--mixing-knn` | 30 |
| `--near-zero-threshold` | 0.05 |
| `--swd-projections` | 256 |
| `--geometry-timesteps` | 0,1,5,10,20,50,100,200 |
| `--snapshot-timesteps` | 999,900,700,500,300,200,100,50,20,10,5,1,0 |
| `--cpu-threads` | 4 |
| `--device` | cuda |
| `--output` | `results/<UTC>_<random-id>`、既存 directory は拒否 |

snapshot と geometry の timestep の和集合を保存・診断する。
sampler/diffusion 設定を任意変更する CLI は提供しない。Stage1 一致を優先する。
seed は固定されるが、GPU/backend や sampling batch を変えた場合の bitwise 一致は保証しない。

GPU はモデル inference に使用。PCA/kNN/local SVD/SWD/mode/mixing は CPU。
forward q_sample も独立 CPU torch.Generator を使い、reverse の乱数列を変えない。
raw reverse/pred arrays は float32 memmap、PCA/vector arrays は float64。
全 real X は RAM に読み込む。PCA fit には dense real matrix と追加作業メモリが必要。
raw snapshot 保存量は概ね `8*S*N*G + 4*N*G` bytes
（S=保存時刻数、N=生成細胞数、G=遺伝子数）。PCA arrays と CSV は別途。
全 step の gene state は保存せず、指定 snapshot だけ保存する。

必要環境は既存 Stage1 Python 環境に加え NumPy / SciPy / scikit-learn / PyTorch /
anndata / pandas / matplotlib / threadpoolctl。ローカル検証は Python 3.9 / PyTorch 2.5.1。
依存バージョンは実行ごとの metadata に保存する。scanpy や論文専用 package の追加 import はない。

## 8. Output と図の読み方

```text
results/<run_id>/
 ├─ metadata.json                     # completed/failed, CLI, effective config, sources, versions
 ├─ checkpoint_provenance.json        # checkpoint/model hashes before & after, assertions
 ├─ pca_metadata.json / pca_basis.npz  # fixed PCA transformation and explained variance
 ├─ real_cells.csv / real_pca.npy     # anchor_real_id -> obs_name / annotations
 ├─ reverse_state.npy                # [snapshot, trajectory, gene], input x_t
 ├─ pred_xstart.npy                   # same snapshot/trajectory indexing
 ├─ final_generated.npy              # output after native t=0
 ├─ {state,score,model_drift,noise,total}_pca.npy
 ├─ forward_pca_t<T>_<repeat>.npy / forward_real_ids_t<T>_<repeat>.npy
 ├─ pred_xstart_pca_t<T>.npy / final_generated_pca.npy / swd_directions.npy
 ├─ tangent_normal.csv / tangent_normal_high_noise_reference.csv
 ├─ exposure.csv
 ├─ mode_occupancy.csv / mode_summary.csv / mode_assignments.csv
 ├─ mixing.csv / mixing_points.csv
 └─ figures/                         # 11 PNG files below
```

| Figure | 内容 / 読み方 |
|---|---|
| `01_score_normal_tangent_vs_t.png` | d 別の score tangent/normal 平均 norm |
| `02_model_drift_normal_tangent_vs_t.png` | **主要結果**: actual model drift tangent/normal 平均 norm |
| `03_noise_normal_tangent_vs_t.png` | stochastic displacement の分解。t=0 の noise は0 |
| `04_normal_fraction_vs_t.png` | score / model drift / noise / total の squared normal fraction |
| `05_model_drift_vs_manifold_distance.png` | normal drift norm vs local normal residual |
| `06_tangent_drift_vs_manifold_distance.png` | tangent drift norm vs local normal residual |
| `07_cosine_to_manifold_normal_vs_distance.png` | local normal correction と model drift の cosine |
| `08_exposure_swd_vs_t.png` | forward/reverse SWD、independent forward/forward baseline |
| `09_exposure_centroid_covariance_vs_t.png` | centroid distance と reverse/forward covariance trace ratio |
| `10_mode_occupancy_over_time.png` | annotation 別の generated-real fraction heatmap と TV/JS |
| `11_real_generated_mixing_over_time.png` | mean/median/near-zero fraction と point distribution |

Figures 01–07 は d 別 panel。05–07 は **時刻を混ぜず** t ごとに距離の等頻度 bin の平均を描く。
異なる t の step size と距離の効果を混同しないためである。
グラフの時刻は sampling 順（大きい t から小さい t）に並べる。
mode の `final` は t=0 prediction と重複するが、final sample の解析を明示するため残す。
全 scalar は CSV にあり、図の平均だけでなく trajectory ごとの分布を確認できる。

## 9. 実行コマンド

既存 Stage1 の Python 環境を有効にする。`PYTHON` は任意でその interpreter の絶対パスに設定できる。
以下は**リモートサーバー内**で実行する。

Unit tests（小さい合成 checkpoint、24 cells、4 genes。学習なし）:

```bash
cd /home/suzuki/Projects/scDiffusion-github
"${PYTHON:-python}" -m unittest discover -s work/20260921/tests -v
```

Checkpoint/campaign discovery のみ（dataset 未読込、PCA/inference なし）:

```bash
cd /home/suzuki/Projects/scDiffusion-github
"${PYTHON:-python}" -u work/20260921/analyze.py --dry-run
# 複数候補がある場合は、表示された campaign 名を --campaign に指定する:
# "${PYTHON:-python}" -u work/20260921/analyze.py --campaign x0predict_... --dry-run
```

Full analysis:

```bash
cd /home/suzuki/Projects/scDiffusion-github
"${PYTHON:-python}" -u work/20260921/analyze.py --device cuda
```

Pull → tests → dry-run → background analysis をまとめて実行する例:

```bash
set -euo pipefail
cd /home/suzuki/Projects/scDiffusion-github
git fetch origin feat/20260916-x0predict-hybrid-additive
git switch feat/20260916-x0predict-hybrid-additive
git pull --ff-only origin feat/20260916-x0predict-hybrid-additive
PYTHON="${PYTHON:-python}"
CAMPAIGN="${CAMPAIGN:-}"
args=()
if [[ -n "$CAMPAIGN" ]]; then args+=(--campaign "$CAMPAIGN"); fi
"$PYTHON" -m unittest discover -s work/20260921/tests -v
"$PYTHON" -u work/20260921/analyze.py "${args[@]}" --dry-run
mkdir -p work/20260921/logs
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_$$"
LOG="$PWD/work/20260921/logs/${RUN_ID}.log"
nohup "$PYTHON" -u work/20260921/analyze.py "${args[@]}" --device cuda \
  --output "work/20260921/results/$RUN_ID" > "$LOG" 2>&1 < /dev/null &
PID=$!
ln -sfn "$(basename "$LOG")" work/20260921/logs/latest.log
printf 'PID=%s\nLOG=%s\n' "$PID" "$LOG"
```

Log 確認:

```bash
tail -n 100 -f /home/suzuki/Projects/scDiffusion-github/work/20260921/logs/latest.log
```

campaign が複数なら最初の dry-run が止まる。表示された正しい候補を
`export CAMPAIGN='x0predict_...'` で明示して再実行する。
各 batch 完了、geometry timestep、SWD/mode/mixing timestep、figure 作成を逐次 log に出す。
サーバーでの学習済み checkpoint を使った解析結果は、実行後に初めて得られる。

## 10. Implementation audit

[IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) を参照。
