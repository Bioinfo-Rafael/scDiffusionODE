## Model Aにおける共分散退化問題とAuxiliary Subspace Parameterizationによる解決

### 1. 起きた問題

Model Aでは、stationary Gaussian diffusionの構造を用いてforward SDEを

$$
dy_s=-(Q+D)y_s\,ds+\sqrt{2D}\,dB_s,
$$

$$
Q^\top=-Q,\qquad D\succeq0
$$

と定義していた。

最初の実装では、1024遺伝子に対して

$$
Q,D\in\mathbb R^{1024\times1024}
$$

を直接学習し、

$$
Q=U-U^\top,
$$

$$
D=CC^\top
$$

とparameterizeした。ここで \(U,C\) が学習可能parameterである。

このモデルは初期値

$$
Q=0,\qquad D=\frac12I
$$

では通常のVP diffusionと一致し、学習開始直後は問題なく動作した。しかし、約6000 training steps後にforward transition covarianceのCholesky分解が失敗した。

エラーは概ね

```text
cholesky_ex info=1020
```

であり、1024次元の遷移共分散が数値的に正定値でなくなったことを意味していた。

---

### 2. 原因

Model Aではstationary covarianceを \(I\) としているため、時刻 \(s\) のconditional covarianceは

$$
\Sigma_s
=
I-\Phi_s\Phi_s^\top,
$$

$$
\Phi_s
=
\exp[-s(Q+D)]
$$

と書ける。

score matchingでは、この共分散のCholesky factor

$$
\Sigma_s=L_sL_s^\top
$$

を用いて

$$
s_\phi(y_s,s)
=
-L_s^{-\top}\epsilon
$$

を計算するため、\(\Sigma_s\) が正定値である必要がある。

しかし旧実装では

$$
D=CC^\top
$$

なので保証されるのは

$$
D\succeq0
$$

までであり、

$$
D\succ0
$$

ではない。

学習中に \(C\) の一部のsingular valueが非常に小さくなると、ある方向へのdiffusionが極端に弱くなる。またModel Aにはeffective interaction

$$
Q+D
$$

のGRN外edgeに対してsoft constraint

$$
R_{\mathrm{GRN}}
=
\operatorname{mean}
\left|
(1-M)\odot(Q+D)
\right|
$$

も存在する。

この正則化は \(D\) を直接0にするものではないが、off-maskのmixingを弱くする方向にも働く。その結果、

* ある方向への直接noise injectionが弱い
* \(Q+D\) による他方向からのmixingも弱い

という状態が生じると、有限時刻での

$$
\Sigma_s
$$

が非常に悪条件になる。

特にfloat32では、小さい正の固有値が丸め誤差によって0または負として扱われ、Cholesky分解が失敗する。

したがって今回の問題は、単純な「division by zero」ではなく、

$$
\boxed{
\text{learned diffusionがほぼ退化した方向を作り、
full 1024-dimensional covarianceが悪条件になった}
}
$$

ことが本質である。

---

### 3. なぜ単純な \(\varepsilon\) jitterでは解決しなかったか

数値対策として

$$
\Sigma_s
\rightarrow
\Sigma_s+\varepsilon I
$$

とする方法も考えられる。

これはCholeskyを安定化する一般的な手法であるが、今回の目的には好ましくない。

なぜならforward modelで定義された分布は

$$
q_\phi(y_s|x)
=
\mathcal N(m_s,\Sigma_s)
$$

であるのに、samplingやscore calculationだけ

$$
\mathcal N(m_s,\Sigma_s+\varepsilon I)
$$

を使うと、宣言したSDEと実際に学習しているtransition distributionが一致しなくなるからである。

つまりjitterは単なる数値演算上の処理ではなく、実質的にforward diffusionそのものを変更する。

そこで今回は、Cholesky failureを後処理で隠すのではなく、**forward SDE自体を数値的にwell-conditionedな構造へ再parameterizeする**方針を採用した。

---

## 4. Auxiliary subspaceを介した新しいModel A

新しいModel Aでは、gene-gene interactionを直接 \(d\times d\) matrixとして学習せず、\(K\ll d\) 次元のauxiliary spaceを介して表現する。

遺伝子数を \(d\)、auxiliary dimensionを \(K\) とし、

$$
Z\in\mathbb R^{d\times K},
\qquad
Z^\top Z=I_K
$$

というlearnable gene embeddingを導入する。

各gene \(i\) は

$$
z_i=Z_{i,:}\in\mathbb R^K
$$

というK次元representationを持つ。

gene-gene interactionは直接parameterを持たず、このauxiliary representationを介してのみ生じる。

---

## 5. 新しい \(Q\) のparameterization

auxiliary space上に

$$
Q_K\in\mathbb R^{K\times K}
$$

を持ち、

$$
Q_K^\top=-Q_K
$$

となるようにparameterizeする。

gene space上のeffective \(Q\) は

$$
\boxed{
Q=ZQ_KZ^\top
}
$$

と定義する。

すると

$$
Q^\top
=
ZQ_K^\top Z^\top
=
-ZQ_KZ^\top
=
-Q
$$

なので、

$$
\boxed{
Q^\top=-Q
}
$$

は厳密に保証される。

gene \(i\) と gene \(j\) のinteractionは

$$
Q_{ij}
=
z_i^\top Q_Kz_j
$$

で与えられる。

これはordinary inner productではなく、skew-symmetric bilinear formであり、

$$
Q_{ij}=-Q_{ji}
$$

というModel Aに必要な非対称interactionを保持できる。

---

## 6. 新しい \(D\) のparameterization

diffusion matrixは

$$
\boxed{
D
=
\sigma^2I_d
+
ZBB^\top Z^\top
}
$$

とした。

ここで

$$
B\in\mathbb R^{K\times K}
$$

はlearnable parameterであり、

$$
\sigma^2>0
$$

はexplicitなisotropic diffusion componentである。

任意の非zero vector \(v\) に対して、

$$
v^\top Dv
=
\sigma^2\|v\|^2
+
\|B^\top Z^\top v\|^2
>0
$$

なので、

$$
\boxed{
D\succ0
}
$$

が厳密に保証される。

これは旧実装の

$$
D=CC^\top\succeq0
$$

との重要な違いである。

また \(i\neq j\) ではidentity termは寄与しないため、

$$
D_{ij}
=
z_i^\top BB^\top z_j.
$$

ここで

$$
h_i=B^\top z_i
$$

と定義すれば、

$$
\boxed{
D_{ij}
=
h_i^\top h_j
}
$$

となる。

したがってgene-gene interactionは、auxiliary spaceにおけるgene representation同士のinner productとして解釈できる。

---

## 7. GRNとの対応

effective stationary operatorは

$$
A=Q+D
$$

である。

off-diagonal成分については

$$
A_{ij}
=
z_i^\top Q_Kz_j
+
(B^\top z_i)^\top(B^\top z_j).
$$

したがって全てのgene-gene interactionは

$$
\boxed{
\text{gene}
\rightarrow
\text{K-dimensional auxiliary space}
\rightarrow
\text{gene}
}
$$

という経路を介して生成される。

一方でgene-to-auxiliary embedding \(Z\) 自体にはGRN maskをかけない。

GRN priorは、最終的に得られるeffective gene-gene matrix

$$
A_{\mathrm{interaction}}
=
Z(Q_K+BB^\top)Z^\top
$$

に対して、

$$
\boxed{
R_{\mathrm{GRN}}
=
\operatorname{mean}
\left|
(1-M)\odot
A_{\mathrm{interaction}}
\right|
}
$$

として適用する。

これにより、

* auxiliary representation自体は自由に学習できる
* GRNに存在しないeffective gene-gene interactionだけをsoftに抑制する

という構造になる。

---

## 8. stationary distributionの条件も維持される

新しいparameterizationでもforward driftは

$$
f(y)=-(Q+D)y
$$

であり、diffusion covarianceは

$$
a=2D
$$

である。

stationary covarianceとして \(I\) を代入すると、

$$
-(Q+D)
-
(Q+D)^\top
+
2D
$$

$$
=
-(Q+D)
-
(-Q+D)
+
2D
$$

$$
=0.
$$

したがってLyapunov equation

$$
FI+IF^\top+2D=0,
\qquad
F=-(Q+D)
$$

を満たし、

$$
\boxed{
\mathcal N(0,I)
}
$$

は依然として厳密なstationary distributionである。

つまりauxiliary subspace化によって、元々Model Aが持っていたstationary diffusionとしての数学的条件は失われていない。

---

## 9. full 1024-dimensional Choleskyが不要になる理由

\(Z\) のcolumn spaceとそのorthogonal complementに状態を分解する。

$$
x_\parallel=Z^\top x,
$$

$$
x_\perp=x-Zx_\parallel.
$$

auxiliary subspace上では

$$
D_K
=
\sigma^2I_K+BB^\top,
$$

$$
A_K
=
Q_K+D_K.
$$

したがってK-dimensional transitionは

$$
\Phi_K(s)
=
e^{-sA_K}.
$$

一方、\(Z\) に直交する \((d-K)\) 次元空間では、\(Q\) も \(ZBB^\top Z^\top\) も作用しないため、

$$
\Phi_\perp(s)
=
e^{-\sigma^2s}.
$$

よってmeanは

$$
\boxed{
m_s
=
e^{-\sigma^2s}x_\perp
+
Z\Phi_K(s)x_\parallel
}
$$

とexactに計算できる。

重要なのは、必要なmatrix exponentialが

$$
d\times d
$$

ではなく

$$
\boxed{
K\times K
}
$$

だけになることである。

---

## 10. covarianceもK次元へexactに分解できる

stationary covarianceが \(I\) であるため、

$$
\Sigma_s
=
I-\Phi_s\Phi_s^\top.
$$

orthogonal complementでは

$$
\boxed{
\Sigma_\perp(s)
=
1-e^{-2\sigma^2s}
}
$$

というscalarになる。

auxiliary subspaceでは

$$
\boxed{
\Sigma_K(s)
=
I_K-\Phi_K(s)\Phi_K(s)^\top.
}
$$

したがってfull covarianceはconceptually

$$
\Sigma_s
=
\Sigma_\perp(I-ZZ^\top)
+
Z\Sigma_KZ^\top
$$

と書ける。

ここでfull \(d\times d\) covarianceをmaterializeする必要はない。

Choleskyが必要なのは

$$
\boxed{
L_KL_K^\top=\Sigma_K
}
$$

という \(K\times K\) matrixのみであり、orthogonal complementでは

$$
\sqrt{\Sigma_\perp}
$$

というscalarだけで十分である。

したがって旧実装で失敗した

$$
1024\times1024
$$

のCholesky分解はtraining pathから完全に消える。

---

## 11. なぜ新しい構造では退化問題を防ぎやすいか

新しいModel Aでは

$$
D_K
=
\sigma^2I_K+BB^\top
\succ0
$$

である。

またorthogonal complementにも必ず

$$
\sigma^2>0
$$

のnoiseが存在する。

したがって、全gene spaceにおいてnoise-freeな方向が存在しない。

補空間については

$$
\Sigma_\perp(s)
=
1-e^{-2\sigma^2s}
>0
\qquad(s>0)
$$

が明示的に保証される。

auxiliary subspaceについても \(D_K\succ0\) なので、finite \(s>0\) でtransition covarianceは非退化になる。

つまり旧実装のように、学習によって \(D\) のあるdirectionが完全に消失し、full covarianceがsingularになる問題を構造的に避けることができる。

これはCholeskyに人工的なjitterを加えて問題を隠しているのではなく、

$$
\boxed{
\text{forward SDEそのものを非退化なモデルとして定義している}
}
$$

点が重要である。

---

## 12. 元論文との関係

元論文では、各observed coordinateに対して小さい \(K\)-dimensional auxiliary dynamicsを導入し、主要なmatrix exponentialやCholeskyを \(K\times K\) の小行列で処理する構造を利用している。

そのため、元論文では高次元データを扱っていても、forward diffusionの主要な線形代数は小さいauxiliary space内で完結する。

今回の実装は元論文と完全に同じ構造ではない。

元論文では主に各data coordinateごとのauxiliary variablesを用いるのに対し、今回のモデルでは

$$
Z\in\mathbb R^{d\times K}
$$

というshared auxiliary gene spaceを用いて、**gene間interactionそのものをK-dimensional space経由で表現する**。

したがって今回の変更は、

$$
\boxed{
\text{元論文の「small auxiliary spaceでdiffusionを扱う」という設計思想を、
GRNを持つgene interaction modelへ拡張したもの}
}
$$

と位置付けられる。

これにより、

* \(Q^\top=-Q\)
* \(D\succ0\)
* stationary prior \(N(0,I)\)
* gene-gene interactionのnonzero性
* GRN soft constraint
* small-dimensional matrix exponential
* small-dimensional Cholesky

を同時に実現できる。

---

## 13. 計算量上の改善

旧dense Model Aでは、

$$
Q,D,\Sigma\in\mathbb R^{d\times d}
$$

を直接扱っていたため、

* \(d\times d\) matrix exponential
* \(d\times d\) covariance
* \(d\times d\) Cholesky

が必要であり、主要計算量は概ね

$$
O(d^3)
$$

であった。

新Model Aでは、

* gene \(\leftrightarrow\) auxiliary projection: \(O(dK)\)
* auxiliary matrix operations: \(K\times K\)
* matrix exponential / Cholesky: \(O(K^3)\)

となる。

したがって \(K\ll d\) なら主要なforward computationは

$$
\boxed{
O(dK)+O(K^3)
}
$$

程度まで削減できる。

例えば

$$
d=1024,\qquad K=64
$$

なら、旧実装の1024次元Choleskyの代わりに64次元Choleskyだけで済む。

GRN regularizationを計算するときだけeffective \(d\times d\) interactionをmaterializeする可能性はあるが、これはtransition covarianceやscore calculationのような数値的にcriticalな線形代数とは異なる。

---

## 14. まとめ

旧Model Aで起きた問題は、単純なfloat overflowではなく、

$$
D=CC^\top
$$

がPSDまでしか保証しないことと、1024-dimensional full covarianceを直接扱っていたことにより、学習途中でforward diffusionがほぼ退化したdirectionを形成し、Cholesky分解が失敗したことであった。

これに対し、新しいModel Aでは

$$
Q=ZQ_KZ^\top,
$$

$$
D=\sigma^2I+ZBB^\top Z^\top
$$

というauxiliary-space parameterizationを導入した。

これにより、

$$
Q^\top=-Q,
$$

$$
D\succ0,
$$

$$
FI+IF^\top+2D=0
$$

が厳密に成立し、stationary prior

$$
N(0,I)
$$

も維持される。

同時にgene-gene interactionは

$$
Q_{ij}=z_i^\top Q_Kz_j,
$$

$$
D_{ij}=(B^\top z_i)^\top(B^\top z_j)
$$

としてauxiliary spaceを介して表現されるため、direct gene-gene parameterを持たなくても全gene pairにnonzero interactionを許すことができる。

さらにtransition covarianceをauxiliary subspaceとorthogonal complementへexactに分解できるため、1024-dimensional Choleskyは不要となり、

$$
K\times K
$$

のCholeskyだけで学習可能になる。

したがって今回の変更は、

**数値エラーを単に回避するための処置ではなく、Model Aのstationary diffusionとしての数学的条件を保ちながら、元論文のsmall auxiliary-space computationに近い構造へ再parameterizeし、GRNとしてのgene-gene interactionも維持した解決策**

である。
