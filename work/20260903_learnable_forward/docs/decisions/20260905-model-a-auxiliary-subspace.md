# 設計判断: Model Aをshared auxiliary subspaceへ変更した理由

- 記録日: 2026-09-05 JST
- 状態: 採用・実装済み
- 実装commit: `a9dc52146eaf72982da752d422fd09cbd3abec72`
- 対象: `stationary_qd` (Model A) のみ。`free_affine` (Model B) は従来のモデルを維持。
- 提供資料: [ユーザー提供の説明原文](../sources/20260905-model-a-auxiliary-rationale-original.md)。原文は変更せず保存した。本メモでは、報告・確認済みの性質・原因仮説・設計判断を区別する。

## 決定

元のdense gene-space Q/Dを、**shared K-dimensional auxiliary spaceを介するstationary Q/D**へ再parameterizeする。

\[
Z=\operatorname{thinQR}(E),\quad Z^\top Z=I_K,
\quad Q=ZQ_KZ^\top,
\quad D=\sigma^2 I_d+ZBB^\top Z^\top,
\]
\[
Q_K=R_Q-R_Q^\top,\qquad
\sigma^2=\texttt{isotropic_d_floor}+\operatorname{softplus}(\rho)>0.
\]

この決定には異なる二つの役割がある。

1. **明示的なisotropic diffusionと正のfloor**により、全方向へのnoise injectionをモデルとして保証する。
2. **QとDの非等方成分に共通の部分空間Z**を使い、transition・covariance・score・ELBOをK次元演算と補空間のscalar演算へ厳密に分解する。

低次元化だけで正定値性が保証されるわけではない。また、dense Dに正のisotropic termを追加するだけでは、d×d分解の計算量は解消しない。両方を同時に満たすため、この構造を採用した。

## 問題の背景と、まだ確定していない原因

提供資料では、旧Model Aが1024遺伝子のdense Q/Dを学習し、約6000 training steps後に `cholesky_ex info=1020` で停止したと報告されている。今回の記録作業では、失敗時そのもののcheckpoint・共分散・完全なログは新たに取得していないため、このstep数とerror codeは**ユーザーからの報告**として記録する。

`info=1020` は、分解対象行列の1020次のleading principal minorで正定値性の判定に失敗したことを示す。退化した遺伝子番号・固有値の個数・原因を直接示すコードではない。[PyTorch cholesky_ex仕様](https://docs.pytorch.org/docs/2.14/generated/torch.linalg.cholesky_ex.html)

旧実装の `D=CCᵀ` が構造として保証するのはPSDまでであり、一様な正の固有値下限はなかった。stationary transitionは

\[
F=-(Q+D),\quad \Phi_s=e^{sF},\quad
\Sigma_s=I-\Phi_s\Phi_s^\top
=\int_0^s e^{uF}(2D)e^{uF^\top}\,du
\]

である。考えられる失敗要因には、学習に伴うDまたは有限時間共分散の悪条件化と、float32での `I−Phi Phiᵀ` の差し引きによる桁落ちがある。実装履歴には小時刻の差し引きを積分級数で評価する対策もある。[実装・検証記録](../../validation/README.md)

ただし、**Dがsingularでも、driftによるnoiseの伝播によって有限時間共分散が正定値になる場合がある**。したがって `D=CCᵀ` という式だけから、今回の失敗時に「Dが退化した」とは確定できない。GRN penaltyはeffective operatorのoff-mask成分を抑えるもので、Dの最小固有値やmixingを単調に減少させる制約ではない。GRNが今回の失敗へ寄与したという説明も仮説である。

原因を確定するには、失敗時刻s、D・Sigmaの最小固有値/条件数、Cの特異値、有限性、同一checkpointのfloat64および安定な共分散積分評価、必要に応じてGRNのablationを確認する。

## なぜtransitionへの後付けjitterを採用しなかったか

宣言したSDEを変えず、samplingやscoreだけ `Sigma_s + epsilon I` を使うと、実際に使うtransitionと宣言したSDEが一致しなくなる。今回の目的は、stationary equation・ELBO・samplingで同一のforward diffusionを扱うことである。

そのため、共分散計算に後付けするjitterではなく、SDEのDそのものへ `sigma² I` を含めた。「jitterを試したが必ず失敗した」という実験結果をここで主張するものではなく、**モデルと目的関数の整合性から採用しなかった**という設計判断である。

## 数学的に維持する条件と相互作用の意味

任意の非零vについて

\[
v^\top Dv=\sigma^2\|v\|^2+\|B^\top Z^\top v\|^2>0.
\]

また `Qᵀ=−Q` なので

\[
F+F^\top+2D=0,
\]

が成立し、`N(0,I)` はstationary distributionである。

列ベクトルとして `z_i=Z[i,:]ᵀ`, `h_i=Bᵀz_i` と置くと、i≠jで

\[
D_{ij}=h_i^\top h_j,\qquad Q_{ij}=z_i^\top Q_Kz_j.
\]

Dの相互作用は通常の内積、Qの相互作用はskew bilinear formである。gene-geneのdirect learnable parameterはなく、全gene pairはauxiliary spaceを介してnonzeroになり得る。ただし、各pairを独立に自由設定できるという意味ではない。

GRN maskはE/Zには適用せず、effective interaction

\[
A_{\mathrm{interaction}}=Z(Q_K+BB^\top)Z^\top
\]

にのみ適用する。既存の `[target,source]` orientation、対角の除外、full-matrix mean、weight semanticsを維持する。forward driftの係数は `−(Q+D)` であることにも注意する。

## なぜK次元へ厳密に縮約できるか

状態を `x_K=Zᵀx` と `x_perp=x−Zx_K` に分ける。共通部分空間がQとDの両方で不変なので、

\[
D_K=\sigma^2I_K+BB^\top,\quad A_K=Q_K+D_K,
\quad \Phi_K=e^{-sA_K},\quad c_s=e^{-\sigma^2s},
\]
\[
m_s=c_sx_\perp+Z\Phi_Kx_K,
\]
\[
v_s=1-e^{-2\sigma^2s},\quad
\Sigma_K=I_K-\Phi_K\Phi_K^\top,
\quad \Sigma_s=v_s(I-ZZ^\top)+Z\Sigma_KZ^\top.
\]

この分解は新parameterizationに対してexactであり、学習したdense行列の近似ではない。training pathではfull P/Q/D/Phi/Sigmaを作らず、CholeskyはSigma_Kにのみ適用する。sampling・score transform・weighted DSM・terminal KL・Appendix-I boundary NLLも同じ分解を使う。詳細は[実装README](../../README.md#model-a-exact-auxiliary-stationary-qd)を参照。

D≥sigma²Iから、厳密算術では `||Phi_s||₂≤exp(−sigma²s)`、したがって

\[
\Sigma_s\succeq(1-e^{-2\sigma^2s})I\succ0\quad(s>0)
\]

が導ける。この下限は、モデルとしてnoise-freeな方向を除く保証である。しかしsやsigma²が小さければ下限も小さく、**有限精度で常に良条件・Cholesky失敗ゼロという保証ではない**。

実装では補空間の分散を `-expm1(-2*sigma²*s)` で評価し、小時刻のK次元共分散には同じ積分を丸め精度まで評価する級数を用いる。hidden jitterは追加せず、失敗時は最小固有値・条件数・sigma²・physical timeを報告する。

## 元論文との関係と、採用した制約

元論文のSection 3.3では、各data coordinateに共通のK×K dynamicsを用い、matrix exponential等を小行列で計算する構成が説明されている。一方、今回のZはgene間を結ぶ**共通の部分空間の基底**であり、状態をgeneごとの確率的auxiliary variablesで増次元する構成ではない。小行列で線形代数を行う考え方は参考にしているが、論文のモデルをそのまま再現したものではない。[Singhal et al., Section 3.3](https://arxiv.org/html/2302.07261#S3.SS3)

採用した構造では、非等方成分のrankがK以下になり、補空間のd−K方向は同一scalar diffusionとなる。旧dense Model Aの全表現能力を保持するわけではなく、表現能力と計算量をKで調整する新モデルである。

計算量にはthin QRが必要であり、batch sizeをnとして `O(dK² + K³ + ndK + nK²)` が実装に即した評価である。`O(dK+K³)` のみではQRとbatch依存項が抜ける。GRN stepでeffective行列を作る場合は別途 `O(dK²+d²K)` の時間と `O(d²)` の領域を使う。

## 初期化・互換性・検証範囲

`Q_K=B=0`, `sigma²=0.5` で、Zに依存せずstandard VPと一致する。ただし `B=0` ではBBᵀに対するBの勾配もゼロなので、Bはそのままでは学習開始できない。提供configは `B=0.01 I_K` を選び、厳密VP初期値は `auxiliary_b_init_scale=0` で明示的に選択できる。

新Model Aはschema 2 / `auxiliary_shared_subspace` とKを保存する。**旧dense step-5000 checkpointからresumeせず新規runにする。** Model Bの既存checkpoint互換性は維持する。

実装時の確認は、全55 tests、小次元dense参照との値・勾配一致、d=1024/K=64 CPU benchmark、実データ2-step smoke、sampling/analysis、Model Bの80テンソルbitwise regressionである。[検証結果](../../validation/README.md)

これは構造と実装の確認であり、新Model Aの長時間学習の安定性や生成品質を実証した結果ではない。長時間runの診断とK依存の性能評価は引き続き必要である。
