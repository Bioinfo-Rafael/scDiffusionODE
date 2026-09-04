# Dense learnable forward diffusion

This work directory contains an end-to-end experiment for learning a dense
linear forward diffusion together with the existing cell denoiser, sampling
it with its own reverse SDE, and running read-only post-hoc analyses.  It is
isolated from `guided_diffusion/`: no file outside this directory is modified.

Two forward processes are implemented:

- **Model A (stationary Q/D)**

  \[
  d y_s=-(Q+D)y_s\,ds+\sqrt{2D}\,dB_s,
  \qquad Q^\top=-Q,\quad D\succeq0.
  \]

- **Model B (free affine)**

  \[
  d y_s=(Wy_s+b)\,ds+dB_s,
  \qquad W\in\mathbb R^{d\times d},\quad b\in\mathbb R^d.
  \]

All matrices are dense.  This experiment contains no low-rank factorization,
learned subspace, projection matrix, or `r64` fallback.  If the dense
\(d=1024\) implementation is not practical, the benchmark reports that result;
it does not silently replace the requested model by an approximation.

The default objective is named `paper_elbo`.  Precisely, it is **Equation (7)
truncated to \([\delta,T]\), made into a valid data ELBO by the Appendix I /
Theorem 3 boundary correction** in Singhal, Goldstein, and Ranganath, *Where to
Diffuse, How to Diffuse, and How to Get Back: Automated Learning for
Multivariate Diffusions* (ICLR 2023):

- [paper (arXiv HTML)](https://arxiv.org/html/2302.07261)
- [Equation (7)](https://arxiv.org/html/2302.07261#S3.SS2)
- [Appendix I / Theorem 3](https://arxiv.org/html/2302.07261#A9)

`paper_elbo` does **not** mean the raw Equation (7) integral with its lower
limit merely changed from zero to \(\delta\).  The latter is not a valid lower
bound for the data likelihood.  It also does not mean plain epsilon MSE.  The
exact derivation and the implementation convention are given below.

## Scope

The experiment implements training, a model-specific reverse-SDE generator,
post-hoc analysis, mathematical and integration tests, and an opt-in dense
\(d=1024\) forward/backward benchmark.  The standard `GaussianDiffusion`
posterior, DDPM sampler, and DDIM sampler are still never used: they assume a
scalar beta schedule and an isotropic transition and are mathematically
invalid for the full-covariance processes here.

The existing training infrastructure remains responsible for:

- data loading;
- DDP wrapping;
- optimizer steps;
- EMA updates;
- raw and EMA checkpoints; and
- resume, including optimizer state.

A top-level `nn.Module` owns both the denoiser and the learnable forward
process.  Consequently, the existing `model.parameters()` and `state_dict()`
paths include \(Q,D,W,b\) without modifying `guided_diffusion/train_util.py`.
All uses of forward-process parameters occur inside the DDP module's
`forward()` call; training code must not reach through `model.module` to compute
part of the loss outside DDP.

### Work-local layout

| Path | Responsibility |
| --- | --- |
| `diffusion/stationary_qd.py` | Dense Model A parameterization, exact transition, sampling, and GRN hook |
| `diffusion/free_affine.py` | Dense Model B parameterization and augmented Van Loan transition |
| `diffusion/grn.py` | Common target/source mask and off-mask penalty semantics for Model A/B |
| `diffusion/objectives.py` | Score/noise transforms, weighted quadratic, endpoint KL, and Appendix-I likelihood |
| `diffusion/training_diffusion.py` | Training-only `training_losses` facade and the two explicitly named loss modes |
| `diffusion/time_mapping.py`, `timestep_sampler.py` | VP-compatible physical clock and one-time-per-rank sampler |
| `models/wrapper.py` | Top-level denoiser + forward-process `nn.Module` used by DDP/optimizer/EMA/checkpoints |
| `models/factory.py` | Config validation, existing denoiser construction, and GRN mask orientation conversion |
| `training/train_loop.py`, `loss_logging.py` | Work-local `TrainLoop` subclass and buffered common-schema component CSV |
| `sampling/reverse_sde.py`, `artifacts.py` | Custom Euler--Maruyama reverse SDE, Appendix-I decode, and provenance-safe archives |
| `analysis/` | Loss, parameter, timestep, learned-drift/scVelo, and generated-cell UMAP analyses |
| `scripts/train.py`, `launch.py` | One-run and two-model training entrypoints using the unchanged core `TrainLoop` contract |
| `scripts/sample.py`, `analyze.py` | Standalone sampling and read-only analysis entrypoints |
| `scripts/run_full_pipeline.py` | Resumable Model A/B training -> sampling -> analysis orchestration |
| `scripts/benchmark_dense.py` | Opt-in exact dense forward/backward benchmark; no fallback |
| `scripts/verify_protected.py` | Read-only hash check for the seven protected core files |
| `configs/` | Shared training defaults and one dense config per model |
| `tests/` | Mathematical, gradient, transition, sampling, provenance, analysis, pipeline, and core training-state tests |

## Notation

For either model, let the transition conditional on a data sample \(x\) be

\[
q_\phi(y_s\mid x)=\mathcal N(m_s,\Sigma_s),
\qquad \Sigma_s=L_sL_s^\top,
\]

where \(L_s\) is the lower Cholesky factor.  Reparameterized sampling is

\[
\xi\sim\mathcal N(0,I),
\qquad y_s=m_s+L_s\xi.
\]

The forward SDE is written generally as

\[
d y_s=f_\phi(y_s,s)\,ds+g_\phi(s)\,dB_s,
\qquad a_s:=g_\phi(s)g_\phi(s)^\top.
\]

The conditional forward score is

\[
s_\phi(y_s,s;x)
=\nabla_{y_s}\log q_\phi(y_s\mid x)
=-\Sigma_s^{-1}(y_s-m_s)
=-L_s^{-\top}\xi.
\]

The existing network remains an epsilon predictor.  Its score interpretation
for the current learned transition is defined by

\[
s_\theta(y_s,s):=-L_s^{-\top}\xi_\theta(y_s,s).
\]

Let

\[
r_s:=\xi_\theta(y_s,s)-\xi,
\qquad M_s:=L_s^{-1}a_sL_s^{-\top}.
\]

Then the score mismatch in Equation (7) becomes

\[
\begin{aligned}
s_\theta-s_\phi
  &=-L_s^{-\top}r_s,\\
\frac12\lVert s_\theta-s_\phi\rVert_{a_s}^2
  &=\frac12(s_\theta-s_\phi)^\top
       a_s(s_\theta-s_\phi)\\
  &=\frac12r_s^\top L_s^{-1}a_sL_s^{-\top}r_s\\
  &=\frac12r_s^\top M_sr_s.
\end{aligned}
\]

This is a matrix-weighted epsilon error.  It reduces to plain epsilon MSE only
when \(M_s=c_sI\), and even in that case only up to a time-dependent scalar
weight.  No matrix inverse is materialized in code.  We solve

\[
z_s=L_s^{-\top}r_s
\]

with a triangular solve and evaluate \(z_s^\top a_sz_s/2\).

> Matrix order matters: in general
> \(L^{-1}L^{-\top}\ne L^{-\top}L^{-1}=\Sigma^{-1}\).

## From Equation (7) to a valid truncated ELBO

There are no auxiliary variables in this experiment, so the paper's
\(-\log q_\gamma(v_0\mid x)\) term is absent.  Equation (7), written as a
maximized quantity, contains

\[
\log\pi(y_T)
+\int_0^T
\left[
  \frac12\lVert s_\phi\rVert_{a_s}^2
  -\frac12\lVert s_\theta-s_\phi\rVert_{a_s}^2
  +\nabla_y\!\cdot f_\phi
\right]ds.
\]

The integrand is singular at zero.  Replacing \([0,T]\) by
\([\delta,T]\) alone produces a bound for the latent \(y_\delta\), not a
valid bound for the observed data \(x\).  Theorem 3 composes this latent bound
with a variational boundary likelihood.  In reward/maximization form the
boundary correction is

\[
\log p_\theta(x\mid y_\delta)
-\log q_\phi(y_\delta\mid x).
\]

Equivalently, the negative valid truncated ELBO, which the optimizer minimizes,
has the following direct form:

\[
\boxed{
\begin{aligned}
\mathcal J_{\mathrm{valid,direct}}
=\mathbb E\Bigg[&
 -\log p_\theta(x\mid y_\delta)
 +\log q_\phi(y_\delta\mid x)
 -\log\pi(y_T)\\
&+\int_\delta^T
\left{
 \frac12\lVert s_\theta-s_\phi\rVert_{a_s}^2
 -\frac12\lVert s_\phi\rVert_{a_s}^2
 -\nabla_y\!\cdot f_\phi
\right\}ds
\Bigg].
\end{aligned}}
\tag{1}
\]

Thus, a direct-form equivalence test must include both
\(-\log p_\theta(x\mid y_\delta)\) and
\(+\log q_\phi(y_\delta\mid x)\).  Testing against a raw truncated Equation
(7) without that correction would test the wrong objective.

### Entropy-flow identity

For a state-independent diffusion covariance \(a_s\), the Fokker--Planck
equation implies

\[
\frac{d}{ds}H(q_\phi(y_s\mid x))
=\mathbb E\left[\nabla_y\!\cdot f_\phi(y_s,s)\right]
+\frac12\mathbb E\left[
  \lVert s_\phi(y_s,s;x)\rVert_{a_s}^2
\right].
\tag{2}
\]

For the Gaussian transition,

\[
H(q_s)=\frac12
\left[d(1+\log 2\pi)+\log\det\Sigma_s\right].
\]

Integrating (2) from \(\delta\) to \(T\) gives

\[
\int_\delta^T\mathbb E\left[
 -\frac12\lVert s_\phi\rVert_{a_s}^2
 -\nabla_y\!\cdot f_\phi
\right]ds
=H(q_\delta)-H(q_T).
\tag{3}
\]

Meanwhile,

\[
\mathbb E[\log q_\phi(y_\delta\mid x)]=-H(q_\delta),
\]

and

\[
\mathbb E[-\log\pi(y_T)]-H(q_T)
=\operatorname{KL}(q_\phi(y_T\mid x)\|\pi).
\tag{4}
\]

Substituting (3)--(4) into (1) cancels the lower-boundary entropy and yields

\[
\boxed{
\begin{aligned}
\mathcal J_{\mathrm{paper\_elbo}}
=\mathbb E\big[-\log p_\theta(x\mid y_\delta)\big]
&+\mathbb E_x\operatorname{KL}
  \left(q_\phi(y_T\mid x)\|\pi\right)\\
&+\int_\delta^T
\frac12\mathbb E
\left[r_s^\top M_sr_s\right]ds.
\end{aligned}}
\tag{5}
\]

Equation (5) is the compact form used by `loss_mode="paper_elbo"`.  It is not
a surrogate: it is Equation (7), truncated and made valid by Appendix I /
Theorem 3, after the exact entropy cancellation above.  The direct form (1) is
kept as a small-dimensional reference for tests.  Direct and compact forms are
equal after taking the indicated expectations; they are not required to be
equal sample by sample.

It is important not to add the direct-form forward-score and divergence terms
on top of (5).  Doing that together with terminal KL double-counts the entropy
terms.  Conversely, replacing the direct-form terminal cross-entropy by KL but
retaining those correction terms is not Equation (7).

### Boundary likelihood

Suppose the lower-time affine transition is

\[
q_\phi(y_\delta\mid x)
=\mathcal N(\Phi_\delta x+h_\delta,\Sigma_\delta).
\]

Theorem 3 uses a Gaussian likelihood based on Tweedie's formula.  Extending the
paper's zero-bias formula by translating by \(h_\delta\), we use

\[
\begin{aligned}
\mu_{\theta,\delta}
&=\Phi_\delta^{-1}
\left[y_\delta-h_\delta
      +\Sigma_\delta s_\theta(y_\delta,\delta)\right],\\
\Sigma_{\theta,\delta}
&=\Phi_\delta^{-1}\Sigma_\delta\Phi_\delta^{-\top},\\
p_\theta(x\mid y_\delta)
&=\mathcal N(\mu_{\theta,\delta},
             \Sigma_{\theta,\delta}).
\end{aligned}
\tag{6}
\]

Since

\[
s_\theta(y_\delta,\delta)
=-L_\delta^{-\top}\xi_\theta(y_\delta,\delta),
\]

the boundary loss uses the same epsilon-prediction network.  Linear solves and
Cholesky factors are used in place of explicit inverses.  The boundary NLL is
part of `paper_elbo`; omitting it turns a truncated latent bound into an
objective that is no longer a valid data ELBO.

## Model A: stationary dense Q/D

### Exact parameterization

The skew matrix is represented without redundant diagonal or symmetric
parameters.  A packed strict-upper-triangle raw vector defines \(U\), and

\[
Q=U-U^\top.
\]

Therefore \(Q^\top=-Q\) by construction.

The paper-faithful default for the dissipative matrix is

\[
\boxed{D=CC^\top\succeq0.}
\]

`C` is a dense-capacity triangular factor whose entries, including its
diagonal, are unconstrained in the default mode.  There is **no** default
softplus diagonal and no default \(d_{\min}\).  Consequently the constraint is
PSD, not SPD, exactly as in the paper's parameterization.

An explicitly selected numerical-stability option may instead set

\[
C_{ii}=\operatorname{softplus}(\rho_i)+d_{\min},
\qquad d_{\min}>0,
\]

which makes \(D\succ0\).  This option must be recorded in the config and run
metadata and must not be described as the paper-faithful default.  With the PSD
default, a singular \(D\) can still produce a positive-definite finite-time
transition covariance when \((Q,D)\) is controllable.  If it does not, the
Cholesky/score calculation reports the failure rather than silently changing
the model.

### Transition and stationarity

Let

\[
A=-(Q+D),\qquad \Phi_s=e^{sA}=e^{-s(Q+D)}.
\]

Then

\[
m_s=\Phi_sx.
\]

Since

\[
A+A^\top+2D
=-(Q+D)-(-Q+D)+2D=0,
\]

the covariance \(I\) solves the stationary Lyapunov equation.  Starting from a
deterministic \(x\), the exact transition covariance is

\[
\boxed{\Sigma_s=I-\Phi_s\Phi_s^\top.}
\tag{7}
\]

This also proves that \(N(0,I)\) is invariant.  Invariance alone does not imply
that \(q(y_T\mid x)=N(0,I)\) at finite \(T\), so the terminal KL in (5) remains
necessary.

At the first 1000-step diffusion index, \(s\) is approximately \(10^{-4}\).
For a dense 1024-dimensional float32 product, directly subtracting the two
near-identity matrices in (7) can lose positive definiteness through roundoff,
even while \(D\) is well-conditioned and the mathematical covariance is SPD.
This is cancellation in the evaluation formula, not a violation of the Model A
parameterization.

The implementation therefore evaluates the same covariance from its exact
convergent integral series whenever
\(s\leq 8d\,\epsilon_{\mathrm{dtype}}\):

\[
\begin{aligned}
\Sigma_s
 &=\int_0^s e^{uF}(2D)e^{uF^\top}\,du
   =\sum_{n=0}^{\infty}\frac{s^{n+1}}{(n+1)!}K_n,\\
K_0&=2D,\\
K_{n+1}&=FK_n+K_nF^\top,
\qquad F=-(Q+D).
\end{aligned}
\tag{7a}
\]

Terms are accumulated with

\[
P_0=s(2D),\qquad
P_{n+1}=\frac{s}{n+2}(FP_n+P_nF^\top),
\qquad \Sigma_s=\sum_nP_n,
\tag{7b}
\]

until an entrywise-max-norm tail bound is below dtype rounding scale.  Failure to
converge is explicit.  This changes only the numerical evaluation of the exact
SDE covariance: it adds no jitter, eigenvalue clipping, diagonal floor,
low-rank approximation, or hidden SPD parameterization.  Value and gradient
equivalence with (7), plus a perturbed dense \(d=1024\) float32 boundary
Cholesky, are regression-tested.

### Model A loss

Here

\[
a_s=2D,
\qquad \nabla_y\!\cdot f=-\operatorname{tr}D,
\]

because \(\operatorname{tr}Q=0\).  Define

\[
H_s=L_s^{-1}DL_s^{-\top}.
\]

The direct-form path loss in (1), after analytically averaging the forward
score norm over \(\xi\), is

\[
r_s^\top H_sr_s
-\operatorname{tr}H_s
+\operatorname{tr}D.
\tag{8}
\]

After the exact entropy cancellation, the compact path term in (5) is simply

\[
\boxed{r_s^\top H_sr_s.}
\tag{9}
\]

It is evaluated without constructing \(H_s\):

\[
z_s=L_s^{-\top}r_s,
\qquad r_s^\top H_sr_s=z_s^\top Dz_s=\lVert C^\top z_s\rVert^2.
\]

### GRN regularization

For a drift matrix acting as \(Ay\), rows are targets and columns are sources.
The repository's legacy mask is loaded as `[source, target]` and transposed
exactly once to `[target, source]`, matching the recent work experiments.

Let

\[
M_{\mathrm{effective}}=M_{\mathrm{GRN}}\lor I
\]

so that required diagonal damping is not treated as an off-mask edge.  The
default L1 regularizer is

\[
R_{\mathrm{GRN}}
=\operatorname{mean}
\left|(1-M_{\mathrm{effective}})\odot(Q+D)\right|.
\tag{10}
\]

This acts on the composed drift matrix, not on the raw factors.  It is an
experiment-specific prior and is not part of the paper ELBO.  With a nonzero
coefficient the total loss is therefore accurately described as

\[
\mathcal J_{A,\mathrm{total}}
=\mathcal J_{\mathrm{paper\_elbo}}
+\lambda_{\mathrm{GRN}}R_{\mathrm{GRN}}.
\]

### Standard-diffusion initialization

Use the existing beta schedule only to define a physical-time grid

\[
s_t=-\log\bar\alpha_t.
\]

Initializing

\[
Q=0,
\qquad C=\frac1{\sqrt2}I,
\qquad D=\frac12I
\]

gives

\[
\Phi_s=e^{-s/2}I,
\qquad \Sigma_s=(1-e^{-s})I.
\]

At \(s_t=-\log\bar\alpha_t\), this is exactly

\[
m_t=\sqrt{\bar\alpha_t}x,
\qquad \Sigma_t=(1-\bar\alpha_t)I,
\]

the standard VP forward transition.

## Model B: free dense affine diffusion

### Parameterization and initialization

The implementation stores unconstrained dense parameters

\[
W=-\frac12I+W_{\mathrm{raw}},
\qquad b=b_{\mathrm{raw}}.
\]

This centering does not restrict the set of representable \(W\).  Initializing
\(W_{\mathrm{raw}}=0,b=0\) gives the same standard VP transition described
above.  No stability, symmetry, low-rank, sparsity, or spectral constraint is
applied to \(W\) by default.

### Exact affine transition

For time-independent \(W,b\),

\[
\begin{aligned}
\Phi_s&=e^{sW},\\
h_s&=\int_0^s e^{uW}b\,du,\\
m_s&=\Phi_sx+h_s,\\
\Sigma_s&=\int_0^s e^{uW}e^{uW^\top}\,du.
\end{aligned}
\tag{11}
\]

The formula \(W^{-1}(e^{sW}-I)b\) is not used because free \(W\) may be
singular.  Instead, a single augmented Van Loan exponential computes all
quantities:

\[
\exp\left[
s\begin{pmatrix}
W&I&b\\
0&-W^\top&0\\
0&0&0
\end{pmatrix}
\right]
=
\begin{pmatrix}
\Phi_s&V_s&h_s\\
0&\Phi_s^{-\top}&0\\
0&0&1
\end{pmatrix},
\qquad
\boxed{\Sigma_s=V_s\Phi_s^\top.}
\tag{12}
\]

This is differentiable through `torch.matrix_exp` and does not approximate the
dense dynamics.

### Model B loss

Here

\[
a_s=I,
\qquad \nabla_y\!\cdot f=\operatorname{tr}W,
\qquad M_s=L_s^{-1}L_s^{-\top}.
\]

The direct-form path loss in (1), after analytically averaging the forward
score norm, is

\[
\frac12r_s^\top M_sr_s
-\frac12\operatorname{tr}M_s
-\operatorname{tr}W.
\tag{13}
\]

The compact path term is

\[
\boxed{
\frac12r_s^\top M_sr_s
=\frac12\lVert L_s^{-\top}r_s\rVert^2.}
\tag{14}
\]

For \(\pi=N(0,I)\), the terminal contribution in (5) is computed analytically:

\[
\boxed{
\operatorname{KL}(q_\phi(y_T\mid x)\|\pi)
=\frac12\left[
 \lVert m_T\rVert^2
 +\operatorname{tr}\Sigma_T
 -\log\det\Sigma_T
 -d
\right].}
\tag{15}
\]

In `paper_elbo`, this is a core ELBO term rather than an independently weighted
regularizer.  Adding it to the direct expression (1) without removing the
forward-score/divergence entropy terms would double-count entropy.

Model B uses the same loaded GRN mask, one-time legacy transpose, diagonal
exemption, norm, and default weight as Model A.  Its external prior is

\[
R_{\mathrm{GRN},B}
=\operatorname{mean}
\left|(1-M_{\mathrm{effective}})\odot W\right|.
\tag{16}
\]

Thus the controlled comparison changes the stationary parameterization, not
whether a GRN prior is present:

\[
\mathcal J_{B,\mathrm{total}}
=\mathcal J_{\mathrm{paper\_elbo}}
+\lambda_{\mathrm{GRN}}R_{\mathrm{GRN},B}.
\]

For both models, the phrase `paper ELBO` refers only to the first term.  The
GRN contribution is an external experimental prior.

## Monte Carlo time estimator

The coefficients are time-independent in this experiment, but transition
statistics are functions of physical time \(s\).  We estimate

\[
\int_\delta^T h(s)\,ds
=(T-\delta)\,\mathbb E_{s\sim\operatorname{Uniform}(\delta,T)}[h(s)].
\]

A work-local sampler draws one physical time per DDP rank and shares it across
the batch.  Each sample still receives independent Gaussian noise.  Batch
sharing makes the matrix exponential, covariance, and Cholesky reusable within
one model forward.  The sampler returns schedule weight one; the
\((T-\delta)\) factor is applied internally only to the path term.  Boundary
NLL, terminal KL, and GRN regularization are not multiplied by it.

`microbatch=-1` is required so that the core loop does not split a batch into
separate matrix-statistics evaluations.  Cross-step autograd caches are
forbidden: every optimizer step changes the forward parameters.  Detached
caches are evaluation-only.

## Dynamic dimension normalization and recorded loss

Every Gaussian quadratic, KL, and likelihood primitive above remains in its
mathematical sum-over-dimensions form.  The implementation then applies one
common dynamic factor, obtained from `x_start.shape[-1]`, to the three compact
ELBO contributions:

\[
\begin{aligned}
L_{\mathrm{path,final}}
  &=\frac{(T-\delta)L_{\mathrm{path,raw}}}{d},\\
L_{\mathrm{terminal,final}}
  &=\frac{L_{\mathrm{terminal,raw}}}{d},\\
L_{\mathrm{boundary,final}}
  &=\frac{L_{\mathrm{boundary,raw}}}{d}.
\end{aligned}
\tag{17}
\]

The external GRN mean penalty has its own scale and is not divided by the
state dimension.  The exact minimized scalar is therefore

\[
\boxed{
L_{\mathrm{total}}
=L_{\mathrm{path,final}}
+L_{\mathrm{terminal,final}}
+L_{\mathrm{boundary,final}}
+\lambda_{\mathrm{GRN}}R_{\mathrm{GRN}}.}
\tag{18}
\]

There is no hard-coded gene count, including no objective-level constant
`1024`.  `diffusion.objectives.per_dimension` and
`per_dimension_elbo_terms` centralize this operation; tests exercise
\(d=2,3,7\).  In particular, the full-covariance quadratic is never rewritten
as an elementwise `.mean()`.

The work-local training-loop subclass writes
`segments/segment_NNN/loss_components.csv` with a common Model A/B schema.  A
row records training step, learning rate, physical and fractional time,
dynamic \(d\), total loss, all raw quantities, path-after-duration, all final
per-dimension quantities, raw/weight/final GRN values, and diagnostic plain
epsilon MSE.  Rows are buffered and flushed at the configured interval, every
checkpoint, and loop exit.  Validation reconstructs (18) before a row is
written.  The inherited ODE-specific `loss_details.csv` is disabled.

## Loss modes and gradient semantics

### `paper_elbo` (default)

This mode implements (5), including the Theorem 3 boundary likelihood.  All
terms remain connected to autograd:

- \(Q,D,W,b\to m_s,\Sigma_s,L_s\to y_s\to\xi_\theta\);
- \(D,L_s\to L_s^{-1}(2D)L_s^{-\top}\);
- \(W,L_s\to L_s^{-1}L_s^{-\top}\);
- terminal \(m_T,\Sigma_T,\log\det\Sigma_T\); and
- boundary mean, covariance, score, and NLL.

Reparameterizing \(y_s\) is necessary but not sufficient.  Detaching the
metric, covariance, terminal term, or boundary term would discard explicit
\(\phi\)-dependent pieces of the valid ELBO.

The mathematically exact components are not independently reweighted.  A
single global division by \(d\) may be applied consistently to boundary,
terminal, and path terms for scale.  Selectively using the repository's
per-coordinate `mean_flat` for only the path term changes their relative
weights and is not the same ELBO.  AdamW `weight_decay` is set to zero in an
exact run unless it is intentionally recorded as another external
regularizer.

### `epsilon_surrogate` (explicit ablation only)

This mode may retain plain

\[
\operatorname{mean}\lVert\xi_\theta-\xi\rVert^2
\]

for comparison with the repository's standard training.  It is a simplified
noise-prediction surrogate, not Equation (7) and not `paper_elbo`.  Adding a
terminal KL or GRN penalty does not make plain epsilon MSE equivalent to the
valid ELBO because the covariance-dependent metric and boundary likelihood are
still missing.

### Objective summary

| model | exact paper-derived loss (`paper_elbo`) | additional regularization | difference from plain epsilon MSE |
| --- | --- | --- | --- |
| Model A: stationary Q/D | Appendix-I boundary NLL + terminal `KL(q_phi(y_T\|x) \| N(0,I))` + `integral r_s^T L_s^{-1} D L_s^{-T} r_s ds` | Off-mask penalty on `Q+D`; external to the ELBO | Full matrix weighting through both `D` and `Sigma_s`, plus boundary and terminal terms |
| Model B: free affine | Appendix-I boundary NLL + terminal `KL(q_phi(y_T\|x) \| N(0,I))` + `1/2 integral r_s^T L_s^{-1}L_s^{-T}r_s ds` | Off-mask penalty on `W`; external to the ELBO and matched to Model A | Full Cholesky-coordinate metric `L_s^{-1}L_s^{-T}` (generally not `Sigma_s^{-1}`), plus boundary and terminal terms |

Only the `paper_elbo` column is the paper-derived objective: Equation (7)
truncated to \([\delta,T]\), made valid by Appendix I / Theorem 3, and written
in its exactly entropy-cancelled compact form.  `epsilon_surrogate` is
deliberately outside that claim.

## Custom reverse-SDE generation

Let the learned forward process be

\[
dy_s=f_\phi(y_s,s)\,ds+G_\phi(s)\,dB_s,
\qquad a_\phi=G_\phi G_\phi^\top.
\]

In increasing generative time \(\tau=T-s\), the reverse SDE used here is

\[
\boxed{
dz_\tau=
\left[a_\phi(s)s_\theta(z_\tau,s)-f_\phi(z_\tau,s)\right]d\tau
+G_\phi(s)d\bar B_\tau.}
\tag{19}
\]

Sampling starts from \(z_0\sim N(0,I)\).  This matches the declared prior; the
terminal KL in the training objective is what encourages the learned
finite-time forward marginal to match that prior.

For Model A,

\[
f_A(z)=-(Q+D)z,
\quad a_A=2D,
\quad G_A=\sqrt2C,
\]

so the reverse drift is

\[
2D\,s_\theta(z,s)+(Q+D)z.
\tag{20}
\]

For Model B,

\[
f_B(z)=Wz+b,
\quad a_B=I,
\quad G_B=I,
\]

and the reverse drift is

\[
s_\theta(z,s)-(Wz+b).
\tag{21}
\]

At each visited physical-grid index the sampler recomputes the current
full-covariance transition and uses exactly the training score map

\[
s_\theta(z,s)=-L_s^{-\top}\xi_\theta(z,s).
\]

The grid \(s_t=-\log\bar\alpha_t\) is traversed in descending index order.
For \(s_i>s_{i-1}\), Euler--Maruyama uses the positive increment

\[
\Delta\tau=s_i-s_{i-1}>0
\]

and the row-vector implementation is

\[
z_{i-1}=z_i+
\left[a_i s_\theta(z_i,s_i)-f_\phi(z_i,s_i)\right]\Delta\tau
+\sqrt{\Delta\tau}\,\eta_iG_i^\top,
\qquad \eta_i\sim N(0,I).
\tag{22}
\]

This sign and covariance convention is covered by unit tests.  Model A uses
the actual learned factor \(\sqrt2C\), including its sign and triangular
orientation; no new Cholesky of \(2D\) changes the declared diffusion factor.

The reverse integration stops at \(s=\delta\), not at the data directly.  Its
last operation is the same Appendix-I decoder (6) used in `paper_elbo`:

\[
x\sim p_\theta(x\mid y_\delta)
=\mathcal N(\mu_{\theta,\delta},
            \Phi_\delta^{-1}\Sigma_\delta\Phi_\delta^{-\top}).
\tag{23}
\]

Decoder sampling is the default; `decoder_sampling_mode="mean"` is an
explicit deterministic diagnostic option.  Sampling always defaults to the
highest-step checkpoint at the highest configured EMA rate and refuses to
reuse the standard DDPM/DDIM reverse code.

Each `.npz` archive and JSON sidecar contains `cell_gen`, ordered gene names,
their order-sensitive SHA256, checkpoint path and step, EMA rate, model
family, seed, sampler name, executed reverse-step count, and boundary decoder
mode.  A matching completed archive is skipped; a conflicting artifact at the
same path is never overwritten.

## Post-hoc analyses

All analyses are read-only with respect to checkpoints and write only below
the run's `analysis/` directory.  Completion metadata is trusted only when its
raw-source list or selected final EMA still matches the current run.

### Loss history

Segment CSVs are concatenated by training step and checked against (18).
Three plots and machine-readable tables separate:

- raw path, terminal KL, boundary NLL, and GRN quantities;
- the actual weighted/per-dimension contributions added to the optimizer
  loss; and
- signed contribution fractions.

Each curve is summarized by a rolling median and rolling 25th/75th
percentiles, following `work/20260830/analysis/loss_history.py`.

### Parameter evolution

`initial_forward_state.pt` is written before any optimizer update.  It is
combined with 5--10 automatically selected chronological **raw** checkpoints;
EMA files are not used.  Histograms follow the legacy layout of parameter
type by row, training stage by column, 60 bins, and mean/std annotations.

Model A reconstructs \(Q,D,A=Q+D,F=-A\), separates known/unknown GRN entries,
and records skew-symmetry, minimum-\(D\)-eigenvalue, and stationarity
residuals.  Model B reconstructs \(W,b\) and reports all/diagonal/off-diagonal
and exact-mask known/unknown values.  CSV summaries include count, mean, std,
median, q05, q95, Frobenius norm, and maximum absolute value.

### Diffusion-time diagnostics

The final EMA is evaluated on the same real-cell subset and the same fixed
Gaussian noise matrix at every selected timestep.  The default grid is
\(0,20,40,\ldots,999\), with the final index always included.  Statistics are
per-cell MSE, Pearson correlation across genes, L2 norms, and norm ratios for:

- predicted epsilon versus the true reparameterization epsilon;
- forward drift versus true epsilon;
- forward drift versus predicted epsilon; and
- predicted conditional score versus true conditional score.

The sum-form paper integrand diagnostic

\[
\frac12(s_\theta-s_\phi)^\top a_s(s_\theta-s_\phi)
\]

is also stored.  Every drift/noise figure and its metadata states:
`diagnostic comparison; forward drift and Gaussian noise have different
semantics`.  Drift/noise MSE must not be interpreted as denoising accuracy.

### Learned drift field and generated-cell UMAP

For clean saved training values, Model A writes
\(v(x)=-(Q+D)x\) and Model B writes \(v(x)=Wx+b\) to the AnnData layer
`learned_forward_drift`.  It is called a **Learned forward-diffusion drift
field**, never RNA velocity, and its direction is not reversed.  The
hematopoietic subset is the same default union as the historical work:
`Erythropoietic` + `Immune`.

The scVelo workflow follows `work/20260830/hematopoietic_viz/`: construct the
real-cell embedding once with PCA (`arpack`, 50 components), neighbors
(15 neighbors, 40 PCs), and seeded UMAP; preserve that `X_umap`; store saved
training values in `layers["X"]`; run `velocity_graph` with `xkey="X"`,
`backend="loky"`, and default `n_jobs=32`; then run `velocity_embedding` and
write stream, arrow, and grid figures.  Small-data safety caps only reduce
impossible PCA/neighbors values.  The historical lineage palette and scVelo
0.2.5 compatibility shims are retained.

Generated-vs-real UMAP concatenates sampled and real hematopoietic matrices in
the saved training representation.  It performs no renormalization, log
transform, or scaling.  Generated cells are labelled only
`Generated (unconditional)` and receive no inferred cell type.

The resulting run tree is:

```text
<run_dir>/
├── initial_forward_state.pt
├── samples/
│   ├── samples_ema_*.npz
│   └── samples_ema_*.json
└── analysis/
    ├── loss/
    ├── parameter_evolution/
    ├── diffusion_diagnostics/
    ├── drift_velocity/
    └── hematopoietic_umap/
```

`scripts/run_full_pipeline.py` gives Model A and Model B the same batch ID and
runs training, sampling, and these five analyses in order.  Either model may
be selected alone.  It records running/completed/failed status per stage,
safe-resumes an incomplete training run from a complete raw/optimizer/EMA
bundle, skips provenance-matching completed work, offers `--force-analysis`,
and never overwrites a completed sample archive.

## Numerical policy

For a batch-shared time the dense transition matrices are shared across the
batch, but their construction remains cubic in gene dimension.

- Model A requires a \(d\times d\) matrix exponential, dense products, and a
  \(d\times d\) Cholesky.
- Model B requires a \((2d+1)\times(2d+1)\) augmented exponential; for
  \(d=1024\), this is a \(2049\times2049\) matrix, plus covariance and
  Cholesky operations.
- Boundary \(\delta\), sampled \(s\), and terminal \(T\) statistics are reused
  within a forward but may each require a separate exponential.

At \(d=1024\), one dense \(d\times d\) value occupies about 4 MiB in float32
or 8 MiB in float64.  Model B's single \(2049\times2049\) augmented value is
about 16 MiB or 32 MiB respectively.  These are only primal-tensor lower
bounds: three endpoint/path graphs, Padé/scaling-and-squaring intermediates,
Cholesky factors, and matrix-exponential backward require multiple additional
dense buffers.  Runtime remains \(O(d^3)\); sharing a timestep removes a batch
factor from the matrix decompositions but does not change that cubic scaling.

Correctness tests use float64.  The \(d=1024\) benchmark measures float32 and
float64 where supported.  FP16 is disabled for these matrix operations.

The production `scripts/train.py` entrypoint uses float32 forward parameters.
This is required for compatibility with the unchanged legacy
`MixedPrecisionTrainer`, whose gradient/parameter norm calculation requests a
float32 norm and rejects narrowing a float64 parameter tensor.  Float64 remains
available when the forward modules are called directly by the mathematical
correctness tests and `scripts/benchmark_dense.py`; it is not accepted by the
legacy-`TrainLoop` training entrypoint.  This dtype restriction changes no
forward-process equation or parameterization, but it is an explicit numerical
constraint of the inherited training infrastructure.

The production and benchmark entrypoints currently support CPU and CUDA.  MPS
is rejected explicitly because the float64 physical-time/correctness path is
not supported there.  Before importing Scanpy, `train.py` assigns writable
Numba and Matplotlib cache directories below the ignored `runs/` tree unless
the user already supplied those environment variables.

Every covariance is symmetrized as

\[
\Sigma\leftarrow\frac12(\Sigma+\Sigma^\top)
\]

before `torch.linalg.cholesky_ex`.  The returned `info` is checked.  The default
does not silently clip eigenvalues or add jitter.  A configured jitter or the
Model A SPD option is a numerical-stability intervention and must be reported
as such.  Free-\(W\) overflow, a failed factorization, OOM, or non-finite
gradient fails with diagnostics rather than activating an approximation.

## Tests

The test suite covers:

1. the isotropic standard-VP limit, including transition equality and the
   scalar-weight relationship to epsilon MSE;
2. samplewise equality of score-space and noise-space matrix-weighted losses;
3. value and parameter-gradient equality between the **valid direct form (1),
   including `-log p + log q` boundary correction**, and compact form (5);
4. finite gradients to raw \(Q,C,W,b\);
5. exact skew symmetry of \(Q\) and PSD of the paper-default \(D=CC^\top\);
6. Model A stationarity and covariance identities, exact integral-series
   value/gradient equivalence, and the dense 1024-dimensional float32
   lower-boundary cancellation regression;
7. Model B affine transition and analytic terminal KL;
8. GRN mask orientation and diagonal exemption;
9. optimizer, EMA, raw/EMA/optimizer checkpoint, and resume through the actual
   unchanged core `TrainLoop`, plus DDP-compatible ownership of all trainable
   parameters; and
10. dynamic ELBO normalization at \(d=2,3,7\), loss-row reconstruction, and
    absence of a hard-coded objective dimension;
11. common Model A/B mask semantics, Model B off-mask behavior, and identical
    GRN weighting;
12. reverse score transformation, reverse-drift sign, \(GG^\top=2D\) for
    Model A, \(G=I\) for Model B, fixed-seed reproducibility, Appendix-I decode,
    generated shape, and sample gene/provenance integrity;
13. 1D/2D Euler--Maruyama Gaussian mean/covariance sanity checks;
14. checkpoint ordering, Model A/B parameter extraction, every timestep
    metric family, loss rolling/fraction calculations, and the historical
    hematopoietic selection; and
15. full-pipeline dry-run plus training run/resume/skip decisions, while also
    confirming that no subspace/low-rank model is present.

The equivalence in item 3 is expectation-level.  Its small-dimensional float64
test evaluates Gaussian expectations and a high-accuracy time quadrature; it
does not incorrectly demand samplewise cancellation of entropy terms.

`test_objective_equivalence.py` performs this audit at two levels.  First it
compares the objective primitives.  Second, for both Model A and Model B, it
actually calls the production `LearnableForwardModel` together with
`LearnableForwardTrainingDiffusion(loss_mode="paper_elbo",
normalize_elbo_by_dimension=False)` across the quadrature nodes.  The reference
side is assembled independently in score space from the Appendix-I boundary
NLL, analytic \(\mathbb E[\log q_\phi(y_\delta\mid x)]\), terminal
cross-entropy, forward-score energy, drift divergence, and score mismatch.  It
checks equality of both values and all denoiser/forward-parameter gradients.
As a negative guard, the same test removes the complete
\(-\log p_\theta(x\mid y_\delta)+\log q_\phi(y_\delta\mid x)\) correction and
asserts that the resulting raw truncated Equation (7) value and gradient do
*not* equal production `paper_elbo`.

The opt-in \(d=1024\) benchmark reports forward time, full-loss time, backward
time, peak device memory, Cholesky status, loss finiteness, and gradient
finiteness for both models.  Failure at \(d=1024\) is a valid benchmark result,
not permission to introduce low-rank or subspace approximations.

### Measured dense 1024-dimensional result

On 2026-09-04, the exact benchmark was run at the standard-VP initialization
on CPU with PyTorch 2.5.1, four PyTorch threads, batch size 4, one warm-up, and
three measured repetitions.
Each full forward includes path, boundary, and terminal transition statistics;
the denoiser in this diagnostic is a minimal dense linear layer, so these are
forward-process/objective costs rather than end-to-end `Cell_Unet` throughput.

| model | dtype | transition stats median | full `paper_elbo` forward median | backward median | result |
| --- | ---: | ---: | ---: | ---: | --- |
| Model A | float32 | 0.0255 s | 0.0838 s | 0.3499 s | 3/3 finite, Cholesky succeeded |
| Model A | float64 | 0.0710 s | 0.2047 s | 1.2517 s | 3/3 finite, Cholesky succeeded |
| Model B | float32 | 0.1151 s | 0.2978 s | 2.4763 s | 3/3 finite, Cholesky succeeded |
| Model B | float64 | 0.3944 s | 1.0223 s | 10.3652 s | 3/3 finite, Cholesky succeeded |

No approximation or fallback was used.  Model B's backward is materially more
expensive because autograd differentiates three \(2049\times2049\) augmented
matrix exponentials.  The run establishes correctness/feasibility on this CPU,
but production throughput and accelerator memory must be measured on the
target CUDA host before choosing a training budget.  This CPU run did not
provide a reliable phase-local peak-RSS counter; the benchmark reports CUDA
peak allocation when run on CUDA, while the dense-buffer sizes above give only
the CPU memory floor.

## Commands

From the repository root:

```bash
# Mathematical and integration tests.
conda run -n scdiffusion python -m unittest discover \
  -s work/20260903_learnable_forward/tests -p 'test_*.py' -v

# Confirm that no protected guided_diffusion file changed.
conda run -n scdiffusion python \
  work/20260903_learnable_forward/scripts/verify_protected.py

# Inspect the complete two-model execution plan without writing a run.
conda run -n scdiffusion python \
  work/20260903_learnable_forward/scripts/run_full_pipeline.py \
  --models stationary_qd,free_affine \
  --batch-id DRY_RUN_ID \
  --device cuda \
  --dry-run

# Full training -> custom sampling -> all analyses for both models.
conda run -n scdiffusion python \
  work/20260903_learnable_forward/scripts/run_full_pipeline.py \
  --models stationary_qd,free_affine \
  --batch-id RUN_ID \
  --device cuda

# One model is also valid; rerunning safely resumes/skips completed stages.
conda run -n scdiffusion python \
  work/20260903_learnable_forward/scripts/run_full_pipeline.py \
  --models stationary_qd \
  --batch-id RUN_ID \
  --device cuda

# Standalone sample or analysis of an existing run.
conda run -n scdiffusion python \
  work/20260903_learnable_forward/scripts/sample.py \
  --run-dir work/20260903_learnable_forward/runs/model_a_stationary_qd_dense/RUN_ID \
  --device cuda
conda run -n scdiffusion python \
  work/20260903_learnable_forward/scripts/analyze.py \
  --run-dir work/20260903_learnable_forward/runs/model_a_stationary_qd_dense/RUN_ID \
  --stage all --device cuda

# Reproduce the dense benchmark. Output files are ignored runtime artifacts.
conda run -n scdiffusion python \
  work/20260903_learnable_forward/scripts/benchmark_dense.py \
  --dim 1024 --batch-size 4 --models all \
  --dtypes float32,float64 --device cpu --num-threads 4 \
  --warmup 1 --repeats 3 \
  --output work/20260903_learnable_forward/benchmark_results/d1024.json
```

Training outputs are constrained to this suite's `runs/<experiment>/<run-id>/`
tree.  The two provided configs are dense-only and default to `paper_elbo`.
The Model A config uses `d_parameterization="psd"`, zero diagonal floor, and
zero covariance jitter.  Both configs use the same target/source GRN prior and
default weight 5.0.  Generation uses only the custom sampler described above;
the repository's `q_posterior_mean_variance`, `_predict_xstart_from_eps`,
`p_mean_variance`, `p_sample_loop`, and DDIM paths remain outside this
experiment.
