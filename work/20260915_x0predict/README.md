# START_X two-stage experiment — 20260915

This is an isolated diffusion-parameterization experiment. A newly initialized
CellUNet predicts the clean expression vector `x0` directly. Stage2 freezes its
final EMA and trains one explicit ODE field inside Hybrid500. All four ODE
families are **single explicit-parametric ODEs: K=1, no neural expert gating,
no state-dependent mixture of ODEs**. Their parameters have no expert axis.

No full training has been run during implementation. The tests use small
synthetic CPU tensors. Existing experiments, checkpoints, and diffusion core
files remain unchanged.

## Source audit and correction

The initial adapter used `20260803::HillAfterLinearField` for Hill-after-linear,
`20260816::CenteredSignedHillField` / `ShiftedHillRhoField` with K=8 for the direct
Hill families, and `20260830::SimpleSoftplus20260830` for simple-softplus.
The user's correction supersedes that selection. **All four now import the
20260830 single-ODE implementations.**

`work/20260816` contains an older/different K=8 expert-gated direct Hill model;
that ODE version is intentionally excluded here. Commit
`ed376ec7828eea149483a3ac00aba86ebd6ff5d2` ("Add 20260817 direct Hill single ODE
experiments") introduced the direct single-ODE variants in
`work/20260817_singleODE`. Commit `5a3f758c0745730bd0e962804973d039912bd47f`
consolidated the four fields in `work/20260830/models/ode_fields_20260830.py`.
The subsequent `80808be` change extracted `off_mask_penalty_base` while keeping
the field equations and weighted penalty semantics intact; the current file is
reused. Its separate 100k consistency-training experiment is not inherited.

The earlier `softmax(Wx+b)-c*x` phrase was a naming mistake: the intended model
is **simple_softplus**. Searching 549 unique historical Python blobs under
`ODE`, `work`, and `LegacyFiles` found softmax in expert gating and classification,
and no historical softmax-over-genes production-decay model. See
[history evidence](audit/history_search.json). `GeneODE` in
`ODE/ode_20260421_regODEMLratio.py` uses softplus production and softplus gamma
decay; its old descriptive `exp` docstring is not the executed equation.

## Four exact fields

All classes below are imported from
[`work/20260830/models/ode_fields_20260830.py`](../20260830/models/ode_fields_20260830.py)
through that suite's `build_ode_from_config` factory. No field equation is copied.
The 20260830 CellUNet-consistency wrapper and objective are not used.

Let `sp(u)=log(1+exp(u))`, `e=1e-6`, and `x⁺_j=max(x_j,e)`. All matrices are
`[target i, regulator j]`; the shared TF-edge loader's `[source,target]` mask is
transposed exactly once. Soft constraints do not hard-mask the forward field.

| Family | Imported class | Exact forward equation |
|---|---|---|
| `simple_softplus` | `SimpleSoftplus20260830` | `f_i=sp(s*(Σ_j W_ij*x_j+b_i))-sp(gamma_i)*x_i`, `s=1/sqrt(G)` |
| `hill_after_linear` | `HillAfterLinear20260830` | `z_i=sp(Σ_j W_ij*x_j+b_i)`; `f_i=V_i*z_i²/(K_i²+z_i²)-delta_i*x_i` |
| `centered_signed_hill` | `CenteredSignedHill20260830` | `f_i=b_i+Σ_j A_ij*tanh(alpha_ij*(log(x⁺_j)-log(theta_ij)))-delta_i*x_i` |
| `shifted_hill_rho` | `ShiftedHillRho20260830` | `f_i=b_i+Σ_j A_ij*expm1(rho_ij)*sigmoid(2*(log(x⁺_j)-log(theta_ij)))-delta_i*x_i` |

For shifted Hill the sigmoid equals `x⁺_j²/(theta_ij²+x⁺_j²)`. The existing
`expm1` and log/sigmoid calculation is preserved. Centered and shifted fields
sum **directly over regulator genes j**, with no expert index or selector.
The positive-input guard affects regulatory responses; decay uses the original
signed `x`, exactly as in the source.

Positive `A`, `theta`, Hill threshold `K`, capacity `V`, and `delta` use
`sp(raw)+e`. `A`, `theta`, `K`, `V` initialize to 1; `raw_delta` initializes to
0.1. `W`, `alpha`, and `rho` initialize as independent normal values with standard
deviation `1/sqrt(G)`. Biases initialize to zero. Simple-softplus preserves raw
`gamma=0.1`, `s=1/sqrt(G)`, and decay `sp(gamma)` without the additional `e`.
Here the Hill threshold `K_i` is unrelated to the number of components, which
is always one. Direct Hill uses the source target-chunk size 16.

Construction and checkpoint restoration assert exact class identity,
`is_lincomb=False`, `num_components=num_experts=1`, no child neural modules or
gating parameters, matrix shapes `[G,G]`, and decay shape `[G]`. The ODE has no
`coeff_net`, `gate_network`, or `time_emb`. CellUNet remains a neural network
outside the ODE field. `SingleODEHybrid500` uses the unchanged historical
`Hybrid500Mixin` arithmetic, with a thin branch-weight shape adapter.

## START_X semantics and stages

Forward diffusion is unchanged:

```text
x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
```

In epsilon prediction the denoiser's target is `noise`. In this campaign it is
`x0`: `predict_xstart=True` produces `ModelMeanType.START_X` and `LossType.MSE`.
The existing `GaussianDiffusion.training_losses` selects `x_start` as its MSE
target; `p_mean_variance` already accepts the raw prediction as `pred_xstart`.
There is no need to modify `guided_diffusion` or introduce a custom sampler.

**Stage1:** train a fresh `guided_diffusion.cell_model.Cell_Unet`, hidden widths
`[2000,1000,500,500]`, using the recent two-stage defaults: 30,000 updates,
batch 128, AdamW learning rate `1e-4`, weight decay `1e-4`, EMA `0.9999`, seed
1234, linear 1000-step diffusion, no respacing, no FP16, no class conditioning.
Preserve the source post-update LR annealing convention. Load unchanged `X`
using `load_data(train_vae=True, preprocess=False, layer=None)` and unchanged
gene order. Noise follows the source float64 Gaussian convention; model
parameters are float32.

```text
t ~ Uniform{0,...,999}
L_stage1 = diffusion.training_losses(CellUNet, x0, t)["loss"].mean()
         = mean((CellUNet(x_t,t)-x0)^2)
```

**Stage2:** all eight conditions load the same final Stage1 EMA from the same
campaign. CellUNet parameters have `requires_grad=False` and its `train()`
override keeps it in eval mode. AdamW contains exactly the trainable ODE
parameters. SHA256 state hashes verify CellUNet before and after training,
including the EMA copy. Dataset, edge-file, checkpoint-file, and gene-order
hashes prevent accidental mixing. Stage2 also uses 30,000 updates per condition.

## Hybrid500 and the eight conditions

Using original unscaled diffusion timestep `t`:

```text
w_ode(t) = max(0, 1-t/500)
w_cell(t) = 1-w_ode(t)
x0_hat = w_ode(t)*ODE(x_t,t) + w_cell(t)*CellUNet(x_t,t)
```

| t | CellUNet weight | ODE weight |
|---:|---:|---:|
| 999 | 1 | 0 |
| 750 | 1 | 0 |
| 500 | 1 | 0 |
| 250 | 0.5 | 0.5 |
| 0 | 0 | 1 |

The ODE-form function itself is the structured START_X prediction. There is no
epsilon conversion and no velocity-to-state/Euler adapter in the Hybrid.
Inactive ODE rows are skipped in ordinary forward calls; explicit diagnostics
still evaluate their raw output.

| Field | Ordinary START_X objective | Low-noise OT objective |
|---|---|---|
| simple | `simple_softplus_start_x_soft` | `simple_softplus_ot_soft` |
| Hill-after-linear | `hill_after_linear_start_x_soft` | `hill_after_linear_ot_soft` |
| centered | `centered_signed_hill_start_x_soft` | `centered_signed_hill_ot_soft` |
| shifted | `shifted_hill_rho_start_x_soft` | `shifted_hill_rho_ot_soft` |

For `*_start_x_soft`, use `diffusion.training_losses` with ordinary uniform
`t=0..999`. At `t>=500` the data term supplies no ODE gradient because the
CellUNet is frozen; structural regularization remains active. Timesteps are
not truncated or reweighted.

For `*_ot_soft`, sample uniform `t=0..49`, call `q_sample`, and pass the raw model
output directly to Sinkhorn: `prediction=model(x_t,t); pred_x0=prediction`.
The code asserts object identity and never calls `_predict_xstart_from_eps`.
The loss is set-to-set OT, without explicit cell pairing.

```text
L_start_x_soft = mean((model(x_t,t)-x0)^2) + L_soft
L_ot_soft     = SinkhornDivergence(model(x_t,t), x0) + L_soft
L_soft        = 1.0 * ode.off_mask_penalty("l1")
              = 5.0 * mean(abs((1-mask)*P))
P             = W (simple/Hill), alpha (centered), rho (shifted)
```

The Sinkhorn solver is imported unchanged from
`work/20260913_2step/losses/sinkhorn.py`: cell-to-cell mean squared gene-distance
cost, uniform marginals, debiased divergence, float64, epsilon 0.1, marginal
tolerance `1e-5`, maximum 2000 iterations. Training fails on nonconvergence;
independent conditions continue. Evaluation retains the existing epsilon-scaling
retry and explicitly marks missing distances if convergence still fails.
No trajectory-OT, velocity, kinetic, or manifold training loss is present.

## Sampling and analysis

The launcher samples Stage1 separately and all completed Stage2 conditions:
3000 cells each, batches of 50, native ancestral `diffusion.p_sample`, no DDIM,
`clip_denoised=False`, default `nw=0.5` in `exp(nw*log_variance)`.
Sampling asserts START_X and equality of the saved raw output and native
`pred_xstart`. Snapshots are saved every 50 reverse updates.

Files include `sample_state.npy`, `pred_xstart.npy`, `model_x0.npy`,
`cellunet_raw.npy`, and, for Hybrids, `ode_raw.npy`. There is no mislabeled
epsilon array. Metadata distinguishes the input of an update from its resulting
state and records original timesteps.

On identical forward-noised real cells, diagnostics compare Hybrid, raw
CellUNet, and raw ODE against **true x0**, calculating per-cell Pearson across
genes, MSE, cosine, L2 norm, and prediction/x0 norm ratio. Means and population
standard deviations are summarized across cells. The full grid is every 20
steps plus 250/500/750/999; the low-noise grid is 0..50. The seeded CPU Gaussian
stream is reset at each timestep and shared across conditions.

Saved plots include `true_x0_metrics_pearson_full.png`,
`true_x0_metrics_mse_full.png`, `true_x0_metrics_cosine_full.png`,
`true_x0_metrics_l2_full.png`, and `true_x0_metrics_norm_ratio_full.png`, plus
low-noise views, branch comparisons, and weighted/unweighted norm plots.

The recent generated-vs-real pipeline is retained: unchanged Erythropoietic
reference selection, diversity, Sinkhorn-to-real, sliced Wasserstein, independent
UMAP fits at each snapshot, and a joint terminal fit for Stage2. The UMAP uses
`pred_xstart` during diffusion and actual post-integration states afterward.
These analyses can test whether START_X CellUNet still collapses; this
implementation does not claim an outcome before training.

Stage2 also retains **post-hoc dynamics analysis**: 100 Euler steps of size
0.001 after diffusion, with terminal conditioning `t=0`. Snapshots at 50/100
steps and convergence/divergence diagnostics are recorded separately. This
integration is not part of training or the diffusion predictor semantics.
Numerical and UMAP analyses run independently so a failure in one does not
prevent the other or subsequent training conditions.

## Outputs and provenance

```text
work/20260915_x0predict/
  runs/x0predict_<UTC>_<RUNID>/
    campaign.json, source_sha256.json, configs/*.json
    canonical_stage1.json
    <condition>/<attempt>/
      effective_config.json, metadata.json, losses.csv
      checkpoints/modelNNNNNN.pt, ema_0.9999_NNNNNN.pt, optNNNNNN.pt
      checkpoints/bundleNNNNNN.json
      completed.json or failed.json
    steps/<step>/<attempt>/output.log
    invocation_<UTC>_<RUNID>.json
  results/x0predict_<UTC>_<RUNID>/<condition>/<sample_attempt>/
    sampling_metadata.json, snapshot_metadata.csv, *.npy
    analyze/<attempt>/{*.csv,*.json,figures/<attempt>/*.png}
    embed/<attempt>/{*.csv,*.json,figures/<attempt>/*.png}
```

Checkpoints preserve the established `{"state_dict": ..., "metadata": ...}`
format and raw/EMA/optimizer artifacts. Every Stage2 effective config and
checkpoint records family, `work/20260830` source suite, exact source file and
class, `ode_components=1`, `expert_gating=false`, source SHA256, START_X,
Hybrid500, objective/Sinkhorn settings, seed, git commit, and Stage1 provenance.

The campaign stores immutable effective configs and source hashes. Completion
markers prevent retraining finished conditions. Interrupted training resumes
from the latest complete raw/EMA/optimizer bundle into a new attempt directory.
As in the recent suite, continuation restores update count/optimizer/EMA but
restarts the seeded RNG/data stream; it is not bitwise-equivalent to uninterrupted
training. Partial checkpoint writes are never selected. A filesystem lock
prevents concurrent launchers for the same campaign. All paths are confined to
this suite and output files/directories use exclusive creation.

## Launch and resume

Activate the repository's `scdiffusion` Python environment. From the repository
root (Linux checkout or this macOS checkout), inspect the plan without training:

```bash
bash work/20260915_x0predict/scripts/run_all.sh --all-eight --dry-run
```

Train Stage1 once and run its separate sampling/analysis:

```bash
bash work/20260915_x0predict/scripts/run_all.sh --stage1-only --device cuda
```

Train a new Stage1 followed by all eight conditions, sampling, analysis, plots:

```bash
bash work/20260915_x0predict/scripts/run_all.sh --all-eight --device cuda
```

To run Stage2 against the campaign printed by the Stage1 command:

```bash
bash work/20260915_x0predict/scripts/run_all.sh --stage2-only --all-eight \
  --resume-campaign x0predict_<UTC>_<RUNID> --device cuda
```

Resume all unfinished work or select one condition:

```bash
bash work/20260915_x0predict/scripts/run_all.sh \
  --resume-campaign x0predict_<UTC>_<RUNID> --all-eight --device cuda
bash work/20260915_x0predict/scripts/run_all.sh --stage2-only \
  --resume-campaign x0predict_<UTC>_<RUNID> \
  --condition centered_signed_hill_ot_soft --device cuda
```

For relocated input files, provide `--data /absolute/path/Embryonic.h5ad` and
`--edge-tsv /absolute/path/tf_target_edges.tsv` on the **new campaign** command.
Defaults retain the historical Linux dataset paths and are checked before
campaign creation. The requested `/home/suzuki/Projects/scDiffusion-github`
checkout is not present on this machine; implementation was performed in
`/Users/cls-lab/Git/scDiffusionODE`. No dataset was substituted.
Set `PYTHON=/path/to/environment/bin/python` when the desired interpreter is
not active. `--analysis-device cpu` is the default.

## Lightweight verification

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s work/20260915_x0predict/tests -v
git diff --check
```

The initial 17 tests passed; the added plot/distribution test raised the
pre-correction total to 18, all passing. Applicable tests are retained, with
source expectations corrected to K=1. Additional tests explicitly reject the
old K=8 fields and injected gates, inspect named modules/parameters and matrix
shapes, and compare every field to deterministic manual equations (including
negative/zero input guards and direct sums over regulators).
See [test results](audit/test_results.txt),
[pre-correction results](audit/test_results_before_single_ode_correction.txt),
and [implementation report](IMPLEMENTATION_REPORT.md).
