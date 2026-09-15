# 20260913 suite / 20260914 trajectory-occupation OT

This suite implements a new two-stage experiment. **No scientific training,
sampling, dataset analysis, UMAP fitting, or figure generation was performed as
part of implementation.** Only small CPU unit tests and static/CLI checks are
intended during development. All new experiment outputs stay below this suite.
Existing source suites, checkpoints, runs, and figures are read-only dependencies.

## Hypothesis and experiment matrix

The governing ODE is unknown and is learned. The hypothesis is that ordinary
score matching can be unreliable near the low-noise data manifold, and a learned
ODE may supply a useful trajectory/manifold constraint. This implementation does
not establish that hypothesis, or assume that an epsilon-trained branch is a
validated biological vector field.

One shared Stage-1 CellUNet is trained on the full data distribution. Every
Stage-2 model starts from exactly that run's **final EMA CellUNet**, freezes it,
and trains the existing ODE branch. Training remains unconditional and uses all
source training cells; Erythropoietic selection is an analysis reference only.

| Canonical condition | Stage-2 ODE structure | Training objective / t support |
| --- | --- | --- |
| `stage1_cellunet` | none | epsilon-MSE, uniform 0..999 |
| `hill_after_linear_recon_soft` | single, original 20260803 Hill-after-linear | epsilon-MSE + soft, uniform 0..999 |
| `centered_signed_hill_recon_soft` | K=8 original 20260816 centered signed Hill | epsilon-MSE + soft, uniform 0..999 |
| `shifted_hill_rho_recon_soft` | K=8 original 20260816 shifted Hill rho | epsilon-MSE + soft, uniform 0..999 |
| `hill_after_linear_trajectory_ot_soft` | single, original 20260803 Hill-after-linear | trajectory-occupation Sinkhorn + soft; ODE conditioning t=0 |
| `centered_signed_hill_trajectory_ot_soft` | K=8 original 20260816 centered signed Hill | trajectory-occupation Sinkhorn + soft; ODE conditioning t=0 |
| `shifted_hill_rho_trajectory_ot_soft` | K=8 original 20260816 shifted Hill rho | trajectory-occupation Sinkhorn + soft; ODE conditioning t=0 |

There is exactly one common Stage 1 per campaign. A completed campaign writes
`canonical_stage1.json` once. All six Stage-2 conditions resolve that same file;
they cannot specify an alternative Stage-1 checkpoint. The SHA-256 is verified
on every Stage-2 startup. A new Stage-1 experiment requires a new campaign name.
Repeated Stage-2 trials get new run IDs and retain the same canonical backbone.

## Source reuse and immutable conventions

The selections in `../20260911/configs/selected_models.json` define the three
families. Their old EMA weights are **not** loaded: these are new two-stage
experiments using the selected architectures and configurations.

| Purpose | Existing implementation reused |
| --- | --- |
| Hill-after-linear single ODE and source Hybrid | `work/20260803_ODE_hill_exp/models/factory.py`, `models/ode_fields.py`; `standard_hybrid_single__hill_after_linear.json` + source `base.json` |
| K=8 centered/shifted Hill and source Hybrid | `work/20260816/models/factory.py`, `models/ode_fields.py`; `linear_centered_signed_hill.json` / `linear_shifted_hill_rho.json` + source `base.json` |
| Hybrid forward and weights | `ODE/ode_20260609_hybrid5x3.py::UnifiedODEMLHybrid`; factory `branch_outputs` hooks |
| Pure neural branch | `guided_diffusion/cell_model.py::Cell_Unet` |
| Diffusion construction | `guided_diffusion/script_util.py::create_gaussian_diffusion` (same helper underlying the source factory; avoids building an unused CellUNet) |
| Training epsilon loss, x0 conversion, ancestral update | `guided_diffusion/gaussian_diffusion.py::training_losses`, `_predict_xstart_from_eps`, `p_sample` |
| Data loading | `guided_diffusion/cell_datasets_loader.py::load_data` |
| Full timestep sampling | `guided_diffusion/resample.py::create_named_schedule_sampler` |
| Pearson/L2 metrics | `work/20260816/viz/analysis_helpers.py::_sample_corr`, `_sample_norm`; MSE/cosine use the same expressions as `evaluate_model_io` |
| Selection, gene hash, AnnData assembly, PCA/neighbors/UMAP | `work/20260830/hematopoietic_viz/core.py`, imported through `work/20260911/src/source_imports.py::umap_core` |
| Storage/independent UMAP pattern | adapters following `work/20260911/src/trajectory.py` and `umap_adapter.py` |

The 20260830 model factory was inspected but its different single-ODE
regularization architecture is not substituted for the selected Hybrid families.
The 20260816 `analyze_corr_norm_grid.py` was inspected for grids and branch
comparison behavior. Source packages are imported by qualified `importlib` names
so the two source `models` packages can coexist.

Configurations merge the original base and experiment JSON with the suite's
explicit numerical additions. Effective configs and source-config SHA-256 values
are saved. Defaults remain:

- 1000 linear diffusion steps, no respacing or timestep rescaling;
- epsilon prediction, unrescaled MSE, fixed-large variance, no learned sigma;
- CellUNet hidden widths `[2000,1000,500,500]`, unchanged architecture/dropout;
- batch 128, AdamW LR `1e-4`, weight decay `1e-4`, EMA `0.9999`, seed `1234`;
- 30,000 updates per stage, source linear LR annealing (after the update),
  checkpoint interval 5,000, log interval 1,000;
- float32 model parameters and source **float64 Gaussian training noise**;
- `load_data(train_vae=True, preprocess=False, layer=None)`: the existing
  preprocessed `AnnData.X`, no VAE encoding, no new normalization/log1p/scale,
  and exact unique `var['gene_name']` ordering;
- unconditional ancestral DDPM, source `p_sample` with `nw=0.5`,
  `clip_denoised=False`, no DDIM or extra clipping.

The local loop is single-device, full-batch, and full precision. It exists to
separate the two objectives and explicitly exclude frozen weights from optimizer
and EMA arithmetic. It does not add DDP, mixed precision, microbatching, gradient
clipping, or implicit resume. Explicit Stage-2 checkpoint continuation is available through
`--resume-checkpoint` and the recovery launcher below. Source default
`microbatch=-1` is preserved.

## Stage 1 and frozen Stage 2

Stage 1 is exactly `x_t -> Cell_Unet -> epsilon_hat`, with
`L1 = diffusion.training_losses(...)["loss"].mean()` under the original uniform
full-range sampler. It has no ODE module, mask, soft penalty, or ODE regularizer.

Stage 2 uses the source factory's Hybrid. Its `ml_model` is the same CellUNet
architecture with a mode-only subclass whose `train(...)` always leaves it in
`eval()`. Loading is strict. All CellUNet parameters have `requires_grad=False`.
The optimizer is checked to contain exactly the trainable ODE parameters,
including the source LinComb gates. Parent `.train()` cannot re-enable neural
branch dropout. EMA performs arithmetic only on trainable parameters and copies
frozen state exactly, avoiding floating-point rounding of identical weights.

Mode and optimizer invariants run every update. Full CellUNet parameter/buffer
SHA-256 comparisons run at checkpoints and completion. Raw and EMA checkpoint
checks prove their frozen branch equals the Stage-1 initialization. Metadata
records before/after hashes and the canonical Stage-1 path and file SHA-256.

Raw `modelNNNNNN.pt` and `ema_0.9999_NNNNNN.pt` contain a `state_dict` plus
metadata (config, genes/hash, seed, git commit, origin and step). Optimizer state
is saved separately. Stage-2 checkpoints include the full neural branch, ODE,
gates, and mask. `training/checkpoints.py::restore` requires neither the Stage-1
file nor the edge TSV nor the real dataset. Sampling still imports the original
architecture code from this repository. Sidecar JSON files record file hashes.
These metadata-bearing checkpoints should be read through this suite's loader;
a legacy raw-state consumer would need the `state_dict` member.

### Unchanged interpolation and soft constraint

For normal inference, all six models call the **original Hybrid forward**:

```
w_ode(t) = 1 - t / 999
eps_hybrid = w_ode * eps_ode + (1 - w_ode) * eps_cellunet
```

At t=999 the output is CellUNet; at t=0 it is entirely ODE. No rescaling of the
legacy 50-step OT region and no learned replacement interpolation is introduced.

Reconstruction Stage 2 uses `L = epsilon_MSE + ode_reg_lambda * off_mask_penalty`.
It samples the normal full `t=0..999` range, including all high-noise timesteps.

The exact source `ode_model.off_mask_penalty('l1')` is reused. Outer
`ode_reg_lambda=1`, internal `off_mask_lambda=5`; there is no extra factor 5.
Mask orientation is the source loader's `[source,target]` transposed exactly once
to `[target,source]`. The penalized parameter is source `W` for Hill-after-linear,
`alpha` for centered signed Hill, or `rho` for shifted Hill. Masked L1 is averaged
over **all** matrix entries, including on-mask zeros, exactly as source code.
Source positivity transforms, regulator clamps, decay and gating remain intact.
Other auxiliary penalty coefficients stay at their source zero defaults.

## Canonical trajectory-occupation objective

The old one-step `S(pred_x0, x0)` matches a denoised forward-noised real batch
against its own clean cells. It remains in `training/objectives.py` and the three
`*_ot_soft.json` files as a **legacy baseline**, with unchanged numerical defaults.
The new experiment asks whether frozen diffusion first approaches the manifold,
then a learned ODE explores it so that the union of its trajectories covers the
empirical distribution. Closeness to a manifold at x50 is a hypothesis to inspect,
not an established property of the cache or a nearest-neighbor projection target.

New configs explicitly store `objective="trajectory_ot"` and
`objective_schema_version="trajectory_occupation_v1"`. Stage 1, reconstruction
configs/losses, ODE definitions, soft coefficients, optimizer defaults and normal
Hybrid interpolation retain their source behavior. New trajectory OT conditions
initialize fresh ODE parameters using seed 1234 and load the same canonical
frozen Stage-1 EMA; no legacy ODE weights are transplanted.

### Generated x50 source cache

`training/source_cache.py` / `scripts/cache_x50.py` generate Gaussian-start
ancestral samples with the **pure canonical Stage-1 EMA only**, under `no_grad()`.
The exact unrespaced source `p_sample` is called at input timesteps **999..51**:
949 updates, leaving **x50, the input to the next call at t=50**. This explicit
index convention avoids calling the state after t=50 “x50” (that state is x49).
There is no forward noising of real cells and no gradient through these updates.

Default cache size is 8192 cells, float32 `.npy` memmap, streamed in batches of 50.
`cache_metadata.json` records actual schedule/betas, state semantics, Stage-1
checkpoint path/SHA, neural-state hash, genes/gene-order hash, seed, batch size,
count, dtype via NPY, device, commit and array SHA. Finite values are checked during
generation and reload. Canonical pointers `canonical_x50_train.json` and
`canonical_x50_evaluation.json` reference exclusively created directories under
`runs/<campaign>/x50/`. A partial cache is never reused or overwritten. A valid
existing cache is verified and reused; incompatible settings fail explicitly.
All three training families must use the same canonical training-cache path.
Evaluation has a separate cache of 2048 generated states and a different seed.

### Differentiable ODE integration and minibatch occupation OT

Defaults live in `configs/trajectory_defaults.json`, merged **only** into the new
trajectory training conditions. Starting with B=128 independently selected cached
states, compute K=100 explicit Euler updates:

```
z[:, 0] = generated_x50
z[:, k+1] = z[:, k] + trajectory_ot.ode_dt * ode_model(z[:, k], t_cond=0)
trajectory_ot.ode_dt = 0.001
states.shape = [128, 101, G]
```

Only the ODE branch is called. States remain in the differentiable graph; none
are detached or written per update. Non-reentrant activation checkpointing
recomputes blocks of 10 ODE updates during backward (`checkpoint_block=0` disables
it). The complete path influences gradients at selected later times. CellUNet
receives no gradients and its hash remains unchanged, including in EMA.

The conceptual discrete occupation measure has 128×101=12,928 points. Dense OT
never receives that flattened pool. Each path contributes one uniformly sampled
time index from each bin `[0,25)`, `[25,50)`, `[50,75)`, `[75,101)`, giving exactly
**128×4=512 equally weighted OT points**. Choices are resampled every optimizer
step. Boundaries are `floor(i*(K+1)/bins)`, with the final boundary K+1.
`temporal_bins`, `samples_per_trajectory`, `temporal_sampling="stratified"` and
`binning_rule="floor_boundaries_last_inclusive"` are explicit. This version requires
one sample per bin. The final bin has 26 states rather than 25: equal bin masses
therefore approximate uniform discrete time with a small discretization difference.
This is minibatch occupation matching, not an exact unbiased estimator of the
full dense Sinkhorn problem. 512 sampled points do not mean only 512 states exist,
and 12,928 correlated states are not 12,928 independent trajectories/cells.

A separate real-target sampler independently selects 512 cells from full empirical
`AnnData.X`, retaining the existing gene order and float32 loader representation,
without normalize/log1p/scale, VAE, filtering or target reweighting. It samples
without replacement within each minibatch when N>=512, otherwise with replacement
(recorded in checkpoint metadata). Cells may recur across optimizer steps.
Sources are generated, so targets are never permutations of the source batch.
Source, real-target and time streams use separate seeds 1236/1237/1238 and the
optimizer step via `SeedSequence`; source cache seed is 1235. Seeds are configurable
in JSON and saved in effective config. No cluster balancing, inverse-density
weights, coverage reweighting or unbalanced OT is used.

```
L = S_epsilon(stratified_occupation_512, independent_real_512)
    + ode_reg_lambda * ode_model.off_mask_penalty('l1')
```

There is no epsilon-MSE, no one-step pred_x0 objective, and no automatic LR,
epsilon, soft-coefficient or loss-balance tuning. Fresh-training CLI overrides:
`--trajectory-batch-size`, `--trajectory-ode-steps`, `--trajectory-dt`,
`--trajectory-samples-per-path` (also sets the bin count), `--real-ot-points`,
`--trajectory-checkpoint-block`, `--gradient-diagnostic-interval`.
Resume retains stored scientific settings and permits only an increased solver cap.

### Balanced Sinkhorn and diagnostics

The solver uses mean squared gene distance, equal balanced marginals, debiasing,
float64 log-domain iterations and full autodifferentiation. New trajectory OT
uses geometric epsilon scaling **1.6 → 0.8 → 0.4 → 0.2 → 0.1**. Configurable start,
factor and target must form a finite schedule ending exactly at target epsilon.
Log scalings are rescaled by epsilon_old/epsilon_new to preserve physical dual
potentials for the warm start; they are not detached. Cross terms use alternating
log updates. With scaling enabled, symmetric self terms use the averaged symmetric
log fixed-point update (u=v), avoiding an oscillating self update. Legacy direct
solver behavior is retained when scaling is disabled. Each scale stops early
with the original marginal tolerance 1e-5; 2000 iterations **per scale** is a hard
cap. No unconverged scale/plan is accepted and no failing batch is skipped.
Ten-iteration non-reentrant checkpointing limits solver tape memory.

`losses.csv` stores step, primary/trajectory_ot, weighted soft, total, all three
Sinkhorn iteration counts/residuals, scaling flag, trajectory-state mean/max norm
and mean start-to-end displacement. At step 1 and every
`gradient_diagnostic_interval=100`, it also stores:

- OT and weighted-soft gradient norms and `grad_ratio_ot_to_soft`, measured with
  `autograd.grad(..., retain_graph=True)` without altering `.grad`;
- trainable ODE parameter norm and actual optimizer-update norm;
- cross-cost mean/std/median/q90/q99/max/CV, median/epsilon and max/epsilon.

Non-diagnostic rows leave these expensive fields empty. `sinkhorn_scales.csv`
records one row per logged step/term/epsilon with iteration count and residual,
not a large object inside a CSV cell. `soft` already includes the original outer
coefficient and the original internal off-mask factor. These diagnostics inform
future choices; this implementation does not retune either coefficient.

### Evaluation: occupation versus endpoint and additional Sliced Wasserstein

`occupation.py` independently evaluates **every Stage-2 ODE family**, including
reconstruction checkpoints, from the same evaluation x50 cache. It runs under
`no_grad()` and does not change the normal sampler. Default evaluation uses 512
trajectories × 101 states conceptually, four stratified points per path = 2048
occupation candidates, an independent evaluation seed 4321, and source seed 4322.
For trajectory-OT models it inherits the trained ODE steps/dt and binning; for
recon it uses the explicit defaults. Integration is streamed in batches of 32;
only selected occupation points and endpoints are retained in CPU memory.

`occupation_analysis.csv` contains separate `distribution=occupation` and
`distribution=endpoint` rows. Both reference real Erythropoietic cells, matching
existing evaluation policy; the training target remains the full empirical data.
Fields include condition/checkpoint, source cache, path count, ODE steps/dt,
conceptual total/stratified/used counts, real counts, Sinkhorn divergence,
SW2 and SW2 squared, mean/median diversity and seed, plus separate train/evaluation
sample settings. SW/diversity use up to 2048 equally sized samples; dense evaluation
Sinkhorn uses at most 512, explicitly recorded as `sinkhorn_points_used`.
Endpoint sample count is at most the number of trajectories. IDs/time selections
are saved in NPZ; effective evaluation and unchanged training config are stored
separately in `occupation_metadata.json`. No per-step training trajectories are
saved. Occupation Sinkhorn per-scale diagnostics have their own CSV.

Sliced Wasserstein is **evaluation-only**: 256 seeded Gaussian gene-space
projections normalized to unit L2, sorted projected equal-size samples, mean of
squared 1D W2 distances. `sliced_wasserstein2_squared` stores this mean and
`sliced_wasserstein2` its square root. No extra division by gene count is applied.
The NumPy helper is deterministic and permutation invariant, including when
subsampling (canonical lexicographic order before seeded selection). Smaller sets
use the largest equal-size subsample; there is no upsampling of evaluation data.
`--sw-projections`, `--sw-points`, `--eval-seed`, `--eval-trajectories` customize it.

The existing `analyze.py` additionally saves `sliced_wasserstein_snapshots.csv`
for actual states at each reverse/post-ODE snapshot, including Stage 1. Existing
Sinkhorn, diversity, correlations/MSE/cosine, norms, UMAP and post-ODE convergence
remain available. Snapshot SW uses at most the existing selected real reference
(2000 default), even though the SW point limit is 2048. `plot_occupation.py` renders
occupation/endpoint Sinkhorn and SW vs condition and recorded OT/soft gradient
norm/ratio curves. Plotting reads saved numbers only.

The normal inference experiment remains **1000 original Hybrid reverse updates
then +100 ODE-only updates**, `post_ode_dt=0.001`, with the original interpolation
`w_ode=1-t/999`. These `post_ode_*` settings and saved inference states are separate
from `trajectory_ot.ode_steps/ode_dt` and the independent occupation evaluation.
Neither the ODE training time nor its step index is a diffusion timestep.

Standalone evaluation and later plotting (replace emitted paths):

```bash
python work/20260913_2step/scripts/cache_x50.py --campaign two_step_001 --role evaluation --device cuda
python work/20260913_2step/scripts/occupation.py --checkpoint EMA_CHECKPOINT --source-cache EVALUATION_X50_CACHE --device cuda
python work/20260913_2step/scripts/plot_occupation.py --input OCCUPATION_DIR_1 OCCUPATION_DIR_2
```

The last command creates figures; it was not executed during implementation.

### References

These references motivate the loss family and evaluation; they do not prove the
biological hypothesis or independence of time-correlated samples in this experiment.

- [Genevay et al. (2018), Learning Generative Models with Sinkhorn Divergences](https://proceedings.mlr.press/v84/genevay18a.html)
- [Genevay et al. (2019), Sample Complexity of Sinkhorn Divergences](https://proceedings.mlr.press/v89/genevay19a.html)
- [Fatras et al., Minibatch optimal transport distances; analysis and applications](https://arxiv.org/abs/2101.01792)
- [Nadjahi et al. (2019), Asymptotic Guarantees for Learning Generative Models with the Sliced-Wasserstein Distance](https://proceedings.neurips.cc/paper_files/paper/2019/hash/c9e1074f5b3f9fc8ea15d152add07294-Abstract.html)
- [Lezama et al. (2021), Run-Sort-ReRun](https://proceedings.mlr.press/v139/lezama21a.html)
- [Tran et al. (2026), Minimax-Optimal Two-Sample Test with Sliced Wasserstein](https://proceedings.mlr.press/v300/tran26c.html)
- [POT user guide](https://pythonot.github.io/user_guide.html)
- [GeomLoss epsilon-scaling example](https://www.kernel-operations.io/geomloss/_auto_examples/sinkhorn_multiscale/plot_epsilon_scaling.html)

### Legacy one-step OT Stage 2 (noncanonical)

Only the legacy `*_ot_soft` configs use uniform `t in {0,...,49}`. Its dedicated sampler does not change the
diffusion schedule or interpolation. With real x0 and known Gaussian noise:

```
x_t = diffusion.q_sample(x0, t, noise)
eps = model(x_t, t)
pred_x0 = diffusion._predict_xstart_from_eps(x_t, t, eps)
L = S_epsilon(pred_x0, x0) + ode_reg_lambda * off_mask_penalty('l1')
```

There is **no paired epsilon-MSE** in the OT optimization target and no
backpropagation through post-ODE continuation. The two batches are treated as
sets with equal weights. Permuting either batch preserves the objective.

`losses/sinkhorn.py` implements differentiable log-domain Sinkhorn locally;
GeomLoss was not installed and no OT dependency was added. Definitions:

```
C_ij = sum_g (x_ig - y_jg)^2 / G
OT_epsilon(a,b) = min_P <P,C> + epsilon KL(P || a b^T)
S_epsilon(x,y) = OT(x,y) - 0.5 OT(x,x) - 0.5 OT(y,y)
```

The implementation uses uniform marginals and the generalized KL expression
including `-sum(P)+1` (zero at exact unit mass). Costs use a matrix product,
not a batch-by-batch-by-gene tensor. Cost calculation and iterations use float64;
all executed Sinkhorn iterations are differentiated, including self terms.
Ten-iteration blocks use non-reentrant PyTorch activation checkpointing during
backward to limit the saved intermediate tensors. This recomputes the same
iterations; it does not detach the transport plan or approximate its gradient.

| Numerical setting | Default |
| --- | --- |
| Entropic epsilon, in mean-squared-gene-distance units | `0.1` |
| Maximum iterations per cross/self solve | `2000` (early stop every 10) |
| Absolute maximum row/column marginal residual tolerance | `1e-5` |
| Convergence inspection | every 10 iterations and at the final iteration |
| Debiasing | true |
| Compute dtype | float64 |
| Training batch | 128, equal weights |
| Evaluation equal-size subsample | up to 128 cells per distribution |

Every OT training row logs cross/self iteration counts and residuals, as well as
primary, soft and total losses. Solver nonconvergence or nonfinite values abort
with an explicit error; the code does not silently replace the objective or
return an unconverged plan. Epsilon/iteration/tolerance CLI overrides are explicit
and enter the effective config/checkpoint. For comparisons, keep the evaluation
cost and epsilon consistent across conditions (defaults do so).

## Sampling and the 22 snapshots

For each Stage-2 model, `sampling/trajectory.py` calls unchanged `p_sample`
exactly 1000 times. After terminal t=0, it uses only the learned ODE branch:

```
f_k = ode_model(x^(k-1), conditioning_t=0)
x^k = x^(k-1) + post_ode_dt * f_k
```

The default integrator is explicit Euler, `post_ode_dt=0.001`, 100 updates,
for total integration time 0.1. This small initial step is a conservative
experimental choice, **not** a stability guarantee or one diffusion timestep.
No stochastic diffusion noise, neural branch, epsilon-to-x0 conversion, or
negative diffusion time is used in continuation. State values are not projected
or newly clipped. Every update checks finite values and per-cell state L2;
`max_state_norm=1e6` is a configurable failure threshold, not a clipping bound.

| Completed update | phase | diffusion_t | ode_step | integration_time |
| --- | --- | --- | --- | --- |
| 50,100,...,950 | diffusion | 950,900,...,50 | null | null |
| 1000 | diffusion | 0 | null | null |
| 1050 | post_ode | null | 50 | 0.05 at default dt |
| 1100 | post_ode | null | 100 | 0.10 at default dt |

`ode_conditioning_t=0` is a separate field for post-ODE rows. Stage 2 has 22
snapshots; Stage 1 has only the 20 diffusion snapshots and stops at update 1000.

Storage uses streamed batches and `.npy` memmaps; no growing tensor trajectory
lists. Default sample count is the source 3,000 and sample batch is 50.

- `sample_state.npy`: `[cells, 22 or 20, genes]`, state **after** each saved update.
- `pred_xstart.npy`, `epsilon.npy`, `cellunet_raw.npy`: `[cells,20,genes]`, evaluated
  at the **input** to each saved diffusion update. Stage 2 also saves `ode_raw.npy`.
- `post_ode_field.npy`: Stage 2 only, `[cells,2,genes]`, field at input to ODE
  updates 50 and 100. There is no fabricated post-ODE clean prediction.
- `post_displacement.npy`, `post_field_norm.npy`, `post_state_norm.npy`:
  Stage 2 only, `[cells,100]`, float64 per-cell quantities for every ODE update.
- `post_ode_convergence.csv`: global mean/median displacement, mean/max field
  norm, max state norm and phase/physical-time information for each k=1..100.
- `sampling_metadata.json`, `snapshot_metadata.csv`, and completion/failure
  markers describe semantics and progress. Analysis refuses incomplete sampling.

Per-cell convergence values allow a true median across all cells, rather than
an average of batch medians. All sampling is under `no_grad()` and `.eval()`.

## Numerical analysis and separate rendering

`analyze.py` writes numbers, `embed.py` writes UMAP coordinates, and `plot.py`
consumes only those saved results. Rendering never re-evaluates a model or fits
UMAP. Every invocation makes a new directory and refuses overwrite.

Only real `Superclass == Erythropoietic` cells are used as reference, through the
20260911/20260830 selection helper. Gene order must exactly match checkpoint
metadata. The already-preprocessed representation is preserved. No generated
cell-type classifier or inferred biological labels are added.

### Forward-noise and branch diagnostics

`true_noise_metrics.csv` contains per-cell Pearson correlation across genes,
mean squared error across genes, and cosine similarity to known forward Gaussian
noise for Hybrid, frozen CellUNet and raw ODE. Stage 1 has CellUNet only.
`cellunet_vs_ode_metrics.csv` compares the raw two branches on identical x_t.
Metrics are computed per cell, then summarized by mean and population standard
deviation. Pearson and L2 call source helpers; cosine uses source PyTorch
`cosine_similarity` defaults. Constant Pearson inputs follow the source zero
convention with the source denominator clamp.

A fixed float64 CPU noise stream is regenerated at each timestep, shared across
conditions for the same cell IDs, seed and batch size. Up to 2,000 selected real
cells are evaluated in batches of 128. Reference cell IDs/names are saved.

The evaluated grid is the union of `0,20,...,980,999` and every `0..50`.
Full-range plots use the former; low-noise plots use every value in the latter.
`diffusion_schedule.csv` records alpha_bar, sigma=`sqrt(1-alpha_bar)`, SNR and
range flags. Shading distinguishes t=0..10 (approximately sigma <=0.05 for this
schedule) and the legacy OT t=0..49 region. **The entire 50-step OT region is not claimed
to have sigma <0.05.** No true-noise diagnostic is defined for post-ODE updates.

`norm_metrics.csv` includes true noise, Hybrid, raw CellUNet, raw ODE, weighted
CellUNet and weighted ODE L2 norms. It includes each raw output norm divided by
true-noise norm, plus ODE/CellUNet. Ratios use `1e-12` solely to protect division.
Plotting produces full and dense low-noise views, with separate raw/weighted
norm plots and ratio plots.

### Original-gene-space trajectory metrics

All saved **states after updates** are evaluated, including every post-ODE
snapshot and the pure Stage-1 baseline:

- `trajectory_diversity.csv`: exact mean Euclidean distance over distinct
  unordered cell pairs, computed blockwise (128-by-128 distance blocks); exact
  mean squared pairwise distance using `2*sum_i ||x_i-mean(x)||²/(N-1)`;
  median and 5%/95% quantiles from 100,000 deterministic uniform distinct pairs
  sampled with replacement. Counts, seed, phase and estimator definitions are
  saved. This avoids a giant N-by-N-by-G tensor; the exact mean still costs
  O(N²G) time.
- `sinkhorn_to_real.csv`: same debiased normalized-cost solver as training, for
  reconstruction, OT and baseline conditions. Equal-size deterministic sampling
  uses at most 128 cells from generated cells and the selected real reference.
  The same sampled indices are reused across snapshots. Saved fields include
  epsilon, tolerance, iterations limit, marginal residual, seed, cost,
  subsample size, representation and phase. Subsample IDs are saved in NPZ.
- `post_ode_convergence.csv`: copied from sampling for rendering without model
  evaluation; mean/median displacement and field-norm plots show continuation.

Diversity and Sinkhorn plots mark diffusion, terminal t=0, and the distinct
post-ODE region. No covariance effective-rank metric or plot is implemented.

### UMAPs

UMAPs follow 20260911's **clean prediction** convention during diffusion and
use the **actual state** during post-ODE continuation, where no new diffusion
clean prediction exists. This distinction is explicit in metadata. Diversity
and distribution-distance curves consistently use actual after-update states.

Each of the 22 (or baseline 20) snapshots is a fresh independent fit of real
Erythropoietic cells plus that snapshot alone. These coordinate systems are not
aligned trajectories. The original helper uses PCA up to 50 components,
neighbors=15 with up to 40 PCs, ARPACK PCA and UMAP seed 1234, without new
normalization/log1p/scaling. All real selected cells enter UMAP.

Each Stage-2 trajectory also gets exactly one joint fit of real cells plus
steps 950,1000,1050,1100. Real cells are grey; generated labels are `Diffusion
950`, `Diffusion 1000`, `ODE +50`, `ODE +100` with a sequential viridis palette.
All four generated groups share the same fit. No baseline joint/post-ODE
extension is fabricated. `embed.py` saves each fit's coordinates, trajectory IDs,
labels and parameters before `plot.py` creates any figures.

## Output layout and provenance

```
configs/                         source selections + explicit numerical defaults
models/                          source factory / frozen neural mode adapter
training/                        objectives, self-contained checkpoints, loop
losses/sinkhorn.py                local differentiable OT
sampling/trajectory.py            streamed reverse / continuation arrays
analysis/                        diagnostics, distributions, UMAP, rendering
scripts/                         run_all launcher + train/sample/analyze/embed/plot CLIs
launches/<campaign>/             launch manifest, PID, combined/per-stage logs
tests/test_suite.py               small synthetic CPU tests, no figure files
runs/<campaign>/canonical_stage1.json
runs/<campaign>/<condition>/<UTC-run-id>/
    effective_config.json, started.json, losses.csv
    checkpoints/{model...,ema...,opt...}.pt, checkpoint JSON sidecars
    completed.json or failed.json
results/<condition>/<UTC-run-id>/
    sampling arrays, metadata, convergence CSV, completed.json or failed.json
    analyze/<UTC-run-id>/         numeric metrics, reference IDs, provenance
    embed/<UTC-run-id>/           UMAP coordinates and fit metadata
    {analyze,embed}/<run-id>/figures/<UTC-run-id>/
```

Filesystem outputs are confined to the suite. New directories use
`exist_ok=False`; JSON/CSV/checkpoints and figures use exclusive creation.
Run IDs combine UTC timestamp and random suffix. No automatic overwrite,
writing into a partial directory, merging to main, or source-run manifest update
is performed. Preserve the git commit recorded in a run to reproduce its source
imports. Source-config hashes are recorded; Stage-1/Stage-2 gene hashes and edge
TSV hashes are checked across the campaign. The dataset itself is not fully
hashed (large files); copying an h5ad with the same genes but changed expression
values cannot be detected by a gene-order hash alone.

## 全工程をバックグラウンドで実行する（推奨）

remoteのリポジトリ直下で次のブロックを実行する。
`run_all.sh`が既存のconda環境`scdiffusion`を使い、バックグラウンド起動する。
`nohup`や末尾の`&`は不要。起動後はSSHを切断しても処理が継続する。

```bash
git fetch origin && git switch feat/20260914-trajectory-occupation-ot && git merge --ff-only origin/feat/20260914-trajectory-occupation-ot
bash work/20260913_2step/scripts/run_all.sh
```

既定の入力は元スイートの設定を引き継ぐ：

- data: `/home/suzuki/Projects/scDiffusion/work/20260215_embryonic/data/Embryonic.h5ad`
- edges: `/home/suzuki/Projects/scDiffusion/external_data/tf_target_edges.tsv`
- device: `cuda`、sampling: 3,000 cells / batch 50、post-ODE dt: 0.001。

入力の場所が異なる場合は起動コマンドに指定する：

```bash
bash work/20260913_2step/scripts/run_all.sh --data /path/to/Embryonic.h5ad --edge-tsv /path/to/tf_target_edges.tsv
```

実行順は **Stage 1を1回 → Stage 2を6条件 → 各7条件のsampling、
数値解析、UMAP座標計算、数値指標の図、UMAPの図**に加え、共有x50キャッシュ2種と
Stage 2全6条件のoccupation評価・比較図を実行する。全51コマンドを逐次実行する。
後続処理には各CLIが出力したcheckpoint/outputのパスを渡し、古いrunを自動選択しない。
Stage 1にはpost-ODE更新は作らない。この全工程コマンドは実際に図を生成する。

campaign名はUTC日時とランダムsuffixから自動作成する。`--campaign NAME`でも指定可能。
通常起動は既存campaignや同名launchへの上書き・自動resumeを行わない。
中断後は下記の`--resume-campaign`を明示する。
起動時にPID、campaign、launchディレクトリと、そのまま使える`tail -f`コマンドを表示する。

- 統合ログ：`launches/<campaign>/nohup.log`
- 各工程ログ：同ディレクトリの`<condition>.<stage>.log`
- 実行内容とcommit：`launch.json`、プロセスID：`pid`
- 全完了：`completed.json`、失敗：`failed.json`（失敗工程とエラーを記録）

途中のコマンドが失敗したら、その時点で停止する。以降の学習・解析は実行しない。
入力ファイルの存在は起動前、CUDAの利用可否はworker開始時に確認する。
長時間の処理中は同じcheckoutのコードを変更しないこと。

実験を実行せずコマンドだけ確認するには：

```bash
bash work/20260913_2step/scripts/run_all.sh --dry-run
```

`--dry-run`はデータを開かず、ディレクトリも作らない。
`--foreground`を付けるとバックグラウンド化せず終了まで待機する（ログ保存先は同じ）。
conda環境名を変える場合は`TWOSTEP_CONDA_ENV=別の環境名 bash .../run_all.sh`。
既に適切なPython環境を有効化している場合は`python -B .../scripts/run_all.py`で直接起動できる。
このlauncher自体の追加時にも、本学習・sampling・解析・図生成は実行していない。

## OT学習を後回しにし、完了済み4条件だけ解析する

今回のcampaignではStage 1とrecon 3条件の学習が完了している。
以下は**学習を一切起動せず**、その4条件の最終EMAを検証し、
サンプリング → 数値解析 → UMAP座標 → 指標・UMAPの図生成を
バックグラウンドで順次実行する。元のlauncherは全学習の後にsamplingを行うため、
提示された停止時点では解析用の軌跡をまだ作っておらず、まずsamplingから始める。

```bash
git fetch origin && git switch feat/20260914-trajectory-occupation-ot && git merge --ff-only origin/feat/20260914-trajectory-occupation-ot &&
bash work/20260913_2step/scripts/run_all.sh --analyze-campaign two_step_20260913_070659_2e7ec730
```

- 対象は`stage1_cellunet`と`hill_after_linear_recon_soft`、
  `centered_signed_hill_recon_soft`、`shifted_hill_rho_recon_soft`のみ。
- 4条件 × 5工程 = 20コマンド。OTモデルの学習・再開・samplingは行わない。
  必要な完了checkpointがない場合は停止し、学習を自動で補わない。
- 各reconは1000 diffusion + 100 post-ODE更新、Stage 1は1000 diffusionのみ。
- 解析項目は従来どおり。実データへのSinkhorn距離は**評価指標**として残す
  （OTモデルの学習とは別）。評価の反復上限は2000。
- 保存済みrun/checkpoint/失敗ログは変更しない。新しいsampling/analysis出力と
  `launches/<campaign>/analysis_<UTC-id>/`のログを作る。
- 起動時にPIDと`tail -f`コマンドを表示する。`nohup`や`&`は不要。
  `--dry-run`を追加するとcheckpointを検証し、実行計画だけを表示する。
- `--resume-campaign`とは併用不可。後日OT学習を再開したい場合は、下記の
  recoveryコマンドを別途実行する（新trajectory OTを開始し、旧OTは引き継がない）。

## 既存campaignの途中から、新しいtrajectory OTと最後の解析まで実行

remoteのリポジトリ直下で実行する。実装作業中にはこのジョブを起動していない。

```bash
git fetch origin &&
git switch feat/20260914-trajectory-occupation-ot &&
git merge --ff-only origin/feat/20260914-trajectory-occupation-ot &&
bash work/20260913_2step/scripts/run_all.sh --resume-campaign two_step_20260913_070659_2e7ec730
```

`run_all.sh`がconda環境`scdiffusion`で自動的にバックグラウンド起動する。
`nohup`や`&`は不要。PIDと、そのまま使える`tail -f`コマンドを表示する。
完了済みStage 1・recon 3条件を再利用し、以下を順に実行する。

1. canonical Stage-1最終EMAを検証し、学習用x50キャッシュを作成／検証して再利用。
2. **新しいtrajectory OT 3条件**を、それぞれ新しいODE初期値から学習。
3. 別seedの評価用x50キャッシュを作成／再利用。
4. 全7条件の通常sampling → 数値解析（snapshot SW追加）→ UMAP → 図生成。
5. Stage 2全6条件の独立x50 occupation/endpoint評価と比較図・勾配ノルム曲線。

提示された状態では51コマンドから完了済み学習4つを省いた47コマンドになる。
**旧`hill_after_linear_ot_soft`のstep 5000は使用しない。** 目的関数が異なるため、
それを新OTの途中状態として引き継ぐことはできない。旧run・checkpoint・結果を保存したまま、
新しいcondition/runディレクトリに書き込む。

同じコマンドを後日再実行すると、新trajectory OTの完了学習も再利用する。
未完了の**同じobjective/schema・condition・campaign・x50キャッシュ**の
raw + EMA + optimizer bundleだけは再開できる。旧one-step OTを明示的に渡すと
incompatibility errorになる。新OTのsource/target/time抽出はそれぞれ独立した
`(seed, optimizer step)`から決定する。共通checkpoint形式には全Torch/CUDA RNG状態は
含めず、中断なし実行とのbitwise一致は主張しない。

再開ログは`launches/<campaign>/recovery_<id>/`に保存する。
`recovery_selection.json`と`execution_plan.json`が、再利用・初期化・再開対象を記録する。
後処理は新しい出力先で実行し、過去のsampling/解析結果の自動再利用はしない。
未完了状態の既存run/launcherが残っている場合は重複実行を避けて停止する。
複数の完了trialから曖昧な自動選択もしない。

`--source-cache-size 16384`などでキャッシュ数を初回作成時に変更できる。
`--eval-cache-size`は既定2048。再利用時には作成時と同じsize・batch-size・seedが必要で、
不一致なら上書きせず停止する。既存の解析ジョブが動作中のcheckoutは変更しないこと。

実行内容だけ確認するには、同じコマンドに`--dry-run`を追加する。
既存checkpointを読み取り検証するが、h5ad・GPU・worker・出力ディレクトリには触れない。

旧one-step OTは個別CLIでのみ引き続き利用できる。例えば旧raw checkpointの再開は
`train.py --condition hill_after_linear_ot_soft --resume-checkpoint OLD_RAW ...`で行う。
canonical launcherは旧OTを選ばない。旧解析のSinkhorn上限を変更するときは
`analyze.py --trajectory PATH --ot-max-iterations 2000`を明示でき、評価metadataに記録する。

## Commands for later remote execution — not run during implementation

Run from repository root after activating the existing `scdiffusion`
environment. Choose a **new** campaign name. Replace input paths with existing
preprocessed data and TF-target TSV paths on the remote machine.

```bash
python work/20260913_2step/scripts/train.py \
  --campaign two_step_001 --condition stage1_cellunet \
  --data /path/to/Embryonic.h5ad --edge-tsv /path/to/tf_target_edges.tsv \
  --device cuda
```

After Stage 1 completes, run the reconstruction conditions (sequentially or in
separate GPU allocations). They automatically load the campaign's final EMA:

```bash
for family in hill_after_linear centered_signed_hill shifted_hill_rho; do
  python work/20260913_2step/scripts/train.py \
    --campaign two_step_001 --condition "${family}_recon_soft" --device cuda
done
```

Create the shared generated source cache first (same canonical EMA):

```bash
python work/20260913_2step/scripts/cache_x50.py --campaign two_step_001 --device cuda
```

The command prints `X50_CACHE`. Repeating it verifies and reuses the immutable
canonical cache. Training resolves that same cache automatically. One condition:

```bash
python work/20260913_2step/scripts/train.py --campaign two_step_001 \
  --condition hill_after_linear_trajectory_ot_soft --device cuda
```

All three trajectory OT conditions:

```bash
for family in hill_after_linear centered_signed_hill shifted_hill_rho; do
  python work/20260913_2step/scripts/train.py \
    --campaign two_step_001 --condition "${family}_trajectory_ot_soft" --device cuda \
    --ot-epsilon 0.1 --ot-max-iterations 2000 --ot-tolerance 1e-5
done
```

Sampling: use the desired completed run's EMA path printed by training.
Repeat for all six Hybrids and optionally the Stage-1 baseline. The latter
stops automatically at update 1000.

```bash
python work/20260913_2step/scripts/sample.py \
  --checkpoint work/20260913_2step/runs/two_step_001/CONDITION/RUN_ID/checkpoints/ema_0.9999_030000.pt \
  --device cuda --num-samples 3000 --batch-size 50 --post-ode-dt 0.001
```

The sampling command prints `TRAJECTORY_DIR`. Numerical metrics and UMAP fits
are separate operations; neither command renders figures:

```bash
python work/20260913_2step/scripts/analyze.py \
  --trajectory work/20260913_2step/results/CONDITION/SAMPLING_RUN_ID --device cuda
python work/20260913_2step/scripts/embed.py \
  --trajectory work/20260913_2step/results/CONDITION/SAMPLING_RUN_ID
```

An optional `--data /relocated/identical.h5ad` handles relocated files while
checking exact gene order. Keep expression values identical. Both commands
print `ANALYSIS_DIR`. Later, explicitly request rendering from each saved run:

```bash
python work/20260913_2step/scripts/plot.py \
  --input work/20260913_2step/results/CONDITION/SAMPLING_RUN_ID/analyze/ANALYSIS_RUN_ID
python work/20260913_2step/scripts/plot.py \
  --input work/20260913_2step/results/CONDITION/SAMPLING_RUN_ID/embed/EMBED_RUN_ID
```

The final two commands create PNG figures in new figure-run directories. They
were implemented but **not executed** for this task.

Lightweight validation, without real data, GPU, experiment trajectories or plots:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s work/20260913_2step/tests -p 'test_*.py'
python -B work/20260913_2step/scripts/train.py --help
```

## Numerical/design limits to evaluate remotely

Epsilon=0.1 and Euler dt=0.001 are documented starting values, not tuned results.
No real-data memory, runtime, convergence or manifold-coverage result was measured.
100 unrolled ODE steps, three 512×512 Sinkhorn terms over five epsilon levels,
and periodic gradient diagnostics can be substantially more expensive than the
old one-step loss. Checkpointing trades extra backward compute for lower activation
memory; it does not guarantee the defaults fit a particular GPU. Two caches each
require ~949 neural reverse updates per batch. Dense OT remains bounded, but
existing exact diversity and UMAP evaluations can also be expensive.
OT magnitude and its gradient can be small near t=0 relative to the inherited
soft constraint. Monitor primary/soft loss separately. High-dimensional,
poorly conditioned or low-entropy batches can exceed the 2000-iteration cap;
the default fails loudly and requires an explicit new run/config decision.
Backward recomputes ten-iteration blocks to reduce graph memory at the cost of
additional computation. The iteration cap remains finite; solver truncation can cause tiny negative divergence estimates; values are not clipped.

The +ODE continuation uses the **positive raw learned branch** as a vector field
because that is the requested experiment. Epsilon training does not itself
prove this sign, integration time scale, stability, or biological correctness.
Only finite/divergence checks are enforced; do not infer convergence or quality
from a successful finite run. Compare displacement, diversity and distribution
distance, and inspect sensitivity to dt in separately saved sampling runs.
Independent UMAP positions are not aligned and UMAP proximity alone cannot prove
absence of collapse. The exact mean pairwise distance and all UMAP fits can be
expensive; those are explicit future analysis jobs, not implementation checks.

## 2026-09-15: centered/shifted trajectory training runtime fix

The original direct-Hill `component_outputs` computes full `A=softplus(raw_A)`
and `theta=softplus(raw_theta)` **inside each 16-target chunk**. At G=1024 that
repeats both full K×G×G transforms 64 times per ODE call, over 100 ODE calls per
trajectory, with additional backward recomputation. This is unnecessary: the
parameters are constant until the optimizer update after the whole unroll.

`training/direct_hill.py` shares physical A/theta/decay tensors within a single
trajectory graph. It still calls the exact source chunk response, regulator
guards, gates/time embedding/dropout and regularization hooks. It retains
chunk and trajectory checkpointing. It rebuilds the shared tensors on the next
unroll; it never detaches or caches across optimizer updates. Hill-after-linear
uses its unchanged source implementation. Normal Hybrid sampling and independent
occupation evaluation keep their source implementation.

`trajectory_runtime.field_backend=shared_parameters_v1` is now the training
runtime default, including compatible checkpoint recovery. The objective,
128 paths, 100 Euler steps, dt=.001, 512-point OT, soft coefficients, optimizer
and 30,000 updates are unchanged. CPU synthetic tests verify identical states
and loss and matching gradients within floating-point tolerance, not bitwise
identical optimization histories. Real RTX 6000 Ada acceleration/peak memory
has **not** been measured; no particular speedup or finish time is promised.

Trajectory training now prints `seconds_per_step` and `eta_hours` every 50
updates and saves raw/EMA/optimizer every 1000 updates. Runtime settings are
recorded separately and can change on resume. More frequent saves use additional
disk space. Individual train CLI options are `--trajectory-field-backend source`
(for comparison), `--trajectory-log-interval N`, `--trajectory-save-interval N`.
Other conditions retain their original logging/save intervals.

A running Python process does not acquire this fix by fetching code. Stop the
old training with **SIGINT** (`kill -INT TRAIN_PID`, after checking it is still
that training process), wait for the launcher to record failure, then fetch and
use the usual `--resume-campaign` command. The old runner catches SIGINT and
records failure, but **does not save an on-demand checkpoint**. If centered has
not reached the old 5000-update save boundary, only that incomplete condition
must restart. Completed hill-after-linear, Stage1/recon and the x50 cache remain
reusable. Do not use SIGKILL; it leaves no terminal-status marker.

## 重いOT学習を止め、完了済み5条件を先に解析する

`--analyze-completed-campaign NAME`はcanonical条件の完了済み最終EMAだけを
検証して選択する。今回の状態ならStage1、recon 3条件、hill-after-linear
trajectory OTの計5条件。未完了centered/shifted trajectory OTはスキップし、
学習の開始・再開・中間checkpointの読み込みは行わない。
既存の`--analyze-campaign`（Stage1/reconの4条件限定）とは別のオプション。

remoteの現在のcentered学習をSIGINTで停止し、プロセス終了を待ってから実行する。
PIDは`ps`で対象を確認すること。学習子プロセスの異常終了を受けて旧launcherも停止し、
次のshifted学習には進まない。SIGINTは未保存の重みをcheckpointへ保存しない。
完了済みモデル・保存済みcheckpoint・キャッシュは削除しない。

```bash
cd /home/suzuki/Projects/scDiffusion-github &&
ps -p 349558 -o args= | grep -q '[t]rain.py.*--condition centered_signed_hill_trajectory_ot_soft' &&
kill -INT 349558 &&
while kill -0 349558 2>/dev/null; do sleep 1; done &&
sleep 3 &&
git fetch origin &&
git switch feat/20260914-trajectory-occupation-ot &&
git merge --ff-only origin/feat/20260914-trajectory-occupation-ot &&
bash work/20260913_2step/scripts/run_all.sh --analyze-completed-campaign two_step_20260913_070659_2e7ec730
```

すでに学習が停止済みの場合は、停止確認後に`git fetch`以降を実行する。
新launcherはバックグラウンドで動作し、PID・ログ確認コマンドを表示する。
必要なsampling、既存数値指標＋snapshot SW、UMAP、各図を5条件について実行する。
さらに独立の評価用x50キャッシュを作成／再利用し、完了済みStage2の4条件で
occupation/endpoint指標と比較図を作る。現在の選択では合計31工程。
元runや結果には上書きせず、新しいanalysisログと結果ディレクトリを作成する。
解析にもGPU samplingやUMAPなどの時間はかかるが、残り30,000更新の学習は行わない。
`--dry-run`を追加すれば選択結果と計画だけを確認できる。

## 解析中のSinkhorn未収束と、失敗工程からの再開

解析用の距離計算は`analysis/evaluation_ot.py`を通す。最初は指定したsolver設定で
計算し、有限のmarginal residualが未収束なら、epsilon scalingが無効だった場合だけ
有効にして1回再試行する。目標epsilon、許容誤差、各scaleの反復上限、セル選択、
コスト定義は変えない。すでにscaling有効の場合は同じ計算を繰り返さない。

それでも未収束なら**距離は未算出**とし、CSVの数値欄を空欄、
`ot_status=not_converged`、`ot_error`に理由を保存する。未収束の輸送計画から得た
数値を採用したり、ゼロで置き換えたりしない。`ot_attempts`と
`ot_epsilon_scaling`も保存し、`evaluation_ot_status.json`に未算出対象と件数を記録する。
ログと図にも未算出を明示し、他のSW・多様性・UMAP・図生成を続ける。
NaN/Inf、形状不正、OOMなどは引き続きエラー停止する。
**学習用solverの未収束停止は変更しない。** これは評価指標の部分欠損を明示する方針であり、
全てのSinkhorn距離が得られたという意味での完了ではない。

`--resume-analysis-launch PATH`は失敗したanalysis-only launcherの
`execution_plan.json`と工程別completed markerを検証し、完了済み工程を再利用する。
既存のsampling/UMAPを作り直さず、最初の未完了工程から新しいlaunchディレクトリで進む。
checkpoint SHAと出力の存在・完了状態を確認する。失敗した解析工程自体は新しい
解析ディレクトリでやり直し、その工程内の部分CSVは引き継がない。
学習の開始・再開は行わず、動作中や成功済みのlaunchは再開対象にしない。

今回の失敗はhill-after-linear reconの数値解析なので、以下でそこから再開する。
Stage1のsampling・解析・UMAP・図と、reconのsamplingは再利用される。

```bash
cd /home/suzuki/Projects/scDiffusion-github &&
git fetch origin &&
git switch feat/20260914-trajectory-occupation-ot &&
git merge --ff-only origin/feat/20260914-trajectory-occupation-ot &&
bash work/20260913_2step/scripts/run_all.sh --resume-analysis-launch work/20260913_2step/launches/two_step_20260913_070659_2e7ec730/analysis_20260915_055048_eb01ec7e
```

自動バックグラウンド起動。PID・新ログの`tail -f`コマンドを表示する。
`--dry-run`追加時は再利用対象を検証して残りのコマンドだけを表示し、ジョブを起動しない。
再開時のdeviceと残りのsampling/解析引数は元の実行計画を引き継ぐ。


## 完了済みの旧OT 3条件を解析する

`--analyze-legacy-ot-campaign`は `hill_after_linear_ot_soft`、
`centered_signed_hill_ot_soft`、`shifted_hill_rho_ot_soft` の3条件限定。
各条件の完了マーカーと最終EMAのSHA・目的関数・Stage1由来を検証する。
完了済み試行がない、または複数ある場合は停止する。学習やtrajectory OTは起動しない。

```bash
cd /home/suzuki/Projects/scDiffusion-github &&
git fetch origin &&
git switch feat/20260914-trajectory-occupation-ot &&
git merge --ff-only origin/feat/20260914-trajectory-occupation-ot &&
bash work/20260913_2step/scripts/run_all.sh --analyze-legacy-ot-campaign two_step_20260913_070659_2e7ec730
```

自動でバックグラウンド起動し、表示された `LAUNCH_DIR/nohup.log` に進捗を保存する。
3条件それぞれ sampling → analyze → embed → metrics_plot → umap_plot の計15工程。
結果は `results/<旧OT条件>/<実行ID>/`、数値解析はその `analyze/`、
UMAPは `embed/`、図はそれぞれの `figures/` 配下に新規保存する。
既存のrecon解析を再実行しない。評価OTの未収束は再試行後に欠損として記録する。
生成状態の発散など他の異常は停止する。失敗時は表示されたlaunchを
`--resume-analysis-launch <LAUNCH_DIR>` で指定すれば完了工程を再利用できる。


## Hybrid500：既存Stage1を凍結した6条件の新規実験

参照元は `work/20260913_2step`。新しいStage1は学習しない。
`--stage1-campaign` の `canonical_stage1.json` が指す最終EMAをSHA256・重みhash・
完了ステップで検証し、新campaignへ同じ参照情報を保存する。指定元と旧出力は読み取り専用。
今回の参照元は `two_step_20260913_070659_2e7ec730`。
実checkpointのパスとSHAは新campaignの `canonical_stage1.json` と各checkpointの
`originating_stage1` に記録される。リモートの実ファイルはローカル実装時には未検証。

### 条件と補間

`configs/hybrid500_ot.json` は既存条件を参照し、目的関数の設定を継承する。
通常のOTの3条件を先に、trajectory OTの3条件を後に実行する。

| 条件ファミリ | 元の実装 | 目的関数 |
|---|---|---|
| hill_after_linear | 20260803_ODE_hill_exp / standard_hybrid_single__hill_after_linear | ot_soft / trajectory_ot_soft |
| centered_signed_hill | 20260816 / linear_centered_signed_hill | ot_soft / trajectory_ot_soft |
| shifted_hill_rho | 20260816 / linear_shifted_hill_rho | ot_soft / trajectory_ot_soft |

`models/hybrid500.py` が3モデル共通の補間を実装する。
元の拡散時刻（リスケーリングなし）で
`w_ode=max(0,1-t/500)`、`w_cellunet=1-w_ode`。
`t=999,750,500` はODE重み0、`t=250` は0.5、`t=0` は1。
生成forwardは `t>=500` のODE計算を省く。明示的な分岐診断では真のraw ODE出力を
測定するが、重み0の区間では生成結果に混ぜない。raw ODEを未計算のゼロと扱わない。
CellUNetは常時eval・requires_grad=False、optimizerはODE側だけ。

通常OTは `training/objectives.py` の既存定義：実細胞にt=0..49のノイズを付加し、
Hybridから復元したx0の点群と元の実細胞を比較する。batch128、epsilon0.1、
float64、debiased Sinkhorn、tol1e-5、上限2000、既存どおり初回epsilon scalingなし。
trajectory OTは `training/trajectory.py` の既存定義：Stage1生成x50の8192点キャッシュから
128出発点、Euler100段・dt0.001、各軌跡の4区間から1点ずつ選び512点を作り、
独立に選んだ実細胞512点と比較。epsilon scaling 1.6→0.8→0.4→0.2→0.1を維持する。
いずれも `losses/sinkhorn.py` の細胞間・遺伝子平均二乗コストと既存soft制約を使う。
再構成MSEやvelocity/kinetic/manifold等の新しい損失は追加しない。
trajectory OTはODE単独で学習するため、今回の補間変更はその学習損失には直接作用しない。
長時間計算や生成発散が解消されることは保証しない。

### 新規起動と保存先

コードがリモートのcheckoutに反映された後、次を実行する。
この変更自体のpush・mergeや本計算は実装作業中に行わない。

```bash
cd /home/suzuki/Projects/scDiffusion-github &&
bash work/20260913_2step/scripts/run_all.sh \
  --run-six-ot \
  --stage1-campaign two_step_20260913_070659_2e7ec730
```

自動でバックグラウンド起動する。`nohup`・末尾の`&`は不要。
`--foreground` なら終了まで待機する。`--dry-run` は参照元を読み取り検証し、
実行予定を表示するだけでcampaignやworkerを作成しない。
通常起動は毎回新しい `runs/hybrid500_ot_<UTC TIMESTAMP>_<RUNID>/` を作成し、
6条件の子ディレクトリを用意する。`experiment.json` に新実験の設定を保存する。
各学習runの `effective_config.json` とcheckpointには補間式・cutoff・目的関数・
OT設定・seed・git commit・Stage1由来を記録する。

学習からサンプリング3000細胞、数値解析、UMAP、図生成、occupation/endpoint評価まで
順次実行する。学習用と評価用のx50キャッシュは別seed・新campaign内で生成し共有する。
完成したoccupation評価を使った比較図も生成する。再構成条件やStage1学習は含めない。
出力は次のとおり。

- 学習：`runs/<campaign>/<condition>/<runID>/`
- 解析・UMAP・図：`results/<campaign>/<condition>/<samplingID>/`
- occupation：`results/<campaign>/<condition>/occupation/<runID>/`
- 比較図：`results/<campaign>/occupation_comparison/<runID>/`
- 起動ログ：`launches/<campaign>/six_ot_<runID>/`

起動時にPIDと `LAUNCH_DIR`、`tail -f .../nohup.log` コマンドを表示する。
学習進捗は50 stepごとにloss・秒/step・ETAを表示し、1000 stepごとにcheckpointを保存する。
既存のOT・soft・Sinkhorn診断、trajectory OTの勾配・コスト・軌跡診断を維持する。

### 失敗と再開

条件や工程が失敗した場合はその `.failed.json` とログを保存し、依存する工程をskipする。
独立した他条件は続行する。評価OTの有限な未収束は既存方針で再試行し、それでも失敗なら
欠損として記録する。非有限値や生成発散は失敗扱い。
全体が完成すれば `completed.json`、一部失敗・skipがあれば `failed.json` の
`status=partial_failure` を保存する。評価OTの欠損は `evaluation_ot_missing` に別記する。

```bash
bash work/20260913_2step/scripts/run_all.sh \
  --resume-six-ot-campaign hybrid500_ot_<実際に表示されたID>
```

明示的な再開だけは同campaignを使用する。新launchと新runに追記し、既存ファイルを上書きしない。
起動時のdevice・sampling設定を引き継ぎ、完成済み工程は完了マーカーと出力を検証して再利用する。
未完了学習は同campaignの互換raw/EMA/optimizer一式から再開する。
保存済みbundleがなければ未完了条件だけ最初から開始する。
RNGの完全な状態は旧仕様どおり保存しないため、連続実行とのビット単位一致は保証しない。
新旧の補間や目的関数が異なるStage2 checkpointは再開元にできない。
同じcampaignのworkerはOSロックで排他し、二重計算を拒否する。
旧実験用 `--resume-campaign` ではなく、この専用再開オプションを使う。
