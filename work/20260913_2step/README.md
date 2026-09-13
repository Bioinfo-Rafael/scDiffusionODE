# 20260913: common CellUNet, frozen-backbone reconstruction vs OT

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
| `hill_after_linear_ot_soft` | single, original 20260803 Hill-after-linear | Sinkhorn + soft, uniform 0..49 |
| `centered_signed_hill_ot_soft` | K=8 original 20260816 centered signed Hill | Sinkhorn + soft, uniform 0..49 |
| `shifted_hill_rho_ot_soft` | K=8 original 20260816 shifted Hill rho | Sinkhorn + soft, uniform 0..49 |

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
clipping, or automatic resume. Source default `microbatch=-1` is preserved.

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

All six models call the **original Hybrid forward**:

```
w_ode(t) = 1 - t / 999
eps_hybrid = w_ode * eps_ode + (1 - w_ode) * eps_cellunet
```

At t=999 the output is CellUNet; at t=0 it is entirely ODE. No rescaling of the
50-step OT region and no learned replacement interpolation is introduced.

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

### OT Stage 2

Only OT uses uniform `t in {0,...,49}`. Its dedicated sampler does not change the
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

| Numerical setting | Default |
| --- | --- |
| Entropic epsilon, in mean-squared-gene-distance units | `0.1` |
| Maximum iterations per cross/self solve | `200` |
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
schedule) and OT's t=0..49 region. **The entire 50-step OT region is not claimed
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
scripts/                         explicit train/sample/analyze/embed/plot CLIs
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
restart into a partial directory, merging to main, or source-run manifest update
is performed. Preserve the git commit recorded in a run to reproduce its source
imports. Source-config hashes are recorded; Stage-1/Stage-2 gene hashes and edge
TSV hashes are checked across the campaign. The dataset itself is not fully
hashed (large files); copying an h5ad with the same genes but changed expression
values cannot be detected by a gene-order hash alone.

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

OT conditions use the same canonical EMA:

```bash
for family in hill_after_linear centered_signed_hill shifted_hill_rho; do
  python work/20260913_2step/scripts/train.py \
    --campaign two_step_001 --condition "${family}_ot_soft" --device cuda \
    --ot-epsilon 0.1 --ot-max-iterations 200 --ot-tolerance 1e-5
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
PYTHONDONTWRITEBYTECODE=1 python -B work/20260913_2step/tests/test_suite.py
python -B work/20260913_2step/scripts/train.py --help
```

## Numerical/design limits to evaluate remotely

Epsilon=0.1 and Euler dt=0.001 are documented starting values, not tuned results.
OT magnitude and its gradient can be small near t=0 relative to the inherited
soft constraint. Monitor primary/soft loss separately. High-dimensional,
poorly conditioned or low-entropy batches can exceed 200 Sinkhorn iterations;
the default fails loudly and requires an explicit new run/config decision.
Differentiating iterations costs O(iterations * batch²) graph memory. Solver
truncation can cause tiny negative divergence estimates; values are not clipped.

The +ODE continuation uses the **positive raw learned branch** as a vector field
because that is the requested experiment. Epsilon training does not itself
prove this sign, integration time scale, stability, or biological correctness.
Only finite/divergence checks are enforced; do not infer convergence or quality
from a successful finite run. Compare displacement, diversity and distribution
distance, and inspect sensitivity to dt in separately saved sampling runs.
Independent UMAP positions are not aligned and UMAP proximity alone cannot prove
absence of collapse. The exact mean pairwise distance and all UMAP fits can be
expensive; those are explicit future analysis jobs, not implementation checks.
