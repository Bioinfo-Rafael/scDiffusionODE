# Implementation validation — 2026-09-13 JST

Validation used the existing local `scdiffusion` Python environment, PyTorch
2.5.1, CPU only. No real h5ad was opened. No training workflow, reverse sampling
workflow, real-data analysis, UMAP fit, GPU job, or figure-rendering workflow was
executed. No PNG, PDF, or SVG was created.

## Checks

- `PYTHONDONTWRITEBYTECODE=1 /Users/cls-lab/miniconda3/envs/scdiffusion/bin/python -B work/20260913_2step/tests/test_suite.py`
  — **26 tests passed**. Small tensors have four genes and neural hidden widths
  `[8,8,4,4]`; production configs retain `[2000,1000,500,500]`.
- All five CLI `--help` calls (`train`, `sample`, `analyze`, `embed`, `plot`)
  returned exit status 0 without executing their workflows.
- Ruff 0.11.13 lint and format checks passed for all 25 Python files. The
  formatter was installed only in the ignored suite-local test tooling folder;
  no repository/runtime dependency file changed.
- Every suite Python file was parsed with `ast.parse` without compilation or
  experiment execution.
- Staged changes and `git diff --check` were reviewed; all committed additions
  are under `work/20260913_2step/`. Existing user edits in other suites were not
  included or altered.

## Focused coverage

1. Seven canonical conditions map to exactly the three 20260911 selections;
   source architecture/training defaults remain intact.
2. Stage 1 is the actual `Cell_Unet` with no ODE and exactly the source MSE loss.
3. Every Stage-2 condition loads byte-identical neural weights, excludes them
   from optimizer updates and gradients, and forces eval even after `.train()`.
4. A synthetic single optimizer update for each of the six conditions leaves
   the neural hash unchanged, including EMA; illegal optimizer/hash changes fail.
5. Each Hybrid checkpoint restores its full state and identical forward output
   without the original edge TSV or Stage-1 checkpoint.
6. Full-range support is exactly 0..999 for Stage 1/reconstruction, while OT
   support is exactly 0..49 with equal sampling weights.
7. All six models retain the original Hybrid forward and interpolation,
   including t=0 and t=999 endpoint behavior.
8. Reconstruction never calls OT. OT never calls diffusion MSE and compares
   converted pred_x0 with the real set. Out-of-region OT t fails.
9. Sinkhorn tests cover independent batch permutations, zero self divergence,
   nonzero differentiable gradients, finite-difference gradcheck, repeated-gene
   cost normalization, invalid settings and explicit nonconvergence failure.
10. Source mask orientation, the exact L1 coefficient/reduction, and exclusive
    W/alpha/rho penalty gradients are checked.
11. Snapshot tables have exactly 22/20 entries with separate phase, null
    post-ODE diffusion time, terminal ODE conditioning and physical time.
12. A single synthetic Euler update uses only the terminal-conditioned field
    with gradients disabled; NaN, Inf and divergence bounds fail loudly.
13. Joint UMAP assembly contains real + exactly 950/1000/1050/1100, using the
    appropriate clean predictions/states. Mocked dimensionality reduction
    verifies 20 baseline fits or 22 independent + one joint Stage-2 fit.
14. Pearson/L2 match both source helpers; MSE/cosine match source expressions.
    Small mocked-grid diagnostics verify CSV schemas, true-noise/branch/norm
    series, counts and no invented post-ODE metrics.
15. Full and dense low-noise grids and actual schedule sigma shading are checked.
16. Exact mean Euclidean diversity and the squared-distance identity match a
    3-cell example; sampled median is deterministic.
17. Exclusive creation, suite output confinement, canonical EMA verification
    and checkpoint tamper rejection are tested.

These are implementation checks, not evidence that the scientific hypothesis,
training stability, Sinkhorn defaults on real cells, or post-ODE integration
will succeed. The README documents the remaining numerical assumptions and
future remote commands.

## Background launcher follow-up

Added `scripts/run_all.sh` (existing conda environment) and `scripts/run_all.py`
(detached worker, logs, output-path propagation and fail-fast sequencing).

- `python -B -m unittest discover -s work/20260913_2step/tests -p 'test_*.py'`
  — **32 tests passed**, including six launcher checks: all 42 commands and
  dependencies, side-effect-free dry-run, detached launch/no overwrite, stopping
  on first failure, exact output-path capture, and invalid argument rejection.
  Detached spawning was mocked; output capture ran only a one-line Python print.
- `bash -n work/20260913_2step/scripts/run_all.sh` passed.
- `bash work/20260913_2step/scripts/run_all.sh --dry-run` successfully used the
  existing `scdiffusion` conda environment and printed exactly 42 commands.
- Ruff lint and formatting checks passed for 27 Python files.

The full launcher was not started; no experiment training, sampling, real-data
analysis, real UMAP fit or figure generation was executed.

## Sinkhorn convergence and campaign recovery follow-up

The supplied remote log reports a marginal residual of `1.02642e-5` after 200
iterations against a `1e-5` tolerance. This was an iteration-cap failure. The
updated default cap is 2000 with the same epsilon/cost/tolerance and early
convergence checks every ten iterations. Ten-iteration activation checkpointing
recomputes identical iterations during backward to limit saved graph memory.

- **41 tests passed** using CPU synthetic fixtures; no real-data/GPU jobs ran.
- A deterministic 8-cell fixture fails at cap 200, then converges in 450
  iterations at residual `9.492529890062218e-6` with the unchanged `1e-5` tolerance.
  This reproduces the class of failure, not the unavailable remote batch.
- Checkpointed and ordinarily unrolled solver losses and gradients agree.
- Synthetic continuation restores raw model, EMA, AdamW moments/LR and frozen
  CellUNet; the next identical-input optimizer update matches an uninterrupted
  control. Source checkpoint file hashes remain unchanged.
- Recovery selection skips completed training, uses the last complete raw/EMA/
  optimizer bundle, ignores incomplete saves, rejects corruption/wrong campaign/
  wrong Stage-1 origin, and emits a 38-command plan for four completed trainings.
- Recovery dry-run selects the actual recovery plan without spawning a worker.
- The conda wrapper's ordinary dry-run prints 42 commands, including a consistent
  evaluation iteration cap for older reconstruction/Stage-1 checkpoints.
- Ruff lint/format checks and shell syntax validation passed.

No remote campaign was launched by the agent. Old checkpoints lack RNG/loader
state: recovery explicitly records a restarted data/noise stream and does not
claim bitwise equivalence to an uninterrupted experiment. The 2000-iteration cap
is a finite numerical safeguard, not a guarantee of convergence for all real
batches. Existing runs/checkpoints/results are never modified by recovery.

## Completed reconstruction/baseline analysis-only mode — 2026-09-14 JST

`run_all.sh --analyze-campaign NAME` verifies the canonical Stage-1 EMA and three
completed reconstruction EMAs, then launches only sampling, numerical analysis,
UMAP coordinates and both figure stages. OT-model training/sampling is excluded.
Missing completed training fails explicitly; partial bundles are not loaded and
training is never substituted. Logs and all postprocessing outputs are new.

- **44 CPU synthetic tests passed**, including a 20-command postprocessing-only
  plan, exactly three reconstruction checkpoint selections, no OT-model/training
  commands, missing-completion rejection and exclusion of intermediate bundles.
- Ruff lint/format checks and shell syntax validation passed.
- No real sampling, analysis, training, UMAP fitting or figure generation ran
  during implementation. Numerical Sinkhorn distance remains an evaluation
  metric for the four completed models, separate from OT-model training.

## Trajectory-occupation OT — 2026-09-14 JST

Implemented on `feat/20260914-trajectory-occupation-ot`, branched from the fetched
`origin/feat/20260913-2step-ot` (`18590d1`). Changes are confined to this suite.
The new canonical conditions are `hill_after_linear_trajectory_ot_soft`,
`centered_signed_hill_trajectory_ot_soft`, and `shifted_hill_rho_trajectory_ot_soft`.
Old one-step configs and their objective remain available explicitly as legacy.

Implementation scope:

- Immutable canonical Stage-1 EMA Gaussian-start x50 caches, source states after
  reverse input t=51 and before t=50 (949 source updates), memmap plus SHA/genes/
  seeds/schedule/state semantics. Train and evaluation have separate seeds/roles.
- Differentiable ODE-only Euler paths [B,K+1,G] with t_cond=0, dt=.001, K=100,
  non-reentrant 10-update activation checkpointing. Frozen CellUNet and source
  model/sampling definitions are preserved.
- Stratified four-bin occupation sampling, 128 paths -> 512 OT points, versus
  an independent empirical-X target stream of 512 real cells. No paired MSE,
  no forward-noised real sources, no density weights, no coefficient retuning.
- Balanced float64 log-domain debiased Sinkhorn with epsilon warm starts ending
  exactly at target .1; every scale must meet tolerance 1e-5 within cap 2000.
  Scaled self terms use a symmetric averaged log fixed-point update. The legacy
  direct algorithm remains the disabled-scaling path.
- Separate OT/weighted-soft autograd gradient diagnostics (without mutating
  accumulated gradients), parameter/update norms, cross-cost statistics and
  structured per-scale CSVs. Cost/gradient diagnostics run at step 1 and every
  100 updates by default.
- Evaluation-only deterministic projection/sort SW2 and SW2 squared; independent
  generated-x50 occupation versus endpoint CSVs for all Stage-2 families;
  additional SW snapshots; saved-number-only comparison/gradient plots.
- Schema-guarded continuation: legacy objective/schema cannot resume into a
  trajectory condition, even if a checkpoint is renamed. New compatible bundles
  can resume with their original source cache, raw/EMA/optimizer and step.
  Canonical recovery reuses completed Stage1/recon, ignores old OT condition
  directories, and initializes fresh trajectory ODEs when no compatible run exists.

Validation executed (CPU synthetic tensors only, PyTorch 2.5.1):

- `PYTHONDONTWRITEBYTECODE=1 /Users/cls-lab/miniconda3/envs/scdiffusion/bin/python -B -m unittest discover -s work/20260913_2step/tests -p 'test_*.py'`
  — **61 tests passed**, 4.814 seconds. Log is ignored `.test_tmp/trajectory-final-tests.log`.
- Retained Stage1, recon, legacy loss, source Hybrid interpolation, frozen weights,
  EMA/optimizer, checkpoint provenance, unchanged sampling and earlier analysis tests.
- New tests verify all canonical names/default counts, explicit old-schema rejection,
  ODE-only calls, [B,K+1,G] shape and an analytic all-step gradient, checkpointed
  versus ordinary states/loss/gradients in all three original ODE families,
  zero frozen-neural gradients and unchanged neural/EMA hashes after updates.
- Stratified tests check all four bins for every one of 128 paths, exactly 512
  points, fresh choices, and differentiation through selected states. Independent
  real target tests include sparse X, 512 distinct selections when possible,
  separate per-step source/target RNGs and documented small-dataset replacement.
- Cache tests use a tiny fake reverse operator, verifying pure canonical EMA,
  Gaussian starts, exact input indices 999..51, metadata/SHA, reuse, wrong ancestry/
  role/model rejection and no overwrite. No actual diffusion sampling was run.
- Solver tests verify exact epsilon termination, direct/scaled values and gradients,
  cross/target permutation invariance, finite cost statistics and explicit failed
  convergence. Symmetric scaling was added after synthetic self terms exposed slow
  alternating convergence; safeguards were not weakened.
- SW tests verify zero for identical sets, positive shifts, repeatability and
  permutation invariance even under subsampling. A tiny no-grad occupation fixture
  verifies separate endpoint/occupation counts and numeric CSV schemas.
- A synthetic four-gene **two-update** runner fixture (no h5ad, CPU only) checks
  actual log/checkpoint writes, cache reuse and compatible one-update continuation;
  all original files remain hash-identical. These are unit-test fixtures, not an
  experiment training run. Temporary files stay under ignored `.test_tmp/`.
- Launcher tests verify one shared train cache, a separate evaluation cache,
  all three new OT conditions, six occupation evaluations and the final comparison
  plot dependency. Full plan: 51 commands; completed Stage1+recon recovery: 47.
  Analysis-only remains 20 commands. Dry-run never spawns workers or writes outputs.
- All nine CLI `--help` calls returned 0 (`train`, `sample`, `analyze`, `embed`,
  `plot`, `cache_x50`, `occupation`, `plot_occupation`, `run_all`). All new workflow
  imports and AST parsing for 38 Python files passed.
- Conda shell wrapper `run_all.sh --dry-run` succeeded and printed 51 commands;
  shell syntax, Ruff lint/format and `git diff --check` passed.

No full training, actual Stage1 reverse sampling, GPU sampling, real-data analysis,
UMAP fitting, figure rendering or remote/background experiment was executed.
No real h5ad was opened and no PNG/PDF/SVG was generated. Literature URLs were
checked for documentation; no new runtime dependency was added.

Unresolved empirical concerns: production GPU peak memory and runtime are unmeasured;
100-step ODE unrolling plus three 512x512 float64 solves over multiple epsilon
levels can be expensive despite checkpointing. The finite per-scale cap may still
fail on real batches. Euler stability and x50 manifold proximity/coverage remain
hypotheses, and neither epsilon nor dt nor inherited soft-gradient balance was
validated on Embryonic data. SW/occupation samples contain correlated path points.
Training targets the full empirical distribution while evaluation retains the
existing Erythropoietic reference policy. Full Torch/CUDA RNG state is not saved,
so resumed training does not claim bitwise identity to uninterrupted execution.

## Direct-Hill repeated-transform runtime fix — 2026-09-15 JST

Inspected the source direct-Hill component loop: full A/theta transforms are
inside each target chunk, repeated 64 times per field evaluation at G=1024.
The trajectory-only adapter shares these differentiable physical parameters
within one unroll and rebuilds them after every optimizer update. It leaves the
source files, normal Hybrid sampler, objective and scientific batch/time settings
unchanged. New runtime defaults: log every 50 updates with measured seconds/update
and ETA; save every 1000 updates. These operational settings are recorded in
`trajectory_runtime` and may change during compatible recovery.

- **63 CPU synthetic unit tests passed** (7.151 s).
- New tests compare all trajectory states and scalar loss exactly, and gradients
  within rtol=2e-5/atol=1e-8, against the source for centered and shifted fields
  with multiple target chunks and outer activation checkpointing. They exercise
  an autograd diagnostic followed by backward on the same graph.
- A further test takes two synthetic parameter updates and verifies physical
  transforms are rebuilt, with source-identical states on each new unroll.
- Existing objective, legacy, frozen CellUNet, EMA, cache, recovery, Sinkhorn,
  occupation/SW and launcher tests remain green.

No actual data, GPU benchmark, full training, sampling, UMAP or rendering was
executed. This establishes numerical equivalence on small fixtures, not a measured
production acceleration. Existing running remote code cannot be updated in-place;
SIGINT on the old runner records failure but cannot preserve unsaved weights.

## Completed-model analysis including finished trajectory OT — 2026-09-15 JST

Added mutually exclusive `--analyze-completed-campaign`. It selects verified final
canonical EMAs only and skips incomplete conditions without loading intermediate
bundles or scheduling training. The reported campaign selects Stage1, recon3 and
finished hill-after-linear trajectory OT. Its plan has 31 steps: evaluation x50
cache, five sets of sampling/numerics/UMAP/two plots, four occupation evaluations
and one filtered comparison plot. Existing four-condition analysis mode remains.

- **64 CPU synthetic tests passed** in 6.935 s. The new regression verifies five
  selected checkpoints, 31 steps, four occupation inputs, all output dependencies,
  exclusion of the two unfinished models and no training or partial-bundle loads.
- Ruff lint/format, CLI help and diff checks passed.
- No remote process was stopped or launched by the agent; no real-data workflow,
  GPU sampling, UMAP or figure rendering ran. README provides the remote SIGINT,
  fetch and detached analysis command. Unsaved training progress is not persisted
  by the existing SIGINT handler; saved files remain untouched.
