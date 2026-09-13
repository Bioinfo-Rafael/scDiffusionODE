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
