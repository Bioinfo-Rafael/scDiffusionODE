# Implementation and single-ODE correction report

## 1. Branch and worktree

- Branch: `feat/20260915-x0predict`.
- Base: `3108fd1`, on `feat/20260914-trajectory-occupation-ot`.
- Actual checkout: `/Users/cls-lab/Git/scDiffusionODE`; the requested Linux
  checkout path is absent from this environment.
- All new work is under `work/20260915_x0predict`.
- No commit, push, merge, or full GPU experiment was performed.
- Pre-existing unrelated local edits were retained. Their original bytes,
  along with all tracked repository files, are recorded in `audit/baseline.json`
  and checked by the regression test.

## 2. Before the correction

| Family | Initial implementation | Components |
|---|---|---:|
| simple_softplus | `20260830::SimpleSoftplus20260830` | 1 |
| hill_after_linear | `20260803_ODE_hill_exp::HillAfterLinearField` | 1 |
| centered_signed_hill | `20260816::CenteredSignedHillField` | 8 |
| shifted_hill_rho | `20260816::ShiftedHillRhoField` | 8 |

**Yes: centered/shifted incorrectly selected the 20260816 expert-gated models
for the user's corrected scientific intent.** Their adapters have been
replaced, together with Hill-after-linear's adapter, without rebuilding the
experiment infrastructure.

## 3. Exact sources now selected

All four fields come from the unchanged file
`work/20260830/models/ode_fields_20260830.py`:

| Family | Exact class |
|---|---|
| simple_softplus | `SimpleSoftplus20260830` |
| hill_after_linear | `HillAfterLinear20260830` |
| centered_signed_hill | `CenteredSignedHill20260830` |
| shifted_hill_rho | `ShiftedHillRho20260830` |

They are instantiated by the existing `work/20260830/models/factory.py`
`build_ode_from_config` function, then wrapped by the local thin
`SingleODEHybrid500`. Field equations are imported, not duplicated.
The historical Hybrid500 mixin is reused without changing its arithmetic.

The source lineage was inspected: `ed376ec` introduced the 20260817 single
direct Hill variants; `5a3f758` consolidated the four fields in 20260830;
`80808be` factored the penalty helper without changing these equations.

## 4. Exact equations

Let `sp=softplus`, `e=1e-6`, `x⁺_j=max(x_j,e)`, and
`delta_i=sp(raw_delta_i)+e`. All edge matrices are `[target i, regulator j]`.

```text
simple_softplus:
  f_i = sp((Σ_j W_ij*x_j+b_i)/sqrt(G)) - sp(gamma_i)*x_i

hill_after_linear:
  z_i = sp(Σ_j W_ij*x_j+b_i)
  f_i = V_i*z_i²/(K_i²+z_i²) - delta_i*x_i

centered_signed_hill:
  f_i = b_i + Σ_j A_ij*tanh(alpha_ij*(log(x⁺_j)-log(theta_ij)))
            - delta_i*x_i

shifted_hill_rho:
  f_i = b_i + Σ_j A_ij*expm1(rho_ij)
                       *sigmoid(2*(log(x⁺_j)-log(theta_ij)))
            - delta_i*x_i
```

`A`, `theta`, `V`, and Hill threshold `K` also use `sp(raw)+e`. The shifted
sigmoid is the stable evaluation of `x⁺²/(theta²+x⁺²)`. Centered/shifted sum
directly over regulator genes **j**, with no expert dimension. Source
initialization, masking orientation, positive guards, and penalty weighting
are preserved; details are in README.

## 5. Structural guarantees

All four have `is_lincomb=False`, `num_components=1`, `num_experts=1`, no
`coeff_net`, no neural submodules, and no gating/time-embedding parameters
inside the ODE. Matrix parameters are `[G,G]`; decay is `[G]`.
Construction and restoration enforce exact source class identity and these
invariants. The tests also deliberately present old K=8 fields and inject NN
gates to verify rejection.

`audit/model_sources.json` records concrete four-gene model inspections,
including every parameter shape and `named_modules()` result. Metadata now
records `ode_family`, `ode_source_suite`, `ode_source_file`, `ode_class`,
`ode_components=1`, `expert_gating=false`, and source SHA256 both in the effective
config and directly in Stage2 checkpoint metadata.

## 6. Softplus naming clarification

The user clarified that **softplus** is intended. The earlier softmax phrase
was a naming mistake. The historical search found no softmax-over-genes
production-decay field in 549 Python blobs; observed softmax uses were expert
gating/classification. No new softmax field was invented. The existing
`SimpleSoftplus20260830` remains the simple production-decay model.

## 7. START_X and Hybrid500 remain intact

The correction did not change `training/objectives.py`, sampling, diagnostics,
the launcher, optimizer/freezing behavior, or checkpoint recovery mechanics.
The training runner only gained the explicit ODE provenance fields above.

- Stage1 remains a newly initialized CellUNet trained for 30,000 updates with
  the canonical settings and final EMA `0.9999`.
- Exact Stage1 loss: `mean((CellUNet(x_t,t)-x0)^2)`, with uniform `t=0..999`,
  through the existing `diffusion.training_losses` API.
- All diffusion construction asserts **`ModelMeanType.START_X`**, ordinary
  **`LossType.MSE`**, `predict_xstart=True`, 1000 original timesteps.
- Stage2 freezes and shares the same final Stage1 EMA; optimizer membership,
  eval mode, and raw/EMA state hashes are checked.
- `w_ode=max(0,1-t/500)`, `w_cell=1-w_ode`;
  `x0_hat=w_ode*ode_raw+w_cell*cellunet_raw`.
- START_X Stage2 uses uniform `t=0..999`, existing `training_losses`, plus the
  original soft off-mask penalty. High-t ODE inactivity is documented.
- OT uses uniform `t=0..49` and `pred_x0=prediction=model(x_t,t)` directly.
  **`_predict_xstart_from_eps` is not called by x0 OT.**
- Sinkhorn remains debiased, float64, mean squared gene distance, uniform
  marginals, epsilon 0.1, tolerance `1e-5`, cap 2000, existing convergence policy.
- Native ancestral START_X sampling, x0 diagnostics, UMAP/distribution analysis,
  and separately labeled post-hoc dynamics remain available.
- No trajectory OT, epsilon objectives, velocity/kinetic/manifold losses, or
  Euler adapter was added to training.

## 8. Eight conditions and output layout

```text
simple_softplus_start_x_soft
simple_softplus_ot_soft
hill_after_linear_start_x_soft
hill_after_linear_ot_soft
centered_signed_hill_start_x_soft
centered_signed_hill_ot_soft
shifted_hill_rho_start_x_soft
shifted_hill_rho_ot_soft
```

Campaigns write exclusively below:

```text
work/20260915_x0predict/runs/x0predict_<UTC>_<RUNID>/
work/20260915_x0predict/results/x0predict_<UTC>_<RUNID>/
```

Completed work is reused; failures are recorded and independent conditions
continue. Resume restores only complete checkpoint bundles into new attempts.

## 9. Verification before and after

- Original first pass: **17/17 tests passed**.
- Before the source correction: **18/18 passed**, including the subsequent
  distribution-analysis and PNG-rendering test (15.285 seconds).
- After correction: **21/21 passed** (10.258 seconds), retaining every applicable
  original test and adding strict single-ODE structure/provenance, rejection
  of K=8/injected gates, and deterministic forward-formula tests for all four.
- Tests exercise all eight actual backward/optimizer/EMA paths, START_X target,
  OT raw-output identity, Hybrid500, native sampling, Stage1 interruption/resume,
  canonical checkpoint sharing, output isolation, old epsilon defaults, and
  protected-file byte hashes.
- Synthetic integration training was limited to two Stage1 updates and one
  update per Stage2 condition on five cells/four genes, CPU only. Sampling used
  a two-cell constant toy predictor. This is not the full experiment.
- PNG files were generated from synthetic metrics and checked with Pillow.
- `bash -n`, Python AST parsing, launcher `--dry-run`, and `git diff --check`
  completed successfully.
- `guided_diffusion` core and all old experiment source/output paths were
  untouched by this work.

Exact test command used:

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /Users/cls-lab/miniconda3/envs/scdiffusion/bin/python -m unittest discover \
  -s work/20260915_x0predict/tests -v
```

Full logs: `audit/test_results.txt` and
`audit/test_results_before_single_ode_correction.txt`.

## 10. Files changed by the correction

- `common.py`: all four source mappings, K=1 configuration, correct source
  metadata, explicit single-ODE config validation.
- `models/__init__.py`: thin single-ODE Hybrid500 adapter, exact source factory,
  construction/restoration assertions against NN gates and expert dimensions.
- `training/runner.py`: direct checkpoint ODE provenance fields.
- `tests/test_suite.py`: corrected source expectation plus strict structural,
  rejection, and manual-formula tests; existing scientific tests retained.
- `README.md`: completed the requested suite documentation with corrected
  single-ODE semantics, equations, lineage, launch/resume, and analysis details.
- `IMPLEMENTATION_REPORT.md`, `audit/model_sources.json`, and test logs:
  verification and correction records.

The overall new suite also contains `configs/`, `cli.py`, `scripts/run_all.*`,
`training/checkpoints.py`, `training/objectives.py`, `losses/sinkhorn.py`,
`sampling/trajectory.py`, isolated analysis modules, and audit evidence. There
are no task changes outside the new suite. New files are marked intent-to-add
to make them visible to `git diff`; no commit was made.

## 11. Launch commands

With the repository's Python environment active, from its root:

```bash
# Stage1 plus its separate sampling/analysis
bash work/20260915_x0predict/scripts/run_all.sh --stage1-only --device cuda

# Fresh Stage1, then all eight Stage2 conditions and analyses
bash work/20260915_x0predict/scripts/run_all.sh --all-eight --device cuda

# Continue against the SAME Stage1 campaign
bash work/20260915_x0predict/scripts/run_all.sh --stage2-only --all-eight \
  --resume-campaign x0predict_<UTC>_<RUNID> --device cuda
```

The canonical dataset's original Linux paths remain defaults. On another
machine, supply `--data` and `--edge-tsv` with the existing canonical files when
creating the campaign. None of these full-training commands was executed.

## 12. Diff inspection

Use `git diff --stat -- work/20260915_x0predict` for this task alone and
`git diff --stat` for the whole worktree. The latter also includes the user's
pre-existing edits in `work/20260801/.../plot_hybrid_ts_soft_weight.py` and
`work/20260911/README.md`; those are not changes from this implementation.
