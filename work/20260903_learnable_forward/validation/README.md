# Model A auxiliary Q/D validation — 2026-09-05 JST

Target branch: `work/20260816-hill-variants`.
Baseline commit: `9a7e72431ba5a61fcd171627ffddfcffd86c9045`.
All implementation changes are confined to `work/20260903_learnable_forward/`.
Pre-existing changes under `work/20260801/hybrid_ts_soft_weight_curve/` are
excluded from this commit.

## Mathematical implementation

Thin QR of E produces Z. Q = Z(R_Q−R_Qᵀ)Zᵀ and
D = sigma² I + ZBBᵀZᵀ, with an explicit positive softplus scalar and configured
positive floor. Off-diagonal D entries are inner products of rows of H=ZB;
Q entries are skew bilinear forms. Every effective interaction is mediated by
the auxiliary space. Only effective [target,source] interactions enter GRN.

Transitions, sampling and scores split into the learned K-space and its scalar
isotropic complement. DSM applies the exact K-space quadratic plus the scalar
complement quadratic. KL uses split trace, logdet and mean norm. Boundary NLL
uses the algebraically equivalent L_K^{-1}(Phi_K x_K−y_K−Sigma_K score_K)
residual and scalar complement; its log determinant uses tr(A). There is no
gene-space covariance/inverse/Cholesky/exponential in the Model A training or
reverse path. Thin QR costs O(dK²); projection and K-space operation costs are
documented in the main README. GRN alone may materialize a d×d interaction.

## Verification

- Full current suite: **55 tests passed**, using `python -m unittest discover
  -s work/20260903_learnable_forward/tests -v` in the scdiffusion environment.
  The untouched baseline suite also passed its 55 tests before comparison.
- Model A tests compare float64 values and gradients against small dense
  references for multiple d/K pairs, including transition, covariance,
  sampling root, score, weighted DSM, terminal KL, decoder mean/noise and NLL.
  Dense references use the prescribed epsilon coordinate; a dense Cholesky
  defines a different coordinate and is not substituted for it.
- Tests check orthonormal Z, skew Q, positive D, Lyapunov residual, VP limit
  independent of E, empirical sampling covariance, GRN orientation and effect,
  fixed/learned isotropic scalar, finite gradients, small-time covariance,
  failure diagnostics, explicit old-checkpoint rejection, and schema metadata.
- Spies require every production Model A factorization/exponential/solve to be
  K×K. A tensor dispatch guard rejects **any d×d intermediate** in the Model A
  ELBO forward/backward when GRN is disabled.
- [Model B regression](model_b_regression.json): **80 tensors bitwise equal**
  before/after: both losses, all parameter gradients, state/reload, fixed-seed
  reverse samples and parameter-analysis categories. Model B implementation,
  config, objective primitives and its dedicated tests are unchanged.
- `scripts/verify_protected.py`: all seven protected core file hashes match.
- [Model A train dry-run](model_a_dry_run.json) and the Model-A-only full-pipeline
  dry-run succeed and select the new auxiliary experiment.
- Real-data smoke: 1024 genes, K=64, GRN enabled (8221 listed edges),
  batch 4, float32, Cell_Unet hidden sizes [32,16], two optimizer steps.
  Raw/EMA checkpoints were written. The EMA checkpoint produced four finite
  1024-gene samples using ten reverse steps. Parameter evolution reconstructed
  and visualized Z/Q_K/B/sigma²/Q/D/A, with three snapshot NPZs including H and
  on/off-mask interactions. Diffusion diagnostics succeeded for eight real
  cells at timesteps 0, 500 and 999. The histogram was visually inspected.
  This short smoke checks integration, not convergence or scientific quality.
  Runtime outputs remain ignored under `runs/model_a_stationary_qd_aux/smoke-aux-20260905/`.

## Matched d=1024 CPU benchmark

K=64, batch 4, PyTorch 2.5.1, four threads, one warmup, three repetitions per
dtype. Both models start at exact VP initialization; GRN is disabled and the
same minimal linear denoiser is used. Reported values are medians.

| Model | dtype | Transition (ms) | Full ELBO forward (ms) | Backward (ms) |
| --- | --- | ---: | ---: | ---: |
| Old dense A | float32 | 27.591 | 84.959 | 288.057 |
| Old dense A | float64 | 69.177 | 201.222 | 1220.891 |
| Auxiliary A (K=64) | float32 | 1.312 | 5.052 | 5.335 |
| Auxiliary A (K=64) | float64 | 1.776 | 6.594 | 7.760 |

- [Auxiliary results](aux_d1024_k64.json): six successful trials, finite
  losses/gradients, `full_d_cholesky=false`, `full_d_matrix_exponential=false`,
  `approximation=false`; matrix exponential and Cholesky dimension K=64.
- [Old dense results](old_dense_d1024.json): six successful trials using the
  baseline implementation from the commit above, in an isolated temporary
  extraction. This is a capacity-changing reparameterization, so matched
  initialization performance does not establish equal learned models.
- CUDA was not benchmarked; no GPU memory or production convergence claims.

Reproduce the new benchmark:

```bash
conda run -n scdiffusion python work/20260903_learnable_forward/scripts/benchmark_dense.py \
  --dim 1024 --aux-dim 64 --batch-size 4 --models stationary_qd \
  --dtypes float32,float64 --device cpu --num-threads 4 --warmup 1 --repeats 3 \
  --output /tmp/aux_d1024_k64.json
```

## Initialization and new run

The constructor supports Q_K=B=0 and sigma²=0.5 exactly. However, B=0 also has
zero gradient through BBᵀ. The supplied training config initializes B=0.01 I
so anisotropic D can learn; set `auxiliary_b_init_scale=0` explicitly for the
exact VP limit. K and the positive floor are configurable.

Start a **new** `model_a_stationary_qd_aux` run using the command in the main
README. Do not resume the old dense step-5000 checkpoint. Model A raw/EMA
state-dict provenance records schema version 2, parameterization name and K;
persistent buffers enforce compatibility even if provenance metadata is lost.

## Changed files

Paths below are relative to `work/20260903_learnable_forward/`.
The old dense Model A config is replaced with the auxiliary config; the
benchmark script keeps its filename for CLI compatibility.

- [README.md](../README.md)
- [analysis/diffusion_diagnostics.py](../analysis/diffusion_diagnostics.py)
- [analysis/parameter_evolution.py](../analysis/parameter_evolution.py)
- [configs/model_a_stationary_qd_aux.json](../configs/model_a_stationary_qd_aux.json)
- `configs/model_a_stationary_qd_dense.json` (removed)
- [diffusion/stationary_qd.py](../diffusion/stationary_qd.py)
- [models/factory.py](../models/factory.py)
- [models/wrapper.py](../models/wrapper.py)
- [sampling/reverse_sde.py](../sampling/reverse_sde.py)
- [scripts/benchmark_dense.py](../scripts/benchmark_dense.py)
- [scripts/common.py](../scripts/common.py)
- [scripts/train.py](../scripts/train.py)
- [tests/test_normalization_grn_logging.py](../tests/test_normalization_grn_logging.py)
- [tests/test_objective_equivalence.py](../tests/test_objective_equivalence.py)
- [tests/test_posthoc_pipeline.py](../tests/test_posthoc_pipeline.py)
- [tests/test_reverse_sampling.py](../tests/test_reverse_sampling.py)
- [tests/test_stationary_qd.py](../tests/test_stationary_qd.py)
- [tests/test_time_mapping.py](../tests/test_time_mapping.py)
- [tests/test_training_integration.py](../tests/test_training_integration.py)
- [validation/README.md](README.md)
- [validation/aux_d1024_k64.json](../validation/aux_d1024_k64.json)
- [validation/model_a_dry_run.json](../validation/model_a_dry_run.json)
- [validation/model_b_regression.json](../validation/model_b_regression.json)
- [validation/old_dense_d1024.json](../validation/old_dense_d1024.json)
