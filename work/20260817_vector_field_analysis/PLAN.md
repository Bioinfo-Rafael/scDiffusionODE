# Implementation plan

Scope: post-hoc analysis of the raw trained ODE branch from a
`work/20260803_ODE_hill_exp` run whose configuration is exactly
`model_family=standard_hybrid_single` and `ode_type=hill_after_linear`.
Training, sampling, checkpoints, and existing run metadata remain read-only.

## Dynamo source audit

The audited upstream source is `aristoteleo/dynamo-release` at commit
`69ba903000a1dcd5aa7d8b36a9318c1b22c5d650` (2026-08-18 checkout).

- Direct reuse: none at runtime. Dynamo is not installed in the target
  `scdiffusion` environment, and its public `vector_calculus.py` entry points
  expect an AnnData `VecFld` reconstructed by Dynamo. Installing Dynamo or
  fabricating SparseVFC state would add an unnecessary dependency and risk
  violating the no-refit requirement.
- Adapter: expose the trained `model.ode_model` as a read-only vector field
  with Dynamo-style `func(X)` and `get_Jacobian(...)` methods, plus batched
  velocity, exact Jacobian, divergence, and Jacobian-vector-product methods.
- Minimal formula ports: use Dynamo's definitions `div V = tr(J)`,
  `a = J @ V`, normalized sensitivity
  `(I-J)^-1 diag(1 / diag((I-J)^-1))`, its representative-seed/domain/dedup
  fixed-point workflow, and its mean/absolute gene-ranking reductions. Each
  correspondence and source path will be recorded in `README.md`.

## Implementation

1. Reuse the 20260803 `discover_run_inputs`, real/generated subset loaders,
   diffusion builder, and strict checkpoint loader. Reject every other model
   family or ODE type before analysis and evaluate only `model.ode_model`.
2. Compute velocity, exact divergence, and acceleration in the original gene
   space. Use the exact closed-form Jacobian of the trained
   `HillAfterLinearField` for memory-efficient batches, and verify it against
   `torch.func.jacrev` in tests. Expose autograd Jacobian/JVP paths as an
   independent cross-check.
3. Stream full Jacobians only for the CLI-limited real/generated cells. Save
   aggregate mean/mean-absolute matrices as compact NPZ plus CSV rankings;
   never save a cell-by-gene-by-gene tensor.
4. Search for fixed points from k-means representative real cells, reject
   non-converged/out-of-domain solutions, and deduplicate them. Use full
   eigenvalues only in small dimensions; otherwise use sparse rightmost and
   leftmost eigenvalues. Mark unresolved/non-hyperbolic stability explicitly.
5. Fit PCA on real expression only. Project states with PCA transform and
   vectors linearly with `components @ V`; use PCA only for figures.
6. Write strict-finite CSV/NPZ/PNG outputs and an atomic manifest in a new
   timestamped output directory by default. Sensitivity is opt-in and guarded
   by a maximum dimension because dense inversion is cubic in gene count.
7. Run unit tests plus a post-hoc smoke test on a tiny synthetic run fixture
   created without training or sampling, then document commands and artifacts.
