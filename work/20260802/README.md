# 2026-08-02 Hybrid + Softmax gate norm-control experiments

This suite adds only the twelve norm-controlled variants of the successful
20260801 Hybrid + Softmax models.  The three `hybrid_norm_mode="none"` runs
remain in `work/20260801` and are reused as the comparison baselines.

## New conditions

For each base model below, the suite trains four variants:

- `ratio_reg`: log L2-norm ratio regularization.  `ratio_reg_weight=5.0`
  reproduces the effective coefficient of the 20260609 Hybrid5x3 matrix,
  whose `ode_reg_lambda=5.0` multiplied `ratio_reg_weight=1.0`.
- `normed_learned_scale`: L2-normalized ODE/Cell_Unet directions with one
  global positive learned scalar.
- `scale_model_x`: normalized directions with a positive scalar predicted
  from the input gene vector.
- `scale_model_ml_emb`: normalized directions with a positive scalar
  predicted from the Cell_Unet intermediate embedding.

The bases are `hybrid_softmax_gate`, `hybrid_ts_soft_softmax_gate`, and
`hybrid_ts_soft_gentle_tau80_softmax_gate`.  Softmax remains restricted to
LinComb coefficients; Cell_Unet is unchanged.  The two TS bases share a
cache so that `T_s` and `lambda_max` are identical across their norm variants.

## Run

```bash
$HOME/miniconda3/envs/scdiffusion/bin/python \
  work/20260802/scripts/run_experiments_20260802.py
```

This launches the 12 conditions sequentially with the 20260801 training
budget (`100000` steps), then runs sampling, all existing run-dir
visualizations, result summaries, and one integrated UMAP over the three
20260801 baselines plus all twelve new conditions.

Use `--dry-run` to inspect commands.  To resume a named batch, repeat the
same `--batch-id <id>` with `--skip-existing`; the launcher uses deterministic
run directories for that batch.
