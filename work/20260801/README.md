# 2026-08-01 Hybrid + LinComb Softmax gate experiments

This suite adds three experiments while reusing the 2026-07-07 LinComb
training, sampling, and visualization implementations.  All generated files
use the same sibling layout as the earlier suite:

```text
work/20260801/
├── configs/
├── runs/<experiment>/<run_id>/
├── samples/<experiment>/<run_id>/
├── viz/<experiment>/<run_id>/
├── logs/<experiment>/<run_id>/
└── results/
```

## Conditions

- `hybrid_softmax_gate`: standard Hybrid blend and a softmax over LinComb MLP
  coefficients only.  `Cell_Unet` is unchanged.
- `hybrid_ts_soft_softmax_gate`: existing `T_s` sigmoid regime gate
  (`gate_tau=20.0`) and the same LinComb softmax gate.
- `hybrid_ts_soft_gentle_tau80_softmax_gate`: the new gentler `T_s` sigmoid
  variant.  It changes **only** `gate_tau` from `20.0` to `80.0`; a larger
  denominator in `sigmoid((T_s - t) / gate_tau)` makes the transition less
  steep while retaining the existing sigmoid policy.

`gate_mode="softmax"` is consumed by `ConfigurableLinCombField` only, so it
does not alter the `Cell_Unet` branch or its parameters.

## Run all conditions

Use the environment containing PyTorch and Scanpy:

```bash
$HOME/miniconda3/envs/scdiffusion/bin/python \
  work/20260801/scripts/run_experiments_20260801.py
```

The command runs each condition in sequence through `train → sample → all
run-dir visualizations`, then creates the three-model integrated UMAP and a
comparison summary.  The summary retains the existing `lincomb__none`
baseline run/checkpoint/sample references inherited from the shared base
configuration.  Resume an interrupted suite with `--skip-existing`.

For a command-only check, use `--dry-run`.  The per-run entrypoint is
`scripts/run_pipeline_20260801.py`.
