# Tests and smoke artifacts

Run model/unit checks directly with:

```bash
conda run -n scdiffusion python -m unittest -v \
  work/20260803_ODE_hill_exp/tests/test_models.py \
  work/20260803_ODE_hill_exp/tests/test_analysis_metrics.py
```

The supported full smoke entrypoint is the suite launcher:

```bash
conda run -n scdiffusion python work/20260803_ODE_hill_exp/scripts/launch.py \
  --smoke --batch-id 20260803_smoke
```

It records per-experiment JSON, commands, synthetic fixtures, checkpoints,
samples, and minimal plots below `../smoke_results/<batch-id>/`.  Unit tests use
a tiny shape-compatible CellUnet stand-in for exhaustive matrix checks and also
run one explicit compatibility check against the real imported `Cell_Unet`.
`test_analysis_metrics.py` also fixes the regression where a very large but
finite float32 branch ratio overflowed only while NumPy computed its standard
deviation; aggregation must be float64 and genuinely non-finite inputs must
still fail.
