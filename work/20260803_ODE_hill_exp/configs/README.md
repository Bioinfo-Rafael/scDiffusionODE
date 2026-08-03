# Config structure

`base.json` contains every condition shared by the twelve runs.  Each of the
twelve experiment files contains only `experiment`, `model_family`, `ode_type`,
and (for TS-soft) the existing regime-gate selection.  The launcher deep-merges
the selected experiment over `base.json` and writes the fully resolved result
to the run directory before training.

The only intentional training-budget change from the 20260802 inheritance
chain is `lr_anneal_steps=total_steps=30000`, as required for this suite.
`reference_ts_20260802.json` is a read-only provenance record, not a cache that
is written into an earlier experiment.
