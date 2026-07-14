# 20260707 all_models_facets

This directory contains one standalone post-hoc script for the 20260707
LinComb experiment matrix. It does not call training, sampling, visualization,
or pipeline code.

The script reads:

```text
work/20260707_lincomb/runs/<model>/<run_id>/exp_config.json
work/20260707_lincomb/samples/<model>/<run_id>/samples.npz
```

For each of the six canonical models, it checks run directories newest first
and selects the latest run where the same `run_id` has both files. If
`manifest.json` exists in the run directory, runs whose sample/sampling status
clearly indicates failure or error are skipped; otherwise the presence of
`exp_config.json` and `samples.npz` is enough for selection.

## Dry Run

```bash
python work/20260707_lincomb/integrated_analysis/run_all_models_facets_0707.py \
  --work_root work/20260707_lincomb \
  --dry_run
```

Dry-run does not load the real h5ad and does not compute UMAP. It reports the
selected run, sample path, generated cell count, and skip reason for each
model.

## Full Run

```bash
python work/20260707_lincomb/integrated_analysis/run_all_models_facets_0707.py \
  --work_root work/20260707_lincomb \
  --per_model_real_cells 50000 \
  --per_model_gen_cells 3000 \
  --seed 0
```

Output:

```text
work/20260707_lincomb/integrated_analysis/outputs/<YYYYMMDD_HHMMSS>/
├── all_models_facets.png
├── selected_runs.json
└── run_config.json
```

No per-model PNGs, integrated UMAPs, AnnData files, checkpoints, or copied
artifacts are created.

## Plot Semantics

- Each panel is an independent UMAP.
- Each panel combines the same real-data subsample with that model's
  generated sample.
- Real cells are colored by `Superclass`, falling back to `Subclass`,
  `ClassAnn`, `celltype`, then `final_annotation`.
- Generated cells are red and drawn over real cells.
- The figure has a single shared legend.
- For six usable models, the layout is 3 columns by 2 rows.
- Panel coordinates are not shared across models because each panel is an
  independent UMAP.
