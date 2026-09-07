# Hematopoietic post-hoc visualization

This package is independent of the running 100k training pipeline. It reads existing
run artifacts and never calls training, sampling, `backward()`, or an optimizer. The
protected `launch.py`, `train.py`, `sample.py`, `analyze.py`, `training/`,
`guided_diffusion/`, and `ODE/` files are unchanged.

## Current Embryonic data inspection

The repository copy corresponding to the configured Embryonic dataset was inspected
before implementation:

- shape: 156,726 cells × 1,024 genes
- superclass column: `Superclass` (capital S)
- superclass values: `Epithelial`, `Erythropoietic`, `Fibroblast`, `Immune`,
  `Neural`, `Neural crest`, `Other`, `Vascular endothelial`, `Vascular perivascular`
- `Erythropoietic`: 36,639 cells
- `Immune`: 8,517 cells
- default hematopoietic union: 45,156 cells
- `gene_name` is present, unique, and ordered across all 1,024 variables

There is no exact `Hematopoietic` superclass. Therefore the default is the observed
union `Erythropoietic + Immune`, following the historical superclass lineage code.
`--superclass` can override this selection. `metadata.json` records the selected
column/values, available values, cell count, and complete celltype counts.

## Artifact and gene provenance

The real-data path is read only from `<run_dir>/exp_config.json -> data_dir`; no legacy
absolute data path is embedded in the new implementation. The generated archive is
resolved from `<run_dir>/samples/*.npz`, and its JSON sidecar must point to both that
archive and the current sampling checkpoint under the same run. Model restoration
reuses `analysis.runner.load_checkpoint`, `analysis.runner.create_diffusion`, the
20260830 factory, and `scripts.common.choose_sampling_checkpoint`.

Current `sample.py` stores only `cell_gen`, not gene names. Ordering is therefore
verified through the strongest available provenance contract, not width alone:

1. ordered and unique real genes come from the configured data's `var["gene_name"]`;
2. the checkpoint model is strictly constructed/restored with that exact list;
3. the sample sidecar proves the sample belongs to this run/checkpoint;
4. `cell_gen`, real X, ODE output, and CellUnet output must all have that exact width;
5. the ordered names, indices, var names, and SHA-256 order hash are saved.

## Sampling UMAP semantics

The existing `cell_gen` is concatenated with real cells selected from
`Erythropoietic + Immune`. Generated cells are labeled only
`Generated (unconditional)`. They do **not** receive a true or inferred
hematopoietic/superclass label. An inferred classifier was intentionally omitted:
the sample artifact carries no biological labels or embedded gene-name provenance,
and avoiding a false biological label is safer than presenting a post-hoc KNN label
as truth.

The saved Embryonic X is already the 1,024-HVG, normalized/logged/scaled training
representation (its var metadata contains the stored mean/std). Generated values use
that same model representation. The visualization therefore does **not** normalize,
log-transform, or scale either matrix again. It runs:

```text
PCA(svd_solver="arpack", n_comps=50)
→ neighbors(n_neighbors=15, n_pcs=40)
→ UMAP(random_state=seed)
```

Small test datasets automatically cap these parameters at valid dimensions.

## Vector-field semantics

One separate PCA/neighbors/UMAP embedding is computed once on real hematopoietic X.
That exact `X_umap` is asserted unchanged while scVelo embeds every field.

- ODE: `model.ode_model(real_X, None)`. All four 20260830 single ODEs are
  time-independent, and this output is the ODE `dx/dt` field.
- CellUnet: `x_t = diffusion.q_sample(real_X, t, fixed_noise)`, followed by
  `model.ml_model(x_t, diffusion._scale_timesteps(t))`. The default is `t=0`.
  Current linear diffusion has `alpha_cumprod[0] < 1`, so t=0 is closest to the clean
  manifold but is not exact clean X. Additional timesteps regenerate the same fixed
  noise stream for the same cells.

The current model predicts epsilon. Thus the CellUnet field is always titled
`CellUnet diffusion-output field`; it is not claimed to be physical or biological
`dx/dt`.

For each field, the historical scVelo pipeline is used:

```text
layers[vkey] = field
layers["X"] = saved training X
velocity_graph(vkey=vkey, xkey="X", backend="loky", n_jobs=32)
velocity_embedding(basis="umap", vkey=vkey)
velocity_embedding_stream / velocity_embedding / velocity_embedding_grid
```

PAGA is attempted when cell/celltype counts are sufficient; a PAGA failure produces
a warning and does not stop the required figures.

## Historical code reuse map

| Concern | Direct source/reuse |
|---|---|
| data/config loading | `work/20260830/scripts/common.py::read_json`; run `exp_config.json` |
| sample/checkpoint loading | `scripts.common.choose_sampling_checkpoint`; `analysis.runner.load_checkpoint` |
| sampling UMAP concept | `work/20260215_embryonic/visualization_analysis.py::create_combined_anndata`, `plot_umap_analysis` |
| superclass selection | `work/20260215_embryonic/20260224_084326_Lamda5/velocity_by_Superclass.py` |
| lineage palette | exact historical `LINEAGES` table loaded by path; its gradient/category logic is reused for the union |
| PCA/neighbors/UMAP | historical `run_velocity_pipeline`: 50 / 15 / 40 |
| shared multi-field embedding | `work/20260215_embryonic/20260330_Analyze-20260224Lamda5/embed_velocity.py` |
| scVelo graph | historical `velocity_graph(xkey="X", backend="loky", n_jobs=32)` |
| stream/arrow/grid | historical `plot_velocity_panels` settings |
| current model/diffusion | `work/20260830/models`, `analysis.runner.create_diffusion/load_checkpoint/select_device` |

## Outputs

```text
<run_dir>/hematopoietic_viz/
├── metadata.json
├── figures/
│   ├── 01_sampling_umap_real_hema_vs_generated.png
│   ├── 02_sampling_umap_celltypes_plus_generated.png
│   ├── 10_ode_velocity_stream.png
│   ├── 11_ode_velocity_arrow.png
│   ├── 12_ode_velocity_grid.png
│   ├── 20_cellunet_t000_velocity_stream.png
│   ├── 21_cellunet_t000_velocity_arrow.png
│   ├── 22_cellunet_t000_velocity_grid.png
│   └── 30_ode_vs_cellunet_stream.png
├── csv/
│   ├── gene_order.csv
│   ├── sampling_umap_coordinates.csv
│   └── vector_field_umap_coordinates.csv
└── h5ad/
    ├── sampling_umap.h5ad
    └── hematopoietic_vector_fields.h5ad
```

Additional requested timesteps receive matching `t249`, `t499`, etc. filenames.

## Commands

Run only after the selected run already has a completed sample:

```bash
python work/20260830/scripts/hematopoietic_viz.py \
  --run-dir work/20260830/runs/01_centered_signed_hill_lambda0p1/<batch-id>
```

Additional timesteps or superclass override:

```bash
python work/20260830/scripts/hematopoietic_viz.py \
  --run-dir work/20260830/runs/01_centered_signed_hill_lambda0p1/<batch-id> \
  --timesteps 0,249,499,749,999 \
  --superclass Erythropoietic --superclass Immune
```

Run all sample-complete conditions and safely skip unfinished/completed runs:

```bash
python work/20260830/scripts/hematopoietic_viz_all.py \
  --batch-id cell-ode-100k-20260831-071721
```

Use `--force` to regenerate. `--no-h5ad` avoids large h5ad output, and `--no-paga`
disables the optional PAGA attempt.
