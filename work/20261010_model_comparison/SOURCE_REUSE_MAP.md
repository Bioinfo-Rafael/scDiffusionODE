# Source reuse map

No copied model implementations exist in this directory. Existing source files remain authoritative; their SHA256 is saved with each run/checkpoint. `audit/protected_files.json` is the initial local-file baseline including existing user changes, not a clean-HEAD checksum.

| Imported implementation | Used by | Adapter responsibility |
|---|---|---|
| `guided_diffusion/cell_model.py::Cell_Unet` | A/C/D/E and extensions | dimension from ordered gene IDs; architecture checks |
| `work/20260830/models/ode_fields_20260830.py::{SimpleSoftplus20260830,HillAfterLinear20260830,CenteredSignedHill20260830,ShiftedHillRho20260830}` | pilot four ODEs | constructor only, explicit target/source mask; no forward rewrite |
| `work/20260915_x0predict/models/__init__.py::{SingleODEHybrid500,FrozenCellUNet,freeze_from_stage1,assert_frozen,optimizer_for,update_ema}` | C/E, shared freeze and EMA | choose composition/trainability, verify complete state hash |
| `work/20260913_2step/models/hybrid500.py::Hybrid500Mixin` (inherited by 9/15 class) | Blend/Additive | original `max(0,1−t/500)` and inactive-row policy |
| `work/20260916_x0predict_hybrid_additive/models.py::AdditiveHybrid500` | C03/C04/E03/E04 | original additive forward; no duplicated equation |
| `work/20260830/models/cellunet_ode_regularized_20260830.py::CellUNetODERegularized20260830` | D01–D04 | train-only consistency cache, sample C only |
| `work/20260830/training/train_loop_20260830.py::loss_components_20260830` | D01–D04 | supply native START_X diffusion loss instead of EPSILON; same soft/consistency assembly |
| `work/20261009_newBenchmark/prepare_data.py::{prepare,select_hvg}` | prepare | scoped import and output bindings into this suite; unchanged validation/HVG ranking |
| `work/20261009_newBenchmark/train.py::{train,Batches,rng_state,restore_rng}` | A01 complete loop; other families data/RNG | IO/checkpoint adaptation; baseline observational hooks |
| `work/20260915_x0predict/training/objectives.py::{training_loss,timestep_sampler,soft_constraint}` | START_X A/B/C/E, OT extension | source objective config; external soft multiplier for unweighted June fields |
| `work/20260913_2step/training/objectives.py::training_loss` | EPSILON MSE/OT extensions | preserves epsilon-to-x0 conversion only in EPSILON OT |
| `guided_diffusion/script_util.py::create_gaussian_diffusion`, `GaussianDiffusion::{training_losses,p_sample_loop_progressive,p_sample}` and `SpacedDiffusion` | all implemented conditions | assert target enum, steps and rescaling; sparse output storage |
| `ODE/ode_20260707_lincomb.py::LinCombOnlyDenoiser` | ODE-only | generic preexisting `ode_model(x,t)` diffusion wrapper (does not introduce LinComb gating) |
| `ODE/ode_20260609_mathmlp.py::{build_edge_mask,LowRankField,LinCombField,MatSumField,LoRAField}` | GRN and extended fields | Mouse symbol→ID table; transpose exactly once; original penalty preserved |
| `ODE/ode_20260421_regODEMLratio.py::GeneODE` | extension | correct full-order constructor mask; keep `x @ W` source/target forward |
| `ODE/ode_20260707_lincomb.py::ConfigurableLinCombField` | gate extensions | original sparse/entropy checks and losses |
| `work/20260803_ODE_hill_exp/models/ode_fields.py::{HillAfterLinearField,RacipeField,ExpField}` | single/expert extensions | no RACIPE/exp approximation |
| `work/20260816/models/ode_fields.py::{CenteredSignedHillField,ShiftedHillRhoField}` | K8 direct Hill extensions | K8 source defaults and gating retained |
| `work/20260917/src/fields.py::{DirectMessageODE,MultiHopGraphFilterODE}`; `models.py::ComparisonHybrid` | graph / LowRank extensions | edges from mapped mask; source LowRank inactive-cache reset |
| `ODE/ode_20260609_hybrid5x3.py::UnifiedODEMLHybrid`; `ode_20260609_scalemodel.py::SimpleScalarScaleModel` | historical composition extensions | choose standard/ratio/norm/scale/TS policy explicitly |
| `work/20260911/src/source_imports.py::umap_core` → `work/20260830/hematopoietic_viz/core.py::{build_sampling_anndata,compute_common_umap}` | generative visualization | original geometry; ID manifest retained separately; no new normalization |
| `work/20260915_x0predict/analysis/distributions.py::diversity`, `analysis/sliced_wasserstein.py::sliced_wasserstein` | collapse/distribution diagnostics | reuse gene-space metrics; plotting only |
| `data_preparation/20261007/benchmark/{prepare,run}.py` | official normalized 3fold | subprocess CLI only; existing isolated Python and VeloEV; no new metrics |

## New code

- `adapters/legacy.py`: scoped `common` binding for old unqualified imports; restore any prior module binding.
- `adapters/data.py`: shared manifest, ID/hash/scale checks, TSV mapping/coverage; source `prepare` retained.
- `adapters/model.py`: class dispatch and composition; source/source-target mask boundary.
- `adapters/objectives.py`: family dispatch and original consistency composer. No Sinkhorn, diffusion, soft-mask or consistency formula copies.
- `adapters/training.py`: artifact/resume/diagnostic orchestration for B–E; imports losses, batches and EMA. A01 delegates the full source loop. This does not replace historical TrainLoop implementations.
- `adapters/checkpoint.py`: trusted bundle format, strict architecture, ID/data/source and final-EMA checks; all Stage2 conditions register one immutable Stage1 identity.
- `adapters/velocity.py`: separate clean-X outputs and proxy metadata; imported native DDPM sampler.
- `adapters/benchmark.py`: exact three-fold command construction, provenance reuse, failure logs.
- `adapters/visualization.py`: log/parameter/distribution plots and scVelo streams using the official saved postprocess embedding; no metric recomputation.
- `adapters/summary.py`: preserve official summaries and all fold values, split comparison charts by protocol/profile/reference.
- `common.py`, `cli.py`, `scripts/*`: configs, local output confinement, provenance, independent pipeline entry points.

## Extension boundaries

PCA OT needs a fixed-PCA/independent-target cache adapter; trajectory OT needs verified diffusion-start caches; kNN needs a new-data graph and row-ID sampler; learnable forward needs a separate process/sampler family; older parameterizations need dedicated state/mask bindings. Their sources and equations are inventoried and retained as candidates. No approximate substitutes are marked implemented. The old 10/08 dual optimizer needs two optimizer states and is not the requested AdamW pilot.
