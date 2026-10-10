# Repository audit coverage

34 historical work directories; 496 Python files indexed (excluding datasets, virtual environments, vendor and generated runs).
Each directory below was classified using imports, class/forward/loss/update bodies and its configuration/launcher. Python index is a coverage aid, not a claim that every plotting/test line was manually read.

| Directory | Source finding | Python files |
|---|---|---|
| `20260122_1_ReconstructUMAP` | Sigmoid GeneODE + CellUNet; gene/dataset/sampling-UMAP settings. | 3 |
| `20260123_pancreas` | Same 11/06 classes on pancreas; dataset variant. | 2 |
| `20260215_embryonic` | Same 11/06 sigmoid GeneODE hybrid, EPSILON legacy TrainLoop, 2000 CLI-default anneal steps; embryonic data variant. | 13 |
| `20260413_Normalized_Embryonic` | Softplus GeneODE with positive decay; branch unit normalization. | 6 |
| `20260416_Transformer` | FactorMLP low-rank state-conditioned field, not attention. | 3 |
| `20260420_Normalized_LearnScale` | 4/13 GeneODE + detached-summary SmallScaleNet; learned global branch magnitude. | 5 |
| `20260420_TransformerTimeEmbedding` | GeneODE_Time: time-conditioned input before FactorMLP. | 3 |
| `20260421_MoE` | MixtureOfSoftplus / MatrixDictionary / LowRankResidualDictionary via AlphaNet. | 3 |
| `20260421_RegODEMLratio` | Scaled softplus GeneODE; log norm-ratio cached in off_mask_penalty. | 4 |
| `20260609_Hybrid5x3` | Unified Hybrid across five fields × normalization modes; 30k default; scale-model extension. | 22 |
| `20260609_HybridNormModes` | ratio_reg / none / normed_learned_scale configurations, not three independent ODEs. | 9 |
| `20260609_MathMLPHybrid` | Dynamic LowRank, raw LinComb, MatSum, LoRA; time features; static versus cached penalties. | 10 |
| `20260707_lincomb` | raw/softmax gating, sparse/entropy; TS spectral estimate and sigmoid regime scheduler; TrainLoop reuse. | 15 |
| `20260801` | Launcher/settings for 7/07 TS/reversed coefficients; integrated UMAP and weight-curve analysis. | 5 |
| `20260802` | Normalization-mode launcher reusing 7/07 implementations; no new field equation. | 3 |
| `20260803_ODE_hill_exp` | Single/gated Hill-after-linear, RACIPE, clipped exp; factory and legacy train/sample. | 19 |
| `20260803_distribution` | Distribution analysis and protection verifier; no trainable model. | 3 |
| `20260804_raw_count_lincomb` | Raw-input variants of softplus/Hill/exp K8; imports 8/03 trainer/factory; distinct data protocol. | 12 |
| `20260816` | K8 centered/shifted direct Hill, softmax experts and target/source masks. | 14 |
| `20260817_singleODE` | K1 direct Hill; no MLP or expert gate; source present in main and history. | 14 |
| `20260817_vector_field_analysis` | Existing vector-field adapters, Dynamo reconstruction and erythroid UMAP; analysis only. | 6 |
| `20260821_noeqThermo` | Landscapes and probability-current analysis for existing fields; no denoiser training. | 10 |
| `20260830` | K1 four-field factory, CellUNet-only output + consistency; EPSILON, 100k; detailed loss/grad/UMAP analysis. | 36 |
| `20260903_learnable_forward` | FreeAffine / StationaryQD forward processes, ELBO or epsilon surrogate, custom reverse sampler; separate family. | 43 |
| `20260911` | Posthoc trajectories/breakpoints and independent/joint UMAP for existing checkpoints; no new model. | 22 |
| `20260913_2step` | EPSILON Stage1/Stage2, frozen CellUNet, reconstruction/Sinkhorn/trajectory occupation OT + Hybrid500. | 48 |
| `20260915_x0predict` | START_X Stage1/Stage2, exact 8/30 K1 fields, Hybrid500; native DDPM; gene-space OT. | 24 |
| `20260916_x0predict_hybrid_additive` | Additive wrapper, reconstruction or independent-target PCA OT, envelope gradient; 10k stop/30k anneal. | 30 |
| `20260917` | June LowRank/MatSum/LoRA + DirectMessage/MultiHop; reconstruction or replacing kNN loss; frozen additive branch. | 31 |
| `20260921` | Projection and nearest-neighbor/geometric diagnostics of existing models; no new trainable field. | 5 |
| `20260929noneqThermo` | Raw START_X CellUNet velocity adapter for no-equilibrium thermodynamic analysis. | 4 |
| `20261008_ode_optimizer` | 8/30 loss reused; CellUNet AdamW + ODE SGD; 4 lambdas; optional START_X support. Pilot changes both to AdamW. | 18 |
| `20261009_newBenchmark` | Full-population normalized linear HVG selection, sparse Stage1 START_X, native DDPM, clean-X t49 proxy; old full evaluation. | 8 |
| `20261009_reproduce` | Official stochastic scVelo per raw erythroid fold, official-raw profile; not a full-trained normalized baseline. | 1 |

Per-file symbols/hashes: `source_index.json`. Exact config/launcher values: `work_directory_audit.json`. Existing user edits are listed in `initial_git_status.txt` and included in the local source baseline; no reset/stash was performed.
