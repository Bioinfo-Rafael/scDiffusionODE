# Model inventory

Audit: 2026-10-10 JST. Local HEAD equals fetched origin/main: `cddb2dc52f187f4f990d06737b2755e1e96d8450`.
Forward equations below were checked in source bodies, not inferred from directory names. `softplus` parameter transforms, matrix orientation, expert counts and cache penalties are distinct. All implemented models keep their original classes in existing directories.

40 model/composition/analysis entries. See `registry.json` for compatibility and executable config names; `audit/source_index.json` for all scanned Python files, hashes, classes, methods and line numbers.

## Simple softplus

- Family/classification: autonomous ODE / model
- Source: `work/20260830/models/ode_fields_20260830.py:292::SimpleSoftplus20260830`
- Forward: `softplus((W x+b)/sqrt(G)) - softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Hill after linear

- Family/classification: autonomous ODE / model
- Source: `work/20260830/models/ode_fields_20260830.py:244::HillAfterLinear20260830`
- Forward: `V*z^2/(K^2+z^2)-delta*x; z=softplus(Wx+b), delta=softplus(raw_delta)+eps`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Centered signed Hill

- Family/classification: autonomous ODE / model
- Source: `work/20260830/models/ode_fields_20260830.py:194::CenteredSignedHill20260830`
- Forward: `b_i+sum_j A_ij*tanh(alpha_ij*(log(max(x_j,eps))-log(theta_ij)))-delta_i*x_i`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Shifted Hill rho

- Family/classification: autonomous ODE / model
- Source: `work/20260830/models/ode_fields_20260830.py:217::ShiftedHillRho20260830`
- Forward: `b_i+sum_j A_ij*expm1(rho_ij)*sigmoid(2*(log(max(x_j,eps))-log(theta_ij)))-delta_i*x_i`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## CellUNet

- Family/classification: denoiser / model
- Source: `guided_diffusion/cell_model.py:45::Cell_Unet`
- Forward: `time-conditioned residual MLP encoder/decoder + additive skips + linear output`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 0; gating: none
- GRN: none; CellUNet: standalone / CellUNet branch
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## GeneODE 4/21

- Family/classification: autonomous ODE / model
- Source: `ODE/ode_20260421_regODEMLratio.py:13::GeneODE`
- Forward: `softplus((x W+b)/sqrt(G))-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: source_target; unweighted off-mask + optional cached ratio; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: mapped TSV + replace constructor sub_gene-index mask with full-order source_target mask; original forward unchanged

## LowRank

- Family/classification: state/time-conditioned field / model
- Source: `ODE/ode_20260609_mathmlp.py:233::LowRankField`
- Forward: `softplus(U(x,t)V(x,t)^T x+b)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: rank=16; gating: MLP factors
- GRN: target_source; unweighted; LowRank caches batch subsample W; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: 9/17 external off_mask_lambda=5; clear LowRank cache between inactive batches

## LinComb

- Family/classification: state/time-conditioned field / model
- Source: `ODE/ode_20260609_mathmlp.py:300::LinCombField`
- Forward: `sum_k a_k(x,t)*softplus(W_k x+b_k)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: K=8 bases/experts; gating: MLP raw coefficients
- GRN: target_source; unweighted; LowRank caches batch subsample W; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: 9/17 external off_mask_lambda=5; clear LowRank cache between inactive batches

## MatSum

- Family/classification: state/time-conditioned field / model
- Source: `ODE/ode_20260609_mathmlp.py:358::MatSumField`
- Forward: `softplus(sum_k a_k(x,t) A_k x+b)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: K=8 bases/experts; gating: MLP raw coefficients
- GRN: target_source; unweighted; LowRank caches batch subsample W; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: 9/17 external off_mask_lambda=5; clear LowRank cache between inactive batches

## LoRA

- Family/classification: state/time-conditioned field / model
- Source: `ODE/ode_20260609_mathmlp.py:404::LoRAField`
- Forward: `softplus((W0+sum_k a_k(x,t) U_k V_k^T)x+b)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: K=8 bases/experts; gating: MLP raw coefficients
- GRN: target_source; unweighted; LowRank caches batch subsample W; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: 9/17 external off_mask_lambda=5; clear LowRank cache between inactive batches

## Configurable LinComb

- Family/classification: gated field / model
- Source: `ODE/ode_20260707_lincomb.py:15::ConfigurableLinCombField`
- Forward: `sum_k a_k*softplus(W_k x+b_k)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: K=8; gating: raw or softmax; sparse(raw) or entropy(softmax)
- GRN: target_source; internal off_mask_lambda plus gate penalties; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Hill after linear experts

- Family/classification: single or gated ODE / model
- Source: `work/20260803_ODE_hill_exp/models/ode_fields.py:367::HillAfterLinearField`
- Forward: `sum_k softmax(g(x,t))_k*[V_k*z_k^2/(K_k^2+z_k^2)-delta*x]`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: gate only; experts: 1 or K=8; gating: none or softmax
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## RACIPE

- Family/classification: single or gated ODE / model
- Source: `work/20260803_ODE_hill_exp/models/ode_fields.py:565::RacipeField`
- Forward: `G_i*product_j[(1-h_j)+exp(r_ij)*h_j]-delta_i*x_i; h=Hill(softplus(x/scale),K); log production clipped [-20,20]`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: gate only; experts: 1 or K=8; gating: none or softmax
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Exponential

- Family/classification: single or gated ODE / model
- Source: `work/20260803_ODE_hill_exp/models/ode_fields.py:476::ExpField`
- Forward: `exp(clamp(Wx+b,-20,20))-delta*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: gate only; experts: 1 or K=8; gating: none or softmax
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Centered Hill K8

- Family/classification: gated direct Hill / model
- Source: `work/20260816/models/ode_fields.py:372::CenteredSignedHillField`
- Forward: `sum_k a_k(x,t) * [b_ki+sum_j A_kij*tanh(alpha_kij*log(xpos_j/theta_kij))-delta_i*x_i]`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: gate only; experts: K=8 enforced; gating: softmax
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Shifted Hill K8

- Family/classification: gated direct Hill / model
- Source: `work/20260816/models/ode_fields.py:415::ShiftedHillRhoField`
- Forward: `sum_k a_k(x,t)*[b_ki+sum_j A_kij*expm1(rho_kij)*Hill(xpos_j,theta_kij,n=2)-delta_i*x_i]`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: gate only; experts: K=8 enforced; gating: softmax
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Centered Hill 8/17

- Family/classification: single direct Hill / source variant
- Source: `work/20260817_singleODE/models/ode_fields.py:347::CenteredSignedHillField`
- Forward: `same direct-Hill equation; explicit K=1, no gate network`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **compatible / not yet implemented**
- Adaptation/reason: 8/30 exact requested source implemented; 8/17 class has distinct state layout/provenance and needs explicit registry key

## Shifted Hill 8/17

- Family/classification: single direct Hill / source variant
- Source: `work/20260817_singleODE/models/ode_fields.py:390::ShiftedHillRhoField`
- Forward: `same direct-Hill equation; explicit K=1, no gate network`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **compatible / not yet implemented**
- Adaptation/reason: 8/30 exact requested source implemented; 8/17 class has distinct state layout/provenance and needs explicit registry key

## DirectMessageODE

- Family/classification: sparse neural field / model
- Source: `work/20260917/src/fields.py:16::DirectMessageODE`
- Forward: `-softplus(gamma_i)*x_i + sum_(j->i) w_ji*phi(x_i,x_j,t)`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 1; gating: none
- GRN: hard sparse support; absent-edge penalty exactly zero; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## MultiHopGraphFilterODE

- Family/classification: sparse neural field / model
- Source: `work/20260917/src/fields.py:39::MultiHopGraphFilterODE`
- Forward: `f_theta(x_i,(Px)_i,(P^2x)_i,(P^3x)_i,t); P row-normalized incoming GRN`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 1; gating: none
- GRN: hard sparse support; absent-edge penalty exactly zero; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Original GeneODE

- Family/classification: historical field / model
- Source: `ODE/ode_analysis.py:10::GeneODE`
- Forward: `softplus(x_sub W+b)-gamma*x_sub; zero outside selected subgenes`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: source_target; constructor/index/cache policy differs; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately

## Sigmoid GeneODE 11/06

- Family/classification: historical field / model
- Source: `ODE/ode_analysis1106.py:12::GeneODE`
- Forward: `sigmoid((xW+b)*scale)-gamma*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: source_target; constructor/index/cache policy differs; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately

## Normalized GeneODE 4/13

- Family/classification: historical field / model
- Source: `ODE/ode_analysis20260413.py:13::GeneODE`
- Forward: `softplus(xW+b)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: no; experts: 1; gating: none
- GRN: source_target; constructor/index/cache policy differs; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately

## FactorMLP GeneODE 4/16

- Family/classification: historical field / model
- Source: `ODE/ode_analysis20260416.py:75::GeneODE`
- Forward: `softplus(x A(x)B(x)^T/sqrt(rank)+b)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: state only; experts: rank; gating: none
- GRN: source_target; constructor/index/cache policy differs; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately

## Time FactorMLP GeneODE

- Family/classification: historical field / model
- Source: `ODE/ode_analysis20260416.py:367::GeneODE_Time`
- Forward: `time-condition x before A,B and production; decay uses original x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: rank; gating: none
- GRN: source_target; constructor/index/cache policy differs; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately

## MixtureOfSoftplusODE

- Family/classification: historical field / model
- Source: `ODE/ode_analysis20260420.py:184::MixtureOfSoftplusODE`
- Forward: `sum_k softmax(AlphaNet(x,t))_k*softplus(x W_k+b_k)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: optional time; experts: configurable K; gating: none
- GRN: source_target; constructor/index/cache policy differs; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately

## MatrixDictionaryODE

- Family/classification: historical field / model
- Source: `ODE/ode_analysis20260420.py:280::MatrixDictionaryODE`
- Forward: `softplus(x*(sum_k alpha_k A_k)+b)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: optional time; experts: configurable K; gating: none
- GRN: source_target; constructor/index/cache policy differs; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately

## LowRankResidualDictionaryODE

- Family/classification: historical field / model
- Source: `ODE/ode_analysis20260420.py:352::LowRankResidualDictionaryODE`
- Forward: `softplus(xW0+sum_k alpha_k*x U_k V_k^T+b)-softplus(gamma)*x`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: optional time; experts: configurable K/rank; gating: none
- GRN: source_target; constructor/index/cache policy differs; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: needs dedicated full-ID mask/state layout validation and t-less-call adapter for older GeneODE; cannot alias to June fields; hard-mask behavior must be audited separately

## Standard / norm / ratio / scale / TS

- Family/classification: composition / configuration/composition
- Source: `ODE/ode_20260609_hybrid5x3.py:70::UnifiedODEMLHybrid`
- Forward: `r F+(1-r) C; r=1-t/999 or sigmoid((Ts-t)/tau); mode optionally normalizes/scales branches`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: field-dependent; gating: none
- GRN: field-dependent; CellUNet: r F+(1-r) C; r=1-t/999 or sigmoid((Ts-t)/tau); mode optionally normalizes/scales branches
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Hybrid500

- Family/classification: composition / configuration/composition
- Source: `work/20260915_x0predict/models/__init__.py:13::SingleODEHybrid500`
- Forward: `max(0,1-t/500)*F + (1-max(0,1-t/500))*C`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: field-dependent; gating: none
- GRN: field-dependent; CellUNet: max(0,1-t/500)*F + (1-max(0,1-t/500))*C
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Additive Hybrid500

- Family/classification: composition / configuration/composition
- Source: `work/20260916_x0predict_hybrid_additive/models.py:12::AdditiveHybrid500`
- Forward: `C + max(0,1-t/500)*F`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: field-dependent; gating: none
- GRN: field-dependent; CellUNet: C + max(0,1-t/500)*F
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Consistency wrapper

- Family/classification: composition / configuration/composition
- Source: `work/20260830/models/cellunet_ode_regularized_20260830.py:9::CellUNetODERegularized20260830`
- Forward: `output=C; training cache=mean_gene((C-F)^2); eval does not mix F`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: field-dependent; gating: none
- GRN: field-dependent; CellUNet: output=C; training cache=mean_gene((C-F)^2); eval does not mix F
- Direct import: yes; new Mouse status: **implemented**
- Adaptation/reason: direct import; ordered Mouse IDs + explicit mask

## Normalized historical blend

- Family/classification: historical composition / configuration/composition
- Source: `ODE/ode_analysis20260413.py:141::ODE_ML_Hybrid`
- Forward: `norm * (r*unit(F)+(1-r)*unit(C))`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **compatible / not yet implemented**
- Adaptation/reason: direct class can be added; current extension uses explicitly named UnifiedODEMLHybrid modes, not historical checkpoint aliases

## Learned historical scale

- Family/classification: historical composition / configuration/composition
- Source: `ODE/ode_analysis20260413.py:355::ODE_ML_HybridLearnedScale`
- Forward: `s(detached x,F,C,t) * (r*unit(F)+(1-r)*unit(C))`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **compatible / not yet implemented**
- Adaptation/reason: direct class can be added; current extension uses explicitly named UnifiedODEMLHybrid modes, not historical checkpoint aliases

## HybridNormModes

- Family/classification: historical composition / configuration/composition
- Source: `ODE/ode_20260609_hybridnorm.py:38::ODE_ML_HybridNorm`
- Forward: `none: blend; ratio_reg: blend + cached log norm penalty; normed: exp(log_scale)*blend(units)`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **compatible / not yet implemented**
- Adaptation/reason: direct class can be added; current extension uses explicitly named UnifiedODEMLHybrid modes, not historical checkpoint aliases

## MathML Hybrid

- Family/classification: historical composition / configuration/composition
- Source: `ODE/ode_20260609_mathmlp.py:460::MathML_Hybrid`
- Forward: `r F(x,t)+(1-r) C(x,t); cache log norm-ratio penalty`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 1; gating: none
- GRN: target_source; soft off-mask; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **compatible / not yet implemented**
- Adaptation/reason: direct class can be added; current extension uses explicitly named UnifiedODEMLHybrid modes, not historical checkpoint aliases

## FreeAffine forward

- Family/classification: learnable diffusion process / model
- Source: `work/20260903_learnable_forward/diffusion/free_affine.py:74::FreeAffineForward`
- Forward: `dX=((W-0.5I)X+b) ds+dB; dense matrix-exponential Gaussian transition`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: physical diffusion time; experts: not applicable; gating: none
- GRN: forward drift GRN penalty; CellUNet: LearnableForwardModel + CellUNet epsilon denoiser
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: separate experiment family: nonstandard q, ELBO/boundary loss, shared physical timestep and reverse sampler; incompatible with fixed DDPM swap

## Stationary QD forward

- Family/classification: learnable diffusion process / model
- Source: `work/20260903_learnable_forward/diffusion/stationary_qd.py:106::StationaryQDForward`
- Forward: `dX=-(Q+D)X ds+sqrt(2D)dB; Q skew, D positive, low-rank structured operator`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: physical diffusion time; experts: not applicable; gating: none
- GRN: forward drift GRN penalty; CellUNet: LearnableForwardModel + CellUNet epsilon denoiser
- Direct import: yes; new Mouse status: **requires adapter**
- Adaptation/reason: separate experiment family: nonstandard q, ELBO/boundary loss, shared physical timestep and reverse sampler; incompatible with fixed DDPM swap

## Cell classifier

- Family/classification: classifier / auxiliary network
- Source: `guided_diffusion/cell_model.py:124::Cell_classifier`
- Forward: `time-conditioned class logits`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 0; gating: none
- GRN: none; CellUNet: classifier guidance auxiliary
- Direct import: yes; new Mouse status: **not applicable**
- Adaptation/reason: output is class logits, not gene-space field

## CellUNetVelocity

- Family/classification: analysis / analysis
- Source: `work/20260929noneqThermo/cellunet_adapter.py:34::CellUNetVelocity`
- Forward: `V_t=raw START_X CellUNet(clean X,t)`
- Input/output: x: [cells,genes], optional diffusion t → same gene order [cells,genes]
- Time: yes; experts: 0; gating: none
- GRN: none; CellUNet: ODE-only / Hybrid500 / additive
- Direct import: yes; new Mouse status: **not applicable**
- Adaptation/reason: restoration/output interpretation only; model already counted as CellUNet

## Findings that directory names obscure

- 4/16 Transformer directories implement FactorMLP-generated low-rank fields; the inspected code does not contain transformer attention.
- Early 11/06 GeneODE docstrings say exponential production, but `forward` uses sigmoid. 4/21 uses scaled softplus. These are not aliases.
- June LinComb `compute_W` is a visualization proxy: mixing nonlinear expert outputs is not equal to one softplus of a mixed matrix.
- 8/16 enforces eight gated experts; 8/17 explicitly enforces one; 8/30 supplies the four requested autonomous single fields.
- DirectMessage/MultiHop have hard graph support and exactly zero off-support penalty; metadata must not claim an active soft-edge penalty.
- Native CellUNet defines a dropout member but its forward does not call it; no BatchNorm buffers. Freeze is still checked over all state_dict parameters and buffers.
- No named mandatory model source is missing on the inspected main. Git history includes `ed376ec` for 8/17. README-only aliases are not treated as independent implementations; unaudited historical branches are not assumed available.
- Real Mouse data/checkpoints are absent locally. Mouse compatibility means executable adapter + synthetic validation; real coverage/metrics remain unmeasured.
