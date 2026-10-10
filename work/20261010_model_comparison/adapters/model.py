"""Compose source classes. No field, gate, or Hybrid equations are implemented here."""
from __future__ import annotations
import torch
from ..common import module, require

single = module('work.20260915_x0predict.models')
additive = module('work.20260916_x0predict_hybrid_additive.models')
fields = module('work.20260830.models.ode_fields_20260830')
consistency = module('work.20260830.models.cellunet_ode_regularized_20260830')
math_fields = module('ODE.ode_20260609_mathmlp')

# This wrapper already forwards exactly ode_model(x,t), including diffusion's optional y.
ODEOnly = module('ODE.ode_20260707_lincomb').LinCombOnlyDenoiser


def build_field(c, genes, mask, meta=None):
    kind, d = c['ode'], len(genes)
    kw = dict(c.get('field_kwargs', {}))
    if kind in fields.ODE_CLASS_BY_TYPE_20260830:
        return fields.ODE_CLASS_BY_TYPE_20260830[kind](d, mask=mask, soft=True,
            off_mask_lambda=c['off_mask_lambda'], **kw)
    if kind in ('lowrank', 'lincomb', 'matsum', 'lora'):
        cls = dict(lowrank=math_fields.LowRankField, lincomb=math_fields.LinCombField,
                   matsum=math_fields.MatSumField, lora=math_fields.LoRAField)[kind]
        field = cls(d, mask=mask, soft=True, **kw)
    elif kind == 'lincomb_configurable':
        field = module('ODE.ode_20260707_lincomb').ConfigurableLinCombField(
            d, mask=mask, soft=True, off_mask_lambda=c['off_mask_lambda'], **kw)
    elif kind in ('racipe', 'exp', 'hill_after_linear_experts'):
        src = module('work.20260803_ODE_hill_exp.models.ode_fields')
        cls = dict(racipe=src.RacipeField, exp=src.ExpField, hill_after_linear_experts=src.HillAfterLinearField)[kind]
        field = cls(d, mask=mask, soft=True, off_mask_lambda=c['off_mask_lambda'], **kw)
    elif kind in ('centered_hill_experts', 'shifted_hill_experts'):
        src = module('work.20260816.models.ode_fields')
        cls = src.CenteredSignedHillField if kind == 'centered_hill_experts' else src.ShiftedHillRhoField
        field = cls(d, mask=mask, soft=True, off_mask_lambda=c['off_mask_lambda'], **kw)
    elif kind in ('direct_message', 'multihop_graph_filter'):
        src = module('work.20260917.src.fields')
        dst, source = mask.nonzero(as_tuple=True)
        require(len(source)>0, 'graph fields require nonempty graph')
        cls = src.DirectMessageODE if kind == 'direct_message' else src.MultiHopGraphFilterODE
        field = cls(d, source, dst, **kw)
    elif kind == 'geneode':
        require(meta is not None, 'GeneODE needs mapped TSV metadata')
        cls = module('ODE.ode_20260421_regODEMLratio').GeneODE
        field = cls(genes, edge_tsv_path=meta['grn']['mapped_tsv'], soft=True, device='cpu')
        # Source constructor indexes sub_genes into an n*n matrix. Correct the ID-boundary
        # mapping once without changing its source-target equation or parameter layout.
        field.mask = mask.T.contiguous()
    else:
        raise ValueError(f'field needs adapter: {kind}')
    if hasattr(field, 'ratio_reg_weight'):
        field.ratio_reg_weight = c.get('ratio_reg_weight', 0.)
    return field


def build(c, genes, *, mask=None, meta=None, state=None):
    from guided_diffusion.cell_model import Cell_Unet
    kw = dict(input_dim=len(genes), hidden_num=c['cell_unet_hidden_num'])
    # Construct CellUNet before ODE so matched seeds give identical CellUNet initialization.
    cell = None
    if c['family'] != 'ode_only':
        cls = single.FrozenCellUNet if c['freeze_cellunet'] else Cell_Unet
        cell = cls(**kw)
        if c['freeze_cellunet']:
            cell.requires_grad_(False).eval()
    if c['family'] == 'baseline':
        model = cell
    else:
        require(mask is not None and tuple(mask.shape)==(len(genes),len(genes)), 'explicit target/source GRN mask required')
        require(mask.sum()>0, 'zero GRN edges; refusing misleading soft constraint')
        field = build_field(c, genes, mask, meta)
        if c['family'] == 'ode_only':
            model = ODEOnly(field)
        elif c['family'] == 'consistency':
            model = consistency.CellUNetODERegularized20260830(cell, field)
        elif c['hybrid'] in ('blend','additive'):
            cls = single.SingleODEHybrid500 if c['hybrid']=='blend' else additive.AdditiveHybrid500
            # 9/17 clears LowRank's dynamic cache before inactive-timestep batches.
            if c['ode'] == 'lowrank':
                cls = module('work.20260917.models').ComparisonHybrid
            model = cls(ode_model=field, ml_model=cell, timesteps=1000,
                        hybrid_norm_mode='none', reverse_coef=False, regime_gate_mode='none')
            if hasattr(model, 'mode'):
                model.mode = c['hybrid']
        else:
            cls = module('ODE.ode_20260609_hybrid5x3').UnifiedODEMLHybrid
            mode = c['hybrid']
            options = dict(c.get('hybrid_kwargs', {}))
            if mode == 'scale_model':
                options['scale_model'] = module('ODE.ode_20260609_scalemodel').SimpleScalarScaleModel(c['cell_unet_hidden_num'][-1])
            if mode == 'ts_sigmoid':
                options.update(regime_gate_mode='Ts_I_vs_II_III', regime_gate_type='sigmoid')
                mode = 'none'
            if mode == 'standard': mode = 'none'
            model = cls(ode_model=field, ml_model=cell, timesteps=1000, hybrid_norm_mode=mode, **options)
    if state is not None:
        for name, value in state.items():
            require(torch.isfinite(value).all(), f'nonfinite checkpoint tensor: {name}')
        model.load_state_dict(state, strict=True)
    return model


def branches(c, model, x, t):
    """Separate predictions on the identical clean input, with source Hybrid diagnostics."""
    if c['family'] == 'baseline':
        return {'cellunet_raw': model(x, t)}
    if c['family'] == 'ode_only':
        return {'ode_raw': model.ode_model(x, t)}
    if c['family'] == 'consistency':
        return {'cellunet_raw': model.ml_model(x, t), 'ode_raw': model.ode_model(x, t)}
    if hasattr(model, 'branch_outputs'):
        b = model.branch_outputs(x, t)
        return dict(cellunet_raw=b['ml_raw'], ode_raw=b['ode_raw'], hybrid_raw=b['output'])
    return dict(cellunet_raw=model.ml_model(x, t), ode_raw=model.ode_model(x, t), hybrid_raw=model(x, t))
