"""Dispatch imported losses; START_X is the only adaptation to 8/30 consistency."""
from __future__ import annotations
from ..common import module, require
import torch
x0 = module('work.20260915_x0predict.training.objectives')
eps = module('work.20260913_2step.training.objectives')
consistency = module('work.20260830.training.train_loop_20260830')


def source_config(c):
    cfg = dict(c, objective='stage1' if c['family']=='baseline' else
               ('ot' if c['objective']=='ot' else 'start_x'), schedule_sampler='uniform')
    if c['soft_weighting'] == 'external_5':
        # June fields and old GeneODE return an unweighted penalty; 9/17 applies 5 outside.
        cfg['ode_reg_lambda'] = c['ode_reg_lambda'] * c['off_mask_lambda']
    return cfg


def sampler(c, d):
    return x0.timestep_sampler(source_config(c), d)


def loss(c, model, d, x, t, weights, noise=None):
    cfg = source_config(c)
    if c['family'] != 'consistency':
        fn = x0.training_loss if c['prediction']=='START_X' else eps.training_loss
        return fn(model, d, x, t, weights, cfg, noise=noise)
    require(c['prediction']=='START_X', 'pilot consistency adapter requires START_X')
    module('work.20260915_x0predict.common').assert_start_x(d)
    terms = d.training_losses(model, x, t, noise=noise)
    ode = model.ode_model
    parts = consistency.loss_components_20260830(
        (terms['loss']*weights).mean(), ode.off_mask_penalty(c['ode_reg_norm']),
        c['ode_reg_lambda'], model.consistency_penalty_20260830(), weights,
        c['consistency_lambda'], ode_offmask_base_raw=ode.off_mask_penalty_base(c['ode_reg_norm']))
    return parts['total_loss'], {k:float(v.detach()) for k,v in parts.items()}
