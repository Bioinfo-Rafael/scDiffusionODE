"""Exponential production clipped to the audited clean training expression range."""
import math
import torch
from ..common import module, require

ExpField = module('work.20260803_ODE_hill_exp.models.ode_fields').ExpField


class DataBoundedExpField(ExpField):
    def __init__(self, d, *, clip, **kwargs):
        super().__init__(d, **kwargs)
        low, high = clip['production_min'], clip['production_max']
        require(math.isfinite(low) and math.isfinite(high) and 0 <= low < high,
                'invalid exponential production bounds')
        self.register_buffer('production_bounds', torch.tensor([low, high], dtype=torch.float64))
        # Retain the historical inspection interface, but report the actual bounds.
        self.exp_preactivation_min = math.log(low) if low > 0 else None
        self.exp_preactivation_max = math.log(high)

    def component_outputs(self, x, t=None):
        low, high = self.production_bounds.to(dtype=torch.float32).unbind()
        # Clip before exponentiation so even very large preactivations stay finite.
        z = self.preactivation(x).clamp(max=high.log())
        if self.exp_preactivation_min is not None:
            z = z.clamp(min=low.log())
        production = z.exp().clamp(min=low, max=high)
        return production - self._component_decay(x.float())

    def get_model_info(self):
        info = super().get_model_info()
        info.update(production_min=float(self.production_bounds[0]),
                    production_max=float(self.production_bounds[1]),
                    clip_policy='training_range_margin_nice')
        return info
