"""Opt-in late Hybrid interpolation, retaining source modules and state keys."""

import importlib
import torch


def ode_weight(t, *, device=None, dtype=torch.float32):
    t = torch.as_tensor(t, device=device, dtype=dtype)
    if not torch.isfinite(t).all() or ((t < 0) | (t > 999)).any():
        raise ValueError("Hybrid500 requires original diffusion timestep in [0,999]")
    return (1 - t / 500).clamp_min(0)


class Hybrid500Mixin:
    def _scheduler(self, t, device, dtype):
        return ode_weight(t, device=device, dtype=dtype)

    def _outputs(self, x, t, y=None, *, diagnostics=False):
        x = x.float()
        times = torch.as_tensor(t, device=x.device)
        if times.numel() == 1:
            times = times.reshape(1, 1).expand(len(x), 1)
        else:
            times = times.reshape(len(x), 1)
        weight = self.ode_branch_weight(x, times)
        active = weight[:, 0] > 0
        # Forward skips inactive ODE rows. Explicit diagnostics retain true raw outputs.
        ode_raw = torch.zeros_like(x)
        if active.any() and not diagnostics:
            ode_raw = ode_raw.index_copy(
                0, active.nonzero().flatten(), self.ode_model(x[active], times[active])
            )
        if diagnostics:
            ode_raw = self.ode_model(x, times)
        ml_raw = self.ml_model(x, times, y)
        ode_term = torch.zeros_like(x)
        if active.any():
            ode_term = ode_term.index_copy(
                0, active.nonzero().flatten(), weight[active] * ode_raw[active]
            )
        ml_term = (1 - weight) * ml_raw
        return dict(
            ode_raw=ode_raw,
            ml_raw=ml_raw,
            ode_weight=weight,
            ml_weight=1 - weight,
            ode_contribution=ode_term,
            ml_contribution=ml_term,
            output=ode_term + ml_term,
            ode_evaluated=torch.ones_like(active) if diagnostics else active,
        )

    def branch_outputs(self, x, t, y=None):
        return self._outputs(x, t, y, diagnostics=True)

    get_branch_outputs = branch_outputs

    def forward(self, x, t, y=None):
        return self._outputs(x, t, y)["output"]


_SOURCE_A = importlib.import_module(
    "work.20260803_ODE_hill_exp.models.factory"
).InspectableUnifiedODEMLHybrid
_SOURCE_B = importlib.import_module(
    "work.20260816.models.factory"
).InspectableUnifiedODEMLHybrid


class HillAfterLinear500(Hybrid500Mixin, _SOURCE_A):
    pass


class DirectHill500(Hybrid500Mixin, _SOURCE_B):
    pass


def apply(model):
    if (
        model.hybrid_norm_mode != "none"
        or model.reverse_coef
        or model.regime_gate_mode != "none"
    ):
        raise ValueError("Hybrid500 requires the canonical unnormalized blend")
    types = {_SOURCE_A: HillAfterLinear500, _SOURCE_B: DirectHill500}
    if type(model) not in types:
        raise ValueError("unexpected source Hybrid architecture")
    model.__class__ = types[type(model)]
    return model
