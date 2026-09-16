"""Additive composition inside the Hybrid; all fields remain explicit K=1."""
import importlib
import torch
from .common import legacy_config, validate_config

old = importlib.import_module("work.20260915_x0predict.models")
assert_frozen, freeze_from_stage1 = old.assert_frozen, old.freeze_from_stage1
optimizer_for, update_ema = old.optimizer_for, old.update_ema
assert_single_ode = old.assert_single_ode


class AdditiveHybrid500(old.SingleODEHybrid500):
    mode = "additive"
    diagnostic_sink = None

    def _outputs(self, x, t, y=None, *, diagnostics=False):
        if self.mode == "blend":
            return super()._outputs(x, t, y, diagnostics=diagnostics)
        if self.mode != "additive":
            raise ValueError("mode must be additive or blend")
        branches = super()._outputs(x, t, y, diagnostics=diagnostics)
        branches["ml_weight"] = torch.ones_like(branches["ml_weight"])
        branches["ml_contribution"] = branches["ml_raw"]
        branches["output"] = branches["ml_raw"] + branches["ode_contribution"]
        return branches

    def forward(self, x, t, y=None):
        branches = self._outputs(x, t, y, diagnostics=self.diagnostic_sink is not None)
        if self.diagnostic_sink is not None:
            self.diagnostic_sink(t, branches)
        return branches["output"]


def build_model(config, genes, *, state=None):
    validate_config(config)
    model = old.build_model(legacy_config(config), genes, state=state)
    if hasattr(model, "ode_model"):
        model.__class__ = AdditiveHybrid500
        model.mode = "additive"
        assert_single_ode(model.ode_model, config["ode_type"])
    else:
        model.requires_grad_(False).eval()
    return model
