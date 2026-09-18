"""Exact June fields, frozen CellUNet, and the inherited additive forward."""
import importlib
import torch
from ODE.ode_20260609_mathmlp import LowRankField, MatSumField, LoRAField, build_edge_mask
from .common import validate_config, finite
from .src.grn import load_grn
from .src.fields import DirectMessageODE, MultiHopGraphFilterODE

old = importlib.import_module("work.20260915_x0predict.models")
base = importlib.import_module("work.20260916_x0predict_hybrid_additive.models")
freeze_from_stage1, assert_frozen = old.freeze_from_stage1, old.assert_frozen
optimizer_for, update_ema = old.optimizer_for, old.update_ema


class ComparisonHybrid(base.AdditiveHybrid500):
    def _outputs(self, x, t, y=None, *, diagnostics=False):
        # Reset dynamic LowRank cache even when all sampled t >= 500.
        self.ode_model._cached_W_sub = None
        branches = super()._outputs(x, t, y, diagnostics=diagnostics)
        p = branches["ode_weight"].clamp(0, 1)
        branches.update(baseline_weight=1-p, hybrid_cell_weight=p,
                        baseline_contribution=(1-p)*branches["ml_raw"],
                        hybrid_cell_contribution=p*branches["ml_raw"])
        # output = C + p*ODE = (1-p)*C + p*C + p*ODE.
        return branches


def build_model(config, genes, *, state=None, graph=None):
    validate_config(config)
    c, d = config, len(genes)
    ml = old.FrozenCellUNet(input_dim=d, hidden_num=c["cell_unet_hidden_num"])
    ml.requires_grad_(False).eval()
    if c["condition"] == "cellunet_only":
        if state is not None:
            ml.load_state_dict(state, strict=True)
        return ml
    kind = c["model_type"]
    if kind in ("lowrank", "matsum", "lora"):
        # June parser emits mask[source,target]; Sept16 fields use W[target,source].
        # Reuse the loader, explicitly transpose exactly as Sept16's factory does.
        mask = (state["ode_model.mask"].clone() if state is not None else
                build_edge_mask(genes, c["edge_tsv_path"]).T.contiguous())
        kw = dict(mask=mask, soft=True, time_dim=c["time_dim"], hidden=c["field_hidden"],
                  dropout=c["field_dropout"], use_decay=c["use_decay"])
        if kind == "lowrank":
            field = LowRankField(d, rank=c["rank"], lowrank_penalty_subsample=c["lowrank_penalty_subsample"], **kw)
        elif kind == "matsum":
            field = MatSumField(d, K=c["K"], **kw)
        else:
            field = LoRAField(d, K=c["K"], rank=c["rank"], **kw)
        field.ratio_reg_weight = 0.
        field.enable_offmask_cache = c["ode_reg_lambda"] > 0
    else:
        if state is not None:
            src, dst = state["ode_model.src"], state["ode_model.dst"]
        else:
            src, dst, _ = graph if graph is not None else load_grn(genes, c["edge_tsv_path"])
        if kind == "direct_message":
            field = DirectMessageODE(d, src, dst, c["graph_hidden"], c["edge_chunk_size"])
        else:
            field = MultiHopGraphFilterODE(d, src, dst, c["graph_hidden"])
    model = ComparisonHybrid(ode_model=field, ml_model=ml, timesteps=1000,
                             hybrid_norm_mode="none", reverse_coef=False, regime_gate_mode="none")
    if state is not None:
        for name, value in state.items():
            finite(name, value)
        model.load_state_dict(state, strict=True)
    return model
