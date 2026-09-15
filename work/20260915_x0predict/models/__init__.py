"""Reuse exact source architectures; freezing changes mode/optimization only."""

import torch
import importlib
from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260609_hybrid5x3 import UnifiedODEMLHybrid
from ..common import finite, source_factory, state_hash

_schedule = importlib.import_module("work.20260913_2step.models.hybrid500")
_fields = importlib.import_module("work.20260830.models.ode_fields_20260830")


class SingleODEHybrid500(_schedule.Hybrid500Mixin, UnifiedODEMLHybrid):
    """Original Hybrid500 arithmetic around one explicit ODE field."""

    def ode_branch_weight(self, x, t):
        weight = self._scheduler(t, x.device, x.dtype).reshape(-1, 1)
        return weight.expand(len(x), 1)


def assert_single_ode(ode, family):
    expected = _fields.ODE_CLASS_BY_TYPE_20260830[family]
    if type(ode) is not expected:
        raise AssertionError(f"{family} requires exact source class {expected.__name__}")
    if ode.is_lincomb is not False or ode.num_components != 1 or ode.num_experts != 1:
        raise AssertionError("all ODE fields must have exactly one component, without expert mixing")
    if ode.gate_mode is not None:
        raise AssertionError("ODE expert gating must be absent")
    # Inspected source fields contain direct Parameters/buffers and no child
    # modules at all. CellUNet is outside this field and is intentionally a NN.
    if any(name for name, _ in ode.named_modules()):
        raise AssertionError("explicit ODE field cannot contain neural submodules")
    forbidden = ("coeff_net", "gate", "gate_network", "time_emb")
    if any(any(token in name for token in forbidden) for name, _ in ode.named_parameters()):
        raise AssertionError("neural gating parameters are forbidden inside ODE fields")
    matrices = ("W",) if family in ("simple_softplus", "hill_after_linear") else (
        "A", "theta", "alpha" if family == "centered_signed_hill" else "rho")
    for name in matrices:
        if tuple(getattr(ode, name).shape) != (ode.d, ode.d):
            raise AssertionError(f"{name} must be [G,G] without an expert axis")
    decay = ode.gamma if family == "simple_softplus" else ode.delta
    if tuple(decay.shape) != (ode.d,):
        raise AssertionError("decay must be [G]")


class FrozenCellUNet(Cell_Unet):
    def train(self, mode=True):
        # Parent Hybrid.train() recursively calls this override.
        return super().train(False)


def build_model(config, genes, *, state=None):
    if config["objective"] == "stage1":
        model = Cell_Unet(
            input_dim=len(genes), hidden_num=config["cell_unet_hidden_num"]
        )
    else:
        if config["source_suite"] != "20260830" or config["K"] != 1 or config["gate_mode"] != "none":
            raise AssertionError("all fields require the 20260830 single-ODE factory")
        factory = source_factory(config)
        # A restored checkpoint includes its mask; no edge TSV is needed.
        construction = dict(config)
        if state is not None:
            construction["use_mask_reg"] = False
        field = factory.build_ode_from_config(construction, genes, "cpu")
        assert_single_ode(field, config["ode_type"])
        model = SingleODEHybrid500(
                ode_model=field,
                ml_model=Cell_Unet(input_dim=len(genes), hidden_num=config["cell_unet_hidden_num"]),
                timesteps=1000,
                hybrid_norm_mode="none", reverse_coef=False, regime_gate_mode="none",
            )
        model.ml_model = FrozenCellUNet(
            input_dim=len(genes), hidden_num=config["cell_unet_hidden_num"]
        )
        model.ml_model.requires_grad_(False).eval()
        if state is not None and "ode_model.mask" in state:
            model.ode_model.mask = state["ode_model.mask"].detach().clone()
    if state is not None:
        for name, value in state.items():
            finite(name, value)
        model.load_state_dict(state, strict=True)
    if hasattr(model, "ode_model"):
        assert_single_ode(model.ode_model, config["ode_type"])
    return model


def freeze_from_stage1(model, state):
    model.ml_model.load_state_dict(state, strict=True)
    model.ml_model.requires_grad_(False).eval()
    expected = state_hash(state)
    assert_frozen(model, expected=expected)
    return expected


def assert_frozen(model, optimizer=None, expected=None):
    if any(p.requires_grad for p in model.ml_model.parameters()):
        raise AssertionError("CellUNet requires gradients")
    if any(m.training for m in model.ml_model.modules()):
        raise AssertionError("CellUNet left eval mode")
    if optimizer is not None:
        frozen_ids = {id(p) for p in model.ml_model.parameters()}
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if frozen_ids & optimized:
            raise AssertionError("optimizer includes CellUNet")
        if optimized != {
            id(p) for p in model.ode_model.parameters() if p.requires_grad
        }:
            raise AssertionError(
                "optimizer must contain exactly the ODE-side parameters"
            )
    if expected is not None and state_hash(model.ml_model) != expected:
        raise AssertionError("frozen CellUNet parameter/buffer hash changed")


def optimizer_for(model, config):
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    if hasattr(model, "ml_model"):
        assert_frozen(model, opt)
    return opt


@torch.no_grad()
def update_ema(ema_state, model, rate):
    # Never interpolate frozen parameters (even identical floats can round).
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    for name, value in model.state_dict().items():
        if name in trainable:
            ema_state[name].mul_(rate).add_(value.detach(), alpha=1 - rate)
        else:
            ema_state[name].copy_(value.detach())
