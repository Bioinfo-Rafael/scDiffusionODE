"""Reuse exact source architectures; freezing changes mode/optimization only."""

import torch
from guided_diffusion.cell_model import Cell_Unet
from ..common import finite, source_factory, state_hash


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
        factory = source_factory(config)
        # A restored checkpoint includes its mask; no edge TSV is needed.
        construction = dict(config)
        if state is not None:
            construction["use_mask_reg"] = False
        model = factory.build_model_from_config(construction, genes, 1000, "cpu")
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
