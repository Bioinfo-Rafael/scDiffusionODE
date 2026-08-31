#!/usr/bin/env python3
"""Synthetic diffusion/backprop smoke test for all four ODE variants."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

SUITE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for path in (REPO_ROOT, SUITE_ROOT, SUITE_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import load_experiment_config  # noqa: E402
from guided_diffusion.script_util import create_gaussian_diffusion  # noqa: E402
from models import build_model_from_config  # noqa: E402
from training import loss_components_20260830  # noqa: E402
from training import TrainLoop20260830  # noqa: E402


def main():
    torch.manual_seed(20260830)
    diffusion = create_gaussian_diffusion(steps=1000, noise_schedule="linear")
    genes = [f"g{i}" for i in range(5)]
    for experiment in (
        "01_centered_signed_hill_lambda0p1",
        "04_shifted_hill_rho_lambda0p1",
        "07_hill_after_linear_lambda0p1",
        "10_simple_softplus_lambda0p1",
    ):
        config = load_experiment_config(experiment)
        config["cell_unet_hidden_num"] = [16, 12, 8, 8]
        config["cell_unet_dropout"] = 0.0
        model = build_model_from_config(
            config, genes, diffusion.num_timesteps, mask=torch.zeros(5, 5)
        ).train()
        calls = []
        hook = model.ode_model.register_forward_hook(lambda *args: calls.append(1))
        x = torch.rand(4, 5) + 0.1
        t = torch.tensor([0, 249, 499, 999])
        losses = diffusion.training_losses(model, x, t, model_kwargs={})
        components = loss_components_20260830(
            losses["loss"].mean(),
            model.ode_model.off_mask_penalty("l1"),
            config["ode_reg_lambda"],
            model.consistency_penalty_20260830(),
            torch.ones(4),
            config["cell_ode_reg_lambda_20260830"],
        )
        components["total_loss"].backward()
        if model.ml_model.out2.weight.grad is None or model.ode_model.penalty_parameter().grad is None:
            raise AssertionError("both CellUnet and ODE must receive gradients")
        missing = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise AssertionError(f"unused trainable parameters: {missing}")
        if not torch.isfinite(components["total_loss"]):
            raise AssertionError("non-finite smoke loss")
        before_eval = len(calls)
        model.eval()
        with torch.no_grad():
            output = model(x, t)
        if output.shape != x.shape or len(calls) != before_eval:
            raise AssertionError("eval must return CellUnet shape without calling ODE")
        hook.remove()
        print(f"PASS {experiment}: loss={float(components['total_loss']):.6f}")

    # Exercise inherited optimizer/EMA/checkpoint machinery and the local
    # microbatch loss assembly in one real TrainLoop step.
    from guided_diffusion import logger

    config = load_experiment_config("10_simple_softplus_lambda0p1")
    config["cell_unet_hidden_num"] = [16, 12, 8, 8]
    config["cell_unet_dropout"] = 0.0
    model = build_model_from_config(
        config, genes, diffusion.num_timesteps, mask=torch.zeros(5, 5)
    )

    def batches():
        while True:
            yield torch.rand(4, 5) + 0.1, {}

    with tempfile.TemporaryDirectory() as directory:
        logger.configure(dir=str(Path(directory) / "logs"))
        loop = TrainLoop20260830(
            model=model,
            diffusion=diffusion,
            data=batches(),
            batch_size=4,
            microbatch=2,
            lr=1e-4,
            ema_rate="0.9999",
            log_interval=1,
            save_interval=1,
            resume_checkpoint="",
            schedule_sampler=None,
            weight_decay=1e-4,
            lr_anneal_steps=1,
            model_name="smoke",
            save_dir=directory,
            ode_reg_lambda=1.0,
            ode_reg_norm="l1",
            save_loss_details=True,
            cell_ode_reg_lambda_20260830=0.1,
        )
        loop.run_loop()
        detail = Path(directory) / "smoke" / "loss_components_20260830.csv"
        header = detail.read_text(encoding="utf-8").splitlines()[0]
        required = {
            "diffusion_loss",
            "ode_soft_constraint",
            "cell_ode_consistency_20260830",
            "total_loss",
        }
        if not required.issubset(set(header.split(","))):
            raise AssertionError("component CSV is missing independent loss columns")
        if not (Path(directory) / "smoke" / "model000000.pt").is_file():
            raise AssertionError("inherited checkpoint save did not run")
    print("PASS TrainLoop20260830: microbatch/optimizer/EMA/checkpoint/log CSV")
    print("SMOKE_20260830=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
