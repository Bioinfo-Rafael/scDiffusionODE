"""Minimal TrainLoop override for the 20260830 consistency term."""

from __future__ import annotations

import csv
import functools
import os

import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import LossAwareSampler
from guided_diffusion.train_util import TrainLoop, log_loss_dict


def loss_components_20260830(
    diffusion_loss: th.Tensor,
    ode_soft_constraint: th.Tensor,
    ode_reg_lambda: float,
    consistency_per_sample: th.Tensor,
    schedule_weights: th.Tensor,
    cell_ode_reg_lambda_20260830: float,
) -> dict[str, th.Tensor]:
    """Compose weighted losses without changing either existing coefficient."""

    if consistency_per_sample.ndim != 1 or schedule_weights.ndim != 1:
        raise ValueError("consistency and schedule weights must be per-sample vectors")
    if consistency_per_sample.shape != schedule_weights.shape:
        raise ValueError("consistency and schedule weights must have identical shape")
    consistency = (consistency_per_sample * schedule_weights).mean()
    ode_weighted = ode_soft_constraint * float(ode_reg_lambda)
    consistency_weighted = consistency * float(cell_ode_reg_lambda_20260830)
    return {
        "diffusion_loss": diffusion_loss,
        "ode_soft_constraint": ode_soft_constraint,
        "ode_soft_constraint_weighted": ode_weighted,
        "cell_ode_consistency_20260830": consistency,
        "cell_ode_consistency_weighted_20260830": consistency_weighted,
        "total_loss": diffusion_loss + ode_weighted + consistency_weighted,
    }


class TrainLoop20260830(TrainLoop):
    """Inherit optimizer/EMA/checkpoint behavior; override only loss assembly."""

    def __init__(self, *, cell_ode_reg_lambda_20260830: float, **kwargs):
        value = float(cell_ode_reg_lambda_20260830)
        if value < 0:
            raise ValueError("cell_ode_reg_lambda_20260830 must be non-negative")
        self.cell_ode_reg_lambda_20260830 = value
        super().__init__(**kwargs)

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                key: value[i : i + self.microbatch].to(dist_util.dev())
                for key, value in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            diffusion_loss = (losses["loss"] * weights).mean()
            model_ref = getattr(self.ddp_model, "module", self.ddp_model)
            ode_ref = getattr(model_ref, "ode_model", None)
            ode_soft = diffusion_loss.new_zeros(())
            if (
                ode_ref is not None
                and getattr(ode_ref, "soft", False)
                and self.ode_reg_lambda > 0
            ):
                ode_soft = ode_ref.off_mask_penalty(self.ode_reg_norm)

            consistency = model_ref.consistency_penalty_20260830()
            components = loss_components_20260830(
                diffusion_loss,
                ode_soft,
                self.ode_reg_lambda,
                consistency,
                weights,
                self.cell_ode_reg_lambda_20260830,
            )
            loss = components["total_loss"]

            # Preserve the exact schema consumed by the inherited save().
            self._current_loss_info = {
                "step": self.step + self.resume_step,
                "original_loss": float(diffusion_loss.detach()),
                "reg_value": float(ode_soft.detach()),
                "reg_weighted": float(components["ode_soft_constraint_weighted"].detach()),
                "total_loss": float(loss.detach()),
                "ode_reg_lambda": self.ode_reg_lambda,
            }
            self._current_loss_components_20260830 = {
                "step": self.step + self.resume_step,
                **{name: float(value.detach()) for name, value in components.items()},
                "ode_reg_lambda": self.ode_reg_lambda,
                "cell_ode_reg_lambda_20260830": self.cell_ode_reg_lambda_20260830,
            }

            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            for name, value in components.items():
                logger.logkv(name, float(value.detach()))
            self.mp_trainer.backward(loss)
            self._assert_finite_20260830(loss, losses, weights)

    def _assert_finite_20260830(self, loss, losses, weights):
        named = [("total_loss", loss), ("schedule_weights", weights)]
        named.extend((f"diffusion_{key}", value) for key, value in losses.items())
        named.extend((f"parameter_{name}", value) for name, value in self.model.named_parameters())
        named.extend(
            (f"gradient_{name}", value.grad)
            for name, value in self.model.named_parameters()
            if value.grad is not None
        )
        bad = [name for name, value in named if not bool(th.isfinite(value).all())]
        if bad:
            raise FloatingPointError("NaN/Inf detected in: " + ", ".join(bad))

    def save(self):
        super().save()
        if (
            dist_util.get_rank() != 0
            or not self.save_loss_details
            or not hasattr(self, "_current_loss_components_20260830")
        ):
            return
        destination = os.path.join(
            self.save_dir, self.timestamp, "loss_components_20260830.csv"
        )
        row = self._current_loss_components_20260830
        write_header = not os.path.exists(destination)
        with open(destination, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


__all__ = ["TrainLoop20260830", "loss_components_20260830"]
