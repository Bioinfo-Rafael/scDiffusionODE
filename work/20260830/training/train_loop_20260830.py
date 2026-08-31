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
    *,
    ode_offmask_base_raw: th.Tensor | None = None,
) -> dict[str, th.Tensor]:
    """Compose losses while retaining every raw/intermediate/final value."""

    if consistency_per_sample.ndim != 1 or schedule_weights.ndim != 1:
        raise ValueError("consistency and schedule weights must be per-sample vectors")
    if consistency_per_sample.shape != schedule_weights.shape:
        raise ValueError("consistency and schedule weights must have identical shape")
    consistency_raw = consistency_per_sample.mean()
    consistency_sampler_weighted = (consistency_per_sample * schedule_weights).mean()
    ode_base = (
        ode_soft_constraint
        if ode_offmask_base_raw is None
        else ode_offmask_base_raw
    )
    ode_final = ode_soft_constraint * float(ode_reg_lambda)
    consistency_final = (
        consistency_sampler_weighted * float(cell_ode_reg_lambda_20260830)
    )
    total = diffusion_loss + ode_final + consistency_final
    return {
        "diffusion_loss": diffusion_loss,
        "ode_offmask_base_raw": ode_base,
        "ode_offmask_after_internal_lambda": ode_soft_constraint,
        "ode_regularization_final_weighted": ode_final,
        "cell_ode_consistency_raw_20260830": consistency_raw,
        "cell_ode_consistency_sampler_weighted_20260830": consistency_sampler_weighted,
        "cell_ode_consistency_final_weighted_20260830": consistency_final,
        "total_loss": total,
        # Backward-compatible aliases consumed by existing logs/analysis.
        "ode_soft_constraint": ode_soft_constraint,
        "ode_soft_constraint_weighted": ode_final,
        "cell_ode_consistency_20260830": consistency_sampler_weighted,
        "cell_ode_consistency_weighted_20260830": consistency_final,
    }


DETAILED_LOSS_COLUMNS_20260830 = (
    "training_step",
    "diffusion_loss",
    "ode_offmask_base_raw",
    "ode_offmask_after_internal_lambda",
    "ode_regularization_final_weighted",
    "cell_ode_consistency_raw_20260830",
    "cell_ode_consistency_sampler_weighted_20260830",
    "cell_ode_consistency_final_weighted_20260830",
    "total_loss",
    "off_mask_lambda",
    "ode_reg_lambda",
    "cell_ode_reg_lambda_20260830",
    "learning_rate",
    # Legacy aliases retained for analysis of old and new runs with one loader.
    "step",
    "ode_soft_constraint",
    "ode_soft_constraint_weighted",
    "cell_ode_consistency_20260830",
    "cell_ode_consistency_weighted_20260830",
)


class TrainLoop20260830(TrainLoop):
    """Add local loss assembly/logging while retaining optimizer/checkpoints."""

    def __init__(
        self,
        *,
        cell_ode_reg_lambda_20260830: float,
        detailed_loss_flush_interval: int = 100,
        **kwargs,
    ):
        value = float(cell_ode_reg_lambda_20260830)
        if value < 0:
            raise ValueError("cell_ode_reg_lambda_20260830 must be non-negative")
        flush_interval = int(detailed_loss_flush_interval)
        if flush_interval <= 0:
            raise ValueError("detailed_loss_flush_interval must be positive")
        self.cell_ode_reg_lambda_20260830 = value
        self.detailed_loss_flush_interval = flush_interval
        self._detailed_loss_buffer_20260830 = []
        super().__init__(**kwargs)

    def run_loop(self):
        try:
            return super().run_loop()
        finally:
            # Also protects the most recent buffered rows on interruption/error.
            self._flush_detailed_losses_20260830()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        if took_step:
            self._record_detailed_loss_20260830()

    def _anneal_lr(self):
        """Keep linear annealing, with exactly zero LR after the final step."""

        if not self.lr_anneal_steps:
            return
        completed = min(self.step + self.resume_step + 1, self.lr_anneal_steps)
        lr = self.lr * (1.0 - completed / self.lr_anneal_steps)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        aggregate = None
        batch_size = int(batch.shape[0])
        if batch_size <= 0:
            raise ValueError("training batch must be non-empty")
        learning_rate = float(self.opt.param_groups[0]["lr"])
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
            ode_base = diffusion_loss.new_zeros(())
            ode_soft = diffusion_loss.new_zeros(())
            if (
                ode_ref is not None
                and getattr(ode_ref, "soft", False)
                and self.ode_reg_lambda > 0
            ):
                ode_base = ode_ref.off_mask_penalty_base(self.ode_reg_norm)
                ode_soft = ode_base * float(ode_ref.off_mask_lambda)

            consistency = model_ref.consistency_penalty_20260830()
            components = loss_components_20260830(
                diffusion_loss,
                ode_soft,
                self.ode_reg_lambda,
                consistency,
                weights,
                self.cell_ode_reg_lambda_20260830,
                ode_offmask_base_raw=ode_base,
            )
            loss = components["total_loss"]

            # Aggregate by number of samples, never by "last microbatch".
            fraction = float(micro.shape[0]) / float(batch_size)
            detached = {
                name: float(value.detach()) for name, value in components.items()
            }
            if aggregate is None:
                aggregate = {name: 0.0 for name in detached}
            for name, value in detached.items():
                aggregate[name] += fraction * value

            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            for name, value in components.items():
                logger.logkv(name, float(value.detach()))
            self.mp_trainer.backward(loss)
            self._assert_finite_20260830(loss, losses, weights)

        if aggregate is None:
            raise RuntimeError("no microbatch was processed")
        training_step = self.step + self.resume_step + 1
        legacy_step = self.step + self.resume_step
        model_ref = getattr(self.ddp_model, "module", self.ddp_model)
        off_mask_lambda = float(model_ref.ode_model.off_mask_lambda)
        self._current_loss_info = {
            "step": legacy_step,
            "original_loss": aggregate["diffusion_loss"],
            "reg_value": aggregate["ode_soft_constraint"],
            "reg_weighted": aggregate["ode_soft_constraint_weighted"],
            "total_loss": aggregate["total_loss"],
            "ode_reg_lambda": self.ode_reg_lambda,
        }
        self._current_loss_components_20260830 = {
            "training_step": training_step,
            **aggregate,
            "off_mask_lambda": off_mask_lambda,
            "ode_reg_lambda": float(self.ode_reg_lambda),
            "cell_ode_reg_lambda_20260830": self.cell_ode_reg_lambda_20260830,
            "learning_rate": learning_rate,
            "step": legacy_step,
        }

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

    def _record_detailed_loss_20260830(self):
        if dist_util.get_rank() != 0 or not self.save_loss_details:
            return
        row = self._current_loss_components_20260830
        self._detailed_loss_buffer_20260830.append(
            {name: row[name] for name in DETAILED_LOSS_COLUMNS_20260830}
        )
        if len(self._detailed_loss_buffer_20260830) >= self.detailed_loss_flush_interval:
            self._flush_detailed_losses_20260830()

    def _flush_detailed_losses_20260830(self):
        if dist_util.get_rank() != 0 or not self._detailed_loss_buffer_20260830:
            return
        destination = os.path.join(self.save_dir, self.timestamp, "loss_components_20260830.csv")
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        write_header = not os.path.exists(destination)
        if not write_header:
            with open(destination, encoding="utf-8", newline="") as handle:
                existing = next(csv.reader(handle), [])
            if tuple(existing) != DETAILED_LOSS_COLUMNS_20260830:
                raise RuntimeError(f"incompatible detailed loss CSV header: {destination}")
        with open(destination, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=DETAILED_LOSS_COLUMNS_20260830)
            if write_header:
                writer.writeheader()
            writer.writerows(self._detailed_loss_buffer_20260830)
            handle.flush()
            os.fsync(handle.fileno())
        self._detailed_loss_buffer_20260830.clear()

    def save(self):
        super().save()
        self._flush_detailed_losses_20260830()


__all__ = [
    "DETAILED_LOSS_COLUMNS_20260830",
    "TrainLoop20260830",
    "loss_components_20260830",
]
