"""Loss-logging subclass of the unchanged repository TrainLoop."""

from __future__ import annotations

from pathlib import Path

from guided_diffusion import dist_util
from guided_diffusion.train_util import TrainLoop

from .loss_logging import LossComponentWriter


class LearnableForwardTrainLoop(TrainLoop):
    """Add generic component logging without invoking core ODE CSV logic."""

    def __init__(
        self,
        *,
        loss_components_path: str | Path,
        loss_log_flush_interval: int = 100,
        **kwargs,
    ) -> None:
        self._loss_component_writer = (
            LossComponentWriter(
                loss_components_path,
                flush_interval=loss_log_flush_interval,
            )
            if dist_util.get_rank() == 0
            else None
        )
        super().__init__(**kwargs)

    def run_step(self, batch, cond):
        training_step = int(self.step + self.resume_step)
        learning_rate = float(self.opt.param_groups[0]["lr"])
        self.diffusion.set_training_context(
            training_step=training_step,
            learning_rate=learning_rate,
        )
        try:
            super().run_step(batch, cond)
            record = self.diffusion.consume_loss_record()
            if record is None:
                raise RuntimeError("training diffusion did not emit a loss record")
            if self._loss_component_writer is not None:
                self._loss_component_writer.append(record)
        except BaseException:
            self.diffusion.consume_loss_record()
            raise

    def save(self):
        super().save()
        if self._loss_component_writer is not None:
            self._loss_component_writer.flush()

    def run_loop(self):
        try:
            return super().run_loop()
        finally:
            if self._loss_component_writer is not None:
                self._loss_component_writer.close()


__all__ = ["LearnableForwardTrainLoop"]
