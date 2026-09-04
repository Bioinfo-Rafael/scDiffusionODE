"""Work-local training extensions; the repository TrainLoop stays untouched."""

from .loss_logging import LOSS_COMPONENT_FIELDS, LossComponentWriter
from .train_loop import LearnableForwardTrainLoop

__all__ = [
    "LOSS_COMPONENT_FIELDS",
    "LearnableForwardTrainLoop",
    "LossComponentWriter",
]
