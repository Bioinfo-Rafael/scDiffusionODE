"""Buffered, model-family-independent loss-component CSV logging."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Mapping


LOSS_COMPONENT_FIELDS = (
    "training_step",
    "learning_rate",
    "sampled_physical_time",
    "fractional_diffusion_timestep",
    "dimension",
    "total_loss",
    "path_loss_raw",
    "terminal_kl_raw",
    "boundary_nll_raw",
    "path_after_duration",
    "path_final_per_dim",
    "terminal_final_per_dim",
    "boundary_final_per_dim",
    "paper_elbo_per_dim",
    "grn_penalty_raw",
    "grn_penalty_weight",
    "grn_penalty_final_weighted",
    "plain_epsilon_mse",
)


RECONSTRUCTION_TOLERANCE_SCALE = 1e-6


def reconstruction_tolerance(total: float, reconstructed: float) -> float:
    """Absolute tolerance for ``total_loss`` reconstructing from its parts.

    Shared by the write-time gate below and the read-time re-check in
    ``analysis.loss_analysis.load_loss_components`` so the two can never
    drift apart; a row already accepted here must also be accepted there.
    """

    return RECONSTRUCTION_TOLERANCE_SCALE * max(1.0, abs(total), abs(reconstructed))


def validate_loss_record(record: Mapping[str, float | int]) -> dict[str, float | int]:
    missing = [name for name in LOSS_COMPONENT_FIELDS if name not in record]
    if missing:
        raise KeyError("loss record is missing fields: " + ", ".join(missing))
    normalized = {name: record[name] for name in LOSS_COMPONENT_FIELDS}
    for name, value in normalized.items():
        numeric = float(value)
        if not math.isfinite(numeric):
            raise FloatingPointError(f"non-finite loss record field {name}: {value}")
    if int(normalized["dimension"]) <= 0:
        raise ValueError("loss record dimension must be positive")
    reconstructed = sum(
        float(normalized[name])
        for name in (
            "path_final_per_dim",
            "terminal_final_per_dim",
            "boundary_final_per_dim",
            "grn_penalty_final_weighted",
        )
    )
    total = float(normalized["total_loss"])
    tolerance = reconstruction_tolerance(total, reconstructed)
    if abs(total - reconstructed) > tolerance:
        raise ValueError(
            "total_loss does not reconstruct from final contributions: "
            f"total={total}, reconstructed={reconstructed}"
        )
    return normalized


class LossComponentWriter:
    """Append validated rows in bounded batches and flush at checkpoints."""

    def __init__(self, path: str | Path, *, flush_interval: int = 100) -> None:
        self.path = Path(path)
        self.flush_interval = int(flush_interval)
        if self.flush_interval <= 0:
            raise ValueError("flush_interval must be positive")
        self._rows: list[dict[str, float | int]] = []

    def append(self, record: Mapping[str, float | int]) -> None:
        self._rows.append(validate_loss_record(record))
        if len(self._rows) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists()
        mode = "a" if not write_header else "w"
        with self.path.open(mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(LOSS_COMPONENT_FIELDS))
            if write_header:
                writer.writeheader()
            writer.writerows(self._rows)
            handle.flush()
        self._rows.clear()

    def close(self) -> None:
        self.flush()


__all__ = [
    "LOSS_COMPONENT_FIELDS",
    "LossComponentWriter",
    "reconstruction_tolerance",
    "validate_loss_record",
]
