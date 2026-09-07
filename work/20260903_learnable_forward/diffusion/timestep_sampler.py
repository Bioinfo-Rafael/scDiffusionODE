"""Batch-shared physical-time sampler compatible with the legacy TrainLoop."""

from __future__ import annotations

import torch

from .time_mapping import PhysicalTimeMap


class BatchSharedPhysicalTimeSampler:
    """Sample one physical time per rank and repeat it over the local batch.

    Returned importance weights are exactly one.  The work-local training
    diffusion already multiplies the path term by ``T-delta`` and appends the
    endpoint/static terms.  Non-unit external weights would incorrectly
    reweight those terms in the unchanged legacy ``TrainLoop``.
    """

    def __init__(self, time_map: PhysicalTimeMap) -> None:
        self.time_map = time_map

    def sample(self, batch_size: int, device, start_guide_time=None):
        del start_guide_time
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        low = self.time_map.boundary_time
        high = self.time_map.terminal_time
        physical_time = torch.empty((), device=device, dtype=torch.float64).uniform_(
            low, high
        )
        fractional = self.time_map.physical_time_to_fractional_index(physical_time)
        timesteps = fractional.to(dtype=torch.float32).expand(batch_size).clone()
        weights = torch.ones(batch_size, device=device, dtype=torch.float32)
        return timesteps, weights


__all__ = ["BatchSharedPhysicalTimeSampler"]
