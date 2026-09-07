"""Map legacy diffusion indices to the physical SDE clock.

At the VP-compatible initialization both dense forward processes satisfy

    Phi(s) = exp(-s/2) I,  Sigma(s) = (1-exp(-s)) I.

Consequently ``s_t = -log(alpha_bar_t)`` makes their transition agree with
the repository's standard ``q_sample`` at every integer timestep.  Between
integer points we interpolate ``s`` linearly and use fractional timestep
conditioning, which the existing sinusoidal embedding explicitly supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PhysicalTimeMap:
    """Monotone correspondence between fractional indices and physical time."""

    physical_grid: torch.Tensor

    def __post_init__(self) -> None:
        grid = torch.as_tensor(self.physical_grid, dtype=torch.float64).detach().cpu()
        if grid.ndim != 1 or grid.numel() < 2:
            raise ValueError("physical_grid must be a one-dimensional tensor of length >=2")
        if not torch.isfinite(grid).all():
            raise ValueError("physical_grid must be finite")
        if not torch.all(grid[1:] > grid[:-1]):
            raise ValueError("physical_grid must be strictly increasing")
        if not bool(grid[0] > 0):
            raise ValueError("the first physical time must be positive")
        object.__setattr__(self, "physical_grid", grid.contiguous())

    @classmethod
    def from_betas(cls, betas: Sequence[float] | np.ndarray | torch.Tensor):
        beta_tensor = torch.as_tensor(betas, dtype=torch.float64)
        if beta_tensor.ndim != 1 or beta_tensor.numel() < 2:
            raise ValueError("betas must be one-dimensional with length >=2")
        if torch.any((beta_tensor <= 0) | (beta_tensor >= 1)):
            raise ValueError("every beta must lie strictly between zero and one")
        alpha_bar = torch.cumprod(1.0 - beta_tensor, dim=0)
        return cls(-torch.log(alpha_bar))

    @classmethod
    def from_named_schedule(cls, name: str, num_timesteps: int):
        from guided_diffusion.gaussian_diffusion import get_named_beta_schedule

        betas = get_named_beta_schedule(str(name), int(num_timesteps))
        return cls.from_betas(betas)

    @property
    def num_timesteps(self) -> int:
        return int(self.physical_grid.numel())

    @property
    def boundary_time(self) -> float:
        return float(self.physical_grid[0])

    @property
    def terminal_time(self) -> float:
        return float(self.physical_grid[-1])

    @property
    def duration(self) -> float:
        return self.terminal_time - self.boundary_time

    def fractional_index_to_time(self, indices: torch.Tensor) -> torch.Tensor:
        """Piecewise-linear ``index -> s`` mapping on the caller's device."""

        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.float64)
        indices_float = indices.to(dtype=torch.float64)
        if not torch.isfinite(indices_float).all():
            raise ValueError("fractional timestep indices must be finite")
        maximum = self.num_timesteps - 1
        if torch.any((indices_float < 0) | (indices_float > maximum)):
            raise ValueError(f"fractional timestep indices must be in [0,{maximum}]")
        low = torch.floor(indices_float).to(dtype=torch.long)
        high = torch.clamp(low + 1, max=maximum)
        fraction = indices_float - low.to(dtype=indices_float.dtype)
        grid = self.physical_grid.to(device=indices.device)
        return grid[low] + fraction * (grid[high] - grid[low])

    def physical_time_to_fractional_index(self, times: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`fractional_index_to_time` on the interpolated grid."""

        if not torch.is_tensor(times):
            times = torch.as_tensor(times, dtype=torch.float64)
        time_float = times.to(dtype=torch.float64)
        if not torch.isfinite(time_float).all():
            raise ValueError("physical times must be finite")
        grid = self.physical_grid.to(device=times.device)
        if torch.any((time_float < grid[0]) | (time_float > grid[-1])):
            raise ValueError(
                f"physical times must be in [{self.boundary_time},{self.terminal_time}]"
            )
        high = torch.searchsorted(grid, time_float, right=False)
        high = torch.clamp(high, min=1, max=self.num_timesteps - 1)
        low = high - 1
        denominator = grid[high] - grid[low]
        fraction = (time_float - grid[low]) / denominator
        result = low.to(dtype=torch.float64) + fraction
        result = torch.where(time_float == grid[0], torch.zeros_like(result), result)
        return result

    def metadata(self) -> dict[str, float | int]:
        return {
            "num_timesteps": self.num_timesteps,
            "boundary_time": self.boundary_time,
            "terminal_time": self.terminal_time,
            "duration": self.duration,
        }


__all__ = ["PhysicalTimeMap"]
