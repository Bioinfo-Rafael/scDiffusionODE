"""CellUnet-only denoiser with a train-only ODE consistency cache."""

from __future__ import annotations

import torch
import torch.nn as nn


class CellUNetODERegularized20260830(nn.Module):
    """Return only CellUnet output; evaluate ODE only while training.

    ``ml_model`` and ``ode_model`` intentionally retain the legacy attribute
    names consumed by the shared soft-constraint TrainLoop interface.
    """

    def __init__(self, ml_model: nn.Module, ode_model: nn.Module):
        super().__init__()
        self.ml_model = ml_model
        self.ode_model = ode_model
        self._cached_cell_ode_reg_20260830 = None

    def forward(self, x: torch.Tensor, t, y=None) -> torch.Tensor:
        cell_out = self.ml_model(x, t, y)
        if not self.training:
            self._cached_cell_ode_reg_20260830 = None
            return cell_out

        ode_out = self.ode_model(x, t)
        if tuple(cell_out.shape) != tuple(ode_out.shape):
            raise RuntimeError(
                "CellUnet and ODE outputs must have exactly the same shape; "
                f"got {tuple(cell_out.shape)} and {tuple(ode_out.shape)}"
            )
        if cell_out.shape[0] != x.shape[0]:
            raise RuntimeError("model output batch dimension changed unexpectedly")
        self._cached_cell_ode_reg_20260830 = (
            (cell_out - ode_out).square().reshape(x.shape[0], -1).mean(dim=1)
        )
        return cell_out

    def consistency_penalty_20260830(self) -> torch.Tensor:
        value = self._cached_cell_ode_reg_20260830
        if value is None:
            raise RuntimeError(
                "no Cell-ODE consistency value is cached; call a training-mode "
                "forward pass first"
            )
        if value.ndim != 1:
            raise RuntimeError(f"expected per-sample penalty, got {tuple(value.shape)}")
        return value


__all__ = ["CellUNetODERegularized20260830"]
