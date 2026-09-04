"""Top-level DDP module owning both denoiser and learnable forward process."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
from torch import nn

from diffusion.objectives import (
    boundary_gaussian_nll,
    score_from_noise,
    standard_normal_kl,
    weighted_noise_quadratic,
)


class LearnableForwardModel(nn.Module):
    """One registered module for optimizer/DDP/EMA/checkpoint compatibility.

    All computations involving forward-process parameters occur inside this
    ``forward`` call.  The work-local diffusion facade never reaches through
    ``DDP.module`` to read parameters, which keeps DDP reducer bookkeeping
    aligned with the actual autograd graph.
    """

    def __init__(self, denoiser: nn.Module, forward_process: nn.Module) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.forward_process = forward_process

    def _denoiser_dtype(self) -> torch.dtype:
        parameter = next(self.denoiser.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    def _denoise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        model_kwargs: Mapping[str, Any],
    ) -> torch.Tensor:
        denoiser_input = noisy.to(dtype=self._denoiser_dtype())
        # Cell_Unet's existing call site supplies [B,1].  Fractional values are
        # supported by the underlying sinusoidal embedding.
        time_input = timesteps.reshape(-1, 1)
        prediction = self.denoiser(denoiser_input, time_input, **model_kwargs)
        if prediction.shape != noisy.shape:
            raise ValueError(
                "denoiser output must match the noisy input shape, got "
                f"{tuple(prediction.shape)} versus {tuple(noisy.shape)}"
            )
        return prediction.to(dtype=noisy.dtype)

    def predict_noise(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        """Public epsilon-prediction hook shared by training and sampling."""

        return self._denoise(
            noisy,
            timesteps,
            {} if model_kwargs is None else dict(model_kwargs),
        )

    @staticmethod
    def _sample(stats, noise: Optional[torch.Tensor]):
        if stats.cholesky is None:
            raise ValueError("transition stats must include a Cholesky factor")
        if noise is None:
            noise = torch.randn_like(stats.mean)
        else:
            if noise.shape != stats.mean.shape:
                raise ValueError(
                    f"noise must have shape {tuple(stats.mean.shape)}, "
                    f"got {tuple(noise.shape)}"
                )
            noise = noise.to(device=stats.mean.device, dtype=stats.mean.dtype)
        noisy = stats.mean + noise @ stats.cholesky.transpose(0, 1)
        return noisy, noise

    def forward(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        physical_time,
        boundary_time,
        terminal_time,
        path_noise: Optional[torch.Tensor] = None,
        boundary_noise: Optional[torch.Tensor] = None,
        compute_boundary: bool = True,
        compute_terminal: bool = True,
        model_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, torch.Tensor]:
        if x_start.ndim != 2:
            raise ValueError(f"x_start must have shape [B,d], got {tuple(x_start.shape)}")
        if timesteps.ndim != 1 or timesteps.shape[0] != x_start.shape[0]:
            raise ValueError(
                f"timesteps must have shape {(x_start.shape[0],)}, "
                f"got {tuple(timesteps.shape)}"
            )
        model_kwargs = {} if model_kwargs is None else dict(model_kwargs)

        path_stats = self.forward_process.transition_stats(x_start, physical_time)
        noisy, sampling_noise = self._sample(path_stats, path_noise)
        noise_prediction = self._denoise(noisy, timesteps, model_kwargs)
        residual = noise_prediction - sampling_noise
        weighted_mismatch = weighted_noise_quadratic(
            path_stats.cholesky, path_stats.diffusion_covariance, residual
        )
        plain_epsilon_mse = residual.square().mean(dim=1)

        batch_size = int(x_start.shape[0])
        zero_vector = weighted_mismatch.new_zeros(batch_size)
        terminal_kl = zero_vector
        if compute_terminal:
            terminal_stats = self.forward_process.transition_stats(
                x_start, terminal_time
            )
            terminal_kl = standard_normal_kl(
                terminal_stats.mean,
                terminal_stats.covariance,
                terminal_stats.cholesky,
            )

        boundary_nll = zero_vector
        if compute_boundary:
            boundary_stats = self.forward_process.transition_stats(
                x_start, boundary_time
            )
            boundary_sample, boundary_sampling_noise = self._sample(
                boundary_stats, boundary_noise
            )
            boundary_timesteps = torch.zeros_like(timesteps)
            boundary_noise_prediction = self._denoise(
                boundary_sample, boundary_timesteps, model_kwargs
            )
            boundary_model_score = score_from_noise(
                boundary_stats.cholesky, boundary_noise_prediction
            )
            boundary_nll = boundary_gaussian_nll(
                x_start=x_start.to(dtype=boundary_stats.mean.dtype),
                y_boundary=boundary_sample,
                model_score=boundary_model_score,
                transition_matrix=boundary_stats.transition_matrix,
                affine_shift=boundary_stats.affine_shift,
                covariance=boundary_stats.covariance,
                cholesky=boundary_stats.cholesky,
            )

        grn_penalty_raw = self.forward_process.grn_penalty_base()
        grn_penalty_weight = grn_penalty_raw.new_tensor(
            self.forward_process.grn_penalty_weight
        )
        regularization = self.forward_process.additional_regularization()
        if regularization.ndim != 0:
            raise ValueError("additional_regularization() must return a scalar")

        return {
            "weighted_mismatch": weighted_mismatch,
            "plain_epsilon_mse": plain_epsilon_mse,
            "terminal_kl": terminal_kl,
            "boundary_nll": boundary_nll,
            "grn_penalty_raw": grn_penalty_raw,
            "grn_penalty_weight": grn_penalty_weight,
            "forward_regularization": regularization,
        }


__all__ = ["LearnableForwardModel"]
