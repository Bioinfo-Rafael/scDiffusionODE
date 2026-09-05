"""Euler--Maruyama sampler for the model-specific reverse SDE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from diffusion.stationary_qd import StationaryQDForward
from diffusion.objectives import boundary_gaussian_parameters, score_from_noise


@dataclass(frozen=True)
class ReverseSampleResult:
    samples: torch.Tensor
    boundary_states: torch.Tensor
    initial_prior: torch.Tensor
    reverse_steps: int
    visited_indices: tuple[int, ...]
    decoder_sampling_mode: str


def reverse_time_indices(num_timesteps: int, stride: int = 1) -> tuple[int, ...]:
    """Return descending physical-grid indices and always include both ends."""

    num_timesteps = int(num_timesteps)
    stride = int(stride)
    if num_timesteps < 2:
        raise ValueError("reverse SDE requires at least two physical-grid points")
    if stride <= 0:
        raise ValueError("reverse stride must be positive")
    values = list(range(num_timesteps - 1, -1, -stride))
    if values[-1] != 0:
        values.append(0)
    return tuple(values)


def reverse_drift(process, states: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """Evaluate ``a(s) score - f(states,s)`` in generative time."""

    if states.shape != score.shape:
        raise ValueError("states and score must have identical shapes")
    if isinstance(process, StationaryQDForward):
        return process.apply_diffusion_covariance(score) - process.drift(states)
    diffusion_covariance = process.diffusion_covariance()
    score_term = score @ diffusion_covariance.transpose(-1, -2)
    return score_term - process.drift(states)


def euler_maruyama_step(
    states: torch.Tensor,
    drift: torch.Tensor,
    diffusion_factor: torch.Tensor,
    delta_tau: float | torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """One positive-generative-time Euler--Maruyama update."""

    if states.shape != drift.shape or states.shape != noise.shape:
        raise ValueError("states, drift, and noise must have identical shapes")
    if diffusion_factor.shape != (states.shape[-1], states.shape[-1]):
        raise ValueError("diffusion_factor has an incompatible shape")
    dt = torch.as_tensor(delta_tau, device=states.device, dtype=states.dtype)
    if dt.numel() != 1 or not bool(torch.isfinite(dt).detach().item()):
        raise ValueError("delta_tau must be one finite scalar")
    if not bool((dt > 0).detach().item()):
        raise ValueError("delta_tau must be positive in generative time")
    stochastic = noise @ diffusion_factor.transpose(-1, -2)
    return states + drift * dt + stochastic * torch.sqrt(dt)


def _cpu_noise(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float64).to(
        device=device, dtype=dtype
    )


def boundary_decode(
    model,
    boundary_states: torch.Tensor,
    *,
    boundary_time: float,
    decoder_sampling_mode: str,
    generator: torch.Generator,
    model_kwargs: Optional[Mapping] = None,
) -> torch.Tensor:
    """Apply the same Appendix-I decoder used by the training objective."""

    mode = str(decoder_sampling_mode).lower()
    if mode not in {"sample", "mean"}:
        raise ValueError("decoder_sampling_mode must be 'sample' or 'mean'")
    process = model.forward_process
    stats = process.transition_stats(boundary_states, boundary_time)
    timesteps = torch.zeros(
        boundary_states.shape[0], device=boundary_states.device, dtype=torch.float32
    )
    epsilon_prediction = model.predict_noise(
        boundary_states, timesteps, model_kwargs
    )
    if isinstance(process, StationaryQDForward):
        model_score = process.conditional_score(stats, epsilon_prediction)
        decoder_mean = process.boundary_mean(stats, boundary_states, model_score)
        if mode == "mean":
            return decoder_mean
        noise = _cpu_noise(tuple(decoder_mean.shape), generator=generator,
                           device=decoder_mean.device, dtype=decoder_mean.dtype)
        return decoder_mean + process.boundary_noise(stats, noise)
    model_score = score_from_noise(stats.cholesky, epsilon_prediction)
    decoder_mean, decoder_root = boundary_gaussian_parameters(
        y_boundary=boundary_states,
        model_score=model_score,
        transition_matrix=stats.transition_matrix,
        affine_shift=stats.affine_shift,
        covariance=stats.covariance,
        cholesky=stats.cholesky,
    )
    if mode == "mean":
        return decoder_mean
    noise = _cpu_noise(
        tuple(decoder_mean.shape),
        generator=generator,
        device=decoder_mean.device,
        dtype=decoder_mean.dtype,
    )
    return decoder_mean + noise @ decoder_root.transpose(-1, -2)


@torch.no_grad()
def sample_reverse_sde(
    model,
    time_map,
    *,
    batch_size: int,
    seed: int,
    reverse_stride: int = 1,
    decoder_sampling_mode: str = "sample",
    model_kwargs: Optional[Mapping] = None,
) -> ReverseSampleResult:
    """Sample ``N(0,I) -> T..delta -> p_theta(x|y_delta)``."""

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    process = model.forward_process
    parameter = next(process.parameters())
    device, dtype = parameter.device, parameter.dtype
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = reverse_time_indices(time_map.num_timesteps, reverse_stride)
    state = _cpu_noise(
        (batch_size, int(process.dim)),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    initial_prior = state.clone()
    grid = time_map.physical_grid.to(device=device, dtype=dtype)
    was_training = model.training
    model.eval()
    try:
        for current_index, next_index in zip(indices[:-1], indices[1:]):
            current_time = grid[current_index]
            next_time = grid[next_index]
            delta_tau = current_time - next_time
            if not bool((delta_tau > 0).detach().item()):
                raise RuntimeError("reverse grid produced a non-positive step")
            stats = process.transition_stats(state, current_time)
            timesteps = torch.full(
                (batch_size,),
                float(current_index),
                device=device,
                dtype=torch.float32,
            )
            epsilon_prediction = model.predict_noise(
                state, timesteps, model_kwargs
            )
            score = (process.conditional_score(stats, epsilon_prediction)
                     if isinstance(process, StationaryQDForward) else
                     score_from_noise(stats.cholesky, epsilon_prediction))
            drift = reverse_drift(process, state, score)
            noise = _cpu_noise(
                tuple(state.shape),
                generator=generator,
                device=device,
                dtype=dtype,
            )
            if isinstance(process, StationaryQDForward):
                state = state + drift * delta_tau + process.diffusion_noise(noise) * delta_tau.sqrt()
            else:
                state = euler_maruyama_step(
                    state,
                    drift,
                    process.diffusion_factor(),
                    delta_tau,
                    noise,
                )

        boundary_states = state
        samples = boundary_decode(
            model,
            boundary_states,
            boundary_time=time_map.boundary_time,
            decoder_sampling_mode=decoder_sampling_mode,
            generator=generator,
            model_kwargs=model_kwargs,
        )
    finally:
        model.train(was_training)
    if samples.shape != (batch_size, int(process.dim)):
        raise RuntimeError("reverse sampler returned an invalid shape")
    if not bool(torch.isfinite(samples).all().detach().item()):
        raise FloatingPointError("reverse sampler produced NaN or Inf")
    return ReverseSampleResult(
        samples=samples,
        boundary_states=boundary_states,
        initial_prior=initial_prior,
        reverse_steps=len(indices) - 1,
        visited_indices=indices,
        decoder_sampling_mode=str(decoder_sampling_mode).lower(),
    )


__all__ = [
    "ReverseSampleResult",
    "boundary_decode",
    "euler_maruyama_step",
    "reverse_drift",
    "reverse_time_indices",
    "sample_reverse_sde",
]
