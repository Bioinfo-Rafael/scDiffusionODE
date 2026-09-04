"""Training-only diffusion facade for the unchanged legacy TrainLoop."""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from .time_mapping import PhysicalTimeMap


LOSS_MODES = ("paper_elbo", "epsilon_surrogate")


class LearnableForwardTrainingDiffusion:
    """Expose ``training_losses`` and ``num_timesteps`` to ``TrainLoop``.

    ``paper_elbo`` means the Eq. (7) MDSM objective truncated to
    ``[delta,T]`` and made into a valid data ELBO with the complete Appendix-I
    / Theorem-3 boundary correction.  The implementation uses its exact
    compact entropy-cancelled form.

    ``epsilon_surrogate`` intentionally reproduces a plain epsilon-prediction
    MSE plus optional terminal matching.  It is not labelled as Eq. (7).
    """

    def __init__(
        self,
        time_map: PhysicalTimeMap,
        *,
        loss_mode: str = "paper_elbo",
        normalize_elbo_by_dimension: bool = True,
        surrogate_terminal_kl_weight: float = 1.0,
    ) -> None:
        self.time_map = time_map
        self.num_timesteps = time_map.num_timesteps
        mode = str(loss_mode).lower()
        if mode not in LOSS_MODES:
            raise ValueError(f"loss_mode must be one of {LOSS_MODES}, got {loss_mode!r}")
        self.loss_mode = mode
        self.normalize_elbo_by_dimension = bool(normalize_elbo_by_dimension)
        self.surrogate_terminal_kl_weight = float(surrogate_terminal_kl_weight)
        if not torch.isfinite(torch.tensor(self.surrogate_terminal_kl_weight)):
            raise ValueError("surrogate_terminal_kl_weight must be finite")
        if self.surrogate_terminal_kl_weight < 0:
            raise ValueError("surrogate_terminal_kl_weight must be nonnegative")

    def _shared_physical_time(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1 or timesteps.numel() == 0:
            raise ValueError("timesteps must be a nonempty one-dimensional tensor")
        physical = self.time_map.fractional_index_to_time(timesteps)
        if not torch.allclose(physical, physical[:1].expand_as(physical), atol=0, rtol=0):
            raise ValueError(
                "LearnableForwardTrainingDiffusion requires one batch-shared timestep"
            )
        return physical[0]

    def training_losses(
        self,
        model,
        x_start: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Optional[Mapping] = None,
        noise: Optional[torch.Tensor] = None,
        boundary_noise: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute per-example losses with the signature expected by TrainLoop."""

        if x_start.ndim != 2:
            raise ValueError(f"x_start must have shape [B,d], got {tuple(x_start.shape)}")
        if t.shape != (x_start.shape[0],):
            raise ValueError(
                f"t must have shape {(x_start.shape[0],)}, got {tuple(t.shape)}"
            )
        physical_time = self._shared_physical_time(t)
        exact = self.loss_mode == "paper_elbo"
        compute_terminal = exact or self.surrogate_terminal_kl_weight > 0.0
        components = model(
            x_start,
            t,
            physical_time=physical_time,
            boundary_time=self.time_map.boundary_time,
            terminal_time=self.time_map.terminal_time,
            path_noise=noise,
            boundary_noise=boundary_noise,
            compute_boundary=exact,
            compute_terminal=compute_terminal,
            model_kwargs={} if model_kwargs is None else model_kwargs,
        )

        batch_size, dimension = x_start.shape
        regularization = components["forward_regularization"].expand(batch_size)
        if exact:
            normalizer = float(dimension) if self.normalize_elbo_by_dimension else 1.0
            path = self.time_map.duration * components["weighted_mismatch"] / normalizer
            terminal = components["terminal_kl"] / normalizer
            boundary = components["boundary_nll"] / normalizer
            total = path + terminal + boundary + regularization
        else:
            path = components["plain_epsilon_mse"]
            terminal = (
                self.surrogate_terminal_kl_weight
                * components["terminal_kl"]
                / float(dimension)
            )
            boundary = path.new_zeros(path.shape)
            total = path + terminal + regularization

        return {
            "loss": total,
            "path_loss": path,
            "terminal_kl": terminal,
            "boundary_nll": boundary,
            "forward_regularization": regularization,
            "plain_epsilon_mse": components["plain_epsilon_mse"],
        }

    # This object intentionally cannot be passed to existing DDPM sampling
    # scripts.  Those use scalar beta posterior identities that are false for
    # the dense forward processes implemented here.
    def p_sample_loop(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "generation requires a custom dense reverse-SDE sampler; this "
            "work-local diffusion is training-only"
        )


__all__ = ["LOSS_MODES", "LearnableForwardTrainingDiffusion"]
