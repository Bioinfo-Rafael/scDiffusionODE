"""Training-only diffusion facade for the unchanged legacy TrainLoop."""

from __future__ import annotations

from typing import Mapping, Optional

import torch

from .objectives import per_dimension_elbo_terms
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
        self._training_context: Optional[dict[str, float | int]] = None
        self._last_loss_record: Optional[dict[str, float | int]] = None

    def set_training_context(self, *, training_step: int, learning_rate: float) -> None:
        """Attach work-local logging metadata for the next loss evaluation."""

        self._training_context = {
            "training_step": int(training_step),
            "learning_rate": float(learning_rate),
        }

    def consume_loss_record(self) -> Optional[dict[str, float | int]]:
        record = self._last_loss_record
        self._last_loss_record = None
        self._training_context = None
        return record

    def _capture_loss_record(self, losses: Mapping[str, torch.Tensor]) -> None:
        if self._training_context is None:
            return
        fields = (
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
        record: dict[str, float | int] = dict(self._training_context)
        for name in fields:
            value = losses[name].detach().mean()
            if name == "dimension":
                record[name] = int(value.cpu().item())
            else:
                record[name] = float(value.cpu().item())
        self._last_loss_record = record

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
        grn_penalty_raw = components["grn_penalty_raw"].expand(batch_size)
        grn_penalty_weight = components["grn_penalty_weight"].expand(batch_size)
        regularization = components["forward_regularization"].expand(batch_size)
        if exact:
            path_loss_raw = components["weighted_mismatch"]
            terminal_kl_raw = components["terminal_kl"]
            boundary_nll_raw = components["boundary_nll"]
            path_after_duration = self.time_map.duration * path_loss_raw
            if self.normalize_elbo_by_dimension:
                path, terminal, boundary = per_dimension_elbo_terms(
                    path_after_duration,
                    terminal_kl_raw,
                    boundary_nll_raw,
                    dimension=dimension,
                )
            else:
                path, terminal, boundary = (
                    path_after_duration,
                    terminal_kl_raw,
                    boundary_nll_raw,
                )
        else:
            path_loss_raw = components["plain_epsilon_mse"]
            terminal_kl_raw = components["terminal_kl"]
            boundary_nll_raw = components["boundary_nll"]
            path_after_duration = path_loss_raw
            path = components["plain_epsilon_mse"]
            terminal = (
                self.surrogate_terminal_kl_weight
                * terminal_kl_raw
                / float(dimension)
            )
            boundary = path.new_zeros(path.shape)
        paper_elbo = path + terminal + boundary
        total = paper_elbo + regularization
        repeated_time = total.new_full(total.shape, float(physical_time.detach().cpu()))
        repeated_timestep = t.to(device=total.device, dtype=total.dtype)
        repeated_dimension = total.new_full(total.shape, float(dimension))
        losses = {
            "loss": total,
            "total_loss": total,
            "path_loss": path,
            "terminal_kl": terminal,
            "boundary_nll": boundary,
            "forward_regularization": regularization,
            "path_loss_raw": path_loss_raw,
            "terminal_kl_raw": terminal_kl_raw,
            "boundary_nll_raw": boundary_nll_raw,
            "path_after_duration": path_after_duration,
            "path_final_per_dim": path,
            "terminal_final_per_dim": terminal,
            "boundary_final_per_dim": boundary,
            "paper_elbo_per_dim": paper_elbo,
            "grn_penalty_raw": grn_penalty_raw,
            "grn_penalty_weight": grn_penalty_weight,
            "grn_penalty_final_weighted": regularization,
            "plain_epsilon_mse": components["plain_epsilon_mse"],
            "sampled_physical_time": repeated_time,
            "fractional_diffusion_timestep": repeated_timestep,
            "dimension": repeated_dimension,
        }
        self._capture_loss_record(losses)
        return losses

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
