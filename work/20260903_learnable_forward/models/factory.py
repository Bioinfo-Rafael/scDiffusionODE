"""Factory for the isolated dense learnable-forward experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
from torch import nn

from guided_diffusion.cell_model import Cell_Unet
from ODE.ode_20260609_mathmlp import build_edge_mask

from diffusion.free_affine import FreeAffineForward
from diffusion.stationary_qd import StationaryQDForward
from diffusion.time_mapping import PhysicalTimeMap
from diffusion.timestep_sampler import BatchSharedPhysicalTimeSampler
from diffusion.training_diffusion import LearnableForwardTrainingDiffusion

from .wrapper import LearnableForwardModel


FORWARD_MODELS = ("stationary_qd", "free_affine")
DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class ExperimentComponents:
    model: LearnableForwardModel
    diffusion: LearnableForwardTrainingDiffusion
    schedule_sampler: BatchSharedPhysicalTimeSampler
    time_map: PhysicalTimeMap


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"cannot interpret {value!r} as bool")


def target_source_mask(
    config: Mapping,
    genes: Sequence[str],
    explicit_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Return a mask aligned with matrix entries ``[target,source]``."""

    if not _bool(config.get("use_grn_mask", False)):
        return None
    dim = len(genes)
    if explicit_mask is not None:
        mask = torch.as_tensor(explicit_mask, dtype=torch.float64)
        if mask.shape != (dim, dim):
            raise ValueError(
                f"explicit target/source mask must have shape {(dim, dim)}, "
                f"got {tuple(mask.shape)}"
            )
        return mask.contiguous()
    path = config.get("edge_tsv_path")
    if not path:
        raise ValueError("edge_tsv_path is required when use_grn_mask=true")
    # Legacy helper returns [source,target]; transpose exactly once here.
    return build_edge_mask(list(genes), path).transpose(0, 1).contiguous()


def validate_training_policy(config: Mapping) -> None:
    """Reject legacy options that invalidate the unchanged-TrainLoop contract."""

    if int(config.get("microbatch", -1)) != -1:
        raise ValueError("dense learnable-forward training requires microbatch=-1")
    if str(config.get("schedule_sampler", "batch_shared_physical_uniform")) != (
        "batch_shared_physical_uniform"
    ):
        raise ValueError(
            "schedule_sampler must be 'batch_shared_physical_uniform'; endpoint "
            "terms cannot be passed through legacy importance weights"
        )
    if _bool(config.get("use_fp16", False)):
        raise ValueError("use_fp16 must be false for dense matrix exponential/Cholesky")
    if str(config.get("timestep_respacing", "")):
        raise ValueError("timestep_respacing is not supported by the training-only diffusion")
    loss_mode = str(config.get("loss_mode", "paper_elbo")).lower()
    if loss_mode == "paper_elbo" and float(config.get("weight_decay", 0.0)) != 0.0:
        raise ValueError(
            "paper_elbo requires weight_decay=0; nonzero AdamW decay is an "
            "additional forward-parameter regularizer"
        )
    if loss_mode == "paper_elbo" and float(config.get("covariance_jitter", 0.0)) != 0.0:
        raise ValueError(
            "paper_elbo requires covariance_jitter=0 because jitter changes q_phi "
            "without changing the declared SDE; use the exact SPD D option for "
            "Model A or label a jittered run as an approximate surrogate"
        )


def build_experiment_components(
    config: Mapping,
    genes: Sequence[str],
    device,
    *,
    denoiser: Optional[nn.Module] = None,
    explicit_mask: Optional[torch.Tensor] = None,
) -> ExperimentComponents:
    validate_training_policy(config)
    dim = len(genes)
    configured_dim = int(config.get("input_dim", dim))
    if configured_dim != dim:
        raise ValueError(
            f"input_dim={configured_dim} does not match gene count {dim}"
        )
    time_map = PhysicalTimeMap.from_named_schedule(
        str(config.get("noise_schedule", "linear")),
        int(config.get("diffusion_steps", 1000)),
    )
    if denoiser is None:
        denoiser = Cell_Unet(
            input_dim=dim,
            hidden_num=list(config.get("hidden_dim", [512, 512, 256, 128])),
            dropout=float(config.get("dropout", 0.0)),
        )

    process = build_forward_process(
        config,
        genes,
        device,
        explicit_mask=explicit_mask,
    )

    model = LearnableForwardModel(denoiser, process).to(device)
    diffusion = LearnableForwardTrainingDiffusion(
        time_map,
        loss_mode=str(config.get("loss_mode", "paper_elbo")),
        normalize_elbo_by_dimension=_bool(
            config.get("normalize_elbo_by_dimension", True)
        ),
        surrogate_terminal_kl_weight=float(
            config.get("surrogate_terminal_kl_weight", 1.0)
        ),
    )
    sampler = BatchSharedPhysicalTimeSampler(time_map)
    return ExperimentComponents(model, diffusion, sampler, time_map)


def build_forward_process(
    config: Mapping,
    genes: Sequence[str],
    device,
    *,
    explicit_mask: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Construct only the dense forward process for read-only analysis."""

    dim = len(genes)
    configured_dim = int(config.get("input_dim", dim))
    if configured_dim != dim:
        raise ValueError(
            f"input_dim={configured_dim} does not match gene count {dim}"
        )
    dtype_name = str(config.get("forward_dtype", "float64")).lower()
    if dtype_name not in DTYPES:
        raise ValueError(
            f"forward_dtype must be one of {tuple(DTYPES)}, got {dtype_name!r}"
        )
    forward_dtype = DTYPES[dtype_name]
    family = str(config.get("forward_model", "")).lower()
    if family not in FORWARD_MODELS:
        raise ValueError(
            f"forward_model must be one of {FORWARD_MODELS}, got {family!r}"
        )
    mask = target_source_mask(config, genes, explicit_mask)
    common_grn = dict(
        grn_mask_target_source=mask,
        allow_self_edges=_bool(config.get("allow_self_edges", True)),
        grn_penalty_weight=float(config.get("grn_penalty_weight", 0.0)),
        grn_penalty_norm=str(config.get("grn_penalty_norm", "l1")),
    )
    if family == "stationary_qd":
        return StationaryQDForward(
            dim,
            d_parameterization=str(config.get("d_parameterization", "psd")),
            d_diagonal_floor=float(config.get("d_diagonal_floor", 0.0)),
            initial_d_diagonal=float(config.get("initial_d_diagonal", 0.5)),
            covariance_jitter=float(config.get("covariance_jitter", 0.0)),
            **common_grn,
            device=device,
            dtype=forward_dtype,
        )
    return FreeAffineForward(
        dim,
        covariance_jitter=float(config.get("covariance_jitter", 0.0)),
        **common_grn,
        device=device,
        dtype=forward_dtype,
    )


__all__ = [
    "DTYPES",
    "ExperimentComponents",
    "FORWARD_MODELS",
    "build_experiment_components",
    "build_forward_process",
    "target_source_mask",
    "validate_training_policy",
]
