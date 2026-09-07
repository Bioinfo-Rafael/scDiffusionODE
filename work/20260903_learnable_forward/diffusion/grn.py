"""Shared GRN-mask semantics for both dense forward-process families."""

from __future__ import annotations

from typing import Optional

import torch


PENALTY_NORMS = ("l1", "l2")


def effective_grn_mask(
    mask_target_source: Optional[torch.Tensor],
    *,
    dim: int,
    allow_self_edges: bool,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Validate a ``[target, source]`` mask and apply diagonal semantics."""

    if mask_target_source is None:
        return reference.new_empty(0), False
    mask = torch.as_tensor(
        mask_target_source, device=reference.device, dtype=reference.dtype
    )
    if tuple(mask.shape) != (int(dim), int(dim)):
        raise ValueError(
            "grn_mask_target_source must have shape "
            f"{(int(dim), int(dim))}, got {tuple(mask.shape)}"
        )
    if not bool(torch.isfinite(mask).all().detach().item()):
        raise ValueError("grn_mask_target_source must be finite")
    if bool(((mask < 0) | (mask > 1)).any().detach().item()):
        raise ValueError("grn_mask_target_source values must be in [0, 1]")
    mask = mask.clone().contiguous()
    if allow_self_edges:
        mask.diagonal().fill_(1.0)
    return mask, True


def validate_penalty_norm(value: str) -> str:
    norm = str(value).lower()
    if norm not in PENALTY_NORMS:
        raise ValueError(f"grn_penalty_norm must be one of {PENALTY_NORMS}, got {value!r}")
    return norm


def off_mask_penalty(
    matrix_target_source: torch.Tensor,
    mask_target_source: torch.Tensor,
    *,
    norm: str,
) -> torch.Tensor:
    """Return the full-matrix mean outside the effective GRN mask."""

    selected = validate_penalty_norm(norm)
    if mask_target_source.numel() == 0:
        return matrix_target_source.new_zeros(())
    if mask_target_source.shape != matrix_target_source.shape:
        raise ValueError("GRN mask and regularized matrix must have identical shape")
    off_mask = matrix_target_source * (1.0 - mask_target_source)
    return off_mask.abs().mean() if selected == "l1" else off_mask.square().mean()


__all__ = [
    "PENALTY_NORMS",
    "effective_grn_mask",
    "off_mask_penalty",
    "validate_penalty_norm",
]
