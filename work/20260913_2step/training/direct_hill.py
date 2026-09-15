"""Exact direct-Hill trajectory adapter: share physical parameters across Euler calls.

Source component_outputs rebuilds full A/theta inside every target chunk.
Parameters are constant throughout one unroll, so transform them once in the
same differentiable graph. Never cache across optimizer updates.
"""

import importlib
import torch
from torch.utils.checkpoint import checkpoint


def prepare_field(ode, backend):
    if backend == "source":
        return ode
    if backend != "shared_parameters_v1":
        raise ValueError("unknown trajectory field backend")
    source = importlib.import_module("work.20260816.models.ode_fields")
    if type(ode) not in (source.CenteredSignedHillField, source.ShiftedHillRhoField):
        return ode
    # Keep physical transforms in the graph, including their exact source guards.
    A, theta, signed, bias, delta = (
        ode.A,
        ode.theta,
        ode.signed_parameter,
        ode.b,
        ode.delta,
    )

    def forward(x, t):
        x = x.float()
        x_pos = ode.positive_regulators(x)
        chunks = []
        for start in range(0, ode.d, min(ode.target_chunk_size, ode.d)):
            stop = min(start + ode.target_chunk_size, ode.d)
            values = (
                A[:, start:stop],
                theta[:, start:stop],
                signed[:, start:stop],
                bias[:, start:stop],
            )
            use_checkpoint = (
                ode.training
                and torch.is_grad_enabled()
                and any(value.requires_grad for value in (x_pos, *values))
            )
            production = (
                checkpoint(ode._chunk_production, x_pos, *values, use_reentrant=False)
                if use_checkpoint
                else ode._chunk_production(x_pos, *values)
            )
            chunks.append(production)
        components = torch.cat(chunks, dim=-1)
        if ode.use_decay:
            components = components - delta.view(1, 1, ode.d) * x[:, None, :]
        # Preserve per-call time embedding, gates/dropout and source diagnostics.
        gate = ode.get_gate_values(x, t)
        ode._cache_gate_regularization(gate["probabilities"])
        return torch.einsum("bk,bkd->bd", gate["coefficients"], components)

    return forward
