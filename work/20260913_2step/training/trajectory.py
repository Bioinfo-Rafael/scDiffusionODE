"""Differentiable ODE-only paths and independently seeded minibatch samplers."""

import math
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from ..common import finite
from ..losses.sinkhorn import sinkhorn_divergence
from .objectives import soft_constraint

SCHEMA = "trajectory_occupation_v1"


def validate_trajectory(c):
    for key in (
        "ode_steps",
        "trajectory_batch_size",
        "real_target_points",
        "samples_per_trajectory",
        "temporal_bins",
        "gradient_diagnostic_interval",
    ):
        if key in c and (not isinstance(c[key], int) or c[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if not math.isfinite(c["ode_dt"]) or c["ode_dt"] <= 0:
        raise ValueError("ode_dt must be finite and positive")
    if c["checkpoint_block"] < 0:
        raise ValueError("checkpoint_block must be >=0 (0 disables checkpointing)")
    if (
        c["samples_per_trajectory"] != c["temporal_bins"]
        or c["temporal_bins"] > c["ode_steps"] + 1
        or c["temporal_sampling"] != "stratified"
        or c["binning_rule"] != "floor_boundaries_last_inclusive"
    ):
        raise ValueError("requires exactly one sample per nonempty temporal bin")


def integrate(ode, start, *, steps, dt, checkpoint_block=10):
    """Return [B,K+1,G]; no diffusion call, no detaching, fixed conditioning t=0."""
    if steps < 1 or checkpoint_block < 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("invalid trajectory integration settings")
    if start.ndim != 2 or min(start.shape) < 1:
        raise ValueError("start must be a nonempty cell-by-gene tensor")
    finite("trajectory start", start)
    terminal = torch.zeros((len(start), 1), device=start.device, dtype=torch.long)

    def block(x, count):
        states = []
        for _ in range(count):
            field = ode(x, terminal)
            if field.shape != x.shape:
                raise ValueError("ODE field shape differs from state")
            x = x + dt * field
            states.append(x)
        return torch.stack(states, dim=1)

    parts = [start[:, None]]
    x = start
    size = checkpoint_block or steps
    for offset in range(0, steps, size):
        count = min(size, steps - offset)
        if checkpoint_block and torch.is_grad_enabled():
            states = checkpoint(block, x, count, use_reentrant=False)
        else:
            states = block(x, count)
        finite("trajectory states", states)
        parts.append(states)
        x = states[:, -1]
    return torch.cat(parts, dim=1)


def temporal_edges(steps, bins):
    if not 1 <= bins <= steps + 1:
        raise ValueError("bins must partition all K+1 states into nonempty intervals")
    return [i * (steps + 1) // bins for i in range(bins + 1)]


def occupation_sample(states, config, rng):
    validate_trajectory(config)
    if states.shape[1] != config["ode_steps"] + 1:
        raise ValueError("trajectory length/config mismatch")
    edges = temporal_edges(config["ode_steps"], config["temporal_bins"])
    times = np.stack(
        [
            rng.integers(lo, hi, size=len(states))
            for lo, hi in zip(edges[:-1], edges[1:])
        ],
        axis=1,
    )
    ids = torch.as_tensor(times, device=states.device)
    samples = states[torch.arange(len(states), device=states.device)[:, None], ids]
    return samples.reshape(-1, states.shape[-1]), times


def independent_batch(matrix, count, seed, step):
    """Independent per-step RNG; no replacement within a batch unless N<count."""
    if count < 1 or matrix.shape[0] < 1:
        raise ValueError("empty sampler")
    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    ids = rng.choice(matrix.shape[0], count, replace=matrix.shape[0] < count)
    rows = matrix[ids]
    rows = rows.toarray() if hasattr(rows, "toarray") else np.asarray(rows)
    return torch.from_numpy(np.array(rows, dtype=np.float32, copy=True)), ids


def gradient_norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return math.sqrt(
        sum(
            float(finite("diagnostic gradient", g).detach().double().square().sum())
            for g in grads
            if g is not None
        )
    )


def trajectory_loss(model, source, real_target, config, *, step):
    c = config["trajectory_ot"]
    validate_trajectory(c)
    if config.get("objective_schema_version") != SCHEMA:
        raise ValueError("incompatible trajectory objective schema")
    if (
        len(source) != c["trajectory_batch_size"]
        or len(real_target) != c["real_target_points"]
    ):
        raise ValueError("trajectory source/real-target batch size mismatch")
    states = integrate(
        model.ode_model,
        source,
        steps=c["ode_steps"],
        dt=c["ode_dt"],
        checkpoint_block=c["checkpoint_block"],
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([c["temporal_sampler_seed"], step])
    )
    points, _ = occupation_sample(states, c, rng)
    diagnostic = step == 1 or step % c["gradient_diagnostic_interval"] == 0
    primary, info = sinkhorn_divergence(
        points, real_target, config["ot"], return_info=True, cost_diagnostics=diagnostic
    )
    soft = soft_constraint(model, config)
    total = finite("trajectory loss", primary + soft)
    values = dict(
        primary=float(primary.detach()),
        trajectory_ot=float(primary.detach()),
        soft=float(soft.detach()),
        total=float(total.detach()),
        grad_ot=None,
        grad_soft=None,
        grad_ratio_ot_to_soft=None,
        parameter_norm=None,
        update_norm=None,
        epsilon_scaling=bool(config["ot"].get("epsilon_scaling", False)),
    )
    for term, detail in info.items():
        values[f"sinkhorn_{term}_iterations"] = detail["iterations"]
        values[f"sinkhorn_{term}_residual"] = detail["marginal_residual"]
    with torch.no_grad():
        norms = states.double().norm(dim=-1)
        values.update(
            trajectory_state_norm_mean=float(norms.mean()),
            trajectory_state_norm_max=float(norms.max()),
            trajectory_displacement_mean=float(
                (states[:, -1] - source).double().norm(dim=1).mean()
            ),
        )
    if diagnostic:
        parameters = [p for p in model.ode_model.parameters() if p.requires_grad]
        values["grad_ot"] = gradient_norm(primary, parameters)
        values["grad_soft"] = gradient_norm(soft, parameters)
        values["grad_ratio_ot_to_soft"] = values["grad_ot"] / (
            values["grad_soft"] + config["division_epsilon"]
        )
        values["parameter_norm"] = math.sqrt(
            sum(float(p.detach().double().square().sum()) for p in parameters)
        )
        values.update(info["cross"]["cost_statistics"])
    return total, values, info, diagnostic
