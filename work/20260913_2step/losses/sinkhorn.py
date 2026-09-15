"""Differentiable log-domain entropic OT with uniform marginals.

OT = min_P <P,C> + epsilon KL(P || a b^T).
S(x,y) = OT(x,y) - OT(x,x)/2 - OT(y,y)/2.
All iterations are differentiated; no detached transport plan approximation.
"""

import math
import torch
from torch.utils.checkpoint import checkpoint
from ..common import finite


class SinkhornConvergenceError(FloatingPointError):
    """Finite iteration limit reached without satisfying marginal tolerance."""


def cost_matrix(x, y):
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or min(*x.shape, *y.shape) == 0
    ):
        raise ValueError(
            "OT requires nonempty cell-by-gene matrices with equal gene counts"
        )
    finite("OT x", x)
    finite("OT y", y)
    # Matrix multiplication avoids N x N x G allocation. Double precision limits cancellation.
    return (
        (x.square().sum(1)[:, None] + y.square().sum(1)[None, :] - 2 * x @ y.T)
        / x.shape[1]
    ).clamp_min(0)


def epsilon_schedule(target, *, enabled=False, start=1.6, factor=0.5):
    if not math.isfinite(target) or target <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not enabled:
        return [target]
    if (
        not math.isfinite(start)
        or start < target
        or not math.isfinite(factor)
        or not 0 < factor < 1
    ):
        raise ValueError("epsilon scaling requires start>=target and 0<factor<1")
    scales = [float(start)]
    while scales[-1] > target:
        value = max(target, scales[-1] * factor)
        if value >= scales[-1] or len(scales) >= 100:
            raise ValueError("epsilon schedule must terminate in <=100 scales")
        scales.append(value)
    return scales


@torch.no_grad()
def cost_statistics(cost, epsilon):
    flat = cost.detach().reshape(-1)
    mean, std = float(flat.mean()), float(flat.std(unbiased=False))
    median, q90, q99 = [
        float(v) for v in torch.quantile(flat, flat.new_tensor([0.5, 0.9, 0.99]))
    ]
    maximum = float(flat.max())
    return dict(
        cost_mean=mean,
        cost_std=std,
        cost_median=median,
        cost_q90=q90,
        cost_q99=q99,
        cost_max=maximum,
        cost_cv=std / max(mean, 1e-12),
        cost_median_over_epsilon=median / epsilon,
        cost_max_over_epsilon=maximum / epsilon,
    )


def entropic_ot(
    x,
    y,
    *,
    epsilon,
    max_iterations,
    tolerance,
    epsilon_scaling=False,
    epsilon_scaling_start=1.6,
    epsilon_scaling_factor=0.5,
    cost_diagnostics=False,
):
    cost = cost_matrix(x, y)
    log_a, log_b = -math.log(len(x)), -math.log(len(y))
    u, v = torch.zeros_like(x[:, 0]), torch.zeros_like(y[:, 0])
    scales = epsilon_schedule(
        epsilon,
        enabled=epsilon_scaling,
        start=epsilon_scaling_start,
        factor=epsilon_scaling_factor,
    )
    details = []

    symmetric_self = epsilon_scaling and x is y

    def block(kernel, u, v, count):
        for _ in range(count):
            if symmetric_self:
                # Symmetric self-cost: a=b and f=g. Averaging the log scaling
                # with its fixed-point update avoids two-cycle oscillation.
                u = 0.5 * (u + log_a - torch.logsumexp(kernel + u[None, :], dim=1))
                v = u
            else:
                u = log_a - torch.logsumexp(kernel + v[None, :], dim=1)
                v = log_b - torch.logsumexp(kernel + u[:, None], dim=0)
        return u, v

    previous = scales[0]
    for scale in scales:
        # u=f/epsilon, v=g/epsilon. Rescale log scalings to preserve the
        # physical dual potentials f,g across epsilon levels; do not detach.
        u, v = u * (previous / scale), v * (previous / scale)
        kernel = -cost / scale
        iteration, residual = 0, float("inf")
        while iteration < max_iterations:
            count = min(10, max_iterations - iteration)
            if torch.is_grad_enabled() and kernel.requires_grad:
                u, v = checkpoint(block, kernel, u, v, count, use_reentrant=False)
            else:
                u, v = block(kernel, u, v, count)
            iteration += count
            with torch.no_grad():
                checked_plan = kernel + u[:, None] + v[None, :]
                residual = max(
                    (checked_plan.logsumexp(1).exp() - math.exp(log_a))
                    .abs()
                    .max()
                    .item(),
                    (checked_plan.logsumexp(0).exp() - math.exp(log_b))
                    .abs()
                    .max()
                    .item(),
                )
            if residual <= tolerance:
                break
        if not math.isfinite(residual):
            raise FloatingPointError("nonfinite Sinkhorn marginal residual")
        if residual > tolerance:
            raise SinkhornConvergenceError(
                f"Sinkhorn did not converge: epsilon={scale:g}, marginal residual={residual:g}, tolerance={tolerance:g}, iterations={iteration}"
            )
        details.append(
            dict(epsilon=scale, iterations=iteration, marginal_residual=residual)
        )
        previous = scale
    log_plan = kernel + u[:, None] + v[None, :]
    plan = log_plan.exp()
    loss = (
        (plan * (cost + epsilon * (log_plan - log_a - log_b))).sum()
        - epsilon * plan.sum()
        + epsilon
    )
    finite("entropic OT", loss)
    info = dict(
        iterations=sum(d["iterations"] for d in details),
        marginal_residual=residual,
        epsilon_scaling=epsilon_scaling,
        scales=details,
    )
    if cost_diagnostics:
        info["cost_statistics"] = cost_statistics(cost, epsilon)
    return loss, info


def sinkhorn_divergence(x, y, config, *, return_info=False, cost_diagnostics=False):
    if (
        config["cost"] != "mean_squared_gene_distance"
        or config["debias"] is not True
        or config["compute_dtype"] != "float64"
    ):
        raise ValueError(
            "requires debiased OT, mean squared gene cost and float64 solver"
        )
    eps, iters, tol = (
        float(config["epsilon"]),
        int(config["max_iterations"]),
        float(config["tolerance"]),
    )
    if (
        not math.isfinite(eps)
        or eps <= 0
        or iters < 1
        or not math.isfinite(tol)
        or not 0 < tol < 1
    ):
        raise ValueError("invalid Sinkhorn numerical settings")
    args = dict(
        epsilon=eps,
        max_iterations=iters,
        tolerance=tol,
        epsilon_scaling=config.get("epsilon_scaling", False),
        epsilon_scaling_start=config.get("epsilon_scaling_start", 1.6),
        epsilon_scaling_factor=config.get("epsilon_scaling_factor", 0.5),
    )
    x, y = x.double(), y.double()
    xy, a = entropic_ot(x, y, **args, cost_diagnostics=cost_diagnostics)
    xx, b = entropic_ot(x, x, **args)
    yy, c = entropic_ot(y, y, **args)
    value = finite("Sinkhorn divergence", xy - 0.5 * xx - 0.5 * yy)
    info = {"cross": a, "pred_self": b, "real_self": c}
    return (value, info) if return_info else value
