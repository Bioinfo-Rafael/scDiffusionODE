"""Differentiable log-domain entropic OT with uniform marginals.

OT = min_P <P,C> + epsilon KL(P || a b^T).
S(x,y) = OT(x,y) - OT(x,x)/2 - OT(y,y)/2.
All iterations are differentiated; no detached transport plan approximation.
"""

import math
import torch
from torch.utils.checkpoint import checkpoint
from ..common import finite


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


def entropic_ot(x, y, *, epsilon, max_iterations, tolerance):
    cost = cost_matrix(x, y)
    log_a, log_b = -math.log(len(x)), -math.log(len(y))
    kernel = -cost / epsilon
    u, v = torch.zeros_like(x[:, 0]), torch.zeros_like(y[:, 0])
    residual = float("inf")

    # Recompute each ten-iteration block during backward. This preserves the
    # differentiated iterations while avoiding an O(iterations * batch²) tape.
    def block(kernel, u, v, count):
        for _ in range(count):
            u = log_a - torch.logsumexp(kernel + v[None, :], dim=1)
            v = log_b - torch.logsumexp(kernel + u[:, None], dim=0)
        return u, v

    iteration = 0
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
                (checked_plan.logsumexp(1).exp() - math.exp(log_a)).abs().max().item(),
                (checked_plan.logsumexp(0).exp() - math.exp(log_b)).abs().max().item(),
            )
        if residual <= tolerance:
            break
    log_plan = kernel + u[:, None] + v[None, :]
    if not math.isfinite(residual) or residual > tolerance:
        raise FloatingPointError(
            f"Sinkhorn did not converge: marginal residual={residual:g}, tolerance={tolerance:g}, iterations={iteration}"
        )
    plan = log_plan.exp()
    loss = (
        (plan * (cost + epsilon * (log_plan - log_a - log_b))).sum()
        - epsilon * plan.sum()
        + epsilon
    )
    finite("entropic OT", loss)
    return loss, {"iterations": iteration, "marginal_residual": residual}


def sinkhorn_divergence(x, y, config, *, return_info=False):
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
    args = dict(epsilon=eps, max_iterations=iters, tolerance=tol)
    x, y = x.double(), y.double()
    xy, a = entropic_ot(x, y, **args)
    xx, b = entropic_ot(x, x, **args)
    yy, c = entropic_ot(y, y, **args)
    value = finite("Sinkhorn divergence", xy - 0.5 * xx - 0.5 * yy)
    info = {"cross": a, "pred_self": b, "real_self": c}
    return (value, info) if return_info else value
