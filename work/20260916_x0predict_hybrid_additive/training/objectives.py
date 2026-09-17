"""START_X reconstruction or independent-target PCA Sinkhorn gradients."""
import importlib
import math
import torch
from torch.utils.checkpoint import checkpoint
from ..common import assert_start_x, finite

old = importlib.import_module("work.20260915_x0predict.training.objectives")
solver = importlib.import_module("work.20260913_2step.losses.sinkhorn")
timestep_sampler, soft_constraint = old.timestep_sampler, old.soft_constraint


def converged_entropic_ot(prediction, target, config):
    """Existing fixed-epsilon equations, with state retained across budget extensions.

    Keep the legacy cost, KL objective, alternating updates, 10-step convergence
    checks and differentiable checkpoint blocks. Only the budget control differs.
    """
    # Apply the 200-iteration start to immutable campaigns created with 2000 too.
    limit = min(200, int(config["max_iterations"]))
    ceiling = max(limit, int(config.get("retry_max_iterations", 16000)))
    epsilon, tolerance = float(config["epsilon"]), float(config["tolerance"])
    if limit < 1 or not math.isfinite(epsilon) or epsilon <= 0 or not 0 < tolerance < 1:
        raise ValueError("invalid Sinkhorn numerical settings")
    cost = solver.cost_matrix(prediction, target)
    log_a, log_b = -math.log(len(prediction)), -math.log(len(target))
    u, v = torch.zeros_like(prediction[:, 0]), torch.zeros_like(target[:, 0])
    kernel = -cost / epsilon

    def block(kernel, u, v, count):
        for _ in range(count):
            u = log_a - torch.logsumexp(kernel + v[None, :], dim=1)
            v = log_b - torch.logsumexp(kernel + u[:, None], dim=0)
        return u, v

    extensions = []
    iteration, residual = 0, float("inf")
    while iteration < ceiling:
        count = min(10, limit - iteration)
        if torch.is_grad_enabled() and kernel.requires_grad:
            # The block has no random operations: RNG snapshots only add overhead.
            u, v = checkpoint(block, kernel, u, v, count, use_reentrant=False, preserve_rng_state=False)
        else:
            u, v = block(kernel, u, v, count)
        iteration += count
        with torch.no_grad():
            checked_plan = kernel + u[:, None] + v[None, :]
            residual = max(
                (checked_plan.logsumexp(1).exp() - math.exp(log_a)).abs().max().item(),
                (checked_plan.logsumexp(0).exp() - math.exp(log_b)).abs().max().item())
        if not math.isfinite(residual):
            raise FloatingPointError("nonfinite Sinkhorn marginal residual")
        if residual <= tolerance:
            break
        if iteration == limit:
            if limit == ceiling:
                raise solver.SinkhornConvergenceError(
                    f"Sinkhorn did not converge: epsilon={epsilon:g}, marginal residual={residual:g}, "
                    f"tolerance={tolerance:g}, iterations={iteration}")
            limit = min(limit + 200, ceiling)
            extensions.append(dict(iterations=iteration, marginal_residual=residual, next_limit=limit))
            if config.get("verbose", False):
                print(f"[PCA OT] iterations={iteration}, marginal residual={residual:g}, "
                      f"tolerance={tolerance:g}; continuing from current state to {limit}", flush=True)
    log_plan = kernel + u[:, None] + v[None, :]
    plan = log_plan.exp()
    loss = ((plan * (cost + epsilon * (log_plan - log_a - log_b))).sum()
            - epsilon * plan.sum() + epsilon)
    finite("entropic OT", loss)
    info = dict(iterations=iteration, marginal_residual=residual, epsilon_scaling=False,
                scales=[dict(epsilon=epsilon, iterations=iteration, marginal_residual=residual)],
                max_iterations=limit, retry_max_iterations=ceiling, restarts=0,
                budget_extensions=extensions)
    if extensions and config.get("verbose", False):
        print(f"[PCA OT] converged after {iteration} total iterations; marginal residual={residual:g}", flush=True)
    return loss, info


def rectangular_sinkhorn(prediction, target, config):
    if target.requires_grad:
        raise ValueError("cached empirical targets must be fixed")
    if config["objective"] != "sinkhorn_divergence_up_to_target_constant" or config["cost"] != "mean_squared_pca_distance" or config["compute_dtype"] != "float64":
        raise ValueError("unsupported PCA OT definition")
    pred, real = prediction.double(), target.double()
    # The old solver accepts rectangular clouds and divides squared distances by D.
    # No target-target call: OT(real, real)/2 is constant in every model parameter.
    cross, cross_info = converged_entropic_ot(pred, real, config)
    self_cost, self_info = converged_entropic_ot(pred, pred, config)
    loss = finite("Sinkhorn model-dependent part", cross - .5 * self_cost)
    return loss, dict(cross=cross_info, pred_self=self_info,
                     omitted="-0.5*OT(target,target): target-only constant; scalar is NOT full divergence")


def training_loss(model, diffusion, x0, t, weights, config, *, pca=None, target=None, noise=None):
    assert_start_x(diffusion)
    if config["objective"] == "start_x":
        return old.training_loss(model, diffusion, x0, t, weights, config, noise)
    if config["objective"] != "ot" or int(t.min()) < 0 or int(t.max()) > 49:
        raise ValueError("OT requires original t=0..49")
    if pca is None or target is None or not torch.equal(weights, torch.ones_like(weights)):
        raise ValueError("OT needs fixed PCA, independent cached target and uniform timesteps")
    noise = torch.randn_like(x0, dtype=torch.float64) if noise is None else noise
    xt = diffusion.q_sample(x0, t, noise=noise)
    prediction = model(xt, diffusion._scale_timesteps(t).unsqueeze(1))
    projected = pca(prediction)
    primary, info = rectangular_sinkhorn(projected, target, config["pca_ot"])
    soft = soft_constraint(model, config)
    total = finite("loss", primary + soft)
    return total, dict(primary=float(primary.detach()), soft=float(soft.detach()),
                       total=float(total.detach()), sinkhorn=info)
