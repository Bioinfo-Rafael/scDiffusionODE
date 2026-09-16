"""START_X reconstruction or independent-target PCA Sinkhorn gradients."""
import importlib
import torch
from ..common import assert_start_x, finite

old = importlib.import_module("work.20260915_x0predict.training.objectives")
solver = importlib.import_module("work.20260913_2step.losses.sinkhorn")
timestep_sampler, soft_constraint = old.timestep_sampler, old.soft_constraint


def rectangular_sinkhorn(prediction, target, config):
    if target.requires_grad:
        raise ValueError("cached empirical targets must be fixed")
    if config["objective"] != "sinkhorn_divergence_up_to_target_constant" or config["cost"] != "mean_squared_pca_distance" or config["compute_dtype"] != "float64":
        raise ValueError("unsupported PCA OT definition")
    args = {k: config[k] for k in ("epsilon", "max_iterations", "tolerance")}
    pred, real = prediction.double(), target.double()
    # The old solver accepts rectangular clouds and divides squared distances by D.
    # No target-target call: OT(real, real)/2 is constant in every model parameter.
    cross, cross_info = solver.entropic_ot(pred, real, **args)
    self_cost, self_info = solver.entropic_ot(pred, pred, **args)
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
