"""R is the original START_X MSE; K replaces it completely with graph loss."""
import importlib
import time
import torch
from ..common import assert_start_x, finite
from ..src.knn import graph_loss
legacy = importlib.import_module("work.20260915_x0predict.training.objectives")
timestep_sampler = legacy.timestep_sampler


def soft_constraint(model, config):
    # June MathMLPField returns an UNWEIGHTED penalty, unlike Sept16 fields.
    return config["ode_reg_lambda"]*config["off_mask_lambda"]*model.ode_model.off_mask_penalty(config["ode_reg_norm"])


def training_loss(model, diffusion, x0, t, weights, config, *, ids=None, matrix=None, graph=None, rng=None, noise=None):
    assert_start_x(diffusion)
    if not torch.equal(weights, torch.ones_like(weights)):
        raise ValueError("requires original uniform timestep sampler")
    model.ode_model._cached_W_sub = None
    zero = next(model.ode_model.parameters()).sum()*0
    pos = neg = knn = reconstruction = zero
    elapsed = 0.
    if config["aux_loss"] == "reconstruction":
        noise = torch.randn_like(x0, dtype=torch.float64) if noise is None else noise
        terms = diffusion.training_losses(model, x0, t, noise=noise)
        reconstruction = (terms["loss"]*weights).mean()
        primary = reconstruction
    else:
        # Consume the identical diffusion noise RNG stream across R/K conditions.
        # This noise is not used by the K objective; no reconstruction is computed.
        if noise is None:
            torch.randn_like(x0, dtype=torch.float64)
        if config["knn"]["lambda_knn"] > 0:
            if graph is None or ids is None or rng is None:
                raise ValueError("kNN training requires cached graph, row IDs and dedicated RNG")
            if x0.device.type == "cuda":
                torch.cuda.synchronize(x0.device)
            started = time.perf_counter()
            knn, pos, neg = graph_loss(model.ode_model, x0, t.float(), ids, matrix, graph, config["knn"], rng)
            if x0.device.type == "cuda":
                torch.cuda.synchronize(x0.device)
            elapsed = time.perf_counter()-started
        elif (config["model_type"] == "lowrank" and config["ode_reg_lambda"] > 0
              and config["off_mask_lambda"] > 0):
            # Dynamic regularization still needs W(x,t) when the graph term is off.
            # No ODE integration, neighbor lookup or reconstruction is performed.
            model.ode_model(x0, t)
        primary = config["knn"]["lambda_knn"]*knn
    soft = soft_constraint(model, config)
    total = finite("training loss", primary+soft+zero)
    values = dict(primary=primary, soft=soft, total=total, loss_reconstruction=reconstruction,
                  loss_knn=knn, loss_knn_pos=pos, loss_knn_neg=neg,
                  knn_contribution=config["knn"]["lambda_knn"]*knn)
    return total, dict(**{k: float(v.detach()) for k, v in values.items()}, knn_loss_seconds=elapsed)
