import torch
import numpy as np
from guided_diffusion.resample import ScheduleSampler, create_named_schedule_sampler
from ..common import finite
from ..losses.sinkhorn import sinkhorn_divergence


class LowNoiseSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion

    def sample(self, batch_size, device):
        t = (
            torch.from_numpy(np.random.randint(0, 50, size=batch_size))
            .long()
            .to(device)
        )
        return t, torch.ones(batch_size, device=device)

    def weights(self):
        weights = np.zeros(self.diffusion.num_timesteps)
        weights[:50] = 1
        return weights


def timestep_sampler(config, diffusion):
    if config["objective"] == "ot":
        return LowNoiseSampler(diffusion)
    return create_named_schedule_sampler(config["schedule_sampler"], diffusion)


def soft_constraint(model, config):
    if config["objective"] == "stage1":
        return next(model.parameters()).new_zeros(())
    ode = model.ode_model
    if not ode.soft:
        raise AssertionError("Stage 2 requires original soft constraint")
    # off_mask_penalty itself applies off_mask_lambda=5; no double weighting.
    return config["ode_reg_lambda"] * ode.off_mask_penalty(config["ode_reg_norm"])


def training_loss(model, diffusion, x0, t, weights, config, noise=None):
    if config["objective"] == "trajectory_ot":
        raise ValueError(
            "trajectory OT requires cached sources and an independent real-target stream"
        )
    if noise is None:
        noise = torch.randn_like(
            x0, dtype=torch.float64
        )  # exact source training convention
    info = {}
    if config["objective"] == "ot":
        if t.min() < 0 or t.max() > 49:
            raise ValueError("OT timesteps must be 0..49")
        xt = diffusion.q_sample(x0, t, noise=noise)
        prediction = model(xt, diffusion._scale_timesteps(t).unsqueeze(1))
        pred_x0 = diffusion._predict_xstart_from_eps(xt, t, prediction)
        primary, info = sinkhorn_divergence(pred_x0, x0, config["ot"], return_info=True)
    else:
        terms = diffusion.training_losses(model, x0, t, noise=noise)
        primary = (terms["loss"] * weights).mean()
    soft = soft_constraint(model, config)
    total = finite("training loss", primary + soft)
    return total, {
        "primary": float(primary.detach()),
        "soft": float(soft.detach()),
        "total": float(total.detach()),
        "sinkhorn": info,
    }
