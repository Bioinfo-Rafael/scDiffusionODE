"""Evaluation-only recovery: never report an unconverged transport estimate."""

import torch
from ..losses.sinkhorn import sinkhorn_divergence, SinkhornConvergenceError


@torch.no_grad()
def evaluate_sinkhorn(x, y, config, *, cost_diagnostics=False):
    attempts = [dict(config)]
    if not config.get("epsilon_scaling", False):
        attempts.append(
            {
                **config,
                "epsilon_scaling": True,
                "epsilon_scaling_start": max(1.6, config["epsilon"]),
                "epsilon_scaling_factor": 0.5,
            }
        )
    errors = []
    for index, numerical in enumerate(attempts, start=1):
        try:
            value, info = sinkhorn_divergence(
                x, y, numerical, return_info=True, cost_diagnostics=cost_diagnostics
            )
            return (
                float(value),
                info,
                dict(
                    ot_status="converged" if index == 1 else "converged_after_retry",
                    ot_attempts=index,
                    ot_epsilon_scaling=numerical.get("epsilon_scaling", False),
                    ot_error=" | ".join(errors),
                ),
            )
        except SinkhornConvergenceError as exc:
            errors.append(str(exc))
            print(f"[evaluation OT] attempt {index} failed: {exc}", flush=True)
    print(
        "[evaluation OT] NOT COMPUTED: continuing other metrics; no unconverged value accepted",
        flush=True,
    )
    return (
        None,
        {},
        dict(
            ot_status="not_converged",
            ot_attempts=len(attempts),
            ot_epsilon_scaling=attempts[-1].get("epsilon_scaling", False),
            ot_error=" | ".join(errors),
        ),
    )
