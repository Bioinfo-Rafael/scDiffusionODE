"""Source per-cell metrics evaluated on identical forward-noised real inputs."""

import numpy as np
import torch
from ..common import finite, metric_helpers, write_csv, assert_start_x


def timestep_grids():
    # Matches 20260816/scripts/analyze_corr_norm_grid.py::timestep_grid.
    full = sorted(set(range(0, 1000, 20)) | {250, 500, 750, 999})
    low = list(range(51))
    return full, low


def per_cell_metrics(a, b):
    finite("metric input a", a)
    finite("metric input b", b)
    return {
        "pearson": metric_helpers()._sample_corr(a, b),
        "mse": (a - b).reshape(len(a), -1).square().mean(1),
        "cosine": torch.nn.functional.cosine_similarity(
            a.reshape(len(a), -1), b.reshape(len(b), -1), dim=1
        ),
    }


def summarize(values):
    array = finite("metric summary", np.asarray(values, dtype=np.float64))
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "n_cells": len(array),
    }


@torch.no_grad()
def diagnostics(model, diffusion, real, output, *, batch_size, seed, division_epsilon):
    if batch_size < 1 or not len(real) or division_epsilon <= 0:
        raise ValueError("invalid diagnostic settings")
    assert_start_x(diffusion)
    model.eval()
    full, low = timestep_grids()
    truth_rows, comparison_rows, norm_rows, schedule_rows = [], [], [], []
    for timestep in sorted(set(full) | set(low)):
        accum = {}
        # Same CPU RNG stream for every timestep and condition.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for start in range(0, len(real), batch_size):
            x0 = torch.as_tensor(
                real[start : start + batch_size],
                device=next(model.parameters()).device,
                dtype=torch.float32,
            )
            noise = torch.randn(x0.shape, generator=generator, dtype=torch.float64).to(
                x0.device
            )
            t = torch.full((len(x0),), timestep, dtype=torch.long, device=x0.device)
            xt = diffusion.q_sample(x0, t, noise=noise)
            tm = diffusion._scale_timesteps(t).unsqueeze(1)
            if hasattr(model, "ode_model"):
                branches = model.branch_outputs(xt, tm)
                outputs = {
                    "hybrid": model(xt, tm),
                    "cellunet": branches["ml_raw"],
                    "ode": branches["ode_raw"],
                }
                if not torch.allclose(
                    outputs["hybrid"], branches["output"], rtol=1e-5, atol=1e-6
                ):
                    raise AssertionError(
                        "source branch decomposition differs from Hybrid"
                    )
            else:
                outputs = {"cellunet": model(xt, tm)}
            norm = metric_helpers()._sample_norm
            metrics = {("norm", "true_x0", "l2"): norm(x0)}
            for name, value in outputs.items():
                finite(name, value)
                metrics.update(
                    {
                        ("truth", name, key): val
                        for key, val in per_cell_metrics(value, x0).items()
                    }
                )
                metrics[("truth", name, "l2")] = norm(value)
                metrics[("truth", name, "norm_ratio")] = norm(value) / norm(x0).clamp_min(division_epsilon)
                metrics[("norm", name, "l2")] = norm(value)
                metrics[("norm", name + "_to_true_x0", "ratio")] = norm(
                    value
                ) / norm(x0).clamp_min(division_epsilon)
            if hasattr(model, "ode_model"):
                metrics.update(
                    {
                        ("comparison", "cellunet_vs_ode", key): val
                        for key, val in per_cell_metrics(
                            outputs["cellunet"], outputs["ode"]
                        ).items()
                    }
                )
                metrics[("norm", "weighted_cellunet", "l2")] = norm(
                    branches["ml_contribution"]
                )
                metrics[("norm", "weighted_ode", "l2")] = norm(
                    branches["ode_contribution"]
                )
                metrics[("norm", "ode_to_cellunet", "ratio")] = norm(
                    outputs["ode"]
                ) / norm(outputs["cellunet"]).clamp_min(division_epsilon)
            for key, value in metrics.items():
                accum.setdefault(key, []).append(finite(str(key), value).cpu().numpy())
        alpha = float(diffusion.alphas_cumprod[timestep])
        schedule = {
            "t": timestep,
            "alpha_bar": alpha,
            "sigma": float(np.sqrt(1 - alpha)),
            "snr": alpha / (1 - alpha),
            "full_grid": timestep in full,
            "low_grid": timestep in low,
            "ot_training_region": timestep <= 49,
            "very_low_noise_annotation": timestep <= 10,
        }
        schedule_rows.append(schedule)
        for (kind, series, metric), chunks in accum.items():
            row = {
                "t": timestep,
                "phase": "diffusion",
                "series": series,
                "metric": metric,
                **summarize(np.concatenate(chunks)),
                "seed": seed,
            }
            {"truth": truth_rows, "comparison": comparison_rows, "norm": norm_rows}[
                kind
            ].append(row)
    fields = ["t", "phase", "series", "metric", "mean", "std", "n_cells", "seed"]
    for filename, rows in (
        ("true_x0_metrics.csv", truth_rows),
        ("cellunet_vs_ode_metrics.csv", comparison_rows),
        ("norm_metrics.csv", norm_rows),
    ):
        write_csv(output / filename, rows, fields=fields)
    write_csv(output / "diffusion_schedule.csv", schedule_rows)
