import math
import numpy as np
import torch
from ..common import finite, new_dir, write_csv, write_json, assert_start_x


def snapshot_table(hybrid, dt=0.001):
    rows = [
        {
            "snapshot_index": i,
            "reverse_step": step,
            "phase": "diffusion",
            "diffusion_t": 1000 - step,
            "ode_step": None,
            "integration_time": None,
            "ode_conditioning_t": None,
        }
        for i, step in enumerate(range(50, 1001, 50))
    ]
    if hybrid:
        rows.extend(
            {
                "snapshot_index": 20 + i,
                "reverse_step": 1000 + k,
                "phase": "post_ode",
                "diffusion_t": None,
                "ode_step": k,
                "integration_time": k * dt,
                "ode_conditioning_t": 0,
            }
            for i, k in enumerate((50, 100))
        )
    return rows


def check_state(x, max_norm):
    finite("state", x)
    norms = finite("state L2 norm", x.double().norm(dim=1))
    if norms.max() > max_norm:
        raise FloatingPointError(
            f"state norm exceeded divergence threshold {max_norm:g}"
        )
    return norms


@torch.no_grad()
def euler_step(ode, x, dt, max_norm):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("post_ode_dt must be finite and positive")
    check_state(x, max_norm)
    t_terminal = torch.zeros((len(x), 1), dtype=torch.long, device=x.device)
    field = finite("post-ODE vector field", ode(x, t_terminal))
    if field.shape != x.shape:
        raise ValueError("ODE field shape differs from state")
    after = x + dt * field
    norms = check_state(after, max_norm)
    return (
        after,
        field,
        {
            "displacement": (after - x).double().norm(dim=1),
            "field_norm": field.double().norm(dim=1),
            "state_norm": norms,
        },
    )


def sample_to_disk(
    model, diffusion, output, config, *, count, batch_size, device, provenance
):
    assert_start_x(diffusion)
    hybrid = hasattr(model, "ode_model")
    if count < 1 or batch_size < 1 or diffusion.num_timesteps != 1000:
        raise ValueError("invalid sample count/batch size/diffusion schedule")
    if config["post_ode_steps"] != 100 or config["post_ode_integrator"] != "euler":
        raise ValueError("canonical continuation requires 100 Euler steps")
    if not math.isfinite(config["post_ode_dt"]) or config["post_ode_dt"] <= 0:
        raise ValueError("invalid post_ode_dt")
    if not math.isfinite(config["max_state_norm"]) or config["max_state_norm"] <= 0:
        raise ValueError("invalid divergence threshold")
    output = new_dir(output)
    table = snapshot_table(hybrid, config["post_ode_dt"])
    genes = provenance["gene_names"]
    dims = len(genes)
    arrays = {}

    def allocate(name, shape, dtype="float32"):
        arrays[name] = np.lib.format.open_memmap(
            output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        return arrays[name]

    allocate("sample_state", (count, len(table), dims))
    for name in ("pred_xstart", "model_x0", "cellunet_raw"):
        allocate(name, (count, 20, dims))
    if hybrid:
        allocate("ode_raw", (count, 20, dims))
        allocate("post_ode_field", (count, 2, dims))
        for name in ("displacement", "field_norm", "state_norm"):
            allocate("post_" + name, (count, 100), "float64")
    metadata = {
        **provenance,
        "sampling_config": config,
        "model_mean_type": "START_X",
        "predict_xstart": True,
        "ancestral_noise_coefficient_nw": 0.5,
        "post_ode_semantics": "post-hoc dynamics analysis only; separate from START_X training and sampling",
        "snapshot_table": table,
        "count": count,
        "batch_size": batch_size,
        "array_shapes": {k: list(v.shape) for k, v in arrays.items()},
        "state_semantics": "sample_state[:, snapshot_index] is state AFTER update",
        "diffusion_outputs_semantics": "pred_xstart, model_x0, cellunet_raw, ode_raw use INPUT to the saved diffusion update; columns 0..19",
        "post_field_semantics": "post_ode_field columns 0,1 evaluated at INPUT to Euler updates 50,100; terminal conditioning t=0",
        "umap_representation": "diffusion pred_xstart; post_ode actual state; no fabricated clean prediction",
    }
    write_json(output / "sampling_metadata.json", metadata)
    write_csv(output / "snapshot_metadata.csv", table)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, count, batch_size):
                n = min(batch_size, count - start)
                sl = slice(start, start + n)
                x = torch.randn((n, dims), device=device)
                for step in range(1, 1001):
                    t = torch.full((n,), 1000 - step, device=device, dtype=torch.long)
                    # Exact source ancestral update; default noise coefficient nw=0.5.
                    out = diffusion.p_sample(model, x, t, clip_denoised=False)
                    check_state(out["sample"], config["max_state_norm"])
                    finite("clean prediction", out["pred_xstart"])
                    if step % 50 == 0:
                        k = step // 50 - 1
                        tm = diffusion._scale_timesteps(t).unsqueeze(1)
                        branches = (
                            model.branch_outputs(x, tm)
                            if hybrid
                            else {"ml_raw": model(x, tm)}
                        )
                        prediction = branches["output"] if hybrid else branches["ml_raw"]
                        if not torch.allclose(prediction, out["pred_xstart"], rtol=1e-5, atol=1e-6):
                            raise AssertionError("native START_X sampling must use raw prediction")
                        values = {
                            "sample_state": out["sample"],
                            "pred_xstart": out["pred_xstart"],
                            "model_x0": prediction,
                            "cellunet_raw": branches["ml_raw"],
                        }
                        if hybrid:
                            values["ode_raw"] = branches["ode_raw"]
                        for key, value in values.items():
                            arrays[key][sl, k] = (
                                finite(key, value).cpu().float().numpy()
                            )
                    x = out["sample"]
                if not torch.allclose(x, out["pred_xstart"], rtol=1e-5, atol=1e-6):
                    raise AssertionError(
                        "terminal diffusion state differs from pred_xstart"
                    )
                if hybrid:
                    for k in range(1, 101):
                        x, field, metrics = euler_step(
                            model.ode_model,
                            x,
                            config["post_ode_dt"],
                            config["max_state_norm"],
                        )
                        for name, value in metrics.items():
                            arrays["post_" + name][sl, k - 1] = (
                                finite(name, value).cpu().numpy()
                            )
                        if k in (50, 100):
                            j = k // 50 - 1
                            arrays["sample_state"][sl, 20 + j] = x.cpu().float().numpy()
                            arrays["post_ode_field"][sl, j] = (
                                field.cpu().float().numpy()
                            )
                for array in arrays.values():
                    array.flush()
                print(f"saved {start + n}/{count} trajectories", flush=True)
        if hybrid:
            rows = []
            for k in range(100):
                displacement = arrays["post_displacement"][:, k]
                field_norm = arrays["post_field_norm"][:, k]
                rows.append(
                    {
                        "phase": "post_ode",
                        "diffusion_t": None,
                        "ode_conditioning_t": 0,
                        "ode_step": k + 1,
                        "reverse_step": 1001 + k,
                        "integration_time": (k + 1) * config["post_ode_dt"],
                        "n_cells": count,
                        "mean_displacement": float(displacement.mean()),
                        "median_displacement": float(np.median(displacement)),
                        "mean_field_norm": float(field_norm.mean()),
                        "max_field_norm": float(field_norm.max()),
                        "max_state_norm": float(arrays["post_state_norm"][:, k].max()),
                        "all_finite": True,
                    }
                )
            write_csv(output / "post_ode_convergence.csv", rows)
        write_json(
            output / "completed.json",
            {"status": "completed", "count": count, "snapshots": len(table)},
        )
    except BaseException as exc:
        write_json(output / "failed.json", {"error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        for array in arrays.values():
            array.flush()
    return output
