"""Small controlled solver benchmark; optional CUDA sizing, no training/model updates."""
import argparse
import importlib
import time
import platform
import torch
from ..common import effective_config, write_json
from ..training.objectives import solver, converged_entropic_ot


def legacy_retry(x, y, config):
    spent = 0
    limit = 200
    while True:
        try:
            value, info = solver.entropic_ot(x, y, epsilon=config["epsilon"],
                tolerance=config["tolerance"], max_iterations=limit)
            info["executed_iterations_including_retries"] = spent + info["iterations"]
            return value, info
        except solver.SinkhornConvergenceError:
            spent += limit
            if limit >= config["retry_max_iterations"]:
                raise
            limit = min(limit + 200, config["retry_max_iterations"])


def run(args):
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.manual_seed(1234)
    x = torch.randn(args.sources, args.dimension, dtype=torch.float64, device=device) * args.scale
    y = torch.randn(args.targets, args.dimension, dtype=torch.float64, device=device) * args.scale
    config = dict(effective_config("softplus_ot")["pca_ot"], retry_max_iterations=args.max_iterations)
    funcs = {
        "legacy_retry": lambda a, b: legacy_retry(a, b, config),
        "continuation": lambda a, b: converged_entropic_ot(a, b, config),
        "epsilon_scaling_candidate": lambda a, b: solver.entropic_ot(a, b, epsilon=config["epsilon"],
            tolerance=config["tolerance"], max_iterations=args.max_iterations, epsilon_scaling=True),
    }
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    rows, reference_gradient, reference_loss = [], None, None
    # Continuation is the production reference, irrespective of requested order.
    methods = sorted(args.methods, key=lambda name: name != "continuation")
    for name in methods:
        for trial in range(args.repeats):
            prediction = x.clone().requires_grad_()
            sync()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            try:
                value, info = funcs[name](prediction, y)
                sync()
                forward_done = time.perf_counter()
                gradient = torch.autograd.grad(value, prediction)[0]
                sync()
                finished = time.perf_counter()
                if reference_gradient is None and name == "continuation":
                    reference_gradient, reference_loss = gradient.detach(), float(value.detach())
                row = dict(method=name, trial=trial, status="ok", forward_seconds=forward_done-started,
                           backward_seconds=finished-forward_done, total_seconds=finished-started,
                           loss=float(value.detach()), solver=info)
                if reference_gradient is not None:
                    row.update(loss_difference=row["loss"]-reference_loss,
                               gradient_relative_error=float((gradient-reference_gradient).norm()/reference_gradient.norm().clamp_min(1e-30)))
                if device.type == "cuda":
                    row["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                print(f"{name} trial={trial}: {row['total_seconds']:.4f}s, iterations={info['iterations']}, "
                      f"gradient_relative_error={row.get('gradient_relative_error')}", flush=True)
                rows.append(row)
            except solver.SinkhornConvergenceError as exc:
                rows.append(dict(method=name, trial=trial, status="nonconverged", error=str(exc)))
                print(f"{name}: {exc}", flush=True)
    result = dict(torch_version=torch.__version__, python=platform.python_version(), device=str(device),
                  device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
                  shape=[args.sources, args.targets, args.dimension], scale=args.scale, threads=args.threads,
                  scope="one rectangular OT forward/backward; excludes ODE, sampling and evaluation", results=rows)
    if args.output:
        write_json(args.output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sources", type=int, default=4)
    p.add_argument("--targets", type=int, default=7)
    p.add_argument("--dimension", type=int, default=3)
    p.add_argument("--scale", type=float, default=5)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-iterations", type=int, default=16000)
    p.add_argument("--methods", nargs="+", default=["continuation", "legacy_retry", "epsilon_scaling_candidate"],
                   choices=["continuation", "legacy_retry", "epsilon_scaling_candidate"])
    p.add_argument("--output")
    args = p.parse_args()
    if min(args.sources, args.targets, args.dimension, args.threads, args.repeats) < 1 or args.max_iterations < 200:
        p.error("positive sizes required; max iterations must be >=200")
    run(args)


if __name__ == "__main__":
    main()
