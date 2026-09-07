#!/usr/bin/env python3
"""Opt-in exact forward/backward benchmark: auxiliary Model A and dense Model B.

The default dimension is 1024.  Each requested case measures one exact
transition-statistics evaluation and the work-local ``paper_elbo`` wrapper's
full forward and backward passes.  Failures (including OOM, non-finite values,
and Cholesky errors) are JSON results: this script never selects a cheaper
parameterization or approximation as a fallback.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (REPO_ROOT, SUITE_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import torch  # noqa: E402
from torch import nn  # noqa: E402

from diffusion.free_affine import FreeAffineForward  # noqa: E402
from diffusion.stationary_qd import StationaryQDForward  # noqa: E402
from diffusion.time_mapping import PhysicalTimeMap  # noqa: E402
from diffusion.training_diffusion import (  # noqa: E402
    LearnableForwardTrainingDiffusion,
)
from models.wrapper import LearnableForwardModel  # noqa: E402


DTYPES = {"float32": torch.float32, "float64": torch.float64}
MODELS = ("stationary_qd", "free_affine")


class BenchmarkDenoiser(nn.Module):
    """Minimal dense trainable epsilon predictor for wrapper benchmarking."""

    def __init__(self, dim: int, *, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.projection = nn.Linear(dim, dim, device=device, dtype=dtype)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, **kwargs):
        del timesteps, kwargs
        return self.projection(x)


def parse_csv_choices(
    value: str, choices: Iterable[str], *, allow_all: bool = False
) -> List[str]:
    allowed = tuple(choices)
    selected = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    if allow_all and selected == ["all"]:
        return list(allowed)
    if not selected or any(item not in allowed for item in selected):
        suffix = ", or all" if allow_all else ""
        raise argparse.ArgumentTypeError(
            f"expected comma-separated values from {allowed}{suffix}, got {value!r}"
        )
    # Preserve order while preventing accidental duplicate expensive runs.
    return list(dict.fromkeys(selected))


def select_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable")
    if device.type == "mps":
        raise RuntimeError(
            "MPS is not supported by this exact benchmark because its float64 "
            "correctness cases are unavailable on MPS"
        )
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"unsupported device type: {device.type}")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def cuda_peak_begin(device: torch.device) -> Optional[int]:
    if device.type != "cuda":
        return None
    synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    return int(baseline)


def cuda_peak_end(device: torch.device, baseline: Optional[int]) -> Optional[int]:
    if device.type != "cuda" or baseline is None:
        return None
    synchronize(device)
    peak = int(torch.cuda.max_memory_allocated(device))
    return max(0, peak - baseline)


def timed(
    operation: Callable[[], Any], device: torch.device
) -> Tuple[Any, float, Optional[int]]:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    synchronize(device)
    baseline = cuda_peak_begin(device)
    started = time.perf_counter()
    result = operation()
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = cuda_peak_end(device, baseline)
    return result, elapsed, peak


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().detach().cpu().item())


def finite_float(value: torch.Tensor) -> Optional[float]:
    scalar = float(value.detach().cpu().item())
    return scalar if math.isfinite(scalar) else None


def stats_report(stats) -> Dict[str, Any]:
    if hasattr(stats, "z"):
        lk = stats.cholesky_k
        return {
            "all_finite": all(finite_tensor(value) for value in
                              (stats.mean, stats.phi_k, stats.covariance_k, lk, stats.variance_perp)),
            "aux_dim": int(stats.z.shape[1]),
            "variance_perp": finite_float(stats.variance_perp),
            "cholesky": {"status": "success", "dimension": int(lk.shape[0]),
                         "minimum_diagonal": finite_float(lk.diagonal().min()),
                         "relative_reconstruction_error": finite_float(
                             torch.linalg.vector_norm(stats.covariance_k - lk @ lk.T) /
                             torch.linalg.vector_norm(stats.covariance_k))},
            "configured_covariance_jitter": 0.,
            "covariance_evaluation": stats.covariance_evaluation,
            "covariance_series_terms": stats.covariance_series_terms,
        }
    diagonal = torch.diagonal(stats.cholesky)
    covariance_scale = torch.linalg.vector_norm(stats.covariance)
    reconstruction_error = torch.linalg.vector_norm(
        stats.covariance - stats.cholesky @ stats.cholesky.transpose(0, 1)
    )
    relative_error = reconstruction_error / torch.clamp(
        covariance_scale, min=torch.finfo(stats.covariance.dtype).tiny
    )
    return {
        "all_finite": all(
            finite_tensor(value)
            for value in (
                stats.mean,
                stats.transition_matrix,
                stats.affine_shift,
                stats.covariance,
                stats.cholesky,
            )
        ),
        "mean_finite": finite_tensor(stats.mean),
        "transition_matrix_finite": finite_tensor(stats.transition_matrix),
        "covariance_finite": finite_tensor(stats.covariance),
        "cholesky": {
            # transition_stats uses cholesky_ex and raises for any nonzero info.
            "status": "success",
            "cholesky_ex_info": 0,
            "finite": finite_tensor(stats.cholesky),
            "minimum_diagonal": finite_float(diagonal.min()),
            "relative_reconstruction_error": finite_float(relative_error),
        },
        "configured_covariance_jitter": finite_float(stats.covariance_jitter),
        "covariance_evaluation": getattr(
            stats, "covariance_evaluation", "augmented_van_loan"
        ),
        "covariance_series_terms": int(
            getattr(stats, "covariance_series_terms", 0)
        ),
    }


def gradient_report(module: nn.Module) -> Dict[str, Any]:
    parameters: Dict[str, Any] = {}
    all_present = True
    all_finite = True
    total_squared_norm = 0.0
    for name, parameter in module.named_parameters():
        gradient = parameter.grad
        present = gradient is not None
        finite = present and finite_tensor(gradient)
        norm = None
        if present:
            norm_value = float(torch.linalg.vector_norm(gradient).detach().cpu())
            norm = norm_value if math.isfinite(norm_value) else None
            if norm is not None:
                total_squared_norm += norm * norm
        all_present = all_present and present
        all_finite = all_finite and finite
        parameters[name] = {
            "present": present,
            "finite": finite,
            "l2_norm": norm,
        }
    return {
        "all_present": all_present,
        "all_finite": all_finite,
        "total_l2_norm": math.sqrt(total_squared_norm),
        "parameters": parameters,
    }


def build_case(
    model_name: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    aux_dim: Optional[int] = None,
) -> Tuple[LearnableForwardModel, LearnableForwardTrainingDiffusion, PhysicalTimeMap]:
    if model_name == "stationary_qd":
        process = StationaryQDForward(
            dim,
            aux_dim=aux_dim,
            isotropic_d_init=0.5,
            covariance_jitter=0.0,
            grn_penalty_weight=0.0,
            device=device,
            dtype=dtype,
        )
    elif model_name == "free_affine":
        process = FreeAffineForward(
            dim, covariance_jitter=0.0, device=device, dtype=dtype
        )
    else:
        raise ValueError(f"unknown model: {model_name}")

    denoiser = BenchmarkDenoiser(dim, device=device, dtype=dtype)
    model = LearnableForwardModel(denoiser, process).to(device)
    time_map = PhysicalTimeMap.from_named_schedule("linear", 1000)
    diffusion = LearnableForwardTrainingDiffusion(
        time_map,
        loss_mode="paper_elbo",
        normalize_elbo_by_dimension=True,
    )
    return model, diffusion, time_map


def one_full_pass(
    model: LearnableForwardModel,
    diffusion: LearnableForwardTrainingDiffusion,
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    path_noise: torch.Tensor,
    boundary_noise: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    return diffusion.training_losses(
        model,
        x_start,
        timesteps,
        noise=path_noise,
        boundary_noise=boundary_noise,
    )


def benchmark_case(
    *,
    model_name: str,
    dim: int,
    batch_size: int,
    dtype_name: str,
    device: torch.device,
    physical_fraction: float,
    seed: int,
    warmup: int,
    repeat: int,
    aux_dim: Optional[int] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "running",
        "model": model_name,
        "dimension": dim,
        "batch_size": batch_size,
        "dtype": dtype_name,
        "device": str(device),
        "repeat": repeat,
        "dense_exact": True,
        "fallback_used": False,
        "approximation": None,
    }
    dtype = DTYPES[dtype_name]
    try:
        torch.manual_seed(seed + repeat)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + repeat)
        model, diffusion, time_map = build_case(model_name, dim, device, dtype, aux_dim)
        if model_name == "stationary_qd":
            result.update(model.forward_process.provenance())
            result["dense_exact"] = False
            result["exact_reduced"] = True
        result["forward_parameter_count"] = sum(
            parameter.numel()
            for parameter in model.forward_process.parameters()
        )
        result["total_parameter_count"] = sum(
            parameter.numel() for parameter in model.parameters()
        )
        result["matrix_exponential_dimension"] = (
            aux_dim if model_name == "stationary_qd" else 2 * dim + 1
        )
        result["time_map"] = time_map.metadata()

        physical_time = (
            time_map.boundary_time + physical_fraction * time_map.duration
        )
        physical_tensor = torch.tensor(
            physical_time, device=device, dtype=torch.float64
        )
        fractional = time_map.physical_time_to_fractional_index(physical_tensor)
        timesteps = fractional.to(dtype=torch.float32).expand(batch_size).clone()
        x_start = torch.randn(batch_size, dim, device=device, dtype=dtype)
        path_noise = torch.randn_like(x_start)
        boundary_noise = torch.randn_like(x_start)
        result["sampled_physical_time"] = physical_time
        result["fractional_timestep"] = float(fractional.detach().cpu())

        for _ in range(warmup):
            model.zero_grad(set_to_none=True)
            warm_losses = one_full_pass(
                model,
                diffusion,
                x_start,
                timesteps,
                path_noise,
                boundary_noise,
            )
            warm_losses["loss"].mean().backward()
            del warm_losses
        model.zero_grad(set_to_none=True)

        with torch.no_grad():
            stats, seconds, peak = timed(
                lambda: model.forward_process.transition_stats(
                    x_start, physical_time
                ),
                device,
            )
        result["forward_stats"] = {
            "wall_seconds": seconds,
            "peak_cuda_memory_bytes_above_baseline": peak,
            **stats_report(stats),
        }
        del stats

        losses, seconds, peak = timed(
            lambda: one_full_pass(
                model,
                diffusion,
                x_start,
                timesteps,
                path_noise,
                boundary_noise,
            ),
            device,
        )
        loss = losses["loss"].mean()
        component_report = {
            name: {
                "mean": finite_float(values.mean()),
                "finite": finite_tensor(values),
            }
            for name, values in losses.items()
        }
        result["full_paper_elbo_forward"] = {
            "wall_seconds": seconds,
            "peak_cuda_memory_bytes_above_baseline": peak,
            "loss_mean": finite_float(loss),
            "all_components_finite": all(
                item["finite"] for item in component_report.values()
            ),
            "components": component_report,
            "path_boundary_terminal_cholesky_status": "success",
        }
        if not finite_tensor(loss):
            raise FloatingPointError("full paper_elbo loss is non-finite")

        _, seconds, peak = timed(loss.backward, device)
        gradients = gradient_report(model)
        result["backward"] = {
            "wall_seconds": seconds,
            "peak_cuda_memory_bytes_above_baseline": peak,
            **gradients,
        }
        if not gradients["all_present"]:
            raise RuntimeError("at least one model parameter has no gradient")
        if not gradients["all_finite"]:
            raise FloatingPointError("at least one model gradient is non-finite")

        result["status"] = "success"
    except Exception as error:  # A failed dense run is itself benchmark output.
        result["status"] = "failed"
        if isinstance(error, torch.cuda.OutOfMemoryError):
            category = "cuda_out_of_memory"
        elif isinstance(error, MemoryError):
            category = "host_out_of_memory"
        elif isinstance(error, FloatingPointError):
            category = "non_finite"
        elif "cholesky" in str(error).lower():
            category = "cholesky_failure"
        else:
            category = "exception"
        result["exception"] = {
            "category": category,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return result


def system_metadata(device: torch.device) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "requested_execution_device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_runtime": torch.version.cuda,
                "cuda_total_memory_bytes": torch.cuda.get_device_properties(
                    device
                ).total_memory,
            }
        )
    return metadata


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--dim", type=int, default=1024)
    argument_parser.add_argument("--aux-dim", type=int, default=None, help="Model A K; default from its config")
    argument_parser.add_argument("--batch-size", type=int, default=4)
    argument_parser.add_argument(
        "--models",
        default="all",
        help="all or comma-separated stationary_qd,free_affine",
    )
    argument_parser.add_argument(
        "--dtypes",
        default="float32,float64",
        help="comma-separated float32,float64",
    )
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument(
        "--physical-fraction",
        type=float,
        default=0.5,
        help="position in [boundary_time,terminal_time] used for the path term",
    )
    argument_parser.add_argument("--warmup", type=int, default=0)
    argument_parser.add_argument("--repeats", type=int, default=1)
    argument_parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="set and record torch intra-op CPU threads before benchmarking",
    )
    argument_parser.add_argument("--seed", type=int, default=1234)
    argument_parser.add_argument(
        "--output",
        default="-",
        help="JSON path, or '-' for stdout (default)",
    )
    return argument_parser


def main(argv: Optional[List[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.dim <= 0 or args.batch_size <= 0:
        raise SystemExit("--dim and --batch-size must be positive")
    if args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("--warmup must be nonnegative and --repeats positive")
    if args.num_threads is not None and args.num_threads <= 0:
        raise SystemExit("--num-threads must be positive")
    if not 0.0 <= args.physical_fraction <= 1.0:
        raise SystemExit("--physical-fraction must lie in [0,1]")

    try:
        models = parse_csv_choices(args.models, MODELS, allow_all=True)
        dtypes = parse_csv_choices(args.dtypes, DTYPES)
        device = select_device(args.device)
    except (ValueError, RuntimeError, argparse.ArgumentTypeError) as error:
        raise SystemExit(str(error)) from error

    if "stationary_qd" in models and args.aux_dim is None:
        args.aux_dim = json.loads((SUITE_ROOT / "configs/model_a_stationary_qd_aux.json").read_text())["aux_dim"]
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    payload: Dict[str, Any] = {
        "benchmark": "dense_learnable_forward",
        "objective": (
            "Equation (7) truncated and made a valid ELBO with the complete "
            "Appendix-I/Theorem-3 boundary correction; exact compact form"
        ),
        "configuration": {
            "dimension": args.dim,
            "batch_size": args.batch_size,
            "models": models,
            "dtypes": dtypes,
            "device": str(device),
            "physical_fraction": args.physical_fraction,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "num_threads": torch.get_num_threads(),
            "seed": args.seed,
        },
        "system": system_metadata(device),
        "results": [],
    }
    for repeat in range(args.repeats):
        for model_name in models:
            for dtype_name in dtypes:
                payload["results"].append(
                    benchmark_case(
                        model_name=model_name,
                        dim=args.dim,
                        batch_size=args.batch_size,
                        dtype_name=dtype_name,
                        device=device,
                        physical_fraction=args.physical_fraction,
                        seed=args.seed,
                        warmup=args.warmup,
                        repeat=repeat,
                        aux_dim=args.aux_dim,
                    )
                )
    successes = sum(item["status"] == "success" for item in payload["results"])
    payload["summary"] = {
        "case_count": len(payload["results"]),
        "success_count": successes,
        "failure_count": len(payload["results"]) - successes,
        "all_successful": successes == len(payload["results"]),
        "fallback_used": False,
    }

    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output == "-":
        sys.stdout.write(rendered)
    else:
        destination = Path(args.output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(destination.resolve())
    return 0 if payload["summary"]["all_successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
