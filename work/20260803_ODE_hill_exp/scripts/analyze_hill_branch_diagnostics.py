#!/usr/bin/env python3
"""Plot Hill branch outputs and its production/decay decomposition for old runs.

This is a read-only, post-hoc diagnostic: it restores an existing EMA checkpoint
and writes a new timestamped directory below ``run_dir/analysis``.  It does not
update the run manifest, checkpoint, configuration, samples, or training files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (str(REPO_ROOT), str(SUITE_ROOT)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from viz.analysis_helpers import (  # noqa: E402
    _call_ml,
    _checkpoint_label,
    _sample_corr,
    _slug,
    assert_finite,
    build_diffusion,
    load_model,
    load_real_subset,
    parse_t_values,
)


ALLOWED_FAMILIES = {
    "standard_hybrid_single",
    "standard_hybrid_lincomb",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _resolve_candidate(value: Any, run_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [run_dir / raw, REPO_ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _analysis_manifests(run_dir: Path) -> Iterable[Path]:
    return sorted(
        run_dir.glob("analysis/**/analysis_manifest.json"),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )


def _resolve_checkpoint(run_dir: Path, requested: str, manifest: Mapping[str, Any]) -> Path:
    candidates: list[Any] = [requested]
    candidates.extend(
        manifest.get(key)
        for key in ("checkpoint_path", "ema_checkpoint_path", "raw_checkpoint_path")
    )
    for analysis_manifest_path in _analysis_manifests(run_dir):
        analysis_manifest = _read_json(analysis_manifest_path)
        candidates.append(analysis_manifest.get("checkpoint"))
    for candidate in candidates:
        resolved = _resolve_candidate(candidate, run_dir)
        if resolved is not None:
            return resolved

    checkpoints = list(run_dir.glob("train/**/ema_*.pt"))
    checkpoints.extend(run_dir.glob("train/**/model*.pt"))
    checkpoints.extend(run_dir.glob("**/ema_*.pt"))
    existing = [path for path in checkpoints if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"no checkpoint found for {run_dir}; pass --checkpoint explicitly"
        )
    return max(existing, key=lambda path: (path.stat().st_mtime_ns, str(path))).resolve()


def _resolve_data_path(config: Mapping[str, Any], run_dir: Path) -> Path:
    value = config.get("data_dir")
    if not value:
        raise KeyError("exp_config.json has no data_dir")
    path = _resolve_candidate(value, run_dir)
    if path is None:
        raise FileNotFoundError(f"data_dir does not exist: {value}")
    return path


def _select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pearson_flat(a: np.ndarray, b: np.ndarray) -> float:
    a64 = np.asarray(a, dtype=np.float64).reshape(-1)
    b64 = np.asarray(b, dtype=np.float64).reshape(-1)
    a64 = a64 - a64.mean()
    b64 = b64 - b64.mean()
    denominator = np.linalg.norm(a64) * np.linalg.norm(b64)
    return float(np.dot(a64, b64) / max(float(denominator), 1e-12))


def _scatter_panel(
    noise: np.ndarray,
    values: list[tuple[str, np.ndarray, float]],
    *,
    title: str,
    path: Path,
) -> str:
    fig, axes = plt.subplots(1, len(values), figsize=(5.2 * len(values), 4.5), squeeze=False)
    for ax, (label, output, mean_sample_corr) in zip(axes[0], values):
        flat_corr = _pearson_flat(output, noise)
        ax.scatter(noise, output, s=3, alpha=0.16, rasterized=True)
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ax.set(
            xlabel="true noise epsilon",
            ylabel=label,
            title=(
                f"{label} vs noise\n"
                f"mean per-cell r={mean_sample_corr:.4f}, flat r={flat_corr:.4f}"
            ),
        )
        ax.grid(alpha=0.18)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _distribution_panel(
    noise: np.ndarray,
    values: list[tuple[str, np.ndarray]],
    *,
    title: str,
    path: Path,
) -> str:
    fig, axes = plt.subplots(1, len(values), figsize=(5.2 * len(values), 4.3), squeeze=False)
    for ax, (label, output) in zip(axes[0], values):
        combined = np.concatenate([noise, output])
        low, high = np.quantile(combined, [0.005, 0.995])
        if not np.isfinite(low) or not np.isfinite(high):
            raise FloatingPointError(f"non-finite histogram range for {label}")
        if low == high:
            padding = max(abs(float(low)) * 0.01, 1e-6)
            low, high = low - padding, high + padding
        bins = np.linspace(low, high, 100)
        ax.hist(noise, bins=bins, density=True, alpha=0.45, label="noise", color="#7f7f7f")
        ax.hist(output, bins=bins, density=True, alpha=0.50, label=label, color="#1f77b4")
        ax.set(xlabel="value", ylabel="density", title=f"{label} and noise")
        ax.legend(frameon=False)
        ax.grid(alpha=0.18)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _sample_aligned(
    arrays: Mapping[str, torch.Tensor],
    *,
    count: int,
    generator: np.random.Generator,
) -> dict[str, np.ndarray]:
    flattened = {
        name: tensor.detach().float().cpu().numpy().reshape(-1)
        for name, tensor in arrays.items()
    }
    size = next(iter(flattened.values())).size
    if any(value.size != size for value in flattened.values()):
        raise ValueError("diagnostic arrays are not aligned")
    indices = (
        generator.choice(size, size=count, replace=False)
        if count < size
        else np.arange(size)
    )
    return {name: value[indices] for name, value in flattened.items()}


@torch.no_grad()
def _analyze_one_run(args: argparse.Namespace, run_dir_value: str) -> dict[str, Any]:
    run_dir = Path(run_dir_value).expanduser().resolve()
    config_path = run_dir / "exp_config.json"
    manifest_path = run_dir / "manifest.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing run config: {config_path}")
    config = _read_json(config_path)
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    family = str(config.get("model_family", ""))
    ode_type = str(config.get("ode_type", ""))
    if family not in ALLOWED_FAMILIES or ode_type != "hill_after_linear":
        raise ValueError(
            "this diagnostic only accepts standard_hybrid_single or "
            f"standard_hybrid_lincomb hill_after_linear runs, got {family} / {ode_type}"
        )

    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint, manifest)
    data_path = _resolve_data_path(config, run_dir)
    seed = int(config.get("seed", 1234) if args.seed is None else args.seed)
    device = _select_device(args.device)
    real, genes, _, _ = load_real_subset(
        data_path,
        layer=config.get("ts_layer"),
        max_cells=args.max_cells,
        seed=seed,
    )
    diffusion = build_diffusion(config)
    t_values = parse_t_values(args.t_values, diffusion.num_timesteps)
    model = load_model(config, genes, diffusion, checkpoint, device)
    if not hasattr(model, "ode_model") or not hasattr(model, "ml_model"):
        raise RuntimeError("restored model is not a hybrid with ODE and ML branches")
    if not hasattr(model.ode_model, "delta"):
        raise RuntimeError("restored ODE branch does not expose positive decay parameter delta")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / "analysis" / (
        f"hill_branch_diagnostics_{_slug(_checkpoint_label(checkpoint))}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    created: list[str] = []
    generator = np.random.default_rng(seed)
    batches = max(1, math.ceil(len(real) / args.batch_size))
    points_per_batch = max(1, math.ceil(args.max_points / batches))

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    for raw_t in t_values:
        samples: dict[str, list[np.ndarray]] = {
            name: [] for name in ("noise", "model", "ode", "ml", "production", "decay")
        }
        corr_values: dict[str, list[np.ndarray]] = {
            name: [] for name in ("model", "ode", "ml", "production", "decay")
        }
        for start in range(0, len(real), args.batch_size):
            x0 = torch.from_numpy(real[start:start + args.batch_size]).to(
                device=device, dtype=torch.float32
            )
            t_raw = torch.full((len(x0),), int(raw_t), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t_raw, noise=noise)
            model_t = diffusion._scale_timesteps(t_raw)
            total = model(x_t, model_t)
            ode = model.ode_model(x_t, model_t)
            ml = _call_ml(model, x_t, model_t)
            delta = model.ode_model.delta.to(device=x_t.device, dtype=x_t.dtype)
            decay = delta.reshape(1, -1) * x_t
            # Single: ode = production - decay. LinComb: the softmax coefficients
            # sum to one and decay is shared, so the same identity is exact.
            production = ode + decay
            tensors = {
                "noise": noise,
                "model": total,
                "ode": ode,
                "ml": ml,
                "production": production,
                "decay": decay,
            }
            for name, tensor in tensors.items():
                assert_finite(f"{name} at t={raw_t}", tensor)
            selected = _sample_aligned(
                tensors,
                count=min(points_per_batch, noise.numel()),
                generator=generator,
            )
            for name, values in selected.items():
                samples[name].append(values)
            for name in corr_values:
                corr_values[name].append(
                    _sample_corr(tensors[name], noise).detach().cpu().numpy()
                )

        plotted = {
            name: np.concatenate(chunks)[:args.max_points]
            for name, chunks in samples.items()
        }
        mean_corr = {
            name: float(np.concatenate(chunks).astype(np.float64).mean())
            for name, chunks in corr_values.items()
        }
        prefix = f"t{int(raw_t):04d}"
        common_title = (
            f"{config.get('experiment', family)} | t={raw_t} | "
            f"checkpoint={checkpoint.name}"
        )
        created.append(_scatter_panel(
            plotted["noise"],
            [
                ("model output", plotted["model"], mean_corr["model"]),
                ("ODE raw output", plotted["ode"], mean_corr["ode"]),
                ("ML raw output", plotted["ml"], mean_corr["ml"]),
            ],
            title=common_title,
            path=output_dir / f"{prefix}_outputs_vs_noise_scatter.png",
        ))
        created.append(_distribution_panel(
            plotted["noise"],
            [
                ("model output", plotted["model"]),
                ("ODE raw output", plotted["ode"]),
                ("ML raw output", plotted["ml"]),
            ],
            title=common_title,
            path=output_dir / f"{prefix}_output_distributions.png",
        ))
        created.append(_scatter_panel(
            plotted["noise"],
            [
                ("Hill production", plotted["production"], mean_corr["production"]),
                ("decay delta*x_t", plotted["decay"], mean_corr["decay"]),
            ],
            title=common_title,
            path=output_dir / f"{prefix}_hill_terms_vs_noise_scatter.png",
        ))
        created.append(_distribution_panel(
            plotted["noise"],
            [
                ("Hill production", plotted["production"]),
                ("decay delta*x_t", plotted["decay"]),
            ],
            title=common_title,
            path=output_dir / f"{prefix}_hill_term_distributions.png",
        ))

    result = {
        "status": "completed",
        "read_only_source_run": True,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "experiment": config.get("experiment"),
        "model_family": family,
        "ode_type": ode_type,
        "device": str(device),
        "seed": seed,
        "max_cells": args.max_cells,
        "n_cells": len(real),
        "max_points_per_plot": args.max_points,
        "t_values": t_values,
        "created_files": [str(Path(path).relative_to(output_dir)) for path in created],
        "production_identity": "production = ode_raw + delta*x_t",
        "correlation": "per-cell Pearson correlation across genes",
    }
    with (output_dir / "diagnostic_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create read-only Hill branch correlation and distribution diagnostics "
            "for existing standard_hybrid_single/lincomb runs."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="optional checkpoint override (use only when analyzing one run)",
    )
    parser.add_argument("--max-cells", type=int, default=2000)
    parser.add_argument("--max-points", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--t-values",
        default="",
        help="comma-separated raw timesteps; default is 0, midpoint,last",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_cells == 1 or args.max_cells == 2:
        raise ValueError("--max-cells must be <=0 (all) or >=3")
    if args.max_points < 100:
        raise ValueError("--max-points must be >=100")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint and len(args.run_dir) != 1:
        raise ValueError("--checkpoint can only be used with one --run-dir")
    results = [_analyze_one_run(args, run_dir) for run_dir in args.run_dir]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    for result in results:
        print(f"HILL_DIAGNOSTIC_DIR='{result['output_dir']}'")


if __name__ == "__main__":
    main()
