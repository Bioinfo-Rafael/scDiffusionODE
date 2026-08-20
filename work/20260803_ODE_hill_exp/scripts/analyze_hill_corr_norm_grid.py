#!/usr/bin/env python3
"""Plot model/ODE/ML noise correlations and norms on a regular timestep grid.

The script only reads an existing Hill-after-linear hybrid run.  Results are
written to a new timestamped directory below ``run_dir/analysis`` without
updating the run manifest or any training artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


HERE = Path(__file__).resolve().parent
SUITE_ROOT = HERE.parent
REPO_ROOT = SUITE_ROOT.parent.parent
for search_path in (str(REPO_ROOT), str(SUITE_ROOT), str(HERE)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from analyze_hill_branch_diagnostics import (  # noqa: E402
    ALLOWED_FAMILIES,
    _read_json,
    _resolve_checkpoint,
    _resolve_data_path,
    _select_device,
)
from viz.analysis_helpers import (  # noqa: E402
    _checkpoint_label,
    _sample_corr,
    _sample_norm,
    _slug,
    assert_finite,
    build_diffusion,
    load_model,
    load_real_subset,
)


def _timestep_grid(num_timesteps: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("--step must be positive")
    values = list(range(0, num_timesteps, step))
    last = num_timesteps - 1
    if not values or values[-1] != last:
        values.append(last)
    return values


def _moments(chunks: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate(
        [np.asarray(chunk, dtype=np.float64).reshape(-1) for chunk in chunks]
    )
    if not np.isfinite(values).all():
        raise FloatingPointError("metric accumulator contains NaN or Inf")
    return float(values.mean()), float(values.std())


def _plot_metric(
    rows: list[dict[str, float]],
    names: tuple[str, ...],
    *,
    ylabel: str,
    title: str,
    path: Path,
) -> str:
    t = np.asarray([row["t"] for row in rows], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for name in names:
        mean = np.asarray([row[f"{name}_mean"] for row in rows], dtype=np.float64)
        std = np.asarray([row[f"{name}_std"] for row in rows], dtype=np.float64)
        ax.plot(t, mean, marker="o", markersize=3, linewidth=1.4, label=name)
        ax.fill_between(t, mean - std, mean + std, alpha=0.12)
    ax.set(xlabel="forward diffusion timestep t", ylabel=ylabel, title=title)
    ax.set_xticks(t[::max(1, math.ceil(len(t) / 12))])
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(path)


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
    t_values = _timestep_grid(diffusion.num_timesteps, args.step)
    model = load_model(config, genes, diffusion, checkpoint, device)
    if not hasattr(model, "ode_model") or not hasattr(model, "ml_model"):
        raise RuntimeError("restored model is not a hybrid with ODE and ML branches")
    if not hasattr(model, "branch_outputs"):
        raise RuntimeError("restored model does not expose exact weighted branch outputs")
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows: list[dict[str, float]] = []
    for raw_t in t_values:
        accum: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(real), args.batch_size):
            x0 = torch.from_numpy(real[start:start + args.batch_size]).to(
                device=device, dtype=torch.float32
            )
            t_raw = torch.full((len(x0),), int(raw_t), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t_raw, noise=noise)
            model_t = diffusion._scale_timesteps(t_raw)
            branches = model.branch_outputs(x_t, model_t)
            model_output = model(x_t, model_t)
            if not torch.allclose(
                model_output, branches["output"], rtol=1e-5, atol=1e-6
            ):
                raise AssertionError(
                    "weighted ODE and CellUnet branch contributions do not sum "
                    "to the model output"
                )
            outputs = {
                "model": model_output,
                "ode": branches["ode_raw"],
                "ml": branches["ml_raw"],
            }
            for name, output in outputs.items():
                assert_finite(f"{name} output at t={raw_t}", output)
                metrics = {
                    f"{name}_corr": _sample_corr(output, noise),
                    f"{name}_norm": _sample_norm(output),
                }
                for metric_name, values in metrics.items():
                    assert_finite(f"{metric_name} at t={raw_t}", values)
                    accum.setdefault(metric_name, []).append(
                        values.detach().cpu().numpy()
                    )
            noise_norm = _sample_norm(noise)
            assert_finite(f"noise_norm at t={raw_t}", noise_norm)
            accum.setdefault("noise_norm", []).append(noise_norm.cpu().numpy())
            weighted_metrics = {
                "ode_weighted_norm": _sample_norm(branches["ode_contribution"]),
                "ml_weighted_norm": _sample_norm(branches["ml_contribution"]),
                "ode_weight": branches["ode_weight"].reshape(len(x0), -1).mean(dim=1),
                "ml_weight": branches["ml_weight"].reshape(len(x0), -1).mean(dim=1),
            }
            for metric_name, values in weighted_metrics.items():
                assert_finite(f"{metric_name} at t={raw_t}", values)
                accum.setdefault(metric_name, []).append(values.cpu().numpy())

        row: dict[str, float] = {"t": float(raw_t)}
        for name, chunks in accum.items():
            mean, std = _moments(chunks)
            row[f"{name}_mean"] = mean
            row[f"{name}_std"] = std
        rows.append(row)
        print(
            f"[{config.get('experiment', family)}] t={raw_t}: "
            f"model_corr={row['model_corr_mean']:.4f}, "
            f"ode_corr={row['ode_corr_mean']:.4f}, "
            f"ml_corr={row['ml_corr_mean']:.4f}",
            flush=True,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / "analysis" / (
        f"hill_corr_norm_step{args.step}_{_slug(_checkpoint_label(checkpoint))}_{stamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    common_title = (
        f"{config.get('experiment', family)} | every {args.step} steps | "
        f"checkpoint={checkpoint.name}"
    )
    created = [
        _plot_metric(
            rows,
            ("model_corr", "ode_corr", "ml_corr"),
            ylabel="Pearson correlation with true noise",
            title=f"Model and branch noise correlation\n{common_title}",
            path=output_dir / f"correlation_every_{args.step}_steps.png",
        ),
        _plot_metric(
            rows,
            ("noise_norm", "model_norm", "ode_norm", "ml_norm"),
            ylabel="L2 norm per cell",
            title=f"Model, branch, and noise norms\n{common_title}",
            path=output_dir / f"norm_every_{args.step}_steps.png",
        ),
        _plot_metric(
            rows,
            ("ode_weighted_norm", "ml_weighted_norm"),
            ylabel="Weighted branch L2 norm per cell",
            title=f"Weighted ODE and CellUnet branch norms\n{common_title}",
            path=output_dir / "branch_weighted_norm.png",
        ),
    ]
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
        "step": args.step,
        "t_values": t_values,
        "created_files": [str(Path(path).relative_to(output_dir)) for path in created],
        "correlation": "per-cell Pearson correlation across genes",
        "norm": "per-cell L2 norm across genes",
        "weighted_norm": (
            "per-cell L2 norm of the exact ODE and CellUnet contributions "
            "used by the hybrid forward pass"
        ),
        "metrics": rows,
    }
    with (output_dir / "diagnostic_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"saved: {output_dir}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot noise correlations and L2 norms every N diffusion steps for "
            "existing Hill-after-linear hybrid runs."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument(
        "--checkpoint",
        default="",
        help="optional checkpoint override (use only when analyzing one run)",
    )
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--max-cells", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_cells == 1 or args.max_cells == 2:
        raise ValueError("--max-cells must be <=0 (all) or >=3")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint and len(args.run_dir) != 1:
        raise ValueError("--checkpoint can only be used with one --run-dir")
    results = [_analyze_one_run(args, run_dir) for run_dir in args.run_dir]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    for result in results:
        print(f"HILL_CORR_NORM_DIR='{result['output_dir']}'")


if __name__ == "__main__":
    main()
