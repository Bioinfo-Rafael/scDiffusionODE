#!/usr/bin/env python3
"""Post-hoc model/branch diagnostics every N diffusion timesteps."""

from __future__ import annotations

import argparse
import csv
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
for search_path in (str(REPO_ROOT), str(SUITE_ROOT), str(HERE)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

from viz.analysis_helpers import (  # noqa: E402
    _sample_corr,
    _sample_norm,
    assert_finite,
    build_diffusion,
    load_model,
    load_real_subset,
)


ALLOWED_FAMILIES = {"linear_hybrid_lincomb", "ts_soft_tau80_hybrid_lincomb"}
ALLOWED_ODE_TYPES = {"centered_signed_hill", "shifted_hill_rho"}


def timestep_grid(num_timesteps: int, step: int = 20) -> list[int]:
    """Return 0, step, ... and always the actual final valid timestep."""

    num_timesteps = int(num_timesteps)
    step = int(step)
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    values = list(range(0, num_timesteps, step))
    final = num_timesteps - 1
    if values[-1] != final:
        values.append(final)
    return values


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _resolve_candidate(value: Any, run_dir: Path) -> Path | None:
    if value in (None, ""):
        return None
    raw = Path(str(value)).expanduser()
    candidates = [raw] if raw.is_absolute() else [run_dir / raw, REPO_ROOT / raw]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def _resolve_checkpoint(
    run_dir: Path, requested: str, manifest: Mapping[str, Any]
) -> Path:
    for value in (
        requested,
        manifest.get("checkpoint_path"),
        manifest.get("ema_checkpoint_path"),
        manifest.get("raw_checkpoint_path"),
    ):
        path = _resolve_candidate(value, run_dir)
        if path is not None:
            return path
    candidates = list(run_dir.glob("train/**/ema_*.pt"))
    candidates.extend(run_dir.glob("train/**/model*.pt"))
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise FileNotFoundError(f"no checkpoint found below {run_dir}")
    return max(existing, key=lambda path: (path.stat().st_mtime_ns, str(path))).resolve()


def _resolve_data_path(config: Mapping[str, Any], run_dir: Path) -> Path:
    path = _resolve_candidate(config.get("data_dir"), run_dir)
    if path is None:
        raise FileNotFoundError(f"data_dir does not exist: {config.get('data_dir')}")
    return path


def _select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _moments(chunks: Iterable[np.ndarray]) -> tuple[float, float]:
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
    t = np.asarray([row["timestep"] for row in rows], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for name in names:
        mean = np.asarray([row[f"{name}_mean"] for row in rows])
        std = np.asarray([row[f"{name}_std"] for row in rows])
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


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def analyze_run(args: argparse.Namespace, run_dir_value: str) -> dict[str, Any]:
    run_dir = Path(run_dir_value).expanduser().resolve()
    config_path = run_dir / "exp_config.json"
    manifest_path = run_dir / "manifest.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing run config: {config_path}")
    config = _read_json(config_path)
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    family = str(config.get("model_family", ""))
    ode_type = str(config.get("ode_type", ""))
    if family not in ALLOWED_FAMILIES or ode_type not in ALLOWED_ODE_TYPES:
        raise ValueError(f"unsupported 20260816 run: {family} / {ode_type}")

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
    t_values = timestep_grid(diffusion.num_timesteps, args.step)
    model = load_model(config, genes, diffusion, checkpoint, device)
    if not hasattr(model, "branch_outputs"):
        raise RuntimeError("restored model lacks exact Hybrid branch hooks")
    model.eval()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    rows: list[dict[str, float]] = []
    for raw_t in t_values:
        accum: dict[str, list[np.ndarray]] = {}
        for start in range(0, len(real), args.batch_size):
            x0 = torch.from_numpy(real[start : start + args.batch_size]).to(
                device=device, dtype=torch.float32
            )
            t_raw = torch.full((len(x0),), raw_t, device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = diffusion.q_sample(x0, t_raw, noise=noise)
            model_t = diffusion._scale_timesteps(t_raw)
            branches = model.branch_outputs(x_t, model_t)
            full = model(x_t, model_t)
            if not torch.allclose(full, branches["output"], rtol=1e-5, atol=1e-6):
                raise AssertionError("branch contributions do not sum to full output")
            outputs = {
                "hybrid": full,
                "ode": branches["ode_raw"],
                "cell_unet": branches["ml_raw"],
            }
            for name, output in outputs.items():
                assert_finite(f"{name} output at t={raw_t}", output)
                for metric_name, values in (
                    (f"{name}_corr", _sample_corr(output, noise)),
                    (f"{name}_norm", _sample_norm(output)),
                ):
                    assert_finite(f"{metric_name} at t={raw_t}", values)
                    accum.setdefault(metric_name, []).append(values.cpu().numpy())
            extra = {
                "true_noise_norm": _sample_norm(noise),
                "ode_weight": branches["ode_weight"].reshape(len(x0), -1).mean(-1),
                "cell_unet_weight": branches["ml_weight"].reshape(len(x0), -1).mean(-1),
                "weighted_ode_norm": _sample_norm(branches["ode_contribution"]),
                "weighted_cell_unet_norm": _sample_norm(branches["ml_contribution"]),
            }
            for name, values in extra.items():
                assert_finite(f"{name} at t={raw_t}", values)
                accum.setdefault(name, []).append(values.cpu().numpy())

        row: dict[str, float] = {"timestep": int(raw_t)}
        for name, chunks in accum.items():
            row[f"{name}_mean"], row[f"{name}_std"] = _moments(chunks)
        if not np.isfinite(np.asarray(list(row.values()), dtype=np.float64)).all():
            raise FloatingPointError(f"non-finite metric row at t={raw_t}")
        rows.append(row)
        print(
            f"[{config.get('experiment')}] t={raw_t}: "
            f"hybrid_corr={row['hybrid_corr_mean']:.4f}, "
            f"ode_corr={row['ode_corr_mean']:.4f}, "
            f"cell_corr={row['cell_unet_corr_mean']:.4f}",
            flush=True,
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = run_dir / "analysis" / f"corr_norm_step{args.step}_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    title = f"{config.get('experiment')} | every {args.step} steps | {checkpoint.name}"
    csv_path = output_dir / "timestep_metrics.csv"
    _write_csv(csv_path, rows)
    created = [
        str(csv_path),
        _plot_metric(
            rows,
            ("hybrid_corr", "ode_corr", "cell_unet_corr"),
            ylabel="per-cell Pearson correlation with true noise",
            title=f"Noise correlation\n{title}",
            path=output_dir / f"correlation_every_{args.step}_steps.png",
        ),
        _plot_metric(
            rows,
            ("true_noise_norm", "hybrid_norm", "ode_norm", "cell_unet_norm"),
            ylabel="per-cell L2 norm",
            title=f"Raw output norms\n{title}",
            path=output_dir / f"norm_every_{args.step}_steps.png",
        ),
        _plot_metric(
            rows,
            ("weighted_ode_norm", "weighted_cell_unet_norm"),
            ylabel="per-cell L2 norm",
            title=f"Weighted branch contribution norms\n{title}",
            path=output_dir / f"weighted_norm_every_{args.step}_steps.png",
        ),
    ]
    result = {
        "status": "completed",
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "config_path": str(config_path),
        "checkpoint": str(checkpoint),
        "experiment": config.get("experiment"),
        "model_family": family,
        "ode_type": ode_type,
        "diffusion_num_timesteps": int(diffusion.num_timesteps),
        "step": args.step,
        "t_values": t_values,
        "metric_definitions": {
            "correlation": "per-cell Pearson correlation across genes",
            "norm": "per-cell L2 norm across genes",
            "weights": "exact existing Hybrid ODE and CellUNet coefficients",
        },
        "created_files": [str(Path(path).relative_to(output_dir)) for path in created],
        "metrics": rows,
    }
    manifest_path = output_dir / "diagnostic_manifest.json"
    result["created_files"].append(manifest_path.name)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", nargs="+", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--max-cells", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_cells in (1, 2):
        raise ValueError("--max-cells must be <=0 (all) or >=3")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.checkpoint and len(args.run_dir) != 1:
        raise ValueError("--checkpoint can only be used with one --run-dir")
    results = [analyze_run(args, run_dir) for run_dir in args.run_dir]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    for result in results:
        print(f"CORR_NORM_DIR='{result['output_dir']}'")


if __name__ == "__main__":
    main()
