"""Paper-faithful Eq. (7)/(8), Algorithm 1 and Appendix D, arXiv:2608.14067.

No official source implementation was found in the public sources checked at
implementation time (2026-09-12). See PAPER_PROVENANCE.md. Eq. (8) splits by
original diffusion time: t < breakpoint versus t >= breakpoint, even though
stored snapshots are in decreasing-t / increasing-reverse-progress order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import load_trajectory, write_csv, write_json
from .settings import PAPER, Settings


def grouped_trajectories(predictions: np.ndarray, settings: Settings) -> np.ndarray:
    if (
        predictions.ndim != 3
        or predictions.shape[:2] != (settings.trajectories, settings.M)
        or predictions.shape[2] < 1
    ):
        raise ValueError(
            f"expected [{settings.trajectories},{settings.M},G], got {predictions.shape}"
        )
    return predictions.reshape(settings.N, settings.S, settings.M, predictions.shape[2])


def score_curves(predictions: np.ndarray, settings: Settings) -> np.ndarray:
    grouped = grouped_trajectories(predictions, settings)
    scores = np.empty((settings.N, settings.M), dtype=np.float64)
    # Eq. (7): average independent trajectories and all gene features, no norms,
    # squared differences, absolute values, scaling, or ground-truth labels.
    for n in range(settings.N):
        for k in range(settings.M):
            values = grouped[n, :, k, :]
            if not np.isfinite(values).all():
                raise FloatingPointError(f"nonfinite clean prediction in group {n}, snapshot {k}")
            scores[n, k] = values.mean(axis=0, dtype=np.float64).mean()
    return scores


def candidate_indices(count: int) -> np.ndarray:
    trim = int(count * 0.1)
    if trim < 2:
        raise ValueError(
            "at least 20 snapshots required for 10% exclusion and two-point linear fits"
        )
    return np.arange(trim, count - trim)


@dataclass(frozen=True)
class Fit:
    snapshot_index: int
    sse: float
    fitted: np.ndarray


def fit_breakpoint(scores: np.ndarray, diffusion_t: np.ndarray) -> Fit:
    y, x = np.asarray(scores, dtype=np.float64), np.asarray(diffusion_t, dtype=np.float64)
    if y.ndim != 1 or x.shape != y.shape or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("scores and diffusion_t must be finite equal-length vectors")
    if not np.all(np.diff(x) < 0):
        raise ValueError("diffusion_t must decrease in stored reverse-progress order")
    best = None
    for k in candidate_indices(len(y)):
        fitted = np.empty_like(y)
        # Eq. (8) explicitly includes the breakpoint in the higher-t segment.
        for mask in (x < x[k], x >= x[k]):
            design = np.column_stack((x[mask], np.ones(mask.sum())))
            coefficients = np.linalg.lstsq(design, y[mask], rcond=None)[0]
            fitted[mask] = design @ coefficients
        sse = float(np.square(y - fitted).sum())
        if best is None or sse < best.sse:
            best = Fit(int(k), sse, fitted)
    assert best is not None
    return best


def global_breakpoint(fits: list[Fit], table: list[dict]) -> dict:
    median = float(np.median([fit.snapshot_index for fit in fits]))
    indices = np.arange(len(table))
    # Even N may yield a half-index. Preserve it; do not invent a saved snapshot.
    result = {
        "median_snapshot_index": median,
        "reverse_step": float(np.interp(median, indices, [row["reverse_step"] for row in table])),
        "diffusion_t": float(np.interp(median, indices, [row["diffusion_t"] for row in table])),
        "is_saved_snapshot": median.is_integer(),
        "median_policy": "exact median, linearly interpolated coordinates if between saved snapshots; no rounding",
        "tie_policy": "first minimum in increasing snapshot index (reverse-progress order)",
        "SSE": None,
        "SSE_note": "median aggregation is not a fitted curve; SSE is recorded per group",
        "candidate_indices": candidate_indices(len(table)).tolist(),
        "paper": PAPER,
    }
    return result


def analyze(result: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions, table, _, settings = load_trajectory(result)
    scores = score_curves(predictions, settings)
    times = np.array([row["diffusion_t"] for row in table])
    steps = np.array([row["reverse_step"] for row in table])
    fits = [fit_breakpoint(curve, times) for curve in scores]
    output = result / "breakpoint"
    output.mkdir(exist_ok=False)
    write_csv(
        output / "score_curves.csv",
        [
            {"group": n, **row, "score": float(scores[n, k])}
            for n in range(settings.N)
            for k, row in enumerate(table)
        ],
    )
    write_csv(
        output / "per_group_breakpoints.csv",
        [{"group": n, **table[fit.snapshot_index], "SSE": fit.sse} for n, fit in enumerate(fits)],
    )
    summary = global_breakpoint(fits, table)
    write_json(output / "global_breakpoint.json", summary)
    for n, fit in enumerate(fits):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, scores[n], "o-", label="Eq. 7: mean clean prediction")
        k = fit.snapshot_index
        for segment in (slice(0, k + 1), slice(k + 1, None)):
            ax.plot(steps[segment], fit.fitted[segment], "--", color="black")
        ax.axvline(steps[k], color="red", label=f"breakpoint {k}, t={times[k]}")
        ax.set(
            xlabel="Completed reverse updates",
            ylabel="Mean across trajectories and genes",
            title=f"Group {n:02d} | SSE={fit.sse:.6g}",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"breakpoint_fit_group_{n:02d}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(steps, scores.T, alpha=0.4)
    axes[0].axvline(summary["reverse_step"], color="black", linestyle="--")
    axes[0].set(xlabel="Completed reverse updates", ylabel="Eq. 7 score")
    axes[1].hist([fit.snapshot_index for fit in fits], bins=np.arange(settings.M + 1) - 0.5)
    axes[1].axvline(summary["median_snapshot_index"], color="red")
    axes[1].set(xlabel="Breakpoint snapshot index", ylabel="Groups")
    fig.suptitle(
        f"Global median: snapshot {summary['median_snapshot_index']:g}, reverse step {summary['reverse_step']:g}, t={summary['diffusion_t']:g}"
    )
    fig.tight_layout()
    fig.savefig(output / "breakpoint_summary.png", dpi=150)
    plt.close(fig)
    return summary
