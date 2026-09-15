"""Original-gene-space diversity and entropic distribution distance."""

import numpy as np
import torch
from ..common import finite, write_csv
from .sliced_wasserstein import sliced_wasserstein
from .evaluation_ot import evaluate_sinkhorn


def diversity(x, *, pairs=100000, seed=1234, block_size=128):
    from scipy.spatial.distance import cdist

    n = len(x)
    if n < 2 or pairs < 1 or block_size < 1:
        raise ValueError(
            "diversity requires >=2 cells, positive pair count and block size"
        )
    # Exact mean Euclidean distance; O(block_size^2) distances, no N x N x G tensor.
    distance_sum = 0.0
    center = np.zeros(x.shape[1], dtype=np.float64)
    for i in range(0, n, block_size):
        a = finite(
            "diversity block", np.asarray(x[i : i + block_size], dtype=np.float64)
        )
        center += a.sum(0)
        for j in range(i, n, block_size):
            b = np.asarray(x[j : j + block_size], dtype=np.float64)
            d = finite("pairwise distances", cdist(a, b, metric="euclidean"))
            distance_sum += (
                float(d[np.triu_indices(len(a), k=1)].sum())
                if i == j
                else float(d.sum())
            )
    center /= n
    centered_ss = 0.0
    for i in range(0, n, block_size):
        centered_ss += float(
            np.square(
                np.asarray(x[i : i + block_size], dtype=np.float64) - center
            ).sum()
        )
    # Exact mean squared distance over unordered distinct pairs = 2 sum ||x-mean||²/(N-1).
    mean_squared = 2 * centered_ss / (n - 1)
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n, size=pairs)
    second = rng.integers(0, n - 1, size=pairs)
    second += (
        second >= first
    )  # exclude self-pairs; uniform ordered pairs with replacement
    sampled = np.empty(pairs, dtype=np.float64)
    for start in range(0, pairs, block_size):
        sl = slice(start, start + block_size)
        a = np.asarray(x[first[sl]], dtype=np.float64)
        b = np.asarray(x[second[sl]], dtype=np.float64)
        sampled[sl] = np.linalg.norm(a - b, axis=1)
    finite("sampled pair distances", sampled)
    return {
        "n_cells": n,
        "sampled_pairs": pairs,
        "seed": seed,
        "mean": distance_sum / (n * (n - 1) / 2),
        "median": float(np.median(sampled)),
        "mean_squared": mean_squared,
        "q05": float(np.quantile(sampled, 0.05)),
        "q95": float(np.quantile(sampled, 0.95)),
        "mean_method": "exact_blockwise_unordered_pairs",
        "median_method": "uniform_distinct_pairs_with_replacement",
    }


@torch.no_grad()
def trajectory_metrics(states, real, table, output, config, *, seed):
    n = min(config["analysis_ot_cells"], len(real), len(states))
    if n < 2:
        raise ValueError("distribution comparison requires >=2 cells")
    rng = np.random.default_rng(seed)
    generated_ids = np.sort(rng.choice(len(states), n, replace=False))
    real_ids = np.sort(rng.choice(len(real), n, replace=False))
    target = torch.as_tensor(real[real_ids], dtype=torch.float64)
    diversity_rows, distance_rows, sw_rows = [], [], []
    evaluation = config["evaluation"]
    for row in table:
        k = row["snapshot_index"]
        sw_rows.append(
            {
                **row,
                "representation": "state_after_update",
                **sliced_wasserstein(
                    states[:, k],
                    real,
                    projections=evaluation["sliced_wasserstein_projections"],
                    points=evaluation["sliced_wasserstein_points"],
                    seed=evaluation["seed"],
                ),
            }
        )
        diversity_rows.append(
            {
                **row,
                "representation": "state_after_update",
                **diversity(states[:, k], pairs=config["analysis_pairs"], seed=seed),
            }
        )
        pred = torch.as_tensor(
            np.array(states[generated_ids, k], copy=True), dtype=torch.float64
        )
        divergence, info, status = evaluate_sinkhorn(pred, target, config["ot"])
        residual = max((v["marginal_residual"] for v in info.values()), default=None)
        distance_rows.append(
            {
                **row,
                "representation": "state_after_update",
                "sinkhorn_divergence": divergence,
                **status,
                "subsample_size": n,
                "seed": seed,
                "epsilon": config["ot"]["epsilon"],
                "cost": config["ot"]["cost"],
                "debias": True,
                "max_iterations": config["ot"]["max_iterations"],
                "tolerance": config["ot"]["tolerance"],
                "marginal_residual": residual,
            }
        )
    write_csv(output / "sliced_wasserstein_snapshots.csv", sw_rows)
    write_csv(output / "trajectory_diversity.csv", diversity_rows)
    write_csv(output / "sinkhorn_to_real.csv", distance_rows)
    missing = [r for r in distance_rows if r["ot_status"] == "not_converged"]
    from ..common import write_json

    write_json(
        output / "evaluation_ot_status.json",
        dict(
            missing_count=len(missing),
            missing_snapshots=[r["snapshot_index"] for r in missing],
            policy="retry with epsilon scaling; missing values are never replaced by unconverged estimates",
        ),
    )
    with (output / "distribution_subsample_ids.npz").open("xb") as f:
        np.savez(f, generated_ids=generated_ids, real_ids=real_ids)
