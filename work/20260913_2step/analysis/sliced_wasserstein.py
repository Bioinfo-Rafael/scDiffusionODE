"""Evaluation-only squared W2 averaged over seeded unit Gaussian projections."""

import numpy as np
from ..common import finite


def sliced_wasserstein(x, y, *, projections=256, points=2048, seed=4321):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or min(*x.shape, *y.shape, projections, points) < 1
    ):
        raise ValueError("SW requires nonempty equal-gene matrices and positive counts")
    finite("SW x", x)
    finite("SW y", y)
    n = min(len(x), len(y), points)

    # Canonical order before any subsampling makes the finite-seed helper
    # permutation invariant, including when the point limit is active.
    def subset(a):
        if len(a) == n:
            return a
        order = np.lexsort(a.T[::-1])
        ids = np.random.default_rng(seed).choice(len(a), n, replace=False)
        return a[order[ids]]

    x, y = subset(x), subset(y)
    directions = np.random.default_rng(seed + 1).normal(size=(x.shape[1], projections))
    directions /= np.linalg.norm(directions, axis=0, keepdims=True)
    delta = np.sort(x @ directions, axis=0) - np.sort(y @ directions, axis=0)
    squared = float(finite("SW squared distances", delta**2).mean())
    return dict(
        sliced_wasserstein2_squared=squared,
        sliced_wasserstein2=float(np.sqrt(squared)),
        points_used=n,
        projections=projections,
        seed=seed,
    )
