"""Estimate the dynamical-regime boundary from the model input space.

The estimator intentionally subsets rows before converting sparse input to a
dense float32 matrix.  This keeps the peak allocation bounded by
``n_cells * n_vars`` rather than the full AnnData size.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _sample_rows(matrix, indices: np.ndarray) -> np.ndarray:
    subset = matrix[indices]
    if hasattr(subset, "toarray"):
        subset = subset.toarray()
    elif hasattr(subset, "todense"):
        subset = subset.todense()
    return np.asarray(subset, dtype=np.float32)


def estimate_lambda_max_from_adata(
    adata,
    layer: Optional[str] = None,
    n_cells: int = 200_000,
    seed: int = 0,
    use_randomized_pca: bool = True,
) -> Dict[str, Any]:
    """Estimate the largest gene covariance eigenvalue from AnnData input.

    ``adata.X`` is used when ``layer`` is ``None``; otherwise the named layer
    is used.  Rows are sampled before sparse-to-dense conversion.
    """
    if n_cells <= 0:
        raise ValueError(f"n_cells must be > 0, got {n_cells}")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"AnnData layer '{layer}' does not exist")

    source = adata.X if layer is None else adata.layers[layer]
    n_obs_total, n_vars = int(adata.n_obs), int(adata.n_vars)
    if n_obs_total < 2:
        raise ValueError("at least two cells are required to estimate covariance")

    n_used = min(int(n_cells), n_obs_total)
    if n_used == n_obs_total:
        indices = np.arange(n_obs_total, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        # Sorted indices also work with backed/h5py datasets that reject
        # unsorted fancy indexing.
        indices = np.sort(rng.choice(n_obs_total, n_used, replace=False))

    x = _sample_rows(source, indices)
    if x.ndim != 2 or x.shape != (n_used, n_vars):
        raise ValueError(
            f"input subset has unexpected shape {x.shape}; expected {(n_used, n_vars)}"
        )
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    x -= x.mean(axis=0, dtype=np.float64).astype(np.float32, copy=False)

    if use_randomized_pca:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=1, svd_solver="randomized", random_state=int(seed))
        pca.fit(x)
        lambda_max = float(pca.explained_variance_[0])
        method = "sample_randomized_pca1"
    else:
        # Exact SVD is intended only for tests/small inputs.
        singular_value = float(np.linalg.svd(x, full_matrices=False, compute_uv=False)[0])
        lambda_max = singular_value * singular_value / float(max(n_used - 1, 1))
        method = "sample_exact_svd1"

    if not np.isfinite(lambda_max) or lambda_max <= 0:
        raise ValueError(f"invalid lambda_max estimate: {lambda_max}")
    return {
        "lambda_max": lambda_max,
        "n_cells_used": n_used,
        "n_obs_total": n_obs_total,
        "n_vars": n_vars,
        "layer": layer,
        "method": method,
        "seed": int(seed),
    }


def estimate_ts_from_lambda(
    alpha_bar,
    lambda_max: float,
    mode: str = "alpha_bar_threshold",
) -> Dict[str, Any]:
    """Choose t where ``alpha_bar[t] * lambda_max`` is closest to one."""
    if mode != "alpha_bar_threshold":
        raise ValueError(f"unsupported T_s estimation mode: {mode}")
    lam = float(lambda_max)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f"lambda_max must be finite and > 0, got {lambda_max}")
    if hasattr(alpha_bar, "detach"):
        alpha_bar = alpha_bar.detach().cpu().numpy()
    alpha = np.asarray(alpha_bar, dtype=np.float64).reshape(-1)
    if alpha.size == 0:
        raise ValueError("alpha_bar must not be empty")
    alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
    eps = np.finfo(np.float64).tiny
    score = alpha * lam
    t_s = int(np.argmin(np.abs(np.log(np.maximum(score, eps)))))
    return {
        "t_s": t_s,
        "lambda_max": lam,
        "criterion": "argmin_abs_log_alpha_bar_lambda",
        "score_at_ts": float(score[t_s]),
        "alpha_bar_at_ts": float(alpha[t_s]),
    }

