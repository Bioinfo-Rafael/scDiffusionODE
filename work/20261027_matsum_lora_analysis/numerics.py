"""Pure numerical definitions. Axes: cells × components; matrices: target × source."""
import numpy as np


def coefficient_moments(a):
    """Population variance/covariance (ddof=0), plus mean(abs(a)), not abs(mean(a))."""
    a = np.asarray(a, dtype=np.float64)
    mean = a.mean(axis=0)
    centered = a - mean
    covariance = centered.T @ centered / len(a)
    return dict(mean=mean, mean_abs=np.abs(a).mean(axis=0),
                variance=np.diag(covariance).copy(), covariance=covariance,
                std=np.sqrt(np.maximum(np.diag(covariance), 0)))


def normalized_coefficients(a):
    """L1 normalization; zero vectors remain NaN (never manufacture a composition)."""
    total = np.abs(a).sum(axis=1)
    out = np.full(a.shape, np.nan, dtype=np.float64)
    np.divide(a, total[:, None], out=out, where=total[:, None] > 0)
    return out, total


def relative_std(mean, std, relative=1e-6, absolute=1e-12):
    """Signed std/mean; omit means near zero relative to the array's maximum |mean|."""
    threshold = max(absolute, relative * float(np.max(np.abs(mean))))
    valid = np.abs(mean) > threshold
    ratio = np.full(mean.shape, np.nan, dtype=np.float64)
    np.divide(std, mean, out=ratio, where=valid)
    return ratio, valid, threshold


def component_moments(mean_a, std_a, basis):
    """Exact cell moments of C_ck = a_ck B_k, without cell × d × d allocation."""
    return mean_a[:, None, None] * basis, std_a[:, None, None] * np.abs(basis)


def effective_moments(mean_a, covariance_a, basis, base):
    """E[W]=W0+Σ E[a_k]B_k; Var(W_ij)=B_:ijᵀ Cov(a) B_:ij."""
    b = basis.reshape(len(basis), -1).astype(np.float64)
    mean = (mean_a @ b).reshape(base.shape) + base
    variance = np.sum(b * (covariance_a @ b), axis=0).reshape(base.shape)
    return mean, np.sqrt(np.maximum(variance, 0))


def variance_decomposition(a, labels):
    """Total = size-weighted within + between. Returns one result per component."""
    total = np.var(a, axis=0)
    mean = np.mean(a, axis=0)
    within = np.zeros(a.shape[1])
    between = np.zeros(a.shape[1])
    for label in np.unique(labels):
        group = a[labels == label]
        p = len(group) / len(a)
        within += p * np.var(group, axis=0)
        between += p * (group.mean(axis=0) - mean) ** 2
    eta = np.divide(between, total, out=np.full_like(total, np.nan), where=total > 0)
    return dict(total=total, within=within, between=between, eta_squared=eta)


def sample_by_type(labels, count, seed):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(labels == label), min(count, np.sum(labels == label)), replace=False)
        for label in np.unique(labels)
    ]))


def gene_strength(matrix):
    """Off-diagonal |weight| sums: incoming per row, outgoing per column."""
    weight = np.abs(matrix).copy()
    np.fill_diagonal(weight, 0)
    return weight.sum(axis=1), weight.sum(axis=0)
