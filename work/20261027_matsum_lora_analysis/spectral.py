"""Symmetrization, graph Laplacians, low-frequency eigenspaces and 8×8 comparisons."""
import numpy as np
from scipy.linalg import eigh


def laplacians(weight):
    """S=abs((W+W.T)/2), diag(S)=0. Isolates have zero normalized-L diagonal."""
    adjacency = np.abs((weight.astype(np.float64) + weight.T) / 2)
    np.fill_diagonal(adjacency, 0)
    degree = adjacency.sum(axis=1)
    ordinary = np.diag(degree) - adjacency
    inv = np.divide(1, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    normalized = ordinary * inv[:, None] * inv[None, :]
    return adjacency, degree, ordinary, normalized


def spectrum(weight, dimensions):
    """All sorted eigenvalues; first q eigenvectors (including zero modes)."""
    adjacency, degree, ordinary, normalized = laplacians(weight)
    if not 1 <= dimensions < len(weight):
        raise ValueError('spectral_dimensions must be between 1 and d-1')
    result = dict(adjacency=adjacency, degree=degree)
    for name, lap in [('ordinary', ordinary), ('normalized', normalized)]:
        values = eigh(lap, eigvals_only=True, check_finite=True)
        _, vectors = eigh(lap, subset_by_index=[0, dimensions - 1], check_finite=True)
        tolerance = 1e-9 * max(1., np.max(np.abs(values)))
        if values.min() < -tolerance:
            raise ValueError('Laplacian has materially negative eigenvalues')
        gap = float(values[dimensions] - values[dimensions - 1])
        result[name] = dict(eigenvalues=values, eigenvectors=vectors,
                            cutoff_gap=gap, cutoff_degenerate=gap <= tolerance,
                            zero_modes=int(np.sum(np.abs(values) <= tolerance)),
                            zero_mode_tolerance=tolerance, isolated_nodes=int(np.sum(degree == 0)))
    return result


def spectral_comparisons(eigenvalues, eigenvectors, dimensions):
    """Low-q RMSD, ||Qi.T Qj||F, normalized overlap and projector/chordal distance."""
    low = eigenvalues[:, :dimensions]
    rmsd = np.sqrt(np.mean((low[:, None] - low[None, :]) ** 2, axis=-1))
    n = len(low)
    overlap = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            overlap[i, j] = np.linalg.norm(eigenvectors[i].T @ eigenvectors[j], ord='fro')
    similarity = np.clip(overlap / np.sqrt(dimensions), 0, 1)
    distance = np.sqrt(np.maximum(1 - similarity**2, 0))
    np.fill_diagonal(distance, 0)
    return dict(low_eigenvalue_rmsd=rmsd, eigenvector_product_frobenius=overlap,
                eigenspace_similarity=similarity, eigenspace_distance=distance)


def matrix_cosines(basis):
    """Cosine on flattened signed matrices; zero-norm pairs are undefined (NaN)."""
    flat = basis.reshape(len(basis), -1).astype(np.float64)
    norms = np.linalg.norm(flat, axis=1)
    divisor = norms[:, None] * norms[None, :]
    similarity = np.divide(flat @ flat.T, divisor, out=np.full(divisor.shape, np.nan), where=divisor > 0)
    similarity = np.clip(similarity, -1, 1)
    return similarity, 1 - similarity
