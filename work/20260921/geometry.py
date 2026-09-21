"""Independent local PCA and orthogonal projection diagnostics (CPU, float64)."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def project(vectors, basis):
    """Row vectors; basis columns are orthonormal. Never construct a projector."""
    v = np.asarray(vectors, dtype=np.float64)
    tangent = (v @ basis) @ basis.T
    return tangent, v - tangent


def ratio(numerator, denominator):
    a, b = np.broadcast_arrays(numerator, denominator)
    return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b > 0)


class LocalGeometry:
    def __init__(self, real_z, mean_projected, k=50, dimensions=(5, 10, 20)):
        self.real = np.asarray(real_z, dtype=np.float64)
        self.mean_projected = np.asarray(mean_projected, dtype=np.float64)
        self.k, self.dimensions = k, tuple(dimensions)
        if not (1 <= min(dimensions) <= max(dimensions) <= min(k - 1, self.real.shape[1])):
            raise ValueError('Require 1 <= tangent d <= min(local k-1, PCA dimension)')
        if k >= len(self.real):
            raise ValueError('Need more real cells than local k (anchor itself is excluded)')
        self.nn = NearestNeighbors(algorithm='brute', n_jobs=1).fit(self.real)
        self.cache = {}

    def scaled(self, clean_z, a):
        # C(a*x-mu) = a*C(x-mu)+(a-1)*C*mu, with a=sqrt(alpha_bar_t).
        return a * clean_z + (a - 1) * self.mean_projected

    def anchors(self, query_z, a=1.0):
        if not np.isfinite(a) or a <= 0:
            raise ValueError('Scaled-manifold factor must be finite and positive')
        clean_query = (query_z - (a - 1) * self.mean_projected) / a
        distances, indices = self.nn.kneighbors(clean_query, n_neighbors=1)
        return indices[:, 0], distances[:, 0] * a

    def local(self, anchor):
        anchor = int(anchor)
        if anchor not in self.cache:
            indices = self.nn.kneighbors(self.real[anchor:anchor + 1],
                                          n_neighbors=self.k + 1, return_distance=False)[0]
            indices = indices[indices != anchor][:self.k]
            neighborhood = self.real[indices]
            center = neighborhood.mean(axis=0)
            _, singular, vt = np.linalg.svd(neighborhood - center, full_matrices=False)
            self.cache[anchor] = (center, vt[:max(self.dimensions)].T, singular)
        return self.cache[anchor]

    def decompose(self, query_z, vectors, a, timestep, trajectory_ids):
        query_z = np.asarray(query_z, dtype=np.float64)
        anchors, nearest = self.anchors(query_z, a)
        rows = []
        for anchor in np.unique(anchors):
            idx = np.flatnonzero(anchors == anchor)
            center, all_basis, singular = self.local(anchor)
            residual = query_z[idx] - self.scaled(center, a)
            total_variance = np.square(singular).sum()
            for d in self.dimensions:
                basis = all_basis[:, :d]
                _, residual_n = project(residual, basis)
                distance = np.linalg.norm(residual_n, axis=1)
                data = dict(manifold_distance=distance, nearest_real_distance=nearest[idx])
                for name, values in vectors.items():
                    v = np.asarray(values[idx], dtype=np.float64)
                    tangent, normal = project(v, basis)
                    norm = np.linalg.norm(v, axis=1)
                    tn, nn = np.linalg.norm(tangent, axis=1), np.linalg.norm(normal, axis=1)
                    data.update({name + '_norm': norm, name + '_tangent_norm': tn,
                                 name + '_normal_norm': nn,
                                 name + '_tangent_ratio': ratio(tn, norm),
                                 name + '_normal_ratio': ratio(nn, norm),
                                 name + '_tangent_fraction_sq': ratio(tn**2, norm**2),
                                 name + '_normal_fraction_sq': ratio(nn**2, norm**2)})
                drift = vectors['model_drift'][idx]
                data['cos_normal'] = ratio(np.sum(drift * -residual_n, axis=1),
                                           np.linalg.norm(drift, axis=1) * distance)
                rank = int(np.sum(singular > singular[0] * max(self.k, self.real.shape[1])
                                  * np.finfo(float).eps)) if singular[0] > 0 else 0
                for j, position in enumerate(idx):
                    rows.append(dict(t=timestep, trajectory_id=int(trajectory_ids[position]),
                                     anchor_real_id=int(anchor), d=d,
                                     regime='main_low_noise' if timestep <= 200 else 'reference_high_noise',
                                     local_rank=rank, dimension_exceeds_local_rank=d > rank,
                                     local_explained_fraction=float(ratio(np.square(singular[:d]).sum(), total_variance)),
                                     **{key: float(value[j]) for key, value in data.items()}))
        return rows
