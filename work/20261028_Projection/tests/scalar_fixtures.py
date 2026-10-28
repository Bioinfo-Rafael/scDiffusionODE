import numpy as np


class PlaneScalar:
    valid_range = (-1e6, 1e6)
    support_radius = 1e6
    def values(self, x): return np.atleast_2d(x)[:, 0]
    def gradient(self, x):
        result = np.zeros_like(np.atleast_2d(x), dtype=float); result[:, 0] = 1
        return result
    def hessian_vector_product(self, y, v): return np.zeros_like(v, dtype=float)
    def hessian_diagonal(self, y): return np.zeros_like(y, dtype=float)
    def hessian_trace(self, y): return 0.
    def distribution_distance(self, x): return np.zeros(len(np.atleast_2d(x)))


class SphereScalar(PlaneScalar):
    def values(self, x): return np.sum(np.atleast_2d(x)**2, axis=1)/2
    def gradient(self, x): return np.atleast_2d(x).astype(float)
    def hessian_vector_product(self, y, v): return np.asarray(v, dtype=float)
    def hessian_diagonal(self, y): return np.ones_like(y, dtype=float)
    def hessian_trace(self, y): return float(len(y))
