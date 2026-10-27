"""Smooth, bounded-size PCA + Gaussian RBF regression (our extension, not CellRank/ddHodge)."""
import numpy as np
from scipy.linalg import solve
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


class ScalarExtension:
    def __init__(self, n_components=16, n_centers=128, ridge=1e-6, seed=0,
                 max_validation_nrmse=0.25, batch_size=512):
        self.settings = dict(n_components=n_components, n_centers=n_centers, ridge=ridge,
                             seed=seed, max_validation_nrmse=max_validation_nrmse, batch_size=batch_size)

    def _features(self, z):
        r = np.exp(-cdist(z, self.centers, 'sqeuclidean') / (2*self.bandwidth**2))
        return np.column_stack((np.ones(len(z)), z / self.scale, r))

    def fit(self, x, targets):
        x, targets = np.asarray(x, dtype=float), np.asarray(targets, dtype=float)
        if len(x) < 12 or not np.isfinite(x).all() or not np.isfinite(targets).all():
            raise ValueError('Scalar fit needs at least 12 finite eligible cells')
        if np.ptp(targets) <= 1e-12:
            raise ValueError('Constant scalar: no identifiable regular level set')
        cfg = self.settings
        rng = np.random.default_rng(cfg['seed'])
        order = rng.permutation(len(x))
        heldout = order[:max(2, len(x)//5)]
        train = order[len(heldout):]
        pca_rows = train[:min(4096, len(train))]
        dim = min(cfg['n_components'], x.shape[1], len(pca_rows)-1)
        pca = PCA(n_components=dim, svd_solver='randomized', random_state=cfg['seed']).fit(x[pca_rows])
        self.mean, self.basis = pca.mean_, pca.components_.T
        z = (x-self.mean) @ self.basis
        self.scale = max(float(np.sqrt(np.mean(z[train]**2))), 1e-12)
        self.centers = z[train[:min(cfg['n_centers'], len(train))]].copy()
        distances = cdist(self.centers, self.centers)
        positive = distances[distances > 0]
        if not len(positive):
            raise ValueError('Degenerate scalar coordinates')
        self.bandwidth = float(np.median(positive))
        width = 1 + dim + len(self.centers)
        gram, rhs = np.zeros((width, width)), np.zeros(width)
        for start in range(0, len(train), cfg['batch_size']):
            rows = train[start:start+cfg['batch_size']]
            phi = self._features(z[rows])
            gram += phi.T @ phi
            rhs += phi.T @ targets[rows]
        penalty = np.eye(width)*cfg['ridge']*len(train)
        penalty[0, 0] = 0
        self.coefficients = solve(gram+penalty, rhs, assume_a='pos')
        prediction = self.values(x)
        span = float(np.ptp(targets))
        self.metrics = dict(training_rmse=float(np.sqrt(np.mean((prediction[train]-targets[train])**2))),
                            validation_rmse=float(np.sqrt(np.mean((prediction[heldout]-targets[heldout])**2))),
                            max_abs_error=float(np.max(abs(prediction-targets))),
                            validation_count=len(heldout), fit_count=len(train),
                            target_range=span, pca_explained_variance=float(pca.explained_variance_ratio_.sum()))
        self.metrics['validation_nrmse'] = self.metrics['validation_rmse']/span
        if self.metrics['validation_nrmse'] > cfg['max_validation_nrmse']:
            raise ValueError(f'Scalar held-out interpolation error exceeds limit: {self.metrics}')
        self.valid_range = (float(prediction.min()), float(prediction.max()))
        # Original-expression anchors give an explicit, conservative support diagnostic.
        self.anchors = x[order[:min(4096, len(x))]].copy()
        self.tree = cKDTree(self.anchors)
        nearest = self.tree.query(x, k=2)[0][:, 1]
        self.support_radius = max(float(np.quantile(nearest, .99))*2, 1e-10)
        self.metrics['support_radius'] = self.support_radius
        self.metrics['support_definition'] = 'Euclidean distance to nearest of at most 4096 original-space anchors'
        self.training_rows, self.validation_rows = train, heldout
        return self

    def values(self, x):
        x = np.atleast_2d(x)
        values = np.empty(len(x))
        for start in range(0, len(x), self.settings['batch_size']):
            z = (x[start:start+self.settings['batch_size']]-self.mean) @ self.basis
            values[start:start+len(z)] = self._features(z) @ self.coefficients
        return values

    def gradient(self, x):
        x = np.atleast_2d(x)
        z = (x-self.mean) @ self.basis
        r = self._features(z)[:, 1+self.basis.shape[1]:]
        weights = r * self.coefficients[1+self.basis.shape[1]:]
        grad_z = (weights @ self.centers - weights.sum(axis=1)[:, None]*z)/self.bandwidth**2
        grad_z += self.coefficients[1:1+self.basis.shape[1]]/self.scale
        return grad_z @ self.basis.T

    def distribution_distance(self, x):
        return self.tree.query(np.atleast_2d(x))[0]

    def arrays(self):
        return {key: np.asarray(getattr(self, key)) for key in
                ('mean', 'basis', 'centers', 'coefficients', 'bandwidth', 'scale', 'anchors',
                 'training_rows', 'validation_rows')}

    @classmethod
    def restore(cls, arrays, metadata):
        obj = cls(**metadata['settings'])
        for key, value in arrays.items():
            setattr(obj, key, value)
        obj.metrics = metadata['metrics']
        obj.valid_range = tuple(metadata['valid_range'])
        obj.support_radius = obj.metrics['support_radius']
        obj.tree = cKDTree(obj.anchors)
        return obj

    def metadata(self):
        return dict(method='PCA Gaussian RBF ridge with affine term; held-out split',
                    settings=self.settings, metrics=self.metrics, valid_range=list(self.valid_range))
