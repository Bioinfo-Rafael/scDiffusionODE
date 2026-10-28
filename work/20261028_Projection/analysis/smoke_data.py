"""Explicit synthetic integration fixture. Never used as a fallback for absent real data."""
import numpy as np
import pandas as pd
import torch
import anndata as ad
import scanpy as sc
from .diagnostics import cell_classes
from .surface_divergence import TorchDerivatives


class SyntheticVelocity:
    def __init__(self, dimension):
        self.feature_ids = [f'SYN{i+1:04d}' for i in range(dimension)]
        self.provenance = dict(identity='synthetic affine field; no trained checkpoint', data_kind='synthetic')
        rates = np.linspace(.05, .25, dimension)
        matrix = -np.diag(rates)
        matrix[0, 1] += .15; matrix[1, 0] -= .15
        self.matrix = torch.tensor(matrix, dtype=torch.float64)
        self.direction = torch.tensor(np.linspace(.4, 1., dimension), dtype=torch.float64)

    def torch_single(self, x):
        return self.direction+self.matrix @ (x-2)

    def __call__(self, x):
        value = torch.as_tensor(np.asarray(x), dtype=torch.float64)
        return (self.direction+(value-2) @ self.matrix.T).numpy()


def make_smoke(seed=28):
    rng = np.random.default_rng(seed)
    n, d = 200, 10
    time = np.linspace(-1, 1, n)
    direction = np.linspace(.4, 1., d)
    x = 2+time[:, None]*direction+rng.normal(0, .12, (n, d))
    velocity = SyntheticVelocity(d)
    labels = np.repeat(cell_classes(), n//5)
    a = ad.AnnData(x, obs=pd.DataFrame({'celltype': pd.Categorical(labels)},
        index=[f'synthetic_{i:04d}' for i in range(n)]), var=pd.DataFrame(index=velocity.feature_ids))
    a.uns['data_kind'] = 'synthetic: class names are test labels, not biological observations'
    sc.pp.neighbors(a, n_neighbors=15, use_rep='X', random_state=seed)
    sc.tl.umap(a, random_state=seed)
    return velocity, a, TorchDerivatives(velocity.torch_single)
