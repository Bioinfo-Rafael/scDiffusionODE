import os
import sys
import unittest
from unittest import mock

import anndata
import numpy as np
import scipy.sparse as sp

WORK = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORK not in sys.path:
    sys.path.insert(0, WORK)

from utils.regime_time import estimate_lambda_max_from_adata, estimate_ts_from_lambda


class RegimeTimeTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.x = rng.normal(size=(100, 8)).astype(np.float32)

    def test_dense_and_sparse(self):
        dense = anndata.AnnData(self.x)
        sparse = anndata.AnnData(sp.csr_matrix(self.x))
        a = estimate_lambda_max_from_adata(dense, n_cells=40, seed=2)
        b = estimate_lambda_max_from_adata(sparse, n_cells=40, seed=2)
        self.assertAlmostEqual(a["lambda_max"], b["lambda_max"], places=4)
        self.assertEqual(a["n_cells_used"], 40)

    def test_sparse_is_subset_before_dense(self):
        adata = anndata.AnnData(sp.csr_matrix(self.x))
        original = sp.csr_matrix.toarray
        seen_shapes = []

        def record(matrix, *args, **kwargs):
            seen_shapes.append(matrix.shape)
            return original(matrix, *args, **kwargs)

        with mock.patch.object(sp.csr_matrix, "toarray", record):
            estimate_lambda_max_from_adata(adata, n_cells=25, seed=1)
        self.assertTrue(seen_shapes)
        self.assertTrue(all(shape[0] <= 25 for shape in seen_shapes))

    def test_ts_threshold(self):
        alpha = np.array([1.0, 0.5, 0.25, 0.125])
        result = estimate_ts_from_lambda(alpha, lambda_max=4.0)
        self.assertEqual(result["t_s"], 2)
        self.assertAlmostEqual(result["score_at_ts"], 1.0)


if __name__ == "__main__":
    unittest.main()

