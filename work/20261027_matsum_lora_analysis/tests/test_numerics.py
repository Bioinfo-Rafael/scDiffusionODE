"""Numerical identities and invariances; python -m unittest discover -s .../tests -v."""
import sys
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from numerics import coefficient_moments, normalized_coefficients, component_moments, effective_moments, relative_std, variance_decomposition, gene_strength
from spectral import laplacians, spectrum, spectral_comparisons, matrix_cosines
from plots import histogram_edges


class NumericalTests(unittest.TestCase):
    def test_exact_moments_match_explicit_cell_matrices(self):
        rng = np.random.default_rng(10)
        a, b, w0 = rng.normal(size=(17, 8)), rng.normal(size=(8, 5, 5)), rng.normal(size=(5, 5))
        moments = coefficient_moments(a)
        c = a[:, :, None, None] * b
        cm, cs = component_moments(moments['mean'], moments['std'], b)
        np.testing.assert_allclose(cm, c.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(cs, c.std(axis=0), atol=1e-14)
        wm, ws = effective_moments(moments['mean'], moments['covariance'], b, w0)
        w = w0 + c.sum(axis=1)
        np.testing.assert_allclose(wm, w.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(ws, w.std(axis=0), atol=1e-14)

    def test_signed_mean_is_not_mean_absolute(self):
        m = coefficient_moments(np.array([[-2., 0], [2., 0]]))
        np.testing.assert_equal(m['mean'], [0, 0])
        np.testing.assert_equal(m['mean_abs'], [2, 0])
        np.testing.assert_equal(m['variance'], [4, 0])

    def test_zero_normalization_and_ratio(self):
        values, totals = normalized_coefficients(np.array([[0., 0], [-2., 1]]))
        self.assertTrue(np.isnan(values[0]).all())
        np.testing.assert_allclose(values[1], [-2 / 3, 1 / 3])
        ratio, valid, threshold = relative_std(np.array([0., 1e-13, -2., 2.]), np.ones(4))
        np.testing.assert_equal(valid, [False, False, True, True])
        np.testing.assert_allclose(ratio[2:], [-.5, .5])

    def test_variance_decomposition(self):
        a = np.array([[0., 3], [2., 3], [5., 3], [7., 3], [9., 3]])
        result = variance_decomposition(a, np.array(['A', 'A', 'B', 'B', 'B']))
        np.testing.assert_allclose(result['total'], result['within'] + result['between'])
        self.assertTrue(np.isnan(result['eta_squared'][1]))

    def test_laplacian_definition_scaling_and_isolates(self):
        w = np.array([[100., 2, 0, 0], [-4., 0, 3, 0], [0, 1, 0, 0], [0, 0, 0, 99.]])
        s, d, l, normalized = laplacians(w)
        self.assertEqual(s[0, 1], 1.)  # abs AFTER symmetrization
        self.assertTrue((np.diag(s) == 0).all())
        np.testing.assert_allclose(l.sum(axis=1), 0)
        np.testing.assert_allclose(normalized[-1], 0)
        scaled = laplacians(-3 * w)
        np.testing.assert_allclose(scaled[2], 3 * l)
        np.testing.assert_allclose(scaled[3], normalized)
        self.assertGreaterEqual(np.linalg.eigvalsh(l).min(), -1e-12)

    def test_eigenspace_invariant_to_sign_and_rotation(self):
        rng = np.random.default_rng(0)
        q = np.linalg.qr(rng.normal(size=(12, 3)))[0]
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        vectors = np.array([q, q * [-1, 1, -1], q @ rotation])
        result = spectral_comparisons(np.tile(np.arange(12), (3, 1)), vectors, 3)
        np.testing.assert_allclose(result['eigenspace_similarity'], 1, atol=1e-14)
        np.testing.assert_allclose(result['eigenspace_distance'], 0, atol=3e-8)
        np.testing.assert_allclose(result['low_eigenvalue_rmsd'], 0)

    def test_spectrum_cutoff_degeneracy_reported(self):
        result = spectrum(np.zeros((5, 5)), 2)
        self.assertTrue(result['ordinary']['cutoff_degenerate'])
        self.assertEqual(result['normalized']['isolated_nodes'], 5)

    def test_cosine_and_gene_orientation(self):
        w = np.array([[8., -2], [3., 6.]])
        similarity, distance = matrix_cosines(np.array([w, 2 * w, -w, np.zeros_like(w)]))
        self.assertAlmostEqual(similarity[0, 1], 1)
        self.assertAlmostEqual(distance[0, 2], 2)
        self.assertTrue(np.isnan(similarity[3]).all())
        incoming, outgoing = gene_strength(w)
        np.testing.assert_equal(incoming, [2, 3])
        np.testing.assert_equal(outgoing, [3, 2])

    def test_near_constant_histogram_has_positive_bin_width(self):
        edges = histogram_edges(np.array([1., 1. + 1e-15, np.nan]), 60)
        self.assertTrue((np.diff(edges) > 0).all())

    def test_original_models_reconstruct_from_exported_basis(self):
        import torch
        import anndata as ad
        from ODE.ode_20260609_mathmlp import MatSumField, LoRAField
        from load_model import shared_matrices, extract_coefficients
        from run import verify_reconstruction
        torch.manual_seed(1)
        a = ad.AnnData(np.random.default_rng(1).gamma(2, 1, size=(5, 7)).astype(np.float32))
        for cls in (MatSumField, LoRAField):
            field = cls(7, K=8, time_dim=8, hidden=12).eval()
            b, base = shared_matrices(field)
            coeff = extract_coefficients(field, a, 49, 2)
            report = verify_reconstruction(field, a, coeff, b, base, 49)
            self.assertLess(report['max_absolute_forward_error'], 1e-5)


if __name__ == '__main__': unittest.main()
