#!/usr/bin/env python3
"""Unit tests for original-space vector-field calculations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parent
REPO_ROOT = ANALYSIS_ROOT.parent.parent
SOURCE_SUITE = REPO_ROOT / "work" / "20260803_ODE_hill_exp"
for search_path in (str(REPO_ROOT), str(SOURCE_SUITE), str(ANALYSIS_ROOT)):
    if search_path in sys.path:
        sys.path.remove(search_path)
    sys.path.insert(0, search_path)

from models.ode_fields import HillAfterLinearField  # noqa: E402
from dynamo_analysis import (  # noqa: E402
    acceleration_gene_summary,
    aggregate_jacobians,
    evaluate_dataset,
    find_fixed_points,
    jacobian_gene_summary,
    jacobian_top_interactions,
    summarize_metrics,
)
from vector_field_adapter import TrainedODEVectorField  # noqa: E402


class VectorFieldMathTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260817)
        self.field = HillAfterLinearField(
            d=5,
            is_lincomb=False,
            num_components=1,
            mask=None,
            use_decay=True,
            soft=False,
            off_mask_lambda=0.0,
            hill_n=2.0,
        ).eval()
        self.adapter = TrainedODEVectorField(self.field, device="cpu", batch_size=2)
        rng = np.random.default_rng(20260817)
        self.X = rng.lognormal(mean=0.1, sigma=0.4, size=(4, 5)).astype(np.float32)

    def test_analytic_jacobian_matches_jacrev(self) -> None:
        analytical = self.adapter.jacobian_tensor(self.X, method="analytical").numpy()
        autograd = self.adapter.jacobian_tensor(self.X, method="autograd").numpy()
        np.testing.assert_allclose(analytical, autograd, rtol=2e-5, atol=2e-6)

        single = self.adapter.get_Jacobian()(self.X[0])
        batched = self.adapter.get_Jacobian()(self.X)
        self.assertEqual(single.shape, (5, 5))
        self.assertEqual(batched.shape, (5, 5, 4))
        np.testing.assert_allclose(single, analytical[0], rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(
            np.moveaxis(batched, 2, 0), analytical, rtol=1e-6, atol=1e-7
        )

    def test_divergence_and_acceleration_match_full_jacobian(self) -> None:
        evaluated = self.adapter.evaluate(self.X)
        jacobians = self.adapter.jacobian_tensor(self.X).numpy().astype(np.float64)
        expected_divergence = np.trace(jacobians, axis1=1, axis2=2)
        expected_acceleration = np.einsum(
            "nij,nj->ni", jacobians, evaluated.velocity.astype(np.float64)
        )
        np.testing.assert_allclose(
            evaluated.divergence, expected_divergence, rtol=2e-5, atol=2e-6
        )
        np.testing.assert_allclose(
            evaluated.acceleration, expected_acceleration, rtol=2e-5, atol=2e-6
        )
        product = self.adapter.jvp(self.X[0], evaluated.velocity[0])
        np.testing.assert_allclose(
            product, expected_acceleration[0], rtol=2e-5, atol=2e-6
        )

    def test_metrics_and_streamed_rankings_are_finite(self) -> None:
        real_table, real_result = evaluate_dataset(
            self.adapter,
            self.X,
            dataset="real",
            annotations=np.array(["A", "A", "B", "B"]),
        )
        generated_table, generated_result = evaluate_dataset(
            self.adapter, self.X[::-1].copy(), dataset="generated"
        )
        metrics = summarize_metrics(
            __import__("pandas").concat([real_table, generated_table], ignore_index=True)
        )
        self.assertFalse(metrics.empty)
        self.assertTrue(np.isfinite(metrics.select_dtypes(include=[np.number])).all().all())

        aggregates = aggregate_jacobians(
            self.adapter, {"real": self.X[:2], "generated": self.X[2:]}
        )
        self.assertEqual(set(aggregates), {"real", "generated", "combined"})
        genes = [f"g{i}" for i in range(5)]
        gene_summary = jacobian_gene_summary(aggregates, genes)
        interactions = jacobian_top_interactions(aggregates, genes, top_n=3)
        acceleration = acceleration_gene_summary(
            {"real": real_result, "generated": generated_result}, genes
        )
        self.assertEqual(len(gene_summary), 15)
        self.assertFalse(interactions.empty)
        self.assertEqual(len(acceleration), 10)

    def test_fixed_point_search_and_stability(self) -> None:
        with torch.no_grad():
            self.field.W.zero_()
            self.field.b.zero_()
        expected = self.adapter.func(np.zeros(5, dtype=np.float32)) / (
            self.field.delta.detach().numpy()
        )
        result = find_fixed_points(
            self.adapter,
            np.full((1, 5), 0.2, dtype=np.float32),
            ["test_seed"],
            np.column_stack((np.zeros(5), np.full(5, 2.0))),
            residual_tolerance=1e-6,
            redundant_tolerance=1e-4,
            max_iterations=10,
            full_eigen_max_dim=32,
            leading_eigenvalues=2,
            eigen_tolerance=1e-6,
            stability_tolerance=1e-7,
        )
        self.assertEqual(len(result.points), 1)
        np.testing.assert_allclose(result.points[0], expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(result.metadata.loc[0, "stability"], "stable")


if __name__ == "__main__":
    unittest.main()
