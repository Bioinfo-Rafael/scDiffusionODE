#!/usr/bin/env python3
"""Focused tests for deterministic grouping and density calculations."""

from __future__ import annotations

import sys
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from analyze_distribution import (  # noqa: E402
    EXPERIMENTS, RunRecord, _matching_existing_sample, _plot_grid, assert_finite,
    assign_groups, discover_runs, histogram_density, real_gene_means,
    real_group_values, select_experiments,
)


class DistributionTests(unittest.TestCase):
    def test_groups_cover_every_gene_once_in_stable_descending_order(self):
        means = np.asarray([4, 1, 4, 3, 0, 2, 2, 5, 1, 0, 3], dtype=np.float64)
        genes = [f"g{i}" for i in range(len(means))]
        assignment, summary, groups = assign_groups(means, genes)
        flattened = np.concatenate(groups)
        self.assertEqual(sorted(flattened.tolist()), list(range(len(means))))
        self.assertEqual(len(set(flattened.tolist())), len(means))
        self.assertEqual(flattened.tolist()[:3], [7, 0, 2])
        self.assertTrue(np.all(np.diff(assignment.real_mean_expression) <= 0))
        self.assertEqual(summary.gene_count.sum(), len(means))
        self.assertLessEqual(summary.gene_count.max() - summary.gene_count.min(), 1)

    def test_density_uses_all_values_including_out_of_range_denominator(self):
        values = np.asarray([-10, 0, 0.5, 1, 10], dtype=np.float32)
        edges = np.asarray([0, 0.5, 1], dtype=np.float64)
        density = histogram_density(values, edges)
        # np.histogram includes the final right edge (1.0), so 0, 0.5, 1 are in range.
        self.assertAlmostEqual(float(np.sum(density * np.diff(edges))), 3 / 5)

    def test_nonfinite_is_not_silently_removed(self):
        with self.assertRaises(FloatingPointError):
            assert_finite("bad", np.asarray([0, np.nan]))

    def test_selection_preserves_canonical_order(self):
        selected = select_experiments("", "", [EXPERIMENTS[5], EXPERIMENTS[0]])
        self.assertEqual(selected, (EXPERIMENTS[0], EXPERIMENTS[5]))
        self.assertEqual(len(select_experiments("ode_only_lincomb", "", [])), 3)
        self.assertEqual(len(select_experiments("", "racipe", [])), 2)

    def test_six_by_five_png_and_pdf_are_created(self):
        records = [RunRecord(name, *name.split("__", 1), "/r", "/c", "/m", "/e", 30000) for name in EXPERIMENTS]
        edges = [np.linspace(-1, 1, 11) for _ in range(5)]
        hist = {}
        for record in records:
            for group in range(5):
                hist[(record.experiment, group, "real")] = np.ones(10)
                hist[(record.experiment, group, "generated")] = np.ones(10) * .8
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory); (output / "figures").mkdir()
            _plot_grid(hist, edges, [1.0] * 5, records, [2] * 5, output, "test_6x5")
            png, pdf = output / "figures/test_6x5.png", output / "figures/test_6x5.pdf"
            self.assertTrue(png.is_file() and png.stat().st_size > 1000)
            self.assertTrue(pdf.is_file() and pdf.read_bytes().startswith(b"%PDF"))

    def test_chunked_real_statistics_and_flatten_count(self):
        matrix = np.arange(35, dtype=np.float32).reshape(7, 5)
        means = real_gene_means(matrix, 7, 5, chunk_rows=3)
        np.testing.assert_allclose(means, matrix.mean(axis=0))
        values = real_group_values(matrix, 7, np.asarray([0, 3]), chunk_rows=2)
        self.assertEqual(values.size, 7 * 2)
        np.testing.assert_array_equal(values.reshape(7, 2), matrix[:, [0, 3]])

    def test_run_checkpoint_and_sample_are_identity_verified(self):
        experiment = EXPERIMENTS[0]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory); run = source / "runs" / experiment / "batch"
            checkpoint_dir = run / "train/checkpoints/segment_000/model"; checkpoint_dir.mkdir(parents=True)
            for name in ("model000004.pt", "opt000004.pt", "ema_0.9999_000004.pt"):
                (checkpoint_dir / name).write_bytes(b"test")
            config = {
                "experiment": experiment, "model_family": "ode_only_lincomb",
                "ode_type": "hill_after_linear", "ema_rate": "0.9999",
                "total_steps": 4, "num_samples": 3, "seed": 1234,
                "diffusion_steps": 20, "n_genes": 2,
            }
            (run / "exp_config.json").write_text(json.dumps(config))
            (run / "manifest.json").write_text(json.dumps({
                "experiment": experiment, "stages": {"train": {"status": "completed"}},
            }))
            record = discover_runs(source, [experiment], "batch", {})[0]
            sample_dir = run / "samples"; sample_dir.mkdir()
            sample = sample_dir / "samples.npz"; np.savez(sample, cell_gen=np.ones((3, 2), np.float32))
            sidecar = sample.with_suffix(".manifest.json")
            sidecar.write_text(json.dumps({
                "checkpoint": record.checkpoint_path, "sample_path": str(sample),
                "seed": 1234, "diffusion_steps": 20,
            }))
            matched = _matching_existing_sample(record, config)
            self.assertIsNotNone(matched)
            self.assertEqual(matched[0], sample.resolve())
            bad = json.loads(sidecar.read_text()); bad["checkpoint"] = str(checkpoint_dir / "ema_0.9999_000003.pt")
            sidecar.write_text(json.dumps(bad))
            self.assertIsNone(_matching_existing_sample(record, config))


if __name__ == "__main__":
    unittest.main(verbosity=2)
