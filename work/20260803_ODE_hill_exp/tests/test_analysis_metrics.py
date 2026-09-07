#!/usr/bin/env python3
"""Regression checks for numerically stable analysis reductions."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from viz.analysis_helpers import _moments  # noqa: E402


class AnalysisMetricTests(unittest.TestCase):
    def test_float32_finite_ratio_std_is_aggregated_without_overflow(self):
        values = np.asarray([1.0e22, 1.5e22, 2.0e22], dtype=np.float32)
        with np.errstate(over="raise", invalid="raise"):
            mean, std = _moments([values])
        self.assertTrue(np.isfinite(mean))
        self.assertTrue(np.isfinite(std))
        self.assertGreater(std, 0.0)
        self.assertAlmostEqual(mean, float(values.astype(np.float64).mean()))

    def test_nonfinite_input_is_still_rejected(self):
        with self.assertRaises(FloatingPointError):
            _moments([np.asarray([1.0, np.inf], dtype=np.float32)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
