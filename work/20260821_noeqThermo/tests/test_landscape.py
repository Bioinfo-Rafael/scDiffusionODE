#!/usr/bin/env python3
"""Fast mathematical tests independent of model files and Dynamo installation."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "src"))

from noeqthermo.landscape import (
    calibrate_common_sde,
    common_bounds,
    compute_landscape_quantities,
    reflect_into_bounds,
    save_landscape,
    simulate_stationary_distribution,
)


def stable_field(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return np.column_stack((-0.7 * points[:, 0] - 0.2 * points[:, 1], 0.2 * points[:, 0] - 0.7 * points[:, 1]))


class LandscapeTests(unittest.TestCase):
    def setUp(self) -> None:
        theta = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        self.coordinates = np.column_stack((np.cos(theta), np.sin(theta)))
        self.sde = {
            "parameterization": "umap_calibrated",
            "paper_literal_reference": {"dt": [0.1, 0.2], "D": [0.1, 0.2]},
            "dt": None,
            "D": None,
            "drift_step_fraction": 0.05,
            "noise_step_fraction": 0.15,
            "dt_min": 1e-4,
            "dt_max": 1.0,
            "trajectories": 8,
            "steps": 120,
            "burn_in": 20,
            "grid_size": 16,
            "bounds_quantile": 0.0,
            "bounds_margin": 0.1,
            "checkpoint_every": 50,
            "resume": True,
            "potential_cap_above_max": 2.0,
        }

    def test_reflection_handles_large_overshoot(self) -> None:
        bounds = np.array([[-1.0, 1.0], [-2.0, 2.0]])
        points = np.array([[9.5, -14.25], [-8.2, 13.0]])
        reflected = reflect_into_bounds(points, bounds)
        self.assertTrue(np.all(reflected >= bounds[:, 0]))
        self.assertTrue(np.all(reflected <= bounds[:, 1]))

    def test_calibration_is_one_common_pair(self) -> None:
        velocity_a = stable_field(self.coordinates)
        velocity_b = 2.0 * velocity_a
        calibrated = calibrate_common_sde(self.coordinates, [velocity_a, velocity_b], self.sde)
        self.assertGreater(calibrated["dt"], 0)
        self.assertGreater(calibrated["D"], 0)
        self.assertEqual(calibrated["parameterization"], "umap_calibrated")

    def test_simulation_probability_flux_and_resume(self) -> None:
        bounds = common_bounds(self.coordinates, self.sde)
        calibrated = calibrate_common_sde(self.coordinates, [stable_field(self.coordinates)], self.sde)
        with tempfile.TemporaryDirectory() as temporary:
            num, fx, fy, metadata = simulate_stationary_distribution(
                stable_field,
                bounds,
                self.sde,
                calibrated,
                seed=7,
                checkpoint_dir=temporary,
            )
            self.assertEqual(num.sum(), self.sde["trajectories"] * (self.sde["steps"] - self.sde["burn_in"]))
            self.assertFalse(metadata["resumed"])
            num2, fx2, fy2, metadata2 = simulate_stationary_distribution(
                stable_field,
                bounds,
                self.sde,
                calibrated,
                seed=999,
                checkpoint_dir=temporary,
            )
            np.testing.assert_array_equal(num, num2)
            np.testing.assert_allclose(fx, fx2)
            np.testing.assert_allclose(fy, fy2)
            self.assertTrue(metadata2["resumed"])
            result = compute_landscape_quantities(
                stable_field,
                bounds,
                num,
                fx,
                fy,
                diffusion=float(calibrated["D"]),
            )
            self.assertAlmostEqual(result.probability.sum(), 1.0)
            self.assertEqual(result.curl.shape, num.shape)
            self.assertTrue(np.isinf(result.potential[~result.visited]).all())
            save_landscape(result, temporary)
            self.assertTrue((Path(temporary) / "landscape_flux_arrays.npz").is_file())
            for name in ("num_tra", "p_tra", "pot_U", "mean_Fx", "Xgrid", "Ygrid"):
                self.assertTrue((Path(temporary) / f"{name}.csv").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
