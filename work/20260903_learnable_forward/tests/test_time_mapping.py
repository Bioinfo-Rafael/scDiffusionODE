#!/usr/bin/env python3
"""Tests for the physical SDE clock and batch-shared sampler contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


SUITE_ROOT = Path(__file__).resolve().parents[1]
if str(SUITE_ROOT) not in sys.path:
    sys.path.insert(0, str(SUITE_ROOT))

from diffusion.free_affine import FreeAffineForward  # noqa: E402
from diffusion.stationary_qd import StationaryQDForward  # noqa: E402
from diffusion.time_mapping import PhysicalTimeMap  # noqa: E402
from diffusion.timestep_sampler import BatchSharedPhysicalTimeSampler  # noqa: E402
from models.factory import validate_training_policy  # noqa: E402


class PhysicalTimeMapTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260903)

    @staticmethod
    def _time_map() -> PhysicalTimeMap:
        betas = torch.tensor([0.05, 0.11, 0.19, 0.27], dtype=torch.float64)
        return PhysicalTimeMap.from_betas(betas)

    def test_grid_is_negative_log_alpha_bar(self):
        betas = torch.tensor([0.05, 0.11, 0.19, 0.27], dtype=torch.float64)
        time_map = PhysicalTimeMap.from_betas(betas)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        torch.testing.assert_close(time_map.physical_grid, -torch.log(alpha_bar))
        self.assertEqual(time_map.num_timesteps, betas.numel())
        self.assertAlmostEqual(time_map.boundary_time, float(-torch.log(alpha_bar[0])))
        self.assertAlmostEqual(time_map.terminal_time, float(-torch.log(alpha_bar[-1])))
        self.assertAlmostEqual(
            time_map.duration,
            time_map.terminal_time - time_map.boundary_time,
        )

    def test_fractional_index_and_physical_time_round_trip(self):
        time_map = self._time_map()
        fractional_indices = torch.tensor(
            [0.0, 0.23, 1.0, 1.61, 2.37, 3.0], dtype=torch.float64
        )
        physical_times = time_map.fractional_index_to_time(fractional_indices)
        recovered = time_map.physical_time_to_fractional_index(physical_times)
        torch.testing.assert_close(
            recovered, fractional_indices, atol=2e-15, rtol=2e-15
        )

        integer_indices = torch.arange(4, dtype=torch.float64)
        torch.testing.assert_close(
            time_map.fractional_index_to_time(integer_indices),
            time_map.physical_grid,
            atol=0.0,
            rtol=0.0,
        )

    def test_time_grid_recovers_standard_q_transition_at_both_initializations(self):
        betas = torch.tensor([0.04, 0.09, 0.16, 0.23], dtype=torch.float64)
        time_map = PhysicalTimeMap.from_betas(betas)
        alpha_bar = torch.cumprod(1.0 - betas, dim=0)
        x = torch.tensor(
            [[0.2, -0.3, 0.5], [-0.4, 0.1, 0.7]], dtype=torch.float64
        )

        processes = (
            StationaryQDForward(3, aux_dim=2, dtype=torch.float64),
            FreeAffineForward(3, dtype=torch.float64),
        )
        for process in processes:
            with self.subTest(process=type(process).__name__):
                for index in range(time_map.num_timesteps):
                    stats = process.transition_stats(
                        x, time_map.physical_grid[index]
                    )
                    expected_phi = torch.sqrt(alpha_bar[index]) * torch.eye(
                        3, dtype=torch.float64
                    )
                    expected_covariance = (1.0 - alpha_bar[index]) * torch.eye(
                        3, dtype=torch.float64
                    )
                    torch.testing.assert_close(
                        (stats.materialize_for_analysis()["phi"] if isinstance(process, StationaryQDForward) else stats.transition_matrix),
                        expected_phi,
                        atol=1e-11,
                        rtol=1e-11,
                    )
                    torch.testing.assert_close(
                        stats.mean,
                        torch.sqrt(alpha_bar[index]) * x,
                        atol=1e-11,
                        rtol=1e-11,
                    )
                    torch.testing.assert_close(
                        (stats.materialize_for_analysis()["covariance"] if isinstance(process, StationaryQDForward) else stats.covariance),
                        expected_covariance,
                        atol=1e-11,
                        rtol=1e-11,
                    )

    def test_batch_shared_sampler_repeats_one_time_and_returns_unit_weights(self):
        time_map = self._time_map()
        sampler = BatchSharedPhysicalTimeSampler(time_map)
        timesteps, weights = sampler.sample(9, torch.device("cpu"))

        self.assertEqual(timesteps.shape, (9,))
        self.assertEqual(weights.shape, (9,))
        torch.testing.assert_close(
            timesteps, timesteps[:1].expand_as(timesteps), atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(weights, torch.ones_like(weights), atol=0.0, rtol=0.0)
        physical = time_map.fractional_index_to_time(timesteps.to(torch.float64))
        torch.testing.assert_close(
            physical, physical[:1].expand_as(physical), atol=0.0, rtol=0.0
        )
        self.assertGreaterEqual(float(physical[0]), time_map.boundary_time)
        self.assertLessEqual(float(physical[0]), time_map.terminal_time)

        with self.assertRaisesRegex(ValueError, "batch_size"):
            sampler.sample(0, torch.device("cpu"))

    def test_invalid_time_grids_and_out_of_range_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PhysicalTimeMap(torch.tensor([0.1, 0.1], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "first physical time"):
            PhysicalTimeMap(torch.tensor([0.0, 0.2], dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "strictly between"):
            PhysicalTimeMap.from_betas(torch.tensor([0.1, 1.0]))

        time_map = self._time_map()
        with self.assertRaisesRegex(ValueError, "must be in"):
            time_map.fractional_index_to_time(torch.tensor([-0.01]))
        with self.assertRaisesRegex(ValueError, "must be in"):
            time_map.physical_time_to_fractional_index(
                torch.tensor(time_map.terminal_time + 0.01)
            )

    def test_training_policy_rejects_microbatch_fp16_and_loss_aware_sampling(self):
        valid = {
            "microbatch": -1,
            "schedule_sampler": "batch_shared_physical_uniform",
            "use_fp16": False,
            "timestep_respacing": "",
            "loss_mode": "paper_elbo",
            "weight_decay": 0.0,
        }
        validate_training_policy(valid)

        invalid_cases = (
            ({**valid, "microbatch": 2}, "microbatch=-1"),
            ({**valid, "use_fp16": True}, "use_fp16"),
            (
                {**valid, "schedule_sampler": "loss-second-moment"},
                "batch_shared_physical_uniform",
            ),
            ({**valid, "timestep_respacing": "ddim10"}, "not supported"),
            ({**valid, "weight_decay": 1e-4}, "weight_decay=0"),
        )
        for config, message in invalid_cases:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    validate_training_policy(config)


if __name__ == "__main__":
    unittest.main()
