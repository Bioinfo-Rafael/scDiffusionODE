"""CPU unit tests; no trained checkpoint or production sampling required."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[1]
sys.path[:0] = [str(WORK), str(ROOT)]

from guided_diffusion.script_util import create_gaussian_diffusion

from src.breakpoint import (
    Fit,
    candidate_indices,
    fit_breakpoint,
    global_breakpoint,
    grouped_trajectories,
    score_curves,
)
from src.model_runs import resolve_entry
from src.settings import Settings
from src.trajectory import sample_to_disk, snapshot_table
from src.umap_adapter import independent_embeddings, select_snapshots, selected_joint_embedding


class SnapshotTests(unittest.TestCase):
    def test_twenty_completed_updates(self):
        table = snapshot_table(Settings())
        self.assertEqual(len(table), 20)
        self.assertEqual([r["reverse_step"] for r in table], list(range(50, 1001, 50)))
        self.assertEqual([r["diffusion_t"] for r in table], list(range(950, -1, -50)))
        self.assertEqual(table[-1], {"snapshot_index": 19, "reverse_step": 1000, "diffusion_t": 0})

    def test_mapping_and_settings_validation(self):
        settings = Settings(T=100, M=20, S=1, N=1)
        mapping = list(range(0, 1000, 10))
        self.assertEqual(snapshot_table(settings, mapping)[0]["diffusion_t"], 950)
        with self.assertRaises(ValueError):
            snapshot_table(settings, list(reversed(mapping)))
        with self.assertRaises(ValueError):
            Settings(T=999)

    def test_final_matches_existing_pipeline(self):
        # A tiny deterministic denoiser, but the real 1000-step Gaussian sampler.
        class Denoiser(torch.nn.Module):
            def forward(self, x, t):
                return x * 0.01

        model = Denoiser()
        diffusion = create_gaussian_diffusion(steps=1000, noise_schedule="linear")
        settings = Settings(S=2, N=1)
        with tempfile.TemporaryDirectory(dir=WORK / "results") as tmp:
            output = Path(tmp) / "trajectory"
            torch.manual_seed(72)
            info = sample_to_disk(
                model, diffusion, output, settings, gene_count=3, batch_size=2, device="cpu"
            )
            torch.manual_seed(72)
            expected, _ = diffusion.p_sample_loop(
                model, (2, 3), device="cpu", clip_denoised=False, start_time=1000
            )
            states = np.load(output / "sample_state.npy")
            predictions = np.load(output / "pred_xstart.npy")
            np.testing.assert_array_equal(states[:, -1], expected.numpy())
            np.testing.assert_allclose(predictions[:, -1], expected.numpy(), rtol=1e-5, atol=1e-6)
            self.assertEqual(info["final_reverse_step"], 1000)
            self.assertEqual(info["status"], "completed")

    def test_batch_order_and_incomplete_generator(self):
        class FakeDiffusion:
            num_timesteps = 1000

            def __init__(self, stop=1000):
                self.offset, self.stop = 0, stop

            def p_sample_loop_progressive(self, model, shape, **kwargs):
                start = self.offset
                self.offset += shape[0]
                for step in range(1, self.stop + 1):
                    sample = (
                        torch.arange(start, self.offset)[:, None].repeat(1, shape[1]).float()
                        + step / 1000
                    )
                    yield {"sample": sample, "pred_xstart": sample}

        settings = Settings(N=2, S=3)
        with tempfile.TemporaryDirectory(dir=WORK / "results") as tmp:
            output = Path(tmp) / "trajectory"
            sample_to_disk(
                None,
                FakeDiffusion(),
                output,
                settings,
                gene_count=2,
                batch_size=4,
                device="cpu",
                save_states=False,
            )
            array = np.load(output / "pred_xstart.npy")
            np.testing.assert_allclose(array[:, -1, 0], np.arange(6) + 1)
            self.assertFalse((output / "sample_state.npy").exists())
            bad = Path(tmp) / "incomplete"
            with self.assertRaises(RuntimeError):
                sample_to_disk(
                    None,
                    FakeDiffusion(999),
                    bad,
                    settings,
                    gene_count=2,
                    batch_size=4,
                    device="cpu",
                )
            self.assertEqual(
                json.loads((bad / "sampling_metadata.json").read_text())["status"], "failed"
            )


class BreakpointTests(unittest.TestCase):
    def test_shape_order_and_eq7(self):
        settings = Settings()
        predictions = np.arange(2000 * 20 * 3, dtype=np.float32).reshape(2000, 20, 3)
        grouped = grouped_trajectories(predictions, settings)
        self.assertEqual(grouped.shape, (20, 100, 20, 3))
        np.testing.assert_array_equal(grouped[3, 24], predictions[324])
        scores = score_curves(predictions, settings)
        self.assertEqual(scores.shape, (20, 20))
        np.testing.assert_allclose(scores, grouped.mean(axis=(1, 3), dtype=np.float64))
        predictions[0, 0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            score_curves(predictions, settings)

    def test_eq8_segment_boundary(self):
        times = np.arange(950, -1, -50)
        # Discontinuity makes inclusion of breakpoint in t>=boundary unambiguous.
        y = np.where(times >= times[8], 0.1 * times + 20, -0.7 * times - 100)
        fit = fit_breakpoint(y, times)
        self.assertEqual(fit.snapshot_index, 8)
        self.assertLess(fit.sse, 1e-20)

    def test_endpoint_exclusion(self):
        np.testing.assert_array_equal(candidate_indices(20), np.arange(2, 18))
        self.assertEqual(len(candidate_indices(20)), 16)

    def test_exact_half_median(self):
        fits = [Fit(k, 0, np.empty(0)) for k in [7] * 10 + [8] * 10]
        info = global_breakpoint(fits, snapshot_table(Settings()))
        self.assertEqual(info["median_snapshot_index"], 7.5)
        self.assertEqual(info["reverse_step"], 425)
        self.assertEqual(info["diffusion_t"], 575)
        self.assertFalse(info["is_saved_snapshot"])


class UmapTests(unittest.TestCase):
    def setUp(self):
        self.real = SimpleNamespace(n_obs=5)
        self.predictions = np.broadcast_to(np.arange(20)[None, :, None], (7, 20, 3)).copy()

        def build(real, generated):
            return SimpleNamespace(
                X=generated.copy(), obs=pd.DataFrame(index=np.arange(real.n_obs + len(generated)))
            )

        self.core = SimpleNamespace(
            build_sampling_anndata=Mock(side_effect=build),
            compute_common_umap=Mock(return_value={"tested": True}),
        )

    def test_stage1_twenty_distinct_fits(self):
        seen = []
        independent_embeddings(
            self.real,
            self.predictions,
            seed=1234,
            consume=lambda k, obj, params: seen.append((k, obj)),
            core=self.core,
        )
        self.assertEqual(self.core.compute_common_umap.call_count, 20)
        self.assertEqual(len({id(obj) for _, obj in seen}), 20)
        for k, obj in seen:
            self.assertEqual(obj.X.shape, (7, 3))
            np.testing.assert_array_equal(obj.X, np.full((7, 3), k))

    def test_stage2_only_explicit_selection(self):
        table = snapshot_table(Settings())
        joint, _ = selected_joint_embedding(
            self.real, self.predictions, table, [7, 9, 3], seed=1234, core=self.core
        )
        self.assertEqual(self.core.compute_common_umap.call_count, 1)
        self.assertEqual(joint.X.shape, (21, 3))
        np.testing.assert_array_equal(joint.X[:, 0], np.repeat([7, 9, 3], 7))
        np.testing.assert_array_equal(joint.obs["reverse_step"][5:], np.repeat([400, 500, 200], 7))
        self.assertEqual(select_snapshots(table, reverse_steps=[400, 500]), [7, 9])
        for invalid in ([], [0, 0], [-1], [20]):
            with self.assertRaises(ValueError):
                select_snapshots(table, snapshots=invalid)
        with self.assertRaises(ValueError):
            select_snapshots(table)


class RestoreTests(unittest.TestCase):
    def test_both_suites_in_fresh_processes(self):
        for suite in ("20260803", "20260816"):
            with self.subTest(suite=suite):
                result = subprocess.run(
                    [sys.executable, str(WORK / "tests/restore_fixture.py"), suite],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('"status": "fixture_restore_passed"', result.stdout)

    def test_ambiguous_mapping(self):
        entry = {"suite": "20260803", "experiment": "example"}
        report = {
            "suite": "20260803",
            "experiments": {"example": {"candidates": [{"run_dir": "a"}, {"run_dir": "b"}]}},
        }
        with self.assertRaisesRegex(ValueError, "found 2"):
            resolve_entry(entry, report)
        report["experiments"]["example"]["candidates"] = [{"run_dir": "a", "checkpoint": "ema.pt"}]
        self.assertEqual(resolve_entry(entry, report)["run_dir"], "a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
