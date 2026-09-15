"""Synthetic evaluation failures and read-only analysis restart fixtures."""

import contextlib
import csv
import importlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

common = importlib.import_module("work.20260913_2step.common")
ev = importlib.import_module("work.20260913_2step.analysis.evaluation_ot")
dist = importlib.import_module("work.20260913_2step.analysis.distributions")
sink = importlib.import_module("work.20260913_2step.losses.sinkhorn")
spec = importlib.util.spec_from_file_location(
    "eval_restart_launcher", common.SUITE / "scripts/run_all.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class EvaluationRecoveryTests(unittest.TestCase):
    def setUp(self):
        parent = common.SUITE / ".test_tmp"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.config = common.effective_config("hill_after_linear_recon_soft")["ot"]
        self.x = torch.ones(3, 2)

    def test_retry_keeps_target_epsilon_tolerance_and_never_enables_grad(self):
        calls = []

        def solve(x, y, config, **kwargs):
            self.assertFalse(torch.is_grad_enabled())
            calls.append(config)
            if len(calls) == 1:
                raise sink.SinkhornConvergenceError("synthetic finite residual")
            return torch.tensor(0.25), {"cross": {"marginal_residual": 1e-6}}

        with (
            patch.object(ev, "sinkhorn_divergence", side_effect=solve),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            value, info, status = ev.evaluate_sinkhorn(self.x, self.x, self.config)
        self.assertEqual(value, 0.25)
        self.assertEqual(status["ot_status"], "converged_after_retry")
        self.assertTrue(calls[1]["epsilon_scaling"])
        for key in ("epsilon", "tolerance", "max_iterations", "cost", "debias"):
            self.assertEqual(calls[0][key], calls[1][key])
        self.assertNotIn("epsilon_scaling", self.config)

    def test_exhausted_retry_returns_missing_not_unconverged_value(self):
        with (
            patch.object(
                ev,
                "sinkhorn_divergence",
                side_effect=sink.SinkhornConvergenceError("residual too large"),
            ) as solver,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            value, info, status = ev.evaluate_sinkhorn(self.x, self.x, self.config)
        self.assertIsNone(value)
        self.assertEqual(info, {})
        self.assertEqual(solver.call_count, 2)
        self.assertEqual(status["ot_status"], "not_converged")
        self.assertIn("residual too large", status["ot_error"])

    def test_nonfinite_and_programming_errors_remain_fatal(self):
        for error in (
            FloatingPointError("NaN"),
            ValueError("bad shape"),
            RuntimeError("out of memory"),
        ):
            with (
                patch.object(ev, "sinkhorn_divergence", side_effect=error),
                self.assertRaises(type(error)),
            ):
                ev.evaluate_sinkhorn(self.x, self.x, self.config)
        # The training/public solver remains strict, independent of this wrapper.
        with self.assertRaises(sink.SinkhornConvergenceError):
            sink.sinkhorn_divergence(
                torch.arange(6).reshape(3, 2).float(),
                torch.ones(4, 2),
                {
                    **self.config,
                    "epsilon": 1e-8,
                    "max_iterations": 1,
                    "tolerance": 1e-12,
                },
            )

    def test_missing_snapshot_ot_does_not_remove_other_metrics(self):
        output = common.new_dir(self.path / "numbers")
        config = common.effective_config("hill_after_linear_recon_soft")
        config.update(analysis_pairs=5, analysis_ot_cells=3)
        status = dict(
            ot_status="not_converged",
            ot_attempts=2,
            ot_epsilon_scaling=True,
            ot_error="fixture",
        )
        with patch.object(dist, "evaluate_sinkhorn", return_value=(None, {}, status)):
            dist.trajectory_metrics(
                np.arange(12).reshape(3, 2, 2),
                np.ones((3, 2)),
                [{"snapshot_index": 0}, {"snapshot_index": 1}],
                output,
                config,
                seed=1,
            )
        with (output / "sinkhorn_to_real.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(all(row["sinkhorn_divergence"] == "" for row in rows))
        self.assertTrue((output / "sliced_wasserstein_snapshots.csv").exists())
        self.assertTrue((output / "trajectory_diversity.csv").exists())
        self.assertEqual(
            common.read_json(output / "evaluation_ot_status.json")["missing_count"], 2
        )

    def fixture_launch(self):
        launch = common.new_dir(self.path / "launches" / "campaign" / "failed_analysis")
        output = common.new_dir(self.path / "results" / "sample")
        common.write_json(output / "completed.json", {"status": "completed"})
        steps = [
            dict(
                name="model.sample",
                argv=["python", "-B", "-u", "sample.py"],
                capture={"TRAJECTORY_DIR": "@model.trajectory"},
            ),
            dict(
                name="model.analyze",
                argv=[
                    "python",
                    "-B",
                    "-u",
                    "analyze.py",
                    "--trajectory",
                    "@model.trajectory",
                ],
                capture={"ANALYSIS_DIR": "@model.analysis"},
            ),
            dict(
                name="model.embed",
                argv=[
                    "python",
                    "-B",
                    "-u",
                    "embed.py",
                    "--trajectory",
                    "@model.trajectory",
                ],
                capture={},
            ),
        ]
        common.write_json(
            launch / "launch.json",
            dict(analysis_only=True, campaign="campaign", device="cpu"),
        )
        common.write_json(launch / "failed.json", dict(step="model.analyze"))
        common.write_json(
            launch / "execution_plan.json", dict(steps=steps, artifacts={})
        )
        common.write_json(
            launch / "model.sample.completed.json",
            dict(artifacts={"@model.trajectory": str(output)}),
        )
        return launch, output

    def test_restart_reuses_completed_prefix_and_preserves_files(self):
        launch, output = self.fixture_launch()
        before = {str(p): common.file_hash(p) for p in launch.iterdir()}
        with patch.object(launcher, "SUITE", self.path):
            steps, artifacts, _ = launcher.analysis_restart(launch)
            self.assertEqual(
                [s["name"] for s in steps], ["model.analyze", "model.embed"]
            )
            self.assertEqual(artifacts["@model.trajectory"], str(output))
            with (
                patch.object(launcher.subprocess, "Popen") as spawn,
                contextlib.redirect_stdout(io.StringIO()) as log,
            ):
                self.assertEqual(
                    launcher.main(
                        ["--resume-analysis-launch", str(launch), "--dry-run"]
                    ),
                    0,
                )
            spawn.assert_not_called()
            self.assertNotIn("sample.py", log.getvalue())
            self.assertIn("analyze.py", log.getvalue())
        self.assertEqual(
            before, {str(p): common.file_hash(p) for p in launch.iterdir()}
        )

    def test_restart_rejects_live_launch_missing_outputs_and_training(self):
        launch, output = self.fixture_launch()
        with patch.object(launcher, "SUITE", self.path):
            (launch / "failed.json").unlink()
            with self.assertRaises(ValueError):
                launcher.analysis_restart(launch)
            common.write_json(launch / "failed.json", {})
            (output / "completed.json").unlink()
            with self.assertRaises(ValueError):
                launcher.analysis_restart(launch)
            common.write_json(output / "completed.json", {"status": "completed"})
            path = launch / "execution_plan.json"
            data = json.loads(path.read_text())
            data["steps"][1]["argv"][3] = "train.py"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "forbidden"):
                launcher.analysis_restart(launch)
