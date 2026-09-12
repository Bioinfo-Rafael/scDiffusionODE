"""Coordinator failure paths and command isolation without model jobs."""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

WORK = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WORK / "scripts"), str(WORK)]
import run_all


class CoordinatorTests(unittest.TestCase):
    def run_coordinator(self, candidates, extra_args=()):
        experiment = "linear_centered_signed_hill"
        report = {
            "20260816": {
                "suite": "20260816",
                "experiments": {experiment: {"candidates": candidates}},
            }
        }
        with tempfile.TemporaryDirectory(dir=WORK / "results") as tmp:
            mapping = Path(tmp) / "models.json"
            mapping.write_text(
                json.dumps(
                    {
                        "settings": {"T": 1000, "M": 20, "S": 100, "N": 20},
                        "models": [{"suite": "20260816", "experiment": experiment}],
                    }
                )
            )
            with (
                patch.object(sys, "argv", ["run_all.py", "--models", str(mapping), *extra_args]),
                patch.object(run_all, "discover_all", return_value=report),
                patch.object(run_all.subprocess, "run") as run,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if len(candidates) != 1:
                    with self.assertRaisesRegex(ValueError, "No jobs started"):
                        run_all.main()
                    run.assert_not_called()
                else:
                    run_all.main()
                return run.call_args_list

    def test_missing_and_ambiguous_runs_start_no_jobs(self):
        self.run_coordinator([])
        self.run_coordinator([{"run_dir": "one"}, {"run_dir": "two"}])

    def test_all_selections_preflight_before_sampling(self):
        calls = self.run_coordinator(
            [{"run_dir": "/fixture/run", "checkpoint": "/fixture/ema_0.9999_030000.pt"}]
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].args[0][-1], "--inspect-only")
        self.assertNotIn("--inspect-only", calls[1].args[0])
        self.assertTrue(all("run_one_model.py" in call.args[0][1] for call in calls))
        self.assertTrue(all(call.kwargs["check"] for call in calls))

    def test_dry_run_and_restore_only(self):
        candidate = [{"run_dir": "/fixture/run", "checkpoint": "/fixture/ema_0.9999_030000.pt"}]
        self.assertEqual(self.run_coordinator(candidate, ["--dry-run"]), [])
        calls = self.run_coordinator(candidate, ["--restore-only", "--device", "cpu"])
        self.assertEqual(calls[-1].args[0][-1], "--restore-only")

    def test_script_help_and_stage2_requires_selection(self):
        for script in (
            "discover_runs.py",
            "run_all.py",
            "run_one_model.py",
            "analyze_breakpoint.py",
            "plot_independent_umaps.py",
            "plot_selected_joint_umap.py",
        ):
            output = subprocess.run(
                [sys.executable, str(WORK / "scripts" / script), "--help"],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(output.returncode, 0, output.stderr)
        output = subprocess.run(
            [
                sys.executable,
                str(WORK / "scripts/plot_selected_joint_umap.py"),
                "--result-dir",
                str(WORK / "results/does_not_exist"),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(output.returncode, 2)
        self.assertIn("required", output.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
