"""Launcher-only checks; never execute an experiment command."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SUITE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "two_step_launcher", SUITE / "scripts/run_all.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        parent = SUITE / ".test_tmp"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def test_plan_all_conditions_and_output_dependencies(self):
        args = launcher.parser().parse_args(["--campaign", "test"])
        steps = launcher.plan(args)
        self.assertEqual(len(steps), 51)
        self.assertEqual(
            [step["name"] for step in steps if step["name"].endswith(".train")],
            [c + ".train" for c in launcher.CONDITIONS],
        )
        available = set()
        for step in steps:
            for arg in step["argv"]:
                if arg.startswith("@"):
                    self.assertIn(arg, available)
            available.update(step["capture"].values())
        self.assertEqual(sum(s["name"].endswith(".sample") for s in steps), 7)
        self.assertEqual(sum(s["name"].endswith("_plot") for s in steps), 15)

    def test_shared_cache_and_independent_evaluation_dependencies(self):
        args = launcher.parser().parse_args(["--campaign", "test"])
        steps = launcher.plan(args)
        train_cache = [s for s in steps if s["name"] == "x50.train_cache"]
        eval_cache = [s for s in steps if s["name"] == "x50.evaluation_cache"]
        self.assertEqual(len(train_cache), 1)
        self.assertEqual(len(eval_cache), 1)
        new_training = [
            s for s in steps if s["name"].endswith("_trajectory_ot_soft.train")
        ]
        self.assertEqual(len(new_training), 3)
        for s in new_training:
            self.assertEqual(
                s["argv"][s["argv"].index("--source-cache") + 1], "@x50.train"
            )
            self.assertNotIn("--resume-checkpoint", s["argv"])
        evaluation = [s for s in steps if s["name"].endswith(".occupation")]
        self.assertEqual(len(evaluation), 6)
        for s in evaluation:
            self.assertEqual(
                s["argv"][s["argv"].index("--source-cache") + 1], "@x50.evaluation"
            )
        self.assertEqual(steps[-1]["name"], "occupation.comparison_plot")
        self.assertEqual(sum(arg.startswith("@") for arg in steps[-1]["argv"]), 6)
        self.assertFalse(any("hill_after_linear_ot_soft" in s["name"] for s in steps))

    def test_dry_run_does_not_spawn_or_create_directories(self):
        with (
            patch.object(launcher.subprocess, "Popen") as spawn,
            patch.object(launcher.Path, "mkdir") as mkdir,
        ):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(
                    launcher.main(["--dry-run", "--data", "/missing/data.h5ad"]), 0
                )
            self.assertEqual(len(out.getvalue().splitlines()), 51)
            spawn.assert_not_called()
            mkdir.assert_not_called()

    def test_detached_launch_and_no_overwrite(self):
        data = self.path / "dummy-input"
        data.touch()
        argv = ["--campaign", "example", "--data", str(data), "--edge-tsv", str(data)]
        with (
            patch.object(launcher, "SUITE", self.path),
            patch.object(
                launcher.subprocess, "check_output", return_value="test-commit\n"
            ),
            patch.object(launcher.subprocess, "Popen") as spawn,
        ):
            spawn.return_value.pid = 123
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(launcher.main(argv), 0)
            self.assertTrue(spawn.call_args.kwargs["start_new_session"])
            self.assertEqual((self.path / "launches/example/pid").read_text(), "123\n")
            with self.assertRaises(FileExistsError):
                launcher.main(argv)
            self.assertEqual(spawn.call_count, 1)

    def test_worker_stops_on_first_failure(self):
        manifest = self.path / "launch.json"
        launcher.write_json(
            manifest,
            {
                "campaign": "mock",
                "device": "cpu",
                "steps": [{"name": "first"}, {"name": "never"}],
            },
        )
        with patch.object(
            launcher, "run_step", side_effect=RuntimeError("synthetic failure")
        ) as run:
            with (
                contextlib.redirect_stderr(io.StringIO()),
                patch.dict(launcher.os.environ),
            ):
                self.assertEqual(launcher.worker(manifest), 1)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            json.loads((self.path / "failed.json").read_text())["step"], "first"
        )
        self.assertFalse((self.path / "completed.json").exists())

    def test_output_capture_uses_exact_emitted_path(self):
        output = self.path / "dummy-checkpoint"
        output.touch()
        # Only a one-line Python print subprocess, never training/sampling/plotting.
        step = {
            "name": "capture",
            "argv": [
                sys.executable,
                "-c",
                f"print({('EMA_CHECKPOINT=' + str(output))!r})",
            ],
            "capture": {"EMA_CHECKPOINT": "@checkpoint"},
        }
        artifacts = {}
        with contextlib.redirect_stdout(io.StringIO()):
            launcher.run_step(step, artifacts, self.path)
        self.assertEqual(artifacts["@checkpoint"], str(output))

    def test_invalid_settings_fail_before_spawn(self):
        for argv in (
            ["--campaign", "../outside"],
            ["--post-ode-dt", "nan"],
            ["--num-samples", "1"],
        ):
            with (
                self.assertRaises(ValueError),
                patch.object(launcher.subprocess, "Popen") as spawn,
            ):
                launcher.main(argv)
            spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
