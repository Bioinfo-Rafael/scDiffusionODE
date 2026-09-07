"""Launcher failure policy, verified with real short-lived child processes."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
import launch


class LaunchFailurePolicyTests(unittest.TestCase):
    def run_children(self, exits, *, keep_going):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "executed.txt"
            commands = []
            for index, code in enumerate(exits):
                script = (
                    "from pathlib import Path; "
                    f"p=Path({str(marker)!r}); "
                    f"p.open('a').write({str(index) + chr(10)!r}); "
                    f"raise SystemExit({code})"
                )
                commands.append([sys.executable, "-c", script])
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(launch, "_commands", return_value=("test", commands)), \
                    redirect_stdout(stdout), redirect_stderr(stderr):
                try:
                    result = launch.main(["--continue-on-error"] if keep_going else [])
                except subprocess.CalledProcessError as error:
                    result = error
            return result, marker.read_text().splitlines(), stdout.getvalue(), stderr.getvalue()

    def test_a_failure_still_starts_b_and_reports_failure(self):
        result, executed, stdout, stderr = self.run_children([3, 0], keep_going=True)
        self.assertEqual(executed, ["0", "1"])
        self.assertEqual(result, 1)
        self.assertIn("exit code 3", stderr)
        self.assertIn("Continuing to the next model", stdout)

    def test_default_policy_still_stops_at_a_failure(self):
        result, executed, _, _ = self.run_children([3, 0], keep_going=False)
        self.assertEqual(executed, ["0"])
        self.assertIsInstance(result, subprocess.CalledProcessError)
        self.assertEqual(result.returncode, 3)

    def test_b_failure_does_not_report_overall_success(self):
        result, executed, _, stderr = self.run_children([0, 5], keep_going=True)
        self.assertEqual(executed, ["0", "1"])
        self.assertEqual(result, 1)
        self.assertIn("exit code 5", stderr)

    def test_all_success_returns_zero(self):
        result, executed, _, stderr = self.run_children([0, 0], keep_going=True)
        self.assertEqual(executed, ["0", "1"])
        self.assertEqual(result, 0)
        self.assertEqual(stderr, "")

    def test_user_interrupt_does_not_start_another_model(self):
        with patch.object(launch, "_commands", return_value=("test", [["A"], ["B"]])), \
                patch.object(launch.subprocess, "run", side_effect=KeyboardInterrupt) as run, \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                launch.main(["--continue-on-error"])
        self.assertEqual(run.call_count, 1)

    def test_dry_run_reports_policy_without_running_children(self):
        output = io.StringIO()
        with patch.object(launch, "_commands", return_value=("test", [["A"], ["B"]])), \
                patch.object(launch.subprocess, "run") as run, redirect_stdout(output):
            self.assertEqual(launch.main(["--continue-on-error", "--dry-run"]), 0)
        run.assert_not_called()
        self.assertTrue(json.loads(output.getvalue())["continue_on_error"])


if __name__ == "__main__":
    unittest.main()
