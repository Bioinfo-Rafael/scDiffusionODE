"""Detached launcher must run analysis only after successful preflight."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def test_success_and_failed_preflight(self):
        for preflight_exit in (0, 7):
            with self.subTest(preflight_exit=preflight_exit), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp) / 'with spaces'
                folder.mkdir()
                shutil.copy(HERE / 'run_background.sh', folder)
                fake = folder / 'fake python'
                fake.write_text('''#!/bin/sh
echo "$@" >> "$FAKE_CALLS"
case "$*" in
  *--check-only*) exit "$PREFLIGHT_EXIT" ;;
esac
exit 0
''')
                fake.chmod(0o700)
                calls = folder / 'calls.txt'
                env = dict(os.environ, PYTHON=str(fake), FAKE_CALLS=str(calls), PREFLIGHT_EXIT=str(preflight_exit))
                result = subprocess.run(['bash', str(folder / 'run_background.sh'), '--device', 'cpu'],
                                        env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                job = next((folder / 'outputs').glob('job_*'))
                deadline = time.monotonic() + 10
                while not (job / 'exit_code').exists() and time.monotonic() < deadline:
                    time.sleep(.05)
                self.assertTrue((job / 'exit_code').exists(), result.stdout)
                self.assertEqual(int((job / 'exit_code').read_text()), preflight_exit)
                commands = calls.read_text().splitlines()
                self.assertEqual(len(commands), 2 if preflight_exit == 0 else 1)
                self.assertIn('--check-only', commands[0])
                if preflight_exit == 0:
                    self.assertNotIn('--check-only', commands[1])
                    self.assertIn(str(job / 'analysis'), commands[1])


if __name__ == '__main__':
    unittest.main()
