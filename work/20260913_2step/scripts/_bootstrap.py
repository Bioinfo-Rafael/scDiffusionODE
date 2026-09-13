"""Load the date-named package without generic models/analysis import collisions."""

import importlib
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(command):
    importlib.import_module("work.20260913_2step.cli").main(command)
