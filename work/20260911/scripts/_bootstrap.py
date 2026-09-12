"""Keep direct script execution independent of the working directory."""

import sys
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ROOT = WORK.parents[1]
for path in (ROOT, WORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
