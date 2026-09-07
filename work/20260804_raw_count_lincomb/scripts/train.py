#!/usr/bin/env python3
"""Run the tested 20260803 trainer with raw-count-suite root/model adapters."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; SUITE_ROOT = HERE.parent; REPO_ROOT = SUITE_ROOT.parent.parent
for path in (SUITE_ROOT, HERE, REPO_ROOT): sys.path.insert(0, str(path))
import common
sys.modules["common"] = common
source = REPO_ROOT / "work/20260803_ODE_hill_exp/scripts/train.py"
spec = importlib.util.spec_from_file_location("train_20260803_reused", source)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.HERE = HERE; module.SUITE_ROOT = SUITE_ROOT; module.REPO_ROOT = REPO_ROOT

if __name__ == "__main__": raise SystemExit(module.main())
