#!/usr/bin/env python3
"""Run the tested strict EMA sampler with raw-count-suite adapters."""

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; SUITE_ROOT = HERE.parent; REPO_ROOT = SUITE_ROOT.parent.parent
for path in (SUITE_ROOT, HERE, REPO_ROOT): sys.path.insert(0, str(path))
import common
sys.modules["common"] = common
# The reused sampler imports ``models`` lazily after adding its own suite path.
# Preload this suite's package so that strict restore reconstructs the right model.
import models as target_models
sys.modules["models"] = target_models
source = REPO_ROOT / "work/20260803_ODE_hill_exp/scripts/sample.py"
spec = importlib.util.spec_from_file_location("sample_20260803_reused", source)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)

if __name__ == "__main__": raise SystemExit(module.main())
