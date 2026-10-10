"""Load scripts with their own `common` alias; never leak it to other suites."""
from __future__ import annotations
import importlib.util
import sys
from ..common import ROOT, module


def load_script(stem, **bindings):
    common = module('work.20261009_newBenchmark.common')
    name = 'work.20261010_model_comparison._legacy_' + stem
    spec = importlib.util.spec_from_file_location(name, ROOT / 'work/20261009_newBenchmark' / (stem + '.py'))
    result = importlib.util.module_from_spec(spec)
    previous = sys.modules.get('common')
    try:
        sys.modules['common'] = common
        spec.loader.exec_module(result)
    finally:
        if previous is None:
            sys.modules.pop('common', None)
        else:
            sys.modules['common'] = previous
    result.__dict__.update(bindings)
    return result
