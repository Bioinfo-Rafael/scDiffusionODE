"""Private namespace loading; legacy code and module globals stay untouched."""
import importlib.util
import sys
from pathlib import Path


def load(name, source, suite="20260915_x0predict"):
    name = __package__ + "." + name
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / suite / source
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
