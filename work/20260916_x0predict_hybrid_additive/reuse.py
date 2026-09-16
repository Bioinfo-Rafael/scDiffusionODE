"""Load unchanged legacy helpers in this suite's namespace, without monkeypatching.

Relative imports resolve to local adapters (especially output confinement).
Legacy module objects and files are never modified. This avoids copying the
sampling, numerical analysis, UMAP and plotting implementations.
"""
import importlib.util
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "20260915_x0predict"


def load(name, source):
    name = __package__ + "." + name
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SOURCE / source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
