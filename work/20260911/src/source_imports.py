"""Thin imports for date-named suites; one source model suite per process."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "work" / "20260911"
SUITES = {"20260803": "20260803_ODE_hill_exp", "20260816": "20260816"}


def import_file(name: str, path: Path) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def activate_suite(suite: str) -> tuple[ModuleType, ModuleType]:
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected {list(SUITES)}")
    root = ROOT / "work" / SUITES[suite]
    for name in ("common", "models", "sample"):
        loaded = sys.modules.get(name)
        if loaded and root not in Path(loaded.__file__).resolve().parents:
            raise RuntimeError(f"{name} is already loaded from another suite; use a fresh process")
    for path in (ROOT, root, root / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module("common"), importlib.import_module("sample")


def analysis_helpers(suite: str) -> ModuleType:
    activate_suite(suite)
    return import_file(
        f"_posthoc_helpers_{suite}", ROOT / "work" / SUITES[suite] / "viz/analysis_helpers.py"
    )


def umap_core() -> ModuleType:
    # Import the core file directly: package __init__ imports a run/training runner.
    root = ROOT / "work/20260830"
    loaded = sys.modules.get("analysis")
    if loaded and root not in Path(loaded.__file__).resolve().parents:
        raise RuntimeError("UMAP analysis module collision; use a fresh plotting process")
    for path in (ROOT, root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return import_file("_posthoc_umap_core", root / "hematopoietic_viz/core.py")
