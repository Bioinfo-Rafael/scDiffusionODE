"""Unchanged 20260915 implementation with campaign-local imports."""
from ..reuse import load
_source = load("analysis._runner", "analysis/runner.py")
globals().update({k: v for k, v in vars(_source).items() if not k.startswith("__")})
