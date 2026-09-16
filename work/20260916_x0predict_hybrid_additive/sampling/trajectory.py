"""Unchanged 20260915 implementation with campaign-local imports."""
from ..reuse import load
_source = load("sampling._trajectory", "sampling/trajectory.py")
globals().update({k: v for k, v in vars(_source).items() if not k.startswith("__")})
