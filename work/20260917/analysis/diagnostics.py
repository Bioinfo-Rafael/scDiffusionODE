from ..reuse import load
_source = load("analysis._extended_diagnostics", "analysis/diagnostics.py", "20260916_x0predict_hybrid_additive")
globals().update({k: v for k, v in vars(_source).items() if not k.startswith("__")})
