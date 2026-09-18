from ..reuse import load
_source = load("analysis._sliced_wasserstein", "analysis/sliced_wasserstein.py")
globals().update({k: v for k, v in vars(_source).items() if not k.startswith("__")})
