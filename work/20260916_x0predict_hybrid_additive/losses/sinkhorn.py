import importlib
_source = importlib.import_module("work.20260913_2step.losses.sinkhorn")
sinkhorn_divergence = _source.sinkhorn_divergence
SinkhornConvergenceError = _source.SinkhornConvergenceError
