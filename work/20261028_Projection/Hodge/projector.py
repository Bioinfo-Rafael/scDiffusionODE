"""Hodge entry point: velocity -> official ddHodge -> raw U -> smooth U(x)."""
import numpy as np
from common.base import BaseProjector
from common.velocity_adapter import evaluate_velocity
from .ddhodge_bridge import run_ddhodge


class HodgeProjector(BaseProjector):
    method = 'hodge'

    def fit_scalar(self, *, k=12, rdim=4, seed=0, maxiter=5000, ssa=True, julia=None, timeout=3600):
        self._check_free(['projection_hodge', 'projection_hodge_raw'])
        x = self._expression()
        v = evaluate_velocity(self.velocity, x)
        try:
            official, info = run_ddhodge(x, v, k=k, rdim=rdim, seed=seed, maxiter=maxiter,
                                        ssa=ssa, julia=julia, timeout=timeout)
        except Exception as exc:
            self._record_failure('official_ddhodge', exc)
            raise
        raw = official['u']
        self.adata.obs['projection_hodge_raw'] = raw  # Preserve exact official sign and offset.
        self.intermediates = dict(X=x, V=v, **official)
        valid = np.isfinite(raw)
        scalar, main = self._fit_extension(x, raw, valid)
        self.adata.obs['projection_hodge'] = main
        self._record(scalar, dict(official=info, normalization={'policy': 'none; exact ddh.u'},
                                 eligible_cells=self.adata.obs_names[valid].tolist(),
                                 precision='float64', failure_counts={'invalid_scalar': int((~valid).sum())},
                                 stationarity_relative=float(official['stationarity_relative'])))
        return self
