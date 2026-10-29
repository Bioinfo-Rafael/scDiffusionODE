"""CellRank entry point: (x,v) -> P -> all terminal states -> T -> tau -> smooth scalar."""
import numpy as np
from common.base import BaseProjector
from common.velocity_adapter import evaluate_velocity
from .subprocess_bridge import VERSION, run_cellrank


class CellRankProjector(BaseProjector):
    method = 'cellrank'

    def fit_scalar(self, *, terminal_obs_key, terminal_values, lineage=None,
                   n_neighbors=30, similarity='cosine', softmax_scale=10.0,
                   seed=0, solver='gmres', tol=1e-8, min_absorption_probability=1e-6,
                   cellrank_python=None, worker_directory=None, worker_timeout=None):
        keys = ['projection_cellrank', 'projection_cellrank_absorption_steps',
                'projection_cellrank_eligible', 'projection_cellrank_absorption_probability']
        self._check_free(keys, 'projection_cellrank_transition')
        if terminal_obs_key not in self.adata.obs or not terminal_values or isinstance(terminal_values, str):
            raise ValueError('Explicit terminal_obs_key and nonempty terminal_values list required')
        names = [str(value) for value in terminal_values]
        if len(set(names)) != len(names) or any(',' in name for name in names):
            raise ValueError('Unique terminal names without commas required')
        terminal_cells = {name: self.adata.obs_names[self.adata.obs[terminal_obs_key] == value].tolist()
                          for name, value in zip(names, terminal_values)}
        if any(not ids for ids in terminal_cells.values()):
            raise ValueError(f'Terminal group absent: {terminal_cells}')
        if lineage is None:
            if len(names) != 1:
                raise ValueError('Specify lineage separately when multiple terminal states are supplied')
            lineage = names[0]
        if lineage not in names or not 0 < min_absorption_probability <= 1:
            raise ValueError('Unknown lineage or invalid absorption probability threshold')
        x = self._expression()
        v = evaluate_velocity(self.velocity, x)
        try:
            result = run_cellrank(x, v, self.adata.obs_names.tolist(), self.adata.var_names.tolist(), terminal_cells,
                dict(n_neighbors=n_neighbors, seed=seed, similarity=similarity, softmax_scale=softmax_scale,
                     solver=solver, tol=tol), python=cellrank_python, work_directory=worker_directory, timeout=worker_timeout)
        except Exception as exc:
            self._record_failure('official_cellrank', exc)
            raise
        if list(result['cell_ids']) != self.adata.obs_names.tolist():
            raise RuntimeError('Official absorption times cell order changed')
        p, times, probabilities = result['transition'], result['times'], result['probabilities']
        t = np.asarray(times[:, result['time_columns'].index(lineage+'_mean')], dtype=float)
        probability = np.asarray(probabilities[:, result['probability_columns'].index(lineage)]).reshape(-1)
        probability_tolerance = max(1e-6, 100*tol)
        if not np.isfinite(probability).all() or (probability < -probability_tolerance).any() or (probability > 1+probability_tolerance).any():
            raise RuntimeError('Invalid official absorption probabilities')
        # Finite conditional times are undefined/unreliable at zero/tiny selected-lineage probability.
        valid = np.isfinite(t) & (t >= 0) & (probability >= min_absorption_probability)
        if valid.sum() < 12 or np.ptp(t[valid]) <= 1e-12:
            raise ValueError('Insufficient finite nonconstant absorption-time domain')
        tmin, tmax = float(t[valid].min()), float(t[valid].max())
        tau = np.full(len(x), np.nan)
        tau[valid] = 1-(t[valid]-tmin)/(tmax-tmin)
        self.adata.obs['projection_cellrank_absorption_steps'] = t
        self.adata.obs['projection_cellrank_absorption_probability'] = probability
        self.adata.obs['projection_cellrank_eligible'] = valid
        self.adata.obsp['projection_cellrank_transition'] = p
        self.transition_matrix = p
        self.intermediates = dict(X=x, V=v, absorption_steps=t, absorption_probability=probability,
                                  tau_discrete=tau, eligible=valid,
                                  all_absorption_times=times, all_fate_probabilities=np.asarray(probabilities))
        scalar, main = self._fit_extension(x, tau, valid)
        self.adata.obs['projection_cellrank'] = main
        self._record(scalar, dict(terminal_obs_key=terminal_obs_key, terminal_cells=terminal_cells,
            lineage=lineage, all_absorption_time_columns=result['time_columns'],
            all_fate_probability_columns=result['probability_columns'],
            interpretation='conditional mean steps to selected lineage, all competing states retained',
            normalization=dict(T_min=tmin, T_max=tmax, formula='1-(T-T_min)/(T_max-T_min)',
                               min_absorption_probability=min_absorption_probability),
            eligible_cells=self.adata.obs_names[valid].tolist(), transition_checks=result['transition_checks'],
            absorption_checks=result['absorption_checks'], execution=result['runtime'],
            official=dict(version=VERSION, settings=dict(n_neighbors=n_neighbors,
                use_rep='X', xkey='projection_expression', vkey='projection_velocity',
                model='deterministic', similarity=similarity, softmax_scale=softmax_scale,
                seed=seed, solver=solver, tol=tol, use_petsc=False, n_jobs=1)),
            precision='float64; official single-terminal mean output float32',
            failure_counts=dict(nonfinite_time=int((~np.isfinite(t)).sum()),
                negative_time=int((t < 0).sum()), low_probability=int((probability < min_absorption_probability).sum()),
                excluded=int((~valid).sum()))))
        return self
