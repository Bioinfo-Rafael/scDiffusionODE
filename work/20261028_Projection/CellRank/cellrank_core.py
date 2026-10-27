"""Thin official API calls; all probabilities and absorption solves remain in CellRank."""
import importlib.metadata
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import breadth_first_order
import anndata as ad
import scanpy as sc
import cellrank as cr

VERSION = '2.0.7'


def velocity_kernel(x, v, obs, var, *, n_neighbors=30, seed=0):
    if importlib.metadata.version('cellrank') != VERSION:
        raise RuntimeError(f'Tested CellRank {VERSION} required; no alternate solver')
    if x.shape != v.shape or x.shape != (len(obs), len(var)):
        raise ValueError('Expression, velocity and ordered gene IDs must match')
    if not 2 <= n_neighbors < len(x):
        raise ValueError('Need 2 <= n_neighbors < N')
    # Scratch AnnData protects all original neighbors, layers, obsm and estimator annotations.
    scratch = ad.AnnData(X=x, obs=obs.copy(), var=var.copy())
    scratch.layers['projection_expression'] = x
    scratch.layers['projection_velocity'] = v
    sc.pp.neighbors(scratch, n_neighbors=n_neighbors, use_rep='X', random_state=seed)
    kernel = cr.kernels.VelocityKernel(scratch, attr='layers',
        xkey='projection_expression', vkey='projection_velocity', gene_subset=var.index.tolist())
    return kernel


def transition_matrix(kernel, *, similarity='cosine', softmax_scale=10.0, seed=0):
    kernel.compute_transition_matrix(model='deterministic', similarity=similarity,
        softmax_scale=softmax_scale, seed=seed, n_jobs=1, show_progress_bar=False)
    p = sparse.csr_matrix(kernel.transition_matrix)
    sums = np.asarray(p.sum(axis=1)).ravel()
    if not np.isfinite(p.data).all() or (p.data < 0).any() or not np.allclose(sums, 1, atol=1e-10):
        raise RuntimeError('Official transition matrix is not finite nonnegative row-stochastic')
    report = dict(row_sum_max_error=float(np.max(abs(sums-1))), minimum_entry=float(p.data.min()),
                  nnz=int(p.nnz), density=float(p.nnz/(p.shape[0]**2)),
                  self_transition_count=int(np.count_nonzero(p.diagonal())),
                  self_transition_max=float(p.diagonal().max()), shape=list(p.shape))
    return p, report


def terminal_estimator(kernel, terminal_cells):
    estimator = cr.estimators.GPCCA(kernel)
    estimator.set_terminal_states(terminal_cells, allow_overlap=False)
    return estimator


def absorption_times(estimator, *, solver='gmres', tol=1e-8):
    if solver == 'direct' and estimator.adata.n_obs > 2000:
        raise ValueError('CellRank direct solver densifies; use gmres for large datasets')
    # Graph reachability check only: closed nonterminal classes make absorption undefined.
    n = estimator.adata.n_obs
    terminal = np.flatnonzero(estimator.terminal_states.notna().to_numpy())
    reverse = sparse.csr_matrix(estimator.transition_matrix.T)
    reverse.eliminate_zeros()
    seed_edges = sparse.csr_matrix((np.ones(len(terminal)), (np.zeros(len(terminal), dtype=int), terminal)), shape=(1, n+1))
    graph = sparse.vstack([sparse.hstack([reverse, sparse.csr_matrix((n, 1))]), seed_edges], format='csr')
    reachable = breadth_first_order(graph, n, directed=True, return_predecessors=False)
    if len(reachable) != n+1:
        raise RuntimeError('Some cells cannot reach any specified terminal state; no absorption solver fallback')
    kwargs = dict(solver=solver, use_petsc=False, tol=tol, n_jobs=1, show_progress_bar=False)
    # keys=None is essential: retain ALL competing terminal classes for both calls.
    estimator.compute_fate_probabilities(keys=None, **kwargs)
    estimator.compute_absorption_times(keys=None, calculate_variance=False, **kwargs)
    return estimator.absorption_times.copy(), estimator.fate_probabilities


def validate_absorption(estimator, times, probabilities, tol=1e-8):
    """Residual audit only. Does not compute/replace CellRank's probabilities or times."""
    terminal = estimator.terminal_states.notna().to_numpy()
    transient = ~terminal
    p = sparse.csr_matrix(estimator.transition_matrix)
    q = p[transient][:, transient]
    h = np.asarray(probabilities)
    if not np.isfinite(h).all() or not np.allclose(h.sum(axis=1), 1, atol=max(1e-6, 100*tol)):
        raise RuntimeError('Official fate probabilities are invalid or incomplete')
    probability_residual = h[transient] - p[transient] @ h
    max_probability_error = float(np.max(abs(probability_residual)))
    time_errors = {}
    for j, name in enumerate(probabilities.names):
        t = times[name+'_mean'].to_numpy()
        selected = h[transient, j] > 1e-12
        if np.any(~np.isfinite(t[transient][selected])) or np.any(t[transient][selected] < 0):
            raise RuntimeError(f'Invalid conditional absorption time in lineage {name}')
        weighted = np.zeros(transient.sum())
        weighted[selected] = h[transient, j][selected]*t[transient][selected]
        propagated = q @ weighted
        residual = weighted-propagated-h[transient, j]
        scale = max(1., float(np.linalg.norm(weighted)), float(np.linalg.norm(propagated)))
        time_errors[name] = float(np.linalg.norm(residual)/scale)
    if max_probability_error > max(1e-6, 100*tol) or max(time_errors.values()) > max(1e-6, 100*tol):
        raise RuntimeError(f'Official absorption solver residual failed: {max_probability_error}, {time_errors}')
    return dict(probability_max_residual=max_probability_error, conditional_time_relative_residual=time_errors)
