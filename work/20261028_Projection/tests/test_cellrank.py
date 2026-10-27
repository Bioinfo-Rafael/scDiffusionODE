import numpy as np
import pandas as pd
import pytest
from scipy import sparse
import anndata as ad
import cellrank as cr
from CellRank import cellrank_core as official
from CellRank.projector import CellRankProjector


def test_velocity_kernel_official(synthetic, constant_velocity):
    k = official.velocity_kernel(synthetic.X, constant_velocity(synthetic.X), synthetic.obs, synthetic.var,
                                 n_neighbors=12)
    p, report = official.transition_matrix(k)
    assert sparse.issparse(p) and p.nnz < 80*80
    assert report['row_sum_max_error'] < 1e-10 and report['minimum_entry'] >= 0
    assert not synthetic.obsp and not synthetic.layers


def test_absorption_analytic_and_terminal_change():
    a = ad.AnnData(np.zeros((3, 2)), obs=pd.DataFrame(index=['s0', 's1', 's2']))
    p = sparse.csr_matrix([[0., 1, 0], [0, 0, 1], [0, 0, 1]])
    for ids, expected in [(['s2'], [2., 1, 0]), (['s1', 's2'], [1., 0, 0])]:
        e = official.terminal_estimator(cr.kernels.PrecomputedKernel(p, adata=a.copy()), {'end': ids})
        times, _ = official.absorption_times(e, solver='direct')
        np.testing.assert_allclose(times['end_mean'], expected)


def test_competing_conditional_times_keep_all_terminals():
    a = ad.AnnData(np.zeros((4, 2)), obs=pd.DataFrame(index=['s0', 's1', 'a', 'b']))
    p = sparse.csr_matrix([[0., .5, .5, 0], [0, .5, .25, .25], [0, 0, 1, 0], [0, 0, 0, 1]])
    e = official.terminal_estimator(cr.kernels.PrecomputedKernel(p, adata=a), {'A': ['a'], 'B': ['b']})
    times, probs = official.absorption_times(e, solver='direct')
    np.testing.assert_allclose(np.asarray(probs[:, 'A']).ravel(), [.75, .5, 1, 0])
    np.testing.assert_allclose(times['A_mean'][:3], [5/3, 2, 0])
    np.testing.assert_allclose(times['B_mean'].iloc[[0, 1, 3]], [3, 2, 0])
    assert np.isinf(times['A_mean'].iloc[3])


def test_cellrank_full_and_no_overwrite(synthetic, constant_velocity):
    original_x = synthetic.X.copy()
    synthetic.uns['neighbors'] = {'sentinel': 42}
    p = CellRankProjector(constant_velocity, synthetic,
                          interpolation={'n_components': 3, 'n_centers': 40})
    p.fit_scalar(terminal_obs_key='type', terminal_values=['end'], n_neighbors=12)
    assert synthetic.uns['neighbors'] == {'sentinel': 42}
    np.testing.assert_array_equal(synthetic.X, original_x)
    np.testing.assert_allclose(synthetic.obs['projection_cellrank'], p.scalar.values(original_x))
    assert 'projection_cellrank_transition' in synthetic.obsp
    c = p.scalar.values(original_x[:1])[0]
    _, diag = p.get_projected_field(c).evaluate(original_x[0] + 0.005)
    assert diag['displacement'][0] > 0 and diag['surface_residual'][0] < 1e-8
    assert diag['tangent_error'][0] < 1e-8
    with pytest.raises(ValueError, match='Existing|overwrite'):
        p.fit_scalar(terminal_obs_key='type', terminal_values=['end'], n_neighbors=12)


def test_explicit_terminals_and_lineage(synthetic, constant_velocity):
    p = CellRankProjector(constant_velocity, synthetic)
    with pytest.raises(TypeError): p.fit_scalar()
    with pytest.raises(ValueError, match='absent'):
        p.fit_scalar(terminal_obs_key='type', terminal_values=['missing'])
    with pytest.raises(ValueError, match='lineage separately'):
        p.fit_scalar(terminal_obs_key='type', terminal_values=['start', 'end'])


def test_unreachable_terminal_stops_before_solver():
    a = ad.AnnData(np.zeros((3, 2)), obs=pd.DataFrame(index=['s0', 's1', 'end']))
    p = sparse.eye(3, format='csr')
    e = official.terminal_estimator(cr.kernels.PrecomputedKernel(p, adata=a), {'end': ['end']})
    with pytest.raises(RuntimeError, match='cannot reach'):
        official.absorption_times(e)


def test_competing_projector_excludes_other_terminal_but_evaluates_main(synthetic, constant_velocity):
    synthetic.obs['type'] = np.where(synthetic.X[:, 0] > 2.2, 'end',
                                    np.where(synthetic.X[:, 0] < .7, 'other', 'transient'))
    p = CellRankProjector(constant_velocity, synthetic,
                          interpolation={'n_components': 3, 'n_centers': 40})
    p.fit_scalar(terminal_obs_key='type', terminal_values=['end', 'other'], lineage='end',
                 n_neighbors=12, softmax_scale=2)
    other = synthetic.obs['type'] == 'other'
    assert np.isinf(synthetic.obs.loc[other, 'projection_cellrank_absorption_steps']).all()
    assert not synthetic.obs.loc[other, 'projection_cellrank_eligible'].any()
    np.testing.assert_allclose(synthetic.obs['projection_cellrank'], p.scalar.values(synthetic.X))
