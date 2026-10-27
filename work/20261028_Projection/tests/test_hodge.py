import os
import shutil
import numpy as np
import pytest
from Hodge.ddhodge_bridge import run_ddhodge
from Hodge.projector import HodgeProjector


def require_julia():
    executable = os.environ.get('JULIA', 'julia')
    if not shutil.which(executable):
        pytest.skip('Julia unavailable: official Hodge integration not executed')
    return executable


def test_official_gradient_sign_offset_transpose_and_extension(synthetic):
    executable = require_julia()
    x = synthetic.X
    # U=||x||²/2, F=-x: trapezoid edge integration is exact for this quadratic.
    result, info = run_ddhodge(x, -x, k=12, rdim=3, ssa=False, julia=executable)
    target = np.sum(x*x, axis=1)/2
    np.testing.assert_allclose(result['u']-result['u'].mean(), target-target.mean(), atol=2e-5)
    assert info['sign'].startswith('F_gradient=-grad(U)')
    assert result['stationarity_relative'] < 1e-5
    np.testing.assert_array_equal(result['shape_DN'], [3, 80])
    p = HodgeProjector(lambda x: -x, synthetic, interpolation={'n_components': 3, 'n_centers': 64})
    p.fit_scalar(k=12, rdim=3, ssa=False, julia=executable)
    np.testing.assert_allclose(synthetic.obs['projection_hodge_raw'], result['u'])
    np.testing.assert_allclose(synthetic.obs['projection_hodge'], p.scalar.values(x))
    assert p.scalar.metrics['validation_nrmse'] < .05
    c = p.scalar.values(x[:1])[0]
    field = p.get_projected_field(c)
    _, diag = field.evaluate(x[0] + 0.01)  # New, off-surface expression state.
    assert diag['displacement'][0] > 0
    assert diag['surface_residual'][0] < 1e-8 and diag['tangent_error'][0] < 1e-10


def test_missing_julia_is_error(synthetic):
    with pytest.raises(RuntimeError, match='Julia unavailable'):
        run_ddhodge(synthetic.X, -synthetic.X, k=12, rdim=3, julia='/nonexistent/julia')


def test_official_disconnected_graph_stops():
    executable = require_julia()
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal(size=(20, 2)), 100+rng.normal(size=(20, 2))])
    with pytest.raises(RuntimeError, match='Disconnected graph'):
        run_ddhodge(x, -x, k=4, rdim=2, julia=executable)


def test_official_aligned_reduced_workflow(synthetic):
    executable = require_julia()
    x = synthetic.X
    result, info = run_ddhodge(x, -x, k=12, rdim=2, ssa=True, julia=executable)
    target = np.sum(x*x, axis=1)/2
    np.testing.assert_allclose(result['u']-result['u'].mean(), target-target.mean(), atol=2e-5)
    assert info['settings']['ssa'] is True


def test_failed_official_call_records_reason(synthetic):
    p = HodgeProjector(lambda x: -x, synthetic)
    with pytest.raises(RuntimeError, match='Julia unavailable'):
        p.fit_scalar(k=12, rdim=3, julia='/nonexistent/julia')
    failure = synthetic.uns['projection_analysis']['hodge']
    assert failure['status'] == 'failed' and failure['stage'] == 'official_ddhodge'
    with pytest.raises(RuntimeError, match='fit_scalar'): p.get_projected_field(0)
