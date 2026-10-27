import numpy as np
import pytest
from common.levelset_projection import ProjectedField, ProjectionError
from common.scalar_extension import ScalarExtension


class Plane:
    valid_range = (-10., 10.)
    support_radius = 100.
    def values(self, x): return np.atleast_2d(x)[:, 0]
    def gradient(self, x): return np.tile([1., 0.], (len(np.atleast_2d(x)), 1))
    def distribution_distance(self, x): return np.zeros(len(np.atleast_2d(x)))


def test_plane_on_and_off_surface_and_batch():
    f = ProjectedField(lambda x: np.broadcast_to([1., 2.], x.shape), Plane(), .5, 2)
    x = np.array([[.5, 1.], [2., 3.]])
    np.testing.assert_allclose(f.surface_points(x), [[.5, 1.], [.5, 3.]])
    v, d = f.evaluate(x)
    np.testing.assert_allclose(v, [[0., 2.], [0., 2.]])
    np.testing.assert_allclose(f(x[0]), [0., 2.])
    assert d['converged'].all() and max(d['surface_residual']) < 1e-12
    assert max(d['tangent_error']) < 1e-12


def test_evaluates_velocity_at_y_not_x():
    f = ProjectedField(lambda x: np.column_stack([x[:, 1], x[:, 0]]), Plane(), .5, 2)
    np.testing.assert_allclose(f([2., 3.]), [0., .5])


def test_invalid_and_vanishing_gradient():
    with pytest.raises(ValueError): ProjectedField(lambda x: x, Plane(), 20, 2)
    f = ProjectedField(lambda x: x, Plane(), 1, 2)
    with pytest.raises(ValueError): f([np.nan, 1])
    with pytest.raises(ValueError): f([-1, 1])
    class Flat(Plane):
        def gradient(self, x): return np.zeros_like(np.atleast_2d(x))
    f = ProjectedField(lambda x: x, Flat(), 1, 2)
    with pytest.raises(ProjectionError, match='vanishing_gradient'): f([1, 2])


def test_curved_surface_and_nonconvergence():
    class Circle(Plane):
        def values(self, x): return np.sum(np.atleast_2d(x)**2, axis=1)
        def gradient(self, x): return 2*np.atleast_2d(x)
    f = ProjectedField(lambda x: np.column_stack([-x[:, 1], x[:, 0]]), Circle(), 1, 2)
    y = f.surface_points([2., 2.])
    np.testing.assert_allclose(y, [2**-.5, 2**-.5], atol=1e-8)
    assert f.evaluate([2., 2.])[1]['tangent_error'][0] < 1e-12
    f = ProjectedField(lambda x: x, Circle(), 1, 2, max_iterations=1)
    with pytest.raises(ProjectionError, match='max_iterations'): f([2., 2.])


def test_rbf_chain_rule_and_error_gate(synthetic):
    x = synthetic.X
    s = ScalarExtension(n_components=2, n_centers=30, max_validation_nrmse=1).fit(x, x[:, 0]**2)
    point, eps = x[:3], 1e-5
    finite = np.column_stack([(s.values(point+np.eye(3)[j]*eps)-s.values(point-np.eye(3)[j]*eps))/(2*eps)
                             for j in range(3)])
    np.testing.assert_allclose(s.gradient(point), finite, rtol=1e-5, atol=1e-7)
    with pytest.raises(ValueError, match='error exceeds'):
        ScalarExtension(n_components=1, n_centers=2, max_validation_nrmse=1e-12).fit(x, x[:, 0]**2)
    with pytest.raises(ValueError, match='Constant'):
        ScalarExtension().fit(x, np.ones(len(x)))


def test_support_and_invalid_retraction():
    class Bounded(Plane):
        support_radius = 1.
        def distribution_distance(self, x): return np.linalg.norm(np.atleast_2d(x), axis=1)
    f = ProjectedField(lambda x: x, Bounded(), .5, 2)
    with pytest.raises(ProjectionError, match='input_outside_support'): f([2, 2])
    f = ProjectedField(lambda x: x, Plane(), -1, 2)
    with pytest.raises(ProjectionError): f([1, 1])
