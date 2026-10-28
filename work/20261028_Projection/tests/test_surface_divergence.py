import numpy as np
import pytest
import torch
from analysis.surface_divergence import TorchDerivatives, estimate_surface_divergence, scalar_geometry
from common.levelset_projection import ProjectedField
from common.scalar_extension import ScalarExtension
from scalar_fixtures import PlaneScalar, SphereScalar


def test_plane_surface_not_retraction_derivative():
    scalar = PlaneScalar()
    derivatives = TorchDerivatives(lambda x: -x*torch.tensor([1., 2.]))
    field = ProjectedField(lambda x: -np.asarray(x)*[1., 2.], scalar, 1, 2)
    y = field.surface_points([3., 2.])
    result, _ = estimate_surface_divergence(derivatives, scalar, y, probes=8)
    assert result['surface_divergence'] == pytest.approx(-2)
    assert result['jacobian_term'] == pytest.approx(-2) and result['curvature_term'] == 0


@pytest.mark.parametrize('dimension', [2, 3, 10, 1024])
def test_sphere_nonzero_curvature_analytic(dimension):
    scalar = SphereScalar(); y = np.zeros(dimension); y[0] = 1
    # Constant ambient field has J=0 but its tangential projection has nonzero divergence.
    direction = torch.zeros(dimension, dtype=torch.float64); direction[0] = 2
    derivatives = TorchDerivatives(lambda x: direction+x*0)
    result, _ = estimate_surface_divergence(derivatives, scalar, y, probes=16)
    assert result['jacobian_term'] == 0
    assert result['curvature_term'] == pytest.approx(-2*(dimension-1))
    assert result['surface_divergence'] == pytest.approx(-2*(dimension-1))
    assert result['additive_residual'] < 1e-10


@pytest.mark.parametrize('dimension', [3, 10])
def test_surface_matches_full_autograd_jacobian(dimension):
    rng = np.random.default_rng(5)
    y = rng.normal(size=dimension); y /= np.linalg.norm(y)
    a = torch.tensor(rng.normal(size=(dimension, dimension)))
    offset = torch.tensor(rng.normal(size=dimension))
    f = lambda x: a @ x + offset
    def tangent(x):
        n = x/torch.linalg.vector_norm(x)
        return f(x)-n*torch.dot(n, f(x))
    yt = torch.tensor(y)
    p = np.eye(dimension)-np.outer(y, y)
    jt = torch.func.jacrev(tangent)(yt).numpy()
    jf = torch.func.jacrev(f)(yt).numpy()
    expected = np.trace(p @ jt)
    dj = np.trace(p @ jf)
    dc = -float(f(yt) @ yt)*(dimension-1)
    assert dj+dc == pytest.approx(expected, abs=1e-12)
    result, _ = estimate_surface_divergence(TorchDerivatives(f), SphereScalar(), y, probes=2048, seed=3)
    assert abs(result['surface_divergence']-expected) < 5*result['trace_standard_error']
    assert result['curvature_term'] == pytest.approx(dc)


def test_hutchinson_convergence_across_seeds():
    d = 10
    matrix = torch.tensor(np.random.default_rng(18).normal(size=(d, d)))
    derivatives = TorchDerivatives(lambda x: matrix @ x, batch_size=64)
    y = np.ones(d)
    exact = float(matrix.diag()[1:].sum())
    errors = {8: [], 512: []}
    for seed in range(24):
        for probes in errors:
            value, _ = estimate_surface_divergence(derivatives, PlaneScalar(), y, probes=probes, seed=seed)
            errors[probes].append(value['surface_divergence']-exact)
    rmse = {key: float(np.sqrt(np.mean(np.square(value)))) for key, value in errors.items()}
    print('Hutchinson convergence, 24 seeds:', rmse)
    assert rmse[512] < .45*rmse[8]


def test_rbf_hessian_matches_independent_autograd(synthetic):
    x = synthetic.X
    scalar = ScalarExtension(n_components=2, n_centers=20, max_validation_nrmse=1).fit(x, x[:, 0]**2)
    b, mean, centers, coef = map(torch.tensor, (scalar.basis, scalar.mean, scalar.centers, scalar.coefficients))
    def value(point):
        z = (point-mean) @ b
        r = torch.exp(-torch.sum((z-centers)**2, dim=1)/(2*scalar.bandwidth**2))
        return coef[0] + (coef[1:3]*z/scalar.scale).sum() + (coef[3:]*r).sum()
    y = x[0]
    expected = torch.func.hessian(value)(torch.tensor(y)).numpy()
    geometry = scalar_geometry(scalar, y)
    actual = geometry.hessian_product(np.eye(3))
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    np.testing.assert_allclose(geometry.hessian_diagonal, np.diag(expected), atol=1e-10)
    assert geometry.hessian_trace == pytest.approx(np.trace(expected), abs=1e-10)
