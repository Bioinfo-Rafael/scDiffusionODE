import importlib
import numpy as np
import pytest
import torch
from analysis.surface_divergence import TorchDerivatives
from analysis.gene_contributions import exact_gene_contributions
from common.velocity_adapter import ODEVelocity, ROOT
from scalar_fixtures import PlaneScalar, SphereScalar


@pytest.mark.parametrize('dimension', [3, 10, 1024])
def test_exact_contributions_high_dimension_no_dense_jacobian(dimension):
    y = np.zeros(dimension); y[0] = 1
    rates = torch.linspace(.1, 1., dimension, dtype=torch.float64)
    derivatives = TorchDerivatives(lambda x: -rates*x, batch_size=8)
    result = exact_gene_contributions(derivatives, SphereScalar(), y)
    pdiag = np.ones(dimension); pdiag[0] = 0
    np.testing.assert_allclose(result['jacobian'], -rates.numpy()*pdiag, atol=1e-12)
    np.testing.assert_allclose(result['curvature'], .1*pdiag, atol=1e-12)
    assert result['total'].sum() == pytest.approx(result['summary']['surface_divergence'])
    assert result['summary']['gene_sum_residual'] < 1e-9


@pytest.mark.parametrize('dimension', [3, 10])
def test_nonsymmetric_jacobian_contributions_against_dense(dimension):
    rng = np.random.default_rng(38)
    y = rng.normal(size=dimension); y /= np.linalg.norm(y)
    matrix = rng.normal(size=(dimension, dimension))
    offset = rng.normal(size=dimension)
    derivatives = TorchDerivatives(lambda x: torch.tensor(matrix) @ x+torch.tensor(offset))
    result = exact_gene_contributions(derivatives, SphereScalar(), y)
    p = np.eye(dimension)-np.outer(y, y)
    np.testing.assert_allclose(result['jacobian'], np.diag(p @ matrix @ p), atol=1e-12)
    np.testing.assert_allclose(result['curvature'], -((matrix @ y+offset) @ y)*np.diag(p @ p), atol=1e-12)


def test_original_state_dependent_lora_autodiff():
    import sys
    sys.path.insert(0, str(ROOT))
    module = importlib.import_module('ODE.ode_20260609_mathmlp')
    torch.manual_seed(5)
    field = module.LoRAField(10, K=3, rank=2, time_dim=8, hidden=16).eval().requires_grad_(False)
    velocity = ODEVelocity(field, [str(i) for i in range(10)], {})
    derivatives = TorchDerivatives.from_ode(velocity, batch_size=4)
    y = np.linspace(.5, 1.5, 10)
    before = {name: value.clone() for name, value in field.state_dict().items()}
    np.testing.assert_allclose(derivatives.value(y), velocity(y), rtol=1e-6, atol=1e-7)
    full = torch.func.jacrev(velocity.torch_single)(torch.tensor(y, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(derivatives.jvp(y, np.eye(10)), full.T, rtol=1e-5, atol=1e-7)
    result = exact_gene_contributions(derivatives, PlaneScalar(), y)
    np.testing.assert_allclose(result['jacobian'][1:], np.diag(full)[1:], rtol=1e-5, atol=1e-7)
    # Central differences detect accidental detaching of the state-dependent coefficients.
    eps = 1e-3
    fd = np.column_stack([(velocity(y+np.eye(10)[j]*eps)-velocity(y-np.eye(10)[j]*eps))/(2*eps)
                          for j in range(10)])
    np.testing.assert_allclose(full, fd, rtol=2e-2, atol=2e-4)
    for name, value in field.state_dict().items(): assert torch.equal(before[name], value)
