"""D_S = tr(P J_F) - (F.n)/||grad s|| tr(P H_s), evaluated at y on S_c.

Retraction is deliberately outside this module's differentiation. Curvature is
that of the fitted continuous scalar, not the original discrete Hodge/CellRank values.
"""
from dataclasses import dataclass
import numpy as np
import torch
from common.scalar_extension import ScalarExtension


class TorchDerivatives:
    """Official torch.func JVP/VJP; bounded probe batches, no full Jacobian."""
    def __init__(self, function, *, dtype=torch.float64, device='cpu', batch_size=16):
        self.function, self.dtype, self.device = function, dtype, torch.device(device)
        if batch_size < 1:
            raise ValueError('Positive derivative batch_size required')
        self.batch_size = batch_size

    @classmethod
    def from_ode(cls, velocity, batch_size=16):
        parameter = next(velocity.field.parameters())
        return cls(velocity.torch_single, dtype=parameter.dtype, device=parameter.device,
                   batch_size=batch_size)

    def tensor(self, x):
        return torch.as_tensor(np.asarray(x).copy(), dtype=self.dtype, device=self.device)

    def value(self, x):
        with torch.inference_mode(False), torch.enable_grad():
            value = self.function(self.tensor(x))
        result = value.detach().cpu().double().numpy()
        if result.shape != np.asarray(x).shape or not np.isfinite(result).all():
            raise ValueError('Invalid differentiable velocity output')
        return result

    def jvp(self, x, vectors):
        """Rows of vectors -> rows of J_F v, streaming at most batch_size x D."""
        vectors = np.atleast_2d(vectors)
        if vectors.shape[1] != len(x) or not np.isfinite(vectors).all():
            raise ValueError('Invalid JVP vectors')
        output = np.empty_like(vectors, dtype=float)
        with torch.inference_mode(False), torch.enable_grad():
            point = self.tensor(x)
            def one(v):
                return torch.func.jvp(self.function, (point,), (v,))[1]
            for start in range(0, len(vectors), self.batch_size):
                block = self.tensor(vectors[start:start+self.batch_size])
                output[start:start+len(block)] = torch.vmap(one)(block).detach().cpu().double().numpy()
        if not np.isfinite(output).all():
            raise ValueError('Nonfinite JVP')
        return output

    def vjp(self, x, vector):
        with torch.inference_mode(False), torch.enable_grad():
            _, pullback = torch.func.vjp(self.function, self.tensor(x))
            result = pullback(self.tensor(vector))[0].detach().cpu().double().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite VJP')
        return result


@dataclass
class Geometry:
    normal: np.ndarray
    gradient_norm: float
    hessian_diagonal: np.ndarray
    hessian_trace: float
    hessian_product: object

    def project(self, vectors):
        v = np.asarray(vectors)
        return v-np.sum(v*self.normal, axis=-1, keepdims=True)*self.normal

    def projected_hessian_diagonal(self):
        hn = self.hessian_product(self.normal)
        return self.hessian_diagonal-2*self.normal*hn+self.normal**2*(self.normal @ hn)

    def projected_hessian_trace(self):
        return self.hessian_trace-self.normal @ self.hessian_product(self.normal)


def scalar_geometry(scalar, y, gradient_min=1e-10):
    """Reuse existing gradient; analytic PCA Hessian with O(D r + r²) storage."""
    gradient = scalar.gradient(y)[0]
    norm = float(np.linalg.norm(gradient))
    if not np.isfinite(norm) or norm < gradient_min:
        raise ValueError('surface_undefined_vanishing_gradient')
    if isinstance(scalar, ScalarExtension):
        basis, reduced = scalar.basis, scalar.hessian_reduced(y)
        def product(v):
            return (np.asarray(v) @ basis @ reduced) @ basis.T
        diagonal = np.sum((basis @ reduced)*basis, axis=1)
        trace = float(np.trace(reduced))  # basis columns are orthonormal
    else:
        # Explicit analytic fixture interface; no numerical Hessian fallback.
        product = lambda v: scalar.hessian_vector_product(y, v)
        diagonal = scalar.hessian_diagonal(y)
        trace = scalar.hessian_trace(y)
    if not np.isfinite(diagonal).all() or not np.isfinite(trace):
        raise ValueError('Nonfinite scalar Hessian')
    return Geometry(gradient/norm, norm, diagonal, trace, product)


def tangent_field_derivative(geometry, velocity, jv, directions):
    """Directional derivative of P(x)F(x) including dP, never d(retraction)."""
    dn = geometry.project(geometry.hessian_product(directions))/geometry.gradient_norm
    n = geometry.normal
    return geometry.project(jv) - dn*(velocity @ n) - np.sum(dn*velocity, axis=-1, keepdims=True)*n


def estimate_surface_divergence(derivatives, scalar, y, *, probes=64, seed=0):
    if probes < 2:
        raise ValueError('At least two probes required to estimate standard error')
    geometry = scalar_geometry(scalar, y)
    f = derivatives.value(y)
    coefficient = -(f @ geometry.normal)/geometry.gradient_norm
    dc = float(coefficient*geometry.projected_hessian_trace())
    rng = np.random.default_rng(seed)
    jacobian_samples, curvature_samples, direct_samples = [], [], []
    for start in range(0, probes, derivatives.batch_size):
        signs = rng.choice([-1., 1.], size=(min(derivatives.batch_size, probes-start), len(y)))
        tangent = geometry.project(signs)
        jv = derivatives.jvp(y, tangent)
        jacobian_samples.extend(np.sum(tangent*jv, axis=1))
        curvature_samples.extend(coefficient*np.sum(tangent*geometry.hessian_product(tangent), axis=1))
        direct = tangent_field_derivative(geometry, f, jv, tangent)
        direct_samples.extend(np.sum(tangent*direct, axis=1))
    j, c, direct = map(np.asarray, (jacobian_samples, curvature_samples, direct_samples))
    residual = float(np.max(abs(direct-j-c)))
    bound = 1e-9*(1+float(np.max(abs(j)+abs(c))))
    if residual > bound:
        raise RuntimeError(f'Surface derivative decomposition identity failed: {residual}')
    se = float(np.std(j, ddof=1)/np.sqrt(probes))
    result = dict(surface_divergence=float(j.mean()+dc), jacobian_term=float(j.mean()),
                  curvature_term=dc, trace_standard_error=se, probe_count=int(probes), seed=int(seed),
                  additive_residual=residual, direct_trace_estimate=float(direct.mean()),
                  direct_trace_standard_error=float(np.std(direct, ddof=1)/np.sqrt(probes)),
                  curvature_trace_standard_error=float(np.std(c, ddof=1)/np.sqrt(probes)),
                  curvature_trace_discrepancy=float(c.mean()-dc), gradient_norm=geometry.gradient_norm)
    if not all(np.isfinite(v) for v in result.values()):
        raise ValueError('Nonfinite surface divergence result')
    # Samples are retained for probe-convergence plots/tables without any new JVPs.
    return result, {'jacobian_probes': j, 'curvature_probes': c, 'direct_probes': direct}
