"""Exact gene-coordinate contributions, not causal effects or GRN edge strengths."""
import numpy as np
from .surface_divergence import scalar_geometry, tangent_field_derivative


def exact_gene_contributions(derivatives, scalar, y):
    """diag(P J P), -F.n/||g|| diag(P H P), without storing a D x D matrix.

    D coordinate JVPs are expensive, so use only on representative cells. VJP keeps
    both Jn and J^Tn, which differ for a general state-dependent non-symmetric field.
    """
    geometry = scalar_geometry(scalar, y)
    n, d = geometry.normal, len(y)
    f = derivatives.value(y)
    jn = derivatives.jvp(y, n)[0]
    jtn = derivatives.vjp(y, n)
    normal_sensitivity = float(n @ jn)
    diagonal = np.empty(d)
    direct_trace = 0.
    # Independently accumulate tr(P J_{PF}) using the same streamed coordinate products.
    for start in range(0, d, derivatives.batch_size):
        indices = np.arange(start, min(d, start+derivatives.batch_size))
        basis = np.zeros((len(indices), d))
        basis[np.arange(len(indices)), indices] = 1.
        je = derivatives.jvp(y, basis)
        diagonal[indices] = je[np.arange(len(indices)), indices]
        tangent = geometry.project(basis)
        jv = je-n[indices, None]*jn
        derivative = tangent_field_derivative(geometry, f, jv, tangent)
        direct_trace += float(np.sum(tangent*derivative))
    cj = diagonal-n*jn-n*jtn+n*n*normal_sensitivity
    coefficient = -(f @ n)/geometry.gradient_norm
    cc = coefficient*geometry.projected_hessian_diagonal()
    total = cj+cc
    dj = float(diagonal.sum()-normal_sensitivity)
    dc = float(coefficient*geometry.projected_hessian_trace())
    errors = dict(jacobian_sum_residual=float(abs(cj.sum()-dj)),
                  curvature_sum_residual=float(abs(cc.sum()-dc)),
                  gene_sum_residual=float(abs(total.sum()-direct_trace)),
                  decomposition_residual=float(abs(dj+dc-direct_trace)))
    # Float32 original D20 forward/AD, double scalar geometry and contractions.
    tolerance = 5e-5 if str(derivatives.dtype) == 'torch.float32' else 1e-9
    bound = tolerance*(1+np.sum(abs(cj))+np.sum(abs(cc)))
    if max(errors.values()) > bound or not np.isfinite(total).all():
        raise RuntimeError(f'Gene-coordinate additive identity failed: {errors}; bound={bound}')
    return dict(jacobian=cj, curvature=cc, total=total,
                summary=dict(surface_divergence=direct_trace, jacobian_term=dj, curvature_term=dc,
                             numerical_bound=float(bound), **errors))


def grn_sensitivities(derivatives, points, edges, feature_ids, *, candidate_sha256):
    """J[target, source] for an explicit, bounded set of existing candidate edges.

    No novel network inference or causal claim. Returns per-edge mean signed/absolute
    sensitivities over the provided representative points.
    """
    index = {gene: i for i, gene in enumerate(feature_ids)}
    edges = [(source, target) for source, target in edges if source in index and target in index]
    if not edges or not len(points):
        return []
    sources = list(dict.fromkeys(source for source, _ in edges))
    by_source = {source: [i for i, edge in enumerate(edges) if edge[0] == source] for source in sources}
    values = np.empty((len(points), len(edges)))
    for row, y in enumerate(points):
        for start in range(0, len(sources), derivatives.batch_size):
            block = sources[start:start+derivatives.batch_size]
            directions = np.zeros((len(block), len(feature_ids)))
            directions[np.arange(len(block)), [index[source] for source in block]] = 1
            jv = derivatives.jvp(y, directions)
            for k, source in enumerate(block):
                for edge_index in by_source[source]:
                    values[row, edge_index] = jv[k, index[edges[edge_index][1]]]
    return [dict(source=source, target=target, mean_sensitivity=float(values[:, i].mean()),
                 mean_absolute_sensitivity=float(abs(values[:, i]).mean()), n_cells=len(points),
                 candidate_file_sha256=candidate_sha256) for i, (source, target) in enumerate(edges)]
