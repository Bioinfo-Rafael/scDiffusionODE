"""Normal retraction and tangent projection. Retraction is not a closest-point projection."""
import numpy as np
from .velocity_adapter import as_batch, evaluate_velocity


class ProjectionError(RuntimeError):
    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class ProjectedField:
    """F_c(x)=(I-gg^T/||g||²)F(y), y obtained by local normal retraction.

    At the nonnegative boundary, the residual-reducing direction is restricted to
    feasible coordinates and trial points are clipped to the nonnegative orthant.
    Does not guarantee a unique, nearest or reachable surface point. A batch fails
    atomically if any row fails; partial/unconverged velocities are never returned.
    """
    def __init__(self, velocity, scalar, c, dimension, *, tolerance=1e-8,
                 gradient_min=1e-10, max_iterations=100, nonnegative=True,
                 max_distribution_distance=None):
        if not np.isfinite(c) or not scalar.valid_range[0] <= c <= scalar.valid_range[1]:
            raise ValueError(f'c must lie in the fitted-cell range {scalar.valid_range}')
        if not np.isfinite([tolerance, gradient_min]).all() or tolerance <= 0 or gradient_min <= 0 or max_iterations < 1:
            raise ValueError('Positive solver tolerances and iteration budget required')
        self.velocity, self.scalar, self.c, self.dimension = velocity, scalar, float(c), dimension
        self.tolerance, self.gradient_min, self.max_iterations = tolerance, gradient_min, max_iterations
        self.nonnegative = nonnegative
        self.max_distance = scalar.support_radius if max_distribution_distance is None else max_distribution_distance
        if not np.isfinite(self.max_distance) or self.max_distance <= 0:
            raise ValueError('Positive support distance required')
        self.settings = dict(tolerance=tolerance, gradient_min=gradient_min, max_iterations=max_iterations,
                             nonnegative=nonnegative, max_distribution_distance=self.max_distance,
                             retraction_method='active_set_nonnegative' if nonnegative else 'normal')
        self.last_diagnostics = None

    def _retract(self, x):
        y = x.copy()
        if self.nonnegative and (x < 0).any():
            raise ValueError('Negative expression input')
        initial_distance = self.scalar.distribution_distance(x)
        reasons, iterations = [], []
        for i, row in enumerate(y):
            reason = 'max_iterations'
            if initial_distance[i] > self.max_distance:
                reasons.append('input_outside_support'); iterations.append(0); continue
            for iteration in range(self.max_iterations + 1):
                residual = float(self.scalar.values(row)[0] - self.c)
                g = self.scalar.gradient(row)[0]
                norm2 = float(g @ g)
                if not np.isfinite(residual) or not np.isfinite(norm2):
                    reason = 'nonfinite_scalar_or_gradient'; break
                if norm2 < self.gradient_min**2:
                    reason = 'vanishing_gradient'; break
                if abs(residual) <= self.tolerance:
                    reason = 'converged'; break
                if iteration == self.max_iterations:
                    break
                direction = -np.sign(residual)*g
                if self.nonnegative:
                    direction[(row <= 0) & (direction < 0)] = 0
                feasible_norm2 = float(direction @ direction)
                if feasible_norm2 < self.gradient_min**2:
                    reason = 'blocked_feasible_gradient'; break
                step = abs(residual)*direction/feasible_norm2
                # Backtracking avoids invalid expression and rejects non-decreasing residual.
                accepted = False
                for fraction in 0.5**np.arange(30):
                    candidate = row+fraction*step
                    if self.nonnegative:
                        candidate = np.maximum(candidate, 0.)
                    if self.scalar.distribution_distance(candidate)[0] > self.max_distance:
                        continue
                    if abs(self.scalar.values(candidate)[0]-self.c) < abs(residual):
                        row[:] = candidate
                        accepted = True
                        break
                if not accepted:
                    reason = 'line_search_or_domain_failure'; break
            reasons.append(reason); iterations.append(iteration)
        residuals = abs(self.scalar.values(y)-self.c)
        gradients = self.scalar.gradient(y)
        norms = np.linalg.norm(gradients, axis=1)
        distances = self.scalar.distribution_distance(y)
        ok = (np.array(reasons) == 'converged') & (distances <= self.max_distance)
        diagnostics = dict(surface_residual=residuals, displacement=np.linalg.norm(y-x, axis=1),
                           distribution_distance=distances, input_distribution_distance=initial_distance,
                           gradient_norm=norms, converged=ok, status=reasons, iterations=np.array(iterations))
        self.last_diagnostics = diagnostics
        if not ok.all():
            raise ProjectionError(f'Level-set retraction failed: {reasons}', diagnostics)
        return y, gradients, diagnostics

    def surface_points(self, x):
        x, single = as_batch(x, self.dimension)
        y, _, _ = self._retract(x)
        return y[0] if single else y

    def evaluate(self, x):
        x, single = as_batch(x, self.dimension)
        y, g, diagnostics = self._retract(x)
        v = evaluate_velocity(self.velocity, y)
        projected = v - (np.sum(g*v, axis=1)/np.sum(g*g, axis=1))[:, None]*g
        error = abs(np.sum(g*projected, axis=1))
        diagnostics['tangent_error'] = error
        bound = 1e-10*(1+np.linalg.norm(g, axis=1)*np.linalg.norm(v, axis=1))
        if not np.isfinite(projected).all() or (error > bound).any():
            raise ProjectionError('Nonfinite or non-tangent projected velocity', diagnostics)
        return (projected[0] if single else projected), diagnostics

    def __call__(self, x):
        return self.evaluate(x)[0]
