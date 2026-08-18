"""Read-only adapter from a trained 20260803 ODE branch to a vector field.

The public surface intentionally follows Dynamo's ``BaseVectorField`` /
``DifferentiableVectorField`` conventions where useful: ``func`` accepts row
vectors and ``get_Jacobian`` returns a callable whose batched output is
``(dimension, dimension, cells)``.  This module does not import Dynamo and never
fits a vector field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import torch
import torch.nn.functional as F


def assert_finite(name: str, value: Any) -> None:
    """Raise on NaN/Inf instead of repairing an analysis result."""

    if torch.is_tensor(value):
        finite = torch.isfinite(value)
        if not bool(finite.all().item()):
            bad = int((~finite).sum().item())
            raise FloatingPointError(
                f"{name} contains {bad}/{value.numel()} non-finite values"
            )
        return
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.number) and not bool(np.isfinite(array).all()):
        bad = int((~np.isfinite(array)).sum())
        raise FloatingPointError(
            f"{name} contains {bad}/{array.size} non-finite values"
        )


@dataclass(frozen=True)
class VectorFieldBatch:
    """Original-space vector field quantities for a batch of row vectors."""

    velocity: np.ndarray
    divergence: np.ndarray
    acceleration: np.ndarray


class TrainedODEVectorField:
    """Dynamo-compatible view of one trained ``HillAfterLinearField``.

    Parameters are never copied back into the model and gradients are never
    accumulated.  The exact Hill Jacobian is

    ``J[i,j] = slope[i] * W[i,j] - delta[i] * indicator(i == j)``.

    This lets divergence and ``J @ V`` be evaluated without materializing a
    Jacobian for every cell.  ``torch.func.jacrev`` remains available through
    ``get_Jacobian(method="autograd")`` as an independent implementation.
    """

    def __init__(
        self,
        ode_model: torch.nn.Module,
        *,
        device: torch.device | str,
        batch_size: int = 128,
        X: np.ndarray | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.ode_model = ode_model.to(self.device).eval()
        self._validate_supported_model()
        self.dimension = int(self.ode_model.d)
        self.data: dict[str, np.ndarray | None] = {"X": None, "V": None, "Grid": None}
        self.vf_dict: dict[str, Any] = {
            "source": "trained_scdiffusion_ode",
            "ode_type": "hill_after_linear",
            "model_family": "standard_hybrid_single",
        }
        if X is not None:
            self.set_data(X)

    def _validate_supported_model(self) -> None:
        field = self.ode_model
        if str(getattr(field, "ode_type", "")) != "hill_after_linear":
            raise ValueError(
                "TrainedODEVectorField only supports ode_type=hill_after_linear"
            )
        if bool(getattr(field, "is_lincomb", True)):
            raise ValueError("the ODE branch must be a single ODE, not LinComb")
        if int(getattr(field, "num_components", -1)) != 1:
            raise ValueError("the ODE branch must have exactly one component")
        required = ("W", "b", "K", "V", "delta", "hill_n", "use_decay", "d")
        missing = [name for name in required if not hasattr(field, name)]
        if missing:
            raise TypeError(f"unsupported Hill ODE implementation; missing {missing}")
        if tuple(field.W.shape) != (int(field.d), int(field.d)):
            raise ValueError(
                f"single-ODE W must be {(int(field.d), int(field.d))}, "
                f"got {tuple(field.W.shape)}"
            )
        for name, tensor in list(field.named_parameters()) + list(field.named_buffers()):
            if tensor is not None:
                assert_finite(f"ODE tensor {name}", tensor)

    def set_data(self, X: np.ndarray) -> None:
        values, _ = self._validate_array(X)
        self.data["X"] = values.copy()
        self.data["V"] = self.func(values)

    def get_X(self, idx: int | None = None) -> np.ndarray | None:
        value = self.data["X"]
        return value if idx is None or value is None else value[idx]

    def get_V(self, idx: int | None = None) -> np.ndarray | None:
        value = self.data["V"]
        return value if idx is None or value is None else value[idx]

    def get_data(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.data["X"], self.data["V"]

    def _validate_array(self, X: np.ndarray | Iterable[float]) -> tuple[np.ndarray, bool]:
        values = np.asarray(X, dtype=np.float32)
        was_vector = values.ndim == 1
        values = np.atleast_2d(values)
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"X must have shape (cells, {self.dimension}) or ({self.dimension},), "
                f"got {values.shape}"
            )
        assert_finite("vector field input", values)
        return np.ascontiguousarray(values), was_vector

    def _tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.float32, device=self.device)

    def _chunks(self, X: np.ndarray):
        for start in range(0, len(X), self.batch_size):
            yield start, X[start : start + self.batch_size]

    def func(self, X: np.ndarray | Iterable[float]) -> np.ndarray:
        """Evaluate ``V(x)=trained_ODE(x)`` for row-vector input."""

        values, was_vector = self._validate_array(X)
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for _, chunk in self._chunks(values):
                output = self.ode_model(self._tensor(chunk), None)
                assert_finite("ODE velocity", output)
                chunks.append(output.detach().cpu().numpy())
        result = np.concatenate(chunks, axis=0)
        assert_finite("ODE velocity", result)
        return result[0] if was_vector else result

    def _hill_slope(self, x: torch.Tensor) -> torch.Tensor:
        """Return d(production_i)/d((Wx+b)_i) for each cell and target."""

        field = self.ode_model
        preactivation = F.linear(x, field.W, field.b)
        z = F.softplus(preactivation)
        positive = z > 0
        safe_z = torch.where(positive, z, torch.ones_like(z))
        log_ratio = float(field.hill_n) * (
            torch.log(safe_z) - torch.log(field.K).reshape(1, -1)
        )
        hill = torch.sigmoid(log_ratio)
        hill = torch.where(positive, hill, torch.zeros_like(hill))
        derivative_z = float(field.hill_n) * hill * (1.0 - hill) / safe_z
        derivative_z = torch.where(positive, derivative_z, torch.zeros_like(derivative_z))
        slope = field.V.reshape(1, -1) * derivative_z * torch.sigmoid(preactivation)
        assert_finite("Hill local slope", slope)
        return slope

    def evaluate(self, X: np.ndarray | Iterable[float]) -> VectorFieldBatch:
        """Compute velocity, exact divergence, and ``J @ V`` in batches."""

        values, _ = self._validate_array(X)
        velocities: list[np.ndarray] = []
        divergences: list[np.ndarray] = []
        accelerations: list[np.ndarray] = []
        field = self.ode_model
        with torch.no_grad():
            diagonal_w = torch.diagonal(field.W)
            decay = field.delta if bool(field.use_decay) else torch.zeros_like(field.delta)
            for _, chunk in self._chunks(values):
                x = self._tensor(chunk)
                velocity = field(x, None)
                slope = self._hill_slope(x)
                divergence = (slope * diagonal_w.reshape(1, -1)).sum(dim=1) - decay.sum()
                acceleration = slope * F.linear(velocity, field.W) - decay * velocity
                assert_finite("ODE velocity", velocity)
                assert_finite("ODE divergence", divergence)
                assert_finite("ODE acceleration", acceleration)
                velocities.append(velocity.detach().cpu().numpy())
                divergences.append(divergence.detach().cpu().numpy())
                accelerations.append(acceleration.detach().cpu().numpy())
        result = VectorFieldBatch(
            velocity=np.concatenate(velocities, axis=0),
            divergence=np.concatenate(divergences, axis=0),
            acceleration=np.concatenate(accelerations, axis=0),
        )
        assert_finite("velocity batch", result.velocity)
        assert_finite("divergence batch", result.divergence)
        assert_finite("acceleration batch", result.acceleration)
        return result

    def _analytic_jacobian_tensor(self, x: torch.Tensor) -> torch.Tensor:
        slope = self._hill_slope(x)
        jacobian = slope[:, :, None] * self.ode_model.W[None, :, :]
        if bool(self.ode_model.use_decay):
            diagonal = torch.arange(self.dimension, device=x.device)
            jacobian[:, diagonal, diagonal] -= self.ode_model.delta
        assert_finite("analytic Jacobian", jacobian)
        return jacobian

    def _autograd_jacobian_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(torch, "func"):
            raise RuntimeError("torch.func is required for method='autograd'")

        def single_velocity(row: torch.Tensor) -> torch.Tensor:
            return self.ode_model(row.unsqueeze(0), None).squeeze(0)

        # A loop keeps peak memory bounded for D x D Jacobians.
        matrices = [torch.func.jacrev(single_velocity)(row) for row in x]
        jacobian = torch.stack(matrices, dim=0)
        assert_finite("autograd Jacobian", jacobian)
        return jacobian

    def jacobian_tensor(
        self,
        X: np.ndarray | Iterable[float],
        *,
        method: str = "analytical",
    ) -> torch.Tensor:
        """Return Jacobians as a CPU tensor with shape ``(cells, D, D)``."""

        values, _ = self._validate_array(X)
        method = str(method).lower()
        if method not in {"analytical", "analytic", "autograd", "jacrev"}:
            raise ValueError("method must be 'analytical' or 'autograd'")
        chunks: list[torch.Tensor] = []
        for _, chunk in self._chunks(values):
            x = self._tensor(chunk)
            if method in {"analytical", "analytic"}:
                with torch.no_grad():
                    jacobian = self._analytic_jacobian_tensor(x)
            else:
                with torch.enable_grad():
                    jacobian = self._autograd_jacobian_tensor(x)
            chunks.append(jacobian.detach().cpu())
        return torch.cat(chunks, dim=0)

    def get_Jacobian(
        self,
        method: str = "analytical",
        input_vector_convention: str = "row",
        **_: Any,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Return a Dynamo-shaped Jacobian callable.

        A single input returns ``(D,D)``. Batched row input returns
        ``(D,D,cells)``, matching Dynamo's post-hoc utility convention.
        """

        if input_vector_convention != "row":
            raise ValueError("only row-vector input is supported")

        def f_jac(X: np.ndarray) -> np.ndarray:
            values = np.asarray(X)
            was_vector = values.ndim == 1
            tensor = self.jacobian_tensor(values, method=method)
            result = tensor.numpy()
            return result[0] if was_vector else np.moveaxis(result, 0, 2)

        return f_jac

    def jvp(self, x: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Compute an exact Jacobian-vector product without a full Jacobian."""

        point, point_was_vector = self._validate_array(x)
        tangent, tangent_was_vector = self._validate_array(vector)
        if not point_was_vector or not tangent_was_vector:
            raise ValueError("jvp expects one point and one tangent vector")
        x_tensor = self._tensor(point)[0]
        tangent_tensor = self._tensor(tangent)[0]

        def single_velocity(row: torch.Tensor) -> torch.Tensor:
            return self.ode_model(row.unsqueeze(0), None).squeeze(0)

        with torch.enable_grad():
            _, product = torch.func.jvp(single_velocity, (x_tensor,), (tangent_tensor,))
        assert_finite("Jacobian-vector product", product)
        return product.detach().cpu().numpy().astype(np.float64, copy=False)


__all__ = ["TrainedODEVectorField", "VectorFieldBatch", "assert_finite"]
