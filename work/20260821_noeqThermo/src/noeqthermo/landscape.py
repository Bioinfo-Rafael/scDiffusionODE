"""Paper-faithful Langevin simulation and landscape/flux calculations.

The raw histogram arrays intentionally use the authors' ``[x_bin, y_bin]``
orientation.  Plotting arrays use the transposed ``[y_bin, x_bin]`` orientation
that their MATLAB scripts create before plotting.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

ArrayField = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LandscapeResult:
    """Numerical outputs in both upstream/raw and plotting orientations."""

    num_tra: np.ndarray
    total_Fx: np.ndarray
    total_Fy: np.ndarray
    p_tra: np.ndarray
    pot_U: np.ndarray
    mean_Fx: np.ndarray
    mean_Fy: np.ndarray
    Xgrid: np.ndarray
    Ygrid: np.ndarray
    probability: np.ndarray
    potential: np.ndarray
    potential_for_gradient: np.ndarray
    mean_force_x: np.ndarray
    mean_force_y: np.ndarray
    gradient_force_x: np.ndarray
    gradient_force_y: np.ndarray
    probability_flux_x: np.ndarray
    probability_flux_y: np.ndarray
    rotational_force_x: np.ndarray
    rotational_force_y: np.ndarray
    field_x: np.ndarray
    field_y: np.ndarray
    curl: np.ndarray
    visited: np.ndarray
    summary: dict[str, Any]


def _finite(name: str, values: np.ndarray, *, allow_nan: bool = False) -> None:
    array = np.asarray(values)
    valid = ~np.isinf(array) if allow_nan else np.isfinite(array)
    if not bool(valid.all()):
        raise FloatingPointError(f"{name} contains {int((~valid).sum())} invalid values")


def common_bounds(coordinates: np.ndarray, sde: Mapping[str, Any]) -> np.ndarray:
    """Get one robust rectangular domain shared by every compared model."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f"coordinates must have shape (cells, 2), got {coordinates.shape}")
    _finite("UMAP coordinates", coordinates)
    q = float(sde["bounds_quantile"])
    low = np.quantile(coordinates, q, axis=0)
    high = np.quantile(coordinates, 1.0 - q, axis=0)
    span = high - low
    if np.any(span <= 0):
        raise ValueError("UMAP domain has a zero-width axis")
    margin = float(sde["bounds_margin"])
    return np.column_stack((low - margin * span, high + margin * span))


def calibrate_common_sde(
    coordinates: np.ndarray,
    observed_velocities: Sequence[np.ndarray],
    sde: Mapping[str, Any],
) -> dict[str, float | str | list[float]]:
    """Choose one ``dt`` and ``D`` for all models using fixed UMAP geometry.

    For calibrated settings, a typical deterministic step is a configured
    fraction of the median nearest-neighbour distance.  The diffusive RMS step
    is another configured fraction of the same distance.  This preserves the
    paper's SDE while avoiding transfer of literal constants across UMAP scales.
    """

    coordinates = np.asarray(coordinates, dtype=np.float64)
    nearest = NearestNeighbors(n_neighbors=2).fit(coordinates)
    distances, _ = nearest.kneighbors(coordinates)
    positive_nn = distances[:, 1][distances[:, 1] > 0]
    if len(positive_nn) == 0:
        raise ValueError("cannot calibrate SDE: all UMAP coordinates are duplicated")
    median_nn = float(np.median(positive_nn))
    speeds = []
    for velocity in observed_velocities:
        velocity = np.asarray(velocity, dtype=np.float64)
        if velocity.shape != coordinates.shape:
            raise ValueError("all observed UMAP velocities must align with common coordinates")
        _finite("observed UMAP velocity", velocity)
        speeds.append(np.linalg.norm(velocity, axis=1))
    positive_speeds = np.concatenate(speeds)
    positive_speeds = positive_speeds[positive_speeds > 0]
    if len(positive_speeds) == 0:
        raise ValueError("cannot calibrate SDE from an identically zero vector field")
    median_speed = float(np.median(positive_speeds))

    if sde["parameterization"] == "paper_literal":
        dt = float(sde["dt"])
        diffusion = float(sde["D"])
    else:
        dt_raw = float(sde["drift_step_fraction"]) * median_nn / median_speed
        dt = float(np.clip(dt_raw, float(sde["dt_min"]), float(sde["dt_max"])))
        noise_step = float(sde["noise_step_fraction"]) * median_nn
        diffusion = float(noise_step**2 / (2.0 * dt))
    if not np.isfinite(dt) or not np.isfinite(diffusion) or dt <= 0 or diffusion <= 0:
        raise ValueError("calibrated dt and D must be finite and positive")
    return {
        "parameterization": str(sde["parameterization"]),
        "dt": dt,
        "D": diffusion,
        "median_nearest_neighbor_distance": median_nn,
        "median_observed_speed_across_models": median_speed,
        "paper_literal_reference_dt": list(sde["paper_literal_reference"]["dt"]),
        "paper_literal_reference_D": list(sde["paper_literal_reference"]["D"]),
    }


def latin_hypercube(bounds: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Vectorized equivalent of the authors' ``LHSample`` implementation."""

    result = np.empty((size, 2), dtype=np.float64)
    for dim in range(2):
        values = (np.arange(size, dtype=np.float64) + rng.random(size)) / size
        rng.shuffle(values)
        result[:, dim] = bounds[dim, 0] + values * (bounds[dim, 1] - bounds[dim, 0])
    return result


def reflect_into_bounds(points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Reflect points exactly, including overshoots wider than one domain span."""

    points = np.asarray(points, dtype=np.float64)
    result = points.copy()
    for dim in range(2):
        low, high = bounds[dim]
        span = high - low
        phase = np.mod(result[:, dim] - low, 2.0 * span)
        result[:, dim] = low + np.where(phase <= span, phase, 2.0 * span - phase)
    if np.any(result < bounds[:, 0]) or np.any(result > bounds[:, 1]):
        raise RuntimeError("reflection failed to return points to the simulation domain")
    return result


def _checkpoint_paths(directory: Path) -> tuple[Path, Path]:
    return directory / "simulation_checkpoint.npz", directory / "rng_state.json"


def _write_checkpoint(
    directory: Path,
    *,
    step: int,
    positions: np.ndarray,
    num_tra: np.ndarray,
    total_Fx: np.ndarray,
    total_Fy: np.ndarray,
    rng: np.random.Generator,
) -> None:
    checkpoint, rng_path = _checkpoint_paths(directory)
    temporary = checkpoint.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        step=np.array(step, dtype=np.int64),
        positions=positions,
        num_tra=num_tra,
        total_Fx=total_Fx,
        total_Fy=total_Fy,
    )
    os.replace(temporary, checkpoint)
    temporary_json = rng_path.with_suffix(".tmp.json")
    temporary_json.write_text(json.dumps(rng.bit_generator.state), encoding="utf-8")
    os.replace(temporary_json, rng_path)


def simulate_stationary_distribution(
    field: ArrayField,
    bounds: np.ndarray,
    settings: Mapping[str, Any],
    calibrated: Mapping[str, Any],
    *,
    seed: int,
    checkpoint_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run batched Euler-Maruyama paths and accumulate only sufficient statistics."""

    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.shape != (2, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds must be [[xmin,xmax],[ymin,ymax]] with positive spans")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trajectories = int(settings["trajectories"])
    steps = int(settings["steps"])
    burn_in = int(settings["burn_in"])
    grid = int(settings["grid_size"])
    dt = float(calibrated["dt"])
    diffusion = float(calibrated["D"])
    checkpoint_every = int(settings["checkpoint_every"])
    rng = np.random.default_rng(seed)

    checkpoint, rng_path = _checkpoint_paths(checkpoint_dir)
    resumed = False
    if bool(settings.get("resume", True)) and checkpoint.is_file() and rng_path.is_file():
        with np.load(checkpoint) as saved:
            start_step = int(saved["step"])
            positions = saved["positions"].astype(np.float64)
            num_tra = saved["num_tra"].astype(np.int64)
            total_Fx = saved["total_Fx"].astype(np.float64)
            total_Fy = saved["total_Fy"].astype(np.float64)
        rng.bit_generator.state = json.loads(rng_path.read_text(encoding="utf-8"))
        expected = (trajectories, 2)
        if positions.shape != expected or num_tra.shape != (grid, grid):
            raise ValueError("checkpoint dimensions do not match the active configuration")
        if start_step > steps:
            raise ValueError("checkpoint step exceeds configured simulation length")
        resumed = True
    else:
        start_step = 0
        positions = latin_hypercube(bounds, trajectories, rng)
        num_tra = np.zeros((grid, grid), dtype=np.int64)
        total_Fx = np.zeros((grid, grid), dtype=np.float64)
        total_Fy = np.zeros((grid, grid), dtype=np.float64)

    noise_scale = np.sqrt(2.0 * diffusion * dt)
    low = bounds[:, 0]
    span = bounds[:, 1] - bounds[:, 0]
    drift = np.asarray(field(positions), dtype=np.float64)
    if drift.shape != positions.shape:
        raise ValueError(f"field output {drift.shape} does not match points {positions.shape}")
    _finite("Dynamo field at simulation start", drift)
    for step in range(start_step, steps):
        positions = reflect_into_bounds(
            positions + drift * dt + noise_scale * rng.standard_normal(positions.shape),
            bounds,
        )
        # The authors evaluate F again after each position update. Reuse that
        # value as the next step's drift instead of evaluating the same state twice.
        post_drift = np.asarray(field(positions), dtype=np.float64)
        _finite("post-step Dynamo field", post_drift)
        if step >= burn_in:
            bins = np.floor((positions - low) * grid / span).astype(np.int64)
            bins = np.clip(bins, 0, grid - 1)
            np.add.at(num_tra, (bins[:, 0], bins[:, 1]), 1)
            np.add.at(total_Fx, (bins[:, 0], bins[:, 1]), post_drift[:, 0])
            np.add.at(total_Fy, (bins[:, 0], bins[:, 1]), post_drift[:, 1])
        drift = post_drift
        completed = step + 1
        if completed % checkpoint_every == 0 or completed == steps:
            _write_checkpoint(
                checkpoint_dir,
                step=completed,
                positions=positions,
                num_tra=num_tra,
                total_Fx=total_Fx,
                total_Fy=total_Fy,
                rng=rng,
            )

    expected_samples = trajectories * (steps - burn_in)
    if int(num_tra.sum()) != expected_samples:
        raise RuntimeError(
            f"histogram count {num_tra.sum()} != expected post-burn-in samples {expected_samples}"
        )
    metadata = {
        "resumed": resumed,
        "start_step": start_step,
        "completed_steps": steps,
        "expected_samples": expected_samples,
        "observed_samples": int(num_tra.sum()),
        "boundary": "reflecting (periodic triangular reflection for arbitrary overshoot)",
        "integrator": "Euler-Maruyama",
        "noise_increment": "sqrt(2*D*dt) * N(0, I)",
    }
    return num_tra, total_Fx, total_Fy, metadata


def compute_landscape_quantities(
    field: ArrayField,
    bounds: np.ndarray,
    num_tra: np.ndarray,
    total_Fx: np.ndarray,
    total_Fy: np.ndarray,
    *,
    diffusion: float,
    potential_cap_above_max: float = 2.0,
) -> LandscapeResult:
    """Apply the released Python/MATLAB equations without redefining flux."""

    num_tra = np.asarray(num_tra, dtype=np.int64)
    total_Fx = np.asarray(total_Fx, dtype=np.float64)
    total_Fy = np.asarray(total_Fy, dtype=np.float64)
    if num_tra.ndim != 2 or num_tra.shape[0] != num_tra.shape[1]:
        raise ValueError("the trajectory histogram must be a square 2D grid")
    if total_Fx.shape != num_tra.shape or total_Fy.shape != num_tra.shape:
        raise ValueError("force accumulators must match num_tra")
    total = int(num_tra.sum())
    if total <= 0:
        raise ValueError("stationary histogram contains no visited samples")
    p_tra = num_tra.astype(np.float64) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        pot_U = -np.log(p_tra)
        mean_Fx = np.divide(total_Fx, num_tra, where=num_tra > 0)
        mean_Fy = np.divide(total_Fy, num_tra, where=num_tra > 0)
    mean_Fx[num_tra == 0] = np.nan
    mean_Fy[num_tra == 0] = np.nan

    size = num_tra.shape[0]
    xlin = np.linspace(bounds[0, 0], bounds[0, 1], size)
    ylin = np.linspace(bounds[1, 0], bounds[1, 1], size)
    Xgrid, Ygrid = np.meshgrid(xlin, ylin)
    probability = p_tra.T
    potential = pot_U.T
    mean_force_x = mean_Fx.T
    mean_force_y = mean_Fy.T
    visited = num_tra.T > 0

    finite_potential = potential[np.isfinite(potential)]
    cap = float(np.max(finite_potential) + potential_cap_above_max)
    potential_for_gradient = np.where(np.isfinite(potential), potential, cap)
    potential_for_gradient = np.minimum(potential_for_gradient, cap)
    dx = float(xlin[1] - xlin[0])
    dy = float(ylin[1] - ylin[0])
    dU_dy, dU_dx = np.gradient(potential_for_gradient, dy, dx)
    dP_dy, dP_dx = np.gradient(probability, dy, dx)
    gradient_force_x = -float(diffusion) * dU_dx
    gradient_force_y = -float(diffusion) * dU_dy

    safe_mean_x = np.where(visited, mean_force_x, 0.0)
    safe_mean_y = np.where(visited, mean_force_y, 0.0)
    probability_flux_x = safe_mean_x * probability - float(diffusion) * dP_dx
    probability_flux_y = safe_mean_y * probability - float(diffusion) * dP_dy
    rotational_force_x = np.divide(
        probability_flux_x,
        probability,
        out=np.zeros_like(probability_flux_x),
        where=probability > 0,
    )
    rotational_force_y = np.divide(
        probability_flux_y,
        probability,
        out=np.zeros_like(probability_flux_y),
        where=probability > 0,
    )

    points = np.column_stack((Xgrid.ravel(), Ygrid.ravel()))
    grid_field = np.asarray(field(points), dtype=np.float64).reshape(size, size, 2)
    _finite("continuous field on landscape grid", grid_field)
    field_x, field_y = grid_field[..., 0], grid_field[..., 1]
    _, dFy_dx = np.gradient(field_y, dy, dx)
    dFx_dy, _ = np.gradient(field_x, dy, dx)
    curl = dFy_dx - dFx_dy

    summary = {
        "visited_bins": int(visited.sum()),
        "unvisited_bins": int((~visited).sum()),
        "visited_fraction": float(visited.mean()),
        "probability_sum": float(probability.sum()),
        "potential_infinite_bins": int(np.isinf(potential).sum()),
        "potential_cap_for_gradient_and_plot": cap,
        "grid_dx": dx,
        "grid_dy": dy,
        "array_orientation_raw": "[x_bin, y_bin], matching released Python accumulation",
        "array_orientation_plot": "[y_bin, x_bin], matching released MATLAB transpose",
    }
    for name, array in {
        "probability": probability,
        "potential_for_gradient": potential_for_gradient,
        "gradient_force_x": gradient_force_x,
        "gradient_force_y": gradient_force_y,
        "probability_flux_x": probability_flux_x,
        "probability_flux_y": probability_flux_y,
        "rotational_force_x": rotational_force_x,
        "rotational_force_y": rotational_force_y,
        "field_x": field_x,
        "field_y": field_y,
        "curl": curl,
    }.items():
        _finite(name, array)
    if not np.isclose(probability.sum(), 1.0, rtol=1e-10, atol=1e-12):
        raise RuntimeError("stationary probability does not sum to one")
    return LandscapeResult(
        num_tra=num_tra,
        total_Fx=total_Fx,
        total_Fy=total_Fy,
        p_tra=p_tra,
        pot_U=pot_U,
        mean_Fx=mean_Fx,
        mean_Fy=mean_Fy,
        Xgrid=Xgrid,
        Ygrid=Ygrid,
        probability=probability,
        potential=potential,
        potential_for_gradient=potential_for_gradient,
        mean_force_x=mean_force_x,
        mean_force_y=mean_force_y,
        gradient_force_x=gradient_force_x,
        gradient_force_y=gradient_force_y,
        probability_flux_x=probability_flux_x,
        probability_flux_y=probability_flux_y,
        rotational_force_x=rotational_force_x,
        rotational_force_y=rotational_force_y,
        field_x=field_x,
        field_y=field_y,
        curl=curl,
        visited=visited,
        summary=summary,
    )


def save_landscape(result: LandscapeResult, output_dir: str | Path) -> None:
    """Save compact NPZ plus the authors' nine named CSV intermediates."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_arrays = {
        "num_tra": result.num_tra,
        "total_Fx": result.total_Fx,
        "total_Fy": result.total_Fy,
        "p_tra": result.p_tra,
        "pot_U": result.pot_U,
        "mean_Fx": result.mean_Fx,
        "mean_Fy": result.mean_Fy,
        "Xgrid": result.Xgrid,
        "Ygrid": result.Ygrid,
    }
    for name, array in raw_arrays.items():
        np.savetxt(output_dir / f"{name}.csv", array, delimiter=",")
    np.savez_compressed(
        output_dir / "landscape_flux_arrays.npz",
        **raw_arrays,
        probability_plot=result.probability,
        potential_plot=result.potential,
        potential_for_gradient=result.potential_for_gradient,
        mean_force_x_plot=result.mean_force_x,
        mean_force_y_plot=result.mean_force_y,
        gradient_force_x=result.gradient_force_x,
        gradient_force_y=result.gradient_force_y,
        probability_flux_x=result.probability_flux_x,
        probability_flux_y=result.probability_flux_y,
        rotational_force_x=result.rotational_force_x,
        rotational_force_y=result.rotational_force_y,
        continuous_field_x=result.field_x,
        continuous_field_y=result.field_y,
        curl_umap=result.curl,
        visited=result.visited,
    )
    (output_dir / "landscape_summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


__all__ = [
    "LandscapeResult",
    "calibrate_common_sde",
    "common_bounds",
    "compute_landscape_quantities",
    "latin_hypercube",
    "reflect_into_bounds",
    "save_landscape",
    "simulate_stationary_distribution",
]
