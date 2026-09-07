"""Dynamo-traceable post-hoc calculations for a trained ODE vector field.

Only the mathematical post-processing performed after vector-field fitting is
implemented here.  No SparseVFC object is constructed and no vector field is
estimated from real or generated cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse.linalg import (  # noqa: E402
    ArpackNoConvergence,
    LinearOperator,
    eigs,
    gmres,
)
from sklearn.cluster import KMeans  # noqa: E402

from vector_field_adapter import (  # noqa: E402
    TrainedODEVectorField,
    VectorFieldBatch,
    assert_finite,
)


METRICS = (
    "speed",
    "divergence",
    "acceleration_norm",
    "cosine_velocity_acceleration",
)


@dataclass(frozen=True)
class JacobianAggregate:
    dataset: str
    n_cells: int
    mean: np.ndarray
    mean_absolute: np.ndarray


@dataclass(frozen=True)
class FixedPointSearch:
    points: np.ndarray
    metadata: pd.DataFrame
    eigenvalues: pd.DataFrame
    attempts: pd.DataFrame


def evaluate_dataset(
    adapter: TrainedODEVectorField,
    X: np.ndarray,
    *,
    dataset: str,
    annotations: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, VectorFieldBatch]:
    """Evaluate Dynamo velocity/divergence/acceleration definitions."""

    result = adapter.evaluate(X)
    speed = np.linalg.norm(result.velocity.astype(np.float64), axis=1)
    acceleration_norm = np.linalg.norm(result.acceleration.astype(np.float64), axis=1)
    dot = np.einsum(
        "ij,ij->i",
        result.velocity.astype(np.float64),
        result.acceleration.astype(np.float64),
    )
    denominator = speed * acceleration_norm
    cosine_defined = denominator > 0
    cosine = np.zeros(len(X), dtype=np.float64)
    cosine[cosine_defined] = dot[cosine_defined] / denominator[cosine_defined]
    # Roundoff can place a mathematically valid cosine just outside [-1, 1].
    if np.any(cosine < -1.000001) or np.any(cosine > 1.000001):
        raise FloatingPointError("cosine(V,a) is outside its numerical tolerance")
    cosine = np.clip(cosine, -1.0, 1.0)
    labels = (
        np.asarray(annotations, dtype=object)
        if annotations is not None
        else np.repeat(dataset, len(X)).astype(object)
    )
    if labels.shape != (len(X),):
        raise ValueError(f"annotations must have shape {(len(X),)}, got {labels.shape}")
    frame = pd.DataFrame(
        {
            "cell_id": [f"{dataset}_{index:06d}" for index in range(len(X))],
            "dataset": dataset,
            "source_index": np.arange(len(X), dtype=np.int64),
            "annotation": labels.astype(str),
            "speed": speed,
            "divergence": result.divergence.astype(np.float64),
            "acceleration_norm": acceleration_norm,
            "cosine_velocity_acceleration": cosine,
            "cosine_defined": cosine_defined,
        }
    )
    for metric in METRICS:
        assert_finite(f"{dataset} {metric}", frame[metric].to_numpy())
    return frame, result


def summarize_metrics(cell_metrics: pd.DataFrame) -> pd.DataFrame:
    """Create dataset-level and annotation-level summary statistics."""

    rows: list[dict[str, Any]] = []

    def append_group(level: str, dataset: str, group: str, frame: pd.DataFrame) -> None:
        for metric in METRICS:
            values = frame[metric].to_numpy(dtype=np.float64)
            assert_finite(f"summary source {level}/{group}/{metric}", values)
            rows.append(
                {
                    "aggregation": level,
                    "dataset": dataset,
                    "group": group,
                    "metric": metric,
                    "n_cells": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=0)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )

    for dataset, frame in cell_metrics.groupby("dataset", sort=True):
        append_group("dataset", str(dataset), "all", frame)
        for annotation, group in frame.groupby("annotation", sort=True):
            append_group("annotation", str(dataset), str(annotation), group)
    summary = pd.DataFrame(rows)
    for column in ("mean", "median", "std", "minimum", "maximum"):
        assert_finite(f"summary column {column}", summary[column].to_numpy())
    return summary


def aggregate_jacobians(
    adapter: TrainedODEVectorField,
    selected: Mapping[str, np.ndarray],
    *,
    method: str = "analytical",
) -> dict[str, JacobianAggregate]:
    """Stream Jacobians and retain only mean and mean-absolute matrices."""

    raw: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    dimension = adapter.dimension
    for dataset, cells in selected.items():
        cells = np.asarray(cells, dtype=np.float32)
        if cells.ndim != 2 or cells.shape[1] != dimension or len(cells) == 0:
            raise ValueError(f"invalid selected Jacobian cells for {dataset}: {cells.shape}")
        total = np.zeros((dimension, dimension), dtype=np.float64)
        total_absolute = np.zeros_like(total)
        # One cell at a time bounds peak memory independently of adapter batch size.
        for cell in cells:
            jacobian = adapter.jacobian_tensor(cell, method=method)[0].numpy()
            assert_finite(f"{dataset} Jacobian", jacobian)
            total += jacobian
            total_absolute += np.abs(jacobian)
        raw[dataset] = (len(cells), total, total_absolute)

    all_count = sum(value[0] for value in raw.values())
    if all_count <= 0:
        raise ValueError("no Jacobian cells were selected")
    combined_total = sum((value[1] for value in raw.values()), np.zeros((dimension, dimension)))
    combined_absolute = sum(
        (value[2] for value in raw.values()), np.zeros((dimension, dimension))
    )
    raw["combined"] = (all_count, combined_total, combined_absolute)
    aggregates: dict[str, JacobianAggregate] = {}
    for dataset, (count, total, total_absolute) in raw.items():
        mean = total / count
        mean_absolute = total_absolute / count
        assert_finite(f"{dataset} mean Jacobian", mean)
        assert_finite(f"{dataset} mean absolute Jacobian", mean_absolute)
        aggregates[dataset] = JacobianAggregate(
            dataset=dataset,
            n_cells=count,
            mean=mean.astype(np.float32),
            mean_absolute=mean_absolute.astype(np.float32),
        )
    return aggregates


def save_jacobian_aggregates(
    aggregates: Mapping[str, JacobianAggregate], path: Path
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for dataset, aggregate in aggregates.items():
        arrays[f"mean__{dataset}"] = aggregate.mean
        arrays[f"mean_absolute__{dataset}"] = aggregate.mean_absolute
        arrays[f"n_cells__{dataset}"] = np.asarray(aggregate.n_cells, dtype=np.int64)
    np.savez_compressed(path, **arrays)


def jacobian_gene_summary(
    aggregates: Mapping[str, JacobianAggregate], genes: Sequence[str]
) -> pd.DataFrame:
    """Reproduce Dynamo's regulator/effector mean reductions."""

    genes_array = np.asarray(genes, dtype=object)
    rows: list[pd.DataFrame] = []
    for dataset, aggregate in aggregates.items():
        signed = aggregate.mean.astype(np.float64)
        absolute = aggregate.mean_absolute.astype(np.float64)
        values = pd.DataFrame(
            {
                "dataset": dataset,
                "gene": genes_array,
                # J[target, source]: source influence pools over target axis.
                "mean_outgoing_influence": np.mean(signed, axis=0),
                "mean_absolute_outgoing_influence": np.mean(absolute, axis=0),
                # Target response pools over source axis.
                "mean_incoming_influence": np.mean(signed, axis=1),
                "mean_absolute_incoming_influence": np.mean(absolute, axis=1),
                "mean_self_effect": np.diag(signed),
                "n_jacobian_cells": aggregate.n_cells,
            }
        )
        for column in (
            "mean_outgoing_influence",
            "mean_absolute_outgoing_influence",
            "mean_incoming_influence",
            "mean_absolute_incoming_influence",
        ):
            values[f"rank_{column}"] = values[column].rank(
                method="min", ascending=False
            ).astype(np.int64)
        values["rank_most_negative_outgoing_influence"] = values[
            "mean_outgoing_influence"
        ].rank(method="min", ascending=True).astype(np.int64)
        values["rank_most_negative_incoming_influence"] = values[
            "mean_incoming_influence"
        ].rank(method="min", ascending=True).astype(np.int64)
        rows.append(values)
    result = pd.concat(rows, ignore_index=True)
    for column in result.select_dtypes(include=[np.number]).columns:
        assert_finite(f"Jacobian gene summary {column}", result[column].to_numpy())
    return result


def _ranked_flat_indices(
    matrix: np.ndarray,
    *,
    largest: bool,
    top_n: int,
    positive_only: bool = False,
    negative_only: bool = False,
    exclude_diagonal: bool = False,
) -> np.ndarray:
    flat = matrix.reshape(-1)
    eligible = np.ones(flat.shape, dtype=bool)
    if exclude_diagonal:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("diagonal exclusion requires a square matrix")
        diagonal = np.arange(matrix.shape[0])
        eligible[np.ravel_multi_index((diagonal, diagonal), matrix.shape)] = False
    if positive_only:
        eligible &= flat > 0
    if negative_only:
        eligible &= flat < 0
    candidates = np.flatnonzero(eligible)
    if len(candidates) == 0:
        return candidates
    values = flat[candidates]
    order = np.argsort(values)
    if largest:
        order = order[::-1]
    return candidates[order[: min(top_n, len(order))]]


def jacobian_top_interactions(
    aggregates: Mapping[str, JacobianAggregate],
    genes: Sequence[str],
    *,
    top_n: int,
    exclude_diagonal: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    gene_names = np.asarray(genes, dtype=object)
    for dataset, aggregate in aggregates.items():
        signed = aggregate.mean.astype(np.float64)
        absolute = aggregate.mean_absolute.astype(np.float64)
        rankings = {
            "positive_mean": _ranked_flat_indices(
                signed,
                largest=True,
                top_n=top_n,
                positive_only=True,
                exclude_diagonal=exclude_diagonal,
            ),
            "negative_mean": _ranked_flat_indices(
                signed,
                largest=False,
                top_n=top_n,
                negative_only=True,
                exclude_diagonal=exclude_diagonal,
            ),
            "absolute_mean": _ranked_flat_indices(
                absolute,
                largest=True,
                top_n=top_n,
                exclude_diagonal=exclude_diagonal,
            ),
        }
        for rank_type, indices in rankings.items():
            for rank, flat_index in enumerate(indices, start=1):
                target, source = np.unravel_index(int(flat_index), signed.shape)
                rows.append(
                    {
                        "dataset": dataset,
                        "rank_type": rank_type,
                        "rank": rank,
                        "source_gene": str(gene_names[source]),
                        "target_gene": str(gene_names[target]),
                        "source_index": source,
                        "target_index": target,
                        "mean_jacobian": float(aggregate.mean[target, source]),
                        "mean_absolute_jacobian": float(
                            aggregate.mean_absolute[target, source]
                        ),
                        "n_jacobian_cells": aggregate.n_cells,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        assert_finite("top interaction values", result["mean_jacobian"].to_numpy())
        assert_finite(
            "top absolute interaction values",
            result["mean_absolute_jacobian"].to_numpy(),
        )
    return result


def acceleration_gene_summary(
    batches: Mapping[str, VectorFieldBatch], genes: Sequence[str]
) -> pd.DataFrame:
    """Apply Dynamo's raw and absolute gene-ranking reductions to acceleration."""

    frames: list[pd.DataFrame] = []
    for dataset, batch in batches.items():
        signed = np.mean(batch.acceleration.astype(np.float64), axis=0)
        absolute = np.mean(np.abs(batch.acceleration.astype(np.float64)), axis=0)
        frame = pd.DataFrame(
            {
                "dataset": dataset,
                "gene": list(genes),
                "mean_acceleration": signed,
                "mean_absolute_acceleration": absolute,
            }
        )
        frame["rank_mean_acceleration"] = frame["mean_acceleration"].rank(
            method="min", ascending=False
        ).astype(np.int64)
        frame["rank_mean_absolute_acceleration"] = frame[
            "mean_absolute_acceleration"
        ].rank(method="min", ascending=False).astype(np.int64)
        frame["rank_most_negative_acceleration"] = frame[
            "mean_acceleration"
        ].rank(method="min", ascending=True).astype(np.int64)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    assert_finite("acceleration gene summary", result.select_dtypes(include=[np.number]))
    return result


def sensitivity_aggregates(
    adapter: TrainedODEVectorField,
    selected: Mapping[str, np.ndarray],
    *,
    max_dimension: int,
) -> dict[str, JacobianAggregate]:
    """Compute Dynamo's normalized sensitivity for explicitly selected cells."""

    dimension = adapter.dimension
    if dimension > max_dimension:
        raise ValueError(
            f"sensitivity requires dense D x D inversion; D={dimension} exceeds "
            f"--sensitivity-max-dim={max_dimension}"
        )
    identity = np.eye(dimension, dtype=np.float64)
    raw: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
    for dataset, cells in selected.items():
        total = np.zeros((dimension, dimension), dtype=np.float64)
        total_absolute = np.zeros_like(total)
        for cell in cells:
            jacobian = adapter.jacobian_tensor(cell)[0].numpy().astype(np.float64)
            inverse = np.linalg.inv(identity - jacobian)
            diagonal = np.diag(inverse)
            if np.any(diagonal == 0):
                raise np.linalg.LinAlgError(
                    "Dynamo sensitivity normalization has a zero inverse diagonal"
                )
            sensitivity = inverse @ np.diag(1.0 / diagonal)
            assert_finite(f"{dataset} sensitivity", sensitivity)
            total += sensitivity
            total_absolute += np.abs(sensitivity)
        raw[dataset] = (len(cells), total, total_absolute)
    count = sum(value[0] for value in raw.values())
    raw["combined"] = (
        count,
        sum((value[1] for value in raw.values()), np.zeros_like(identity)),
        sum((value[2] for value in raw.values()), np.zeros_like(identity)),
    )
    return {
        dataset: JacobianAggregate(
            dataset=dataset,
            n_cells=n_cells,
            mean=(total / n_cells).astype(np.float32),
            mean_absolute=(total_absolute / n_cells).astype(np.float32),
        )
        for dataset, (n_cells, total, total_absolute) in raw.items()
    }


def representative_real_cells(
    X: np.ndarray,
    pca_coordinates: np.ndarray,
    *,
    n_seeds: int,
    seed: int,
    annotations: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Choose actual representative cells using PCA-space k-means and labels."""

    if n_seeds <= 0:
        return np.empty((0, X.shape[1]), dtype=np.float32), []
    label_values = (
        np.asarray(annotations, dtype=object).astype(str)
        if annotations is not None
        else None
    )
    label_candidates: list[str] = []
    if label_values is not None:
        unique, counts = np.unique(label_values, return_counts=True)
        ordered = sorted(
            zip(unique.tolist(), counts.tolist()),
            key=lambda item: (-item[1], item[0]),
        )
        label_slots = min(len(ordered), int(n_seeds) // 2)
        label_candidates = [label for label, _ in ordered[:label_slots]]
    n_clusters = min(max(1, int(n_seeds) - len(label_candidates)), len(X))
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    kmeans.fit(pca_coordinates)
    chosen: list[int] = []
    sources: list[str] = []
    for cluster, center in enumerate(kmeans.cluster_centers_):
        distances = np.linalg.norm(pca_coordinates - center[None, :], axis=1)
        index = int(np.argmin(distances))
        if index not in chosen:
            chosen.append(index)
            sources.append(f"pca_kmeans_{cluster}")
    if label_values is not None:
        for label in label_candidates:
            indices = np.flatnonzero(label_values == label)
            center = np.mean(pca_coordinates[indices], axis=0)
            index = int(indices[np.argmin(np.linalg.norm(pca_coordinates[indices] - center, axis=1))])
            if index not in chosen:
                chosen.append(index)
                sources.append(f"annotation_{label}")
    return X[np.asarray(chosen, dtype=np.int64)], sources


def _sparse_eigenvalues(
    adapter: TrainedODEVectorField,
    point: np.ndarray,
    *,
    leading: int,
    tolerance: float,
) -> tuple[list[tuple[str, complex]], list[str]]:
    dimension = adapter.dimension
    operator = LinearOperator(
        shape=(dimension, dimension),
        matvec=lambda vector: adapter.jvp(point, np.asarray(vector, dtype=np.float64)),
        dtype=np.float64,
    )
    count = min(max(1, int(leading)), dimension - 2)
    records: list[tuple[str, complex]] = []
    errors: list[str] = []
    for side, which in (("rightmost", "LR"), ("leftmost", "SR")):
        try:
            values = eigs(operator, k=count, which=which, tol=tolerance, return_eigenvectors=False)
        except ArpackNoConvergence as exc:
            values = exc.eigenvalues
            errors.append(f"{side}: {exc}")
        except Exception as exc:  # preserve a found fixed point even if ARPACK fails
            values = np.asarray([], dtype=np.complex128)
            errors.append(f"{side}: {type(exc).__name__}: {exc}")
        for value in np.asarray(values).reshape(-1):
            records.append((side, complex(value)))
    return records, errors


def _classify_eigenvalues(
    values: Sequence[complex], *, tolerance: float
) -> str:
    if not values:
        return "indeterminate"
    real = np.real(np.asarray(values, dtype=np.complex128))
    has_positive = bool(np.any(real > tolerance))
    has_negative = bool(np.any(real < -tolerance))
    has_near_zero = bool(np.any(np.abs(real) <= tolerance))
    if has_near_zero:
        return "nonhyperbolic"
    if has_positive and has_negative:
        return "saddle"
    if has_positive:
        return "unstable"
    if has_negative:
        return "stable"
    return "indeterminate"


def _newton_gmres(
    adapter: TrainedODEVectorField,
    initial: np.ndarray,
    *,
    residual_tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, bool, float, str, int]:
    """Solve V(x)=0 using the trained ODE's exact Jacobian.

    Dynamo's utility uses MINPACK ``fsolve``. Numerical differencing is not
    reliable here because the restored ODE evaluates in float32 and D can be
    1024, so this adapter-specific solver retains Dynamo's root-finding step but
    supplies the exact trained-ODE Jacobian to a memory-bounded GMRES solve.
    """

    point = np.asarray(initial, dtype=np.float64).copy()
    dimension = adapter.dimension
    for iteration in range(max_iterations + 1):
        velocity = np.asarray(adapter.func(point), dtype=np.float64)
        assert_finite("Newton-GMRES point", point)
        assert_finite("Newton-GMRES residual", velocity)
        residual = float(np.linalg.norm(velocity))
        if residual <= residual_tolerance:
            return point, True, residual, "converged", iteration
        if iteration == max_iterations:
            break

        jacobian = adapter.jacobian_tensor(point)[0].numpy().astype(np.float64)
        assert_finite("Newton-GMRES Jacobian", jacobian)
        operator = LinearOperator(
            (dimension, dimension),
            matvec=lambda vector: jacobian @ vector,
            dtype=np.float64,
        )
        diagonal = np.diag(jacobian)
        inverse_diagonal = np.ones(dimension, dtype=np.float64)
        usable = np.abs(diagonal) > np.finfo(np.float64).eps
        inverse_diagonal[usable] = 1.0 / diagonal[usable]
        preconditioner = LinearOperator(
            (dimension, dimension),
            matvec=lambda vector: inverse_diagonal * vector,
            dtype=np.float64,
        )
        step, info = gmres(
            operator,
            -velocity,
            M=preconditioner,
            tol=1e-5,
            atol=max(residual_tolerance * 0.1, 1e-12),
            restart=min(30, dimension),
            maxiter=10,
        )
        assert_finite("Newton-GMRES step", step)
        if info < 0:
            return point, False, residual, f"GMRES illegal input/breakdown ({info})", iteration

        accepted = False
        scale = 1.0
        for _ in range(12):
            candidate = point + scale * step
            candidate_velocity = np.asarray(adapter.func(candidate), dtype=np.float64)
            assert_finite("Newton-GMRES candidate", candidate)
            assert_finite("Newton-GMRES candidate residual", candidate_velocity)
            candidate_residual = float(np.linalg.norm(candidate_velocity))
            if candidate_residual < residual or candidate_residual <= residual_tolerance:
                point = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            detail = f"GMRES info={info}; line search did not reduce residual"
            return point, False, residual, detail, iteration + 1

    final_velocity = np.asarray(adapter.func(point), dtype=np.float64)
    final_residual = float(np.linalg.norm(final_velocity))
    return point, False, final_residual, "maximum iterations reached", max_iterations


def find_fixed_points(
    adapter: TrainedODEVectorField,
    seeds: np.ndarray,
    seed_sources: Sequence[str],
    domain: np.ndarray,
    *,
    residual_tolerance: float,
    redundant_tolerance: float,
    max_iterations: int,
    full_eigen_max_dim: int,
    leading_eigenvalues: int,
    eigen_tolerance: float,
    stability_tolerance: float,
) -> FixedPointSearch:
    """Follow Dynamo's seed/search/domain/dedup/classification workflow."""

    if domain.shape != (adapter.dimension, 2):
        raise ValueError(f"domain must have shape {(adapter.dimension, 2)}")
    accepted: list[np.ndarray] = []
    accepted_meta: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for seed_index, (initial, source) in enumerate(zip(seeds, seed_sources)):
        try:
            point, solver_success, residual, solver_message, iterations = _newton_gmres(
                adapter,
                initial,
                residual_tolerance=residual_tolerance,
                max_iterations=max_iterations,
            )
            velocity = np.asarray(adapter.func(point), dtype=np.float64)
            assert_finite("fixed-point candidate", point)
            assert_finite("fixed-point residual", velocity)
            residual = float(np.linalg.norm(velocity))
            outside = bool(np.any(point < domain[:, 0]) or np.any(point > domain[:, 1]))
            converged = solver_success and residual <= residual_tolerance and not outside
            reason = "accepted" if converged else (
                "outside_domain" if outside else f"not_converged: {solver_message}"
            )
        except Exception as exc:
            point = np.empty(0, dtype=np.float64)
            residual = "not_available"
            iterations = "not_available"
            converged = False
            reason = f"{type(exc).__name__}: {exc}"
        attempts.append(
            {
                "seed_index": seed_index,
                "seed_source": source,
                "accepted": converged,
                "residual_norm": residual,
                "iterations": iterations,
                "reason": reason,
            }
        )
        if not converged:
            continue
        duplicate = next(
            (
                existing_index
                for existing_index, existing in enumerate(accepted)
                if np.linalg.norm(point - existing) <= redundant_tolerance
            ),
            None,
        )
        if duplicate is not None:
            attempts[-1]["accepted"] = False
            attempts[-1]["reason"] = f"duplicate_of_{duplicate}"
            continue
        accepted.append(point)
        accepted_meta.append(
            {
                "fixed_point_id": len(accepted) - 1,
                "seed_index": seed_index,
                "seed_source": source,
                "residual_norm": residual,
            }
        )

    if not accepted:
        return FixedPointSearch(
            points=np.empty((0, adapter.dimension), dtype=np.float64),
            metadata=pd.DataFrame(),
            eigenvalues=pd.DataFrame(),
            attempts=pd.DataFrame(attempts),
        )

    eigen_rows: list[dict[str, Any]] = []
    for fixed_point_id, point in enumerate(accepted):
        errors: list[str] = []
        if adapter.dimension <= full_eigen_max_dim:
            jacobian = adapter.jacobian_tensor(point)[0].numpy().astype(np.float64)
            values = np.linalg.eigvals(jacobian)
            records = [("full", complex(value)) for value in values]
            classification_values = [value for _, value in records]
            eigen_method = "numpy.linalg.eigvals_full"
        else:
            records, errors = _sparse_eigenvalues(
                adapter,
                point,
                leading=leading_eigenvalues,
                tolerance=eigen_tolerance,
            )
            classification_values = [value for _, value in records]
            eigen_method = "scipy.sparse.linalg.eigs_LR_SR"
        if eigen_method == "scipy.sparse.linalg.eigs_LR_SR":
            by_side = {
                side: [value for record_side, value in records if record_side == side]
                for side in ("rightmost", "leftmost")
            }
            if not by_side["rightmost"] or not by_side["leftmost"]:
                stability = "indeterminate"
            else:
                rightmost = max(value.real for value in by_side["rightmost"])
                leftmost = min(value.real for value in by_side["leftmost"])
                if rightmost < -stability_tolerance:
                    stability = "stable"
                elif leftmost > stability_tolerance:
                    stability = "unstable"
                elif (
                    rightmost > stability_tolerance
                    and leftmost < -stability_tolerance
                ):
                    stability = "saddle"
                else:
                    stability = "nonhyperbolic"
        else:
            stability = _classify_eigenvalues(
                classification_values, tolerance=stability_tolerance
            )
        accepted_meta[fixed_point_id].update(
            {
                "stability": stability,
                "eigen_method": eigen_method,
                "eigen_error": " | ".join(errors),
            }
        )
        for eigen_index, (side, value) in enumerate(records):
            eigen_rows.append(
                {
                    "fixed_point_id": fixed_point_id,
                    "eigen_index": eigen_index,
                    "spectrum_part": side,
                    "real": float(value.real),
                    "imaginary": float(value.imag),
                    "modulus": float(abs(value)),
                }
            )
    points = np.vstack(accepted)
    assert_finite("accepted fixed points", points)
    eigen_frame = pd.DataFrame(eigen_rows)
    if not eigen_frame.empty:
        assert_finite(
            "fixed-point eigenvalues",
            eigen_frame[["real", "imaginary", "modulus"]].to_numpy(),
        )
    return FixedPointSearch(
        points=points,
        metadata=pd.DataFrame(accepted_meta),
        eigenvalues=eigen_frame,
        attempts=pd.DataFrame(attempts),
    )


def _arrow_indices(n_cells: int, max_arrows: int, seed: int) -> np.ndarray:
    if n_cells <= max_arrows:
        return np.arange(n_cells)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_cells, max_arrows, replace=False))


def plot_velocity_pca(
    coordinates: np.ndarray,
    projected_velocity: np.ndarray,
    *,
    dataset: str,
    path: Path,
    max_arrows: int,
    seed: int,
    fixed_coordinates: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=8, alpha=0.35, color="#707070")
    indices = _arrow_indices(len(coordinates), max_arrows, seed)
    ax.quiver(
        coordinates[indices, 0],
        coordinates[indices, 1],
        projected_velocity[indices, 0],
        projected_velocity[indices, 1],
        angles="xy",
        scale_units="xy",
        scale=None,
        width=0.0025,
        alpha=0.72,
        color="#1565c0",
    )
    if fixed_coordinates is not None and len(fixed_coordinates):
        ax.scatter(
            fixed_coordinates[:, 0], fixed_coordinates[:, 1], marker="*", s=150,
            color="#d81b60", edgecolor="white", linewidth=0.8, label="fixed point",
        )
        ax.legend(frameon=False)
    ax.set(
        xlabel="PC1 (real-data PCA)",
        ylabel="PC2 (real-data PCA)",
        title=f"{dataset}: trained ODE velocity projected linearly to PCA",
    )
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_pca(
    coordinates: Mapping[str, np.ndarray],
    metrics: Mapping[str, np.ndarray],
    *,
    metric_label: str,
    path: Path,
    fixed_coordinates: np.ndarray | None = None,
) -> None:
    all_values = np.concatenate([np.asarray(value) for value in metrics.values()])
    assert_finite(f"PCA color {metric_label}", all_values)
    low, high = np.quantile(all_values, [0.01, 0.99])
    if low == high:
        padding = max(abs(float(low)) * 0.01, 1e-8)
        low, high = low - padding, high + padding
    datasets = list(coordinates)
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 5.6), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        points = coordinates[dataset]
        scatter = ax.scatter(
            points[:, 0], points[:, 1], c=metrics[dataset], cmap="coolwarm",
            vmin=low, vmax=high, s=10, alpha=0.8, rasterized=True,
        )
        if fixed_coordinates is not None and len(fixed_coordinates):
            ax.scatter(
                fixed_coordinates[:, 0], fixed_coordinates[:, 1], marker="*", s=120,
                color="#ffd600", edgecolor="black", linewidth=0.5,
            )
        ax.set(xlabel="PC1", ylabel="PC2", title=dataset)
        ax.grid(alpha=0.12)
        fig.colorbar(scatter, ax=ax, label=metric_label)
    fig.suptitle(f"{metric_label} in original D-dimensional ODE field")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_real_generated_metrics(cell_metrics: pd.DataFrame, path: Path) -> None:
    datasets = [name for name in ("real", "generated") if name in set(cell_metrics["dataset"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    colors = ("#1976d2", "#ef6c00")
    for ax, metric in zip(axes.flat, METRICS):
        values = [
            cell_metrics.loc[cell_metrics["dataset"] == dataset, metric].to_numpy()
            for dataset in datasets
        ]
        boxes = ax.boxplot(values, patch_artist=True, showfliers=False, widths=0.55)
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.45)
        # Deterministic subsampling shows the empirical distribution and remains
        # valid for a constant metric, unlike a kernel-density violin.
        for position, (dataset_values, color) in enumerate(
            zip(values, colors), start=1
        ):
            count = min(len(dataset_values), 500)
            indices = np.linspace(0, len(dataset_values) - 1, count, dtype=np.int64)
            offsets = np.linspace(-0.13, 0.13, count)
            ax.scatter(
                position + offsets,
                dataset_values[indices],
                s=5,
                alpha=0.18,
                color=color,
                rasterized=True,
            )
        ax.set_xticks(np.arange(1, len(datasets) + 1), datasets)
        ax.set(ylabel=metric, title=metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.18)
    fig.suptitle("Real versus generated: trained ODE vector-field metrics")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "METRICS",
    "FixedPointSearch",
    "JacobianAggregate",
    "acceleration_gene_summary",
    "aggregate_jacobians",
    "evaluate_dataset",
    "find_fixed_points",
    "jacobian_gene_summary",
    "jacobian_top_interactions",
    "plot_metric_pca",
    "plot_real_generated_metrics",
    "plot_velocity_pca",
    "representative_real_cells",
    "save_jacobian_aggregates",
    "sensitivity_aggregates",
    "summarize_metrics",
]
