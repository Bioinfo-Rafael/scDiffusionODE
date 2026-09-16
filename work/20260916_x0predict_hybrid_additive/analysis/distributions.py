"""Unchanged legacy distribution scores plus explicitly additional collapse/coverage metrics."""
import numpy as np
from ..common import write_csv
from ..reuse import load

legacy = load("analysis._distributions", "analysis/distributions.py")
diversity = legacy.diversity


def trajectory_metrics(states, real, table, output, config, *, seed):
    legacy.trajectory_metrics(states, real, table, output, config, seed=seed)
    from sklearn.neighbors import NearestNeighbors
    k = min(5, len(real)-1)
    real_neighbors = NearestNeighbors(n_neighbors=k+1).fit(real)
    radii = real_neighbors.kneighbors(real)[0][:, -1]
    rows, variances = [], []
    for entry in table:
        x = np.asarray(states[:, entry["snapshot_index"]], dtype=np.float64)
        v = x.var(0)
        variances.append(v)
        nearest = NearestNeighbors(n_neighbors=1).fit(x).kneighbors(real)[0][:, 0]
        generated_knn = NearestNeighbors(n_neighbors=min(6, len(x))).fit(x).kneighbors(x)[0][:, 1:]
        rows.append(dict(**entry, aggregate_gene_variance=float(v.mean()),
                         mean_distance_from_population_centroid=float(np.linalg.norm(x-x.mean(0), axis=1).mean()),
                         mean_knn_distance=float(generated_knn.mean()),
                         added_real_knn_radius_coverage=float((nearest <= radii).mean()), coverage_k=k,
                         coverage_definition="fraction of reference cells with generated neighbor inside real kth-neighbor radius"))
    write_csv(output / "collapse_coverage.csv", rows)
    with (output / "snapshot_gene_variance.npz").open("xb") as f:
        np.savez(f, gene_variance=np.array(variances))
