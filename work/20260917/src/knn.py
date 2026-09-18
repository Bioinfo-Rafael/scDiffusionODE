"""One cached exact kNN graph; sampled edge loss without any N x N matrix."""
import hashlib
import json
import time
import fcntl
from pathlib import Path
import numpy as np
import torch
from ..common import SUITE, confined, file_hash, read_json, write_json, finite


def dense(x):
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float32)


class KNNGraph:
    def __init__(self, path, info, hit, prepare_seconds):
        self.path, self.info, self.cache_hit = path, info, hit
        self.prepare_seconds = prepare_seconds
        self.indices = np.load(path / "indices.npy", mmap_mode="r")
        self.weights = np.load(path / "weights.npy", mmap_mode="r")
        self.cell_ids = np.load(path / "cell_ids.npy", mmap_mode="r")
        n, k = self.indices.shape
        if self.weights.shape != (n, k) or self.cell_ids.shape != (n,):
            raise ValueError("invalid graph cache shape")


def prepare_graph(matrix, config, provenance, cache_root=None):
    requested = time.perf_counter()
    from sklearn.neighbors import NearestNeighbors
    n, g = matrix.shape
    k = config["k"]
    if not 1 <= k < n-1:
        raise ValueError("kNN requires 1 <= k < n-1, leaving a non-neighbor negative")
    identity = dict(version=1, **provenance, shape=[n, g], k=k, metric="euclidean",
                    algorithm="kd_tree", sigma="kth_neighbor_distance_clamped_1e-12",
                    row_order="all original training X positional indices")
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = confined(cache_root or SUITE / "cache/knn")
    root.mkdir(parents=True, exist_ok=True)
    path = root / key
    # Serial launcher normally shares the cache; lock also protects concurrent launches.
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = path / "completed.json"
        if marker.exists():
            info = read_json(marker)
            if info["identity"] != identity:
                raise ValueError("kNN cache identity mismatch")
            for name, digest in info["sha256"].items():
                if file_hash(path / name) != digest:
                    raise ValueError("kNN cache checksum mismatch")
            return KNNGraph(path, info, True, time.perf_counter()-requested)
        if path.exists():
            raise ValueError(f"incomplete kNN cache: {path}; remove only that incomplete directory before retry")
        path.mkdir()
        started = time.perf_counter()
        values = finite("kNN real data", dense(matrix))
        # Explicit tree selection prevents sklearn from falling back to all-pairs brute force.
        search = NearestNeighbors(n_neighbors=k+1, algorithm="kd_tree", metric="euclidean").fit(values)
        ids = np.lib.format.open_memmap(path / "indices.npy", mode="w+", dtype="int64", shape=(n, k))
        weights = np.lib.format.open_memmap(path / "weights.npy", mode="w+", dtype="float32", shape=(n, k))
        distances = np.lib.format.open_memmap(path / "distances.npy", mode="w+", dtype="float32", shape=(n, k))
        sigma = np.lib.format.open_memmap(path / "sigma.npy", mode="w+", dtype="float32", shape=(n,))
        for start in range(0, n, config["query_batch_size"]):
            stop = min(n, start+config["query_batch_size"])
            dist, index = search.kneighbors(values[start:stop])
            for offset, anchor in enumerate(range(start, stop)):
                # Duplicate cells may exclude self from the tied returned neighbors.
                keep = index[offset] != anchor
                neighbors, ds = index[offset][keep][:k], dist[offset][keep][:k]
                scale = max(float(ds[-1]), 1e-12)
                ids[anchor], distances[anchor], sigma[anchor] = neighbors, ds, scale
                weights[anchor] = np.exp(-np.square(ds/scale))
        for array in (ids, weights, distances, sigma):
            array.flush()
        np.save(path / "cell_ids.npy", np.arange(n, dtype=np.int64))
        names = ("indices.npy", "weights.npy", "distances.npy", "sigma.npy", "cell_ids.npy")
        info = dict(identity=identity, preprocessing_seconds=time.perf_counter()-started,
                    cached_graph_bytes=sum((path/name).stat().st_size for name in names),
                    sha256={name: file_hash(path/name) for name in names})
        write_json(marker, info)
    return KNNGraph(path, info, False, time.perf_counter()-requested)


def sample_negatives(anchor_ids, positive, n_cells, count, rng):
    """Uniform draws with replacement from each complement, in O(B*k+B*m*log k).

    Rank-select over sorted forbidden IDs avoids rejection stalls for dense kNNs.
    Never builds a B x N membership matrix or an N-sized candidate list per cell.
    """
    forbidden = np.sort(np.concatenate((np.asarray(anchor_ids)[:, None], positive), axis=1), axis=1)
    available = n_cells - forbidden.shape[1]
    if available < 1:
        raise ValueError("no valid negatives")
    ranks = rng.integers(available, size=(len(anchor_ids), count))
    result = np.empty_like(ranks)
    offsets = np.arange(forbidden.shape[1])
    for i in range(len(ranks)):
        result[i] = ranks[i] + np.searchsorted(forbidden[i]-offsets, ranks[i], side="right")
    return result


def integrate(field, x, t, config):
    # Differentiable fixed-step Euler, matching the existing post-hoc solver family.
    h = config["delta_t"] / config["integration_steps"]
    y = x
    for step in range(config["integration_steps"]):
        y = y + h * field(y, t + step*h)
    return finite("ODE evolved anchors", y)


def graph_loss(field, x, t, anchor_ids, matrix, graph, config, rng):
    if config["lambda_knn"] == 0:
        zero = x.new_zeros(())
        return zero, zero, zero
    positive = np.asarray(graph.indices[anchor_ids])
    negative = sample_negatives(anchor_ids, positive, matrix.shape[0], config["num_negatives"], rng)
    y = integrate(field, x, t, config)
    p = np.asarray(graph.weights[anchor_ids])

    def edge_mean(neighbors, edge_weights=None):
        anchor = np.repeat(np.arange(len(x)), neighbors.shape[1])
        target = neighbors.reshape(-1)
        flat_weights = None if edge_weights is None else edge_weights.reshape(-1)
        total = y.new_zeros(())
        chunk = config["edge_chunk_size"]
        for start in range(0, len(target), chunk):
            end = start+chunk
            # Fixed real x_j, NEVER evolved y_j; only sampled edges enter the graph.
            real = torch.from_numpy(dense(matrix[target[start:end]])).to(y.device)
            distance2 = (y[anchor[start:end]]-real).square().sum(-1)
            # Clamp only the zero-distance singularity for fractional b gradients.
            power = distance2.clamp_min(torch.finfo(y.dtype).tiny).pow(config["b"])
            q = 1 / (1 + config["a"]*power)
            if flat_weights is None:
                terms = -torch.log(1-q+config["epsilon"])
            else:
                w = torch.from_numpy(flat_weights[start:end].copy()).to(y.device)
                terms = -w*torch.log(q+config["epsilon"])
            total = total+terms.sum()
        return total/len(target)
    pos, neg = edge_mean(positive, p), edge_mean(negative)
    loss = finite("kNN graph loss", pos + config["lambda_neg"]*neg)
    return loss, pos, neg
