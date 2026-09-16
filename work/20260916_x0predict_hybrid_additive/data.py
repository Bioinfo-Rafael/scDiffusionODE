"""Indexed shuffled source batches, disjoint refreshed targets and fixed PCA."""
from pathlib import Path
import numpy as np
import torch
from .common import finite, read_json, write_json, file_hash, new_dir, run_id


def dense(matrix):
    return np.asarray(matrix.toarray() if hasattr(matrix, "toarray") else matrix, dtype=np.float32)


def source_batches(matrix, batch_size, seed):
    if len(matrix.shape) != 2 or matrix.shape[0] < batch_size:
        raise ValueError("source loader requires one full batch")
    rng = np.random.default_rng(seed)
    while True:
        order = rng.permutation(matrix.shape[0])
        for start in range(0, len(order) - batch_size + 1, batch_size):
            ids = order[start:start + batch_size]
            yield torch.from_numpy(dense(matrix[ids])), ids


class DisjointTargets:
    def __init__(self, count, size=32768, refresh_interval=10, seed=1235):
        if min(size, refresh_interval) < 1 or count <= size:
            raise ValueError("target sampling requires count > size > 0 and positive interval")
        self.count, self.size, self.interval = count, size, refresh_interval
        self.rng = np.random.default_rng(seed)
        self.cached = None
        self.last_step = -1

    def sample(self, source_ids, step):
        source = np.asarray(source_ids, dtype=np.int64)
        if (source.ndim != 1 or len(np.unique(source)) != len(source) or
                not len(source) or source.min() < 0 or source.max() >= self.count):
            raise ValueError("invalid unique source row indices")
        if step <= self.last_step or self.count - len(source) < self.size:
            raise ValueError("steps must increase and enough distinct non-source cells must exist")
        available = np.ones(self.count, dtype=bool)
        available[source] = False
        refresh = self.cached is None or step // self.interval != self.last_step // self.interval
        if refresh:
            self.cached = self.rng.choice(np.flatnonzero(available), self.size, replace=False)
            repaired = 0
        else:
            overlaps = np.isin(self.cached, source)
            repaired = int(overlaps.sum())
            available[self.cached[~overlaps]] = False
            self.cached[overlaps] = self.rng.choice(np.flatnonzero(available), repaired, replace=False)
        self.last_step = step
        if len(np.unique(self.cached)) != self.size or np.intersect1d(source, self.cached).size:
            raise AssertionError("source-target exclusion/unique target count violated")
        return self.cached.copy(), dict(target_count=self.size, overlap=0, refreshed=refresh, repaired=repaired)


class FixedPCA(torch.nn.Module):
    def __init__(self, mean, components):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer("components", torch.as_tensor(components, dtype=torch.float32).clone())

    def forward(self, prediction):
        # Deliberately no detach/no_grad: gradients propagate through this map.
        return (prediction.float() - self.mean) @ self.components.T


def fit_pca(matrix, config, output, provenance):
    from sklearn.decomposition import IncrementalPCA
    n, g = matrix.shape
    dim, batch = config["dimension"], config["batch_size"]
    if not 1 <= dim <= min(n, g) or batch < dim or config["whiten"]:
        raise ValueError("PCA dimension/batch invalid; never silently reduce dimension or whiten")
    output = new_dir(output)
    try:
        estimator = IncrementalPCA(n_components=dim, batch_size=batch, whiten=False)
        starts = list(range(0, n, batch))
        if len(starts) > 1 and n - starts[-1] < dim:
            starts.pop()
        for start, end in zip(starts, starts[1:] + [n]):
            estimator.partial_fit(finite("PCA input", dense(matrix[start:end])))
        mean = estimator.mean_.astype(np.float32)
        components = estimator.components_.astype(np.float32)
        with (output / "transform.npz").open("xb") as f:
            np.savez(f, mean=mean, components=components,
                     explained_variance=estimator.explained_variance_)
        real = np.lib.format.open_memmap(output / "real_pca.npy", mode="w+", dtype="float32", shape=(n, dim))
        for start in range(0, n, batch):
            real[start:start + batch] = finite("real PCA", (dense(matrix[start:start + batch]) - mean) @ components.T)
        real.flush()
        write_json(output / "completed.json", dict(status="completed", **provenance, config=config,
                   n_cells=n, n_genes=g, row_order="original training X positional index",
                   transform_sha256=file_hash(output / "transform.npz"),
                   real_pca_sha256=file_hash(output / "real_pca.npy"),
                   method="IncrementalPCA, one deterministic pass over entire training population"))
    except BaseException as exc:
        write_json(output / "failed.json", dict(error=str(exc)))
        raise
    return output


def load_pca(path, *, data_sha256, gene_order_hash):
    path = Path(path)
    info = read_json(path / "completed.json")
    if info["data_sha256"] != data_sha256 or info["gene_order_hash"] != gene_order_hash:
        raise ValueError("PCA population/gene order mismatch")
    for name, key in (("transform.npz", "transform_sha256"), ("real_pca.npy", "real_pca_sha256")):
        if file_hash(path / name) != info[key]:
            raise ValueError("PCA cache hash changed")
    with np.load(path / "transform.npz") as transform:
        module = FixedPCA(transform["mean"], transform["components"])
    return module, np.load(path / "real_pca.npy", mmap_mode="r"), info
