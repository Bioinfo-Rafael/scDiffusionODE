"""Validation and provenance only; no benchmark metric implementations."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import config as C


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(matrix, name):
    require(matrix is not None, f'{name}: missing')
    if sparse.issparse(matrix):
        values = matrix.data
        require(np.isfinite(values).all(), f'{name}: NaN/Inf')
    else:
        values = np.asarray(matrix)
        for start in range(0, len(values), 256):
            require(np.isfinite(values[start:start + 256]).all(), f'{name}: NaN/Inf')


def ids(index, name):
    require(index.is_unique, f'{name}: duplicate IDs')
    require(len(index) > 0 and all(isinstance(s, str) and s.strip() and s == s.strip()
                                 and s not in ('nan', 'None', '<NA>')
                                 and not any(c in s for c in '\n\r\t') for s in index),
            f'{name}: empty/invalid IDs')


def annotations(adata):
    ids(adata.obs_names, 'cell'); ids(adata.var_names, 'gene')
    for key in ('celltype', 'stage'):
        require(key in adata.obs and not adata.obs[key].isna().any(), f'{key}: missing')
    require(set(adata.obs['celltype'].astype(str)) == set(C.ERYTHROID_CELLTYPES),
            'evaluation requires all five erythroid celltypes only')
    stages = adata.obs['stage'].astype(str)
    require(set(stages) <= set(C.STAGE_TO_DAY), f'unknown stage labels: {set(stages) - set(C.STAGE_TO_DAY)}')
    days = stages.map(C.STAGE_TO_DAY).astype(float)
    require(days.nunique() >= 2, 'at least two stages required')
    return days


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest_ids(values):
    return hashlib.sha256(json.dumps(list(values), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    def numpy_json(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f'not JSON serializable: {type(item)}')
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=numpy_json) + '\n')


def environment():
    return {'python': platform.python_version(), 'executable': sys.executable,
            'platform': platform.platform(),
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}


def check_vendor():
    commit = subprocess.check_output(['git', '-C', str(C.VENDOR), 'rev-parse', 'HEAD'], text=True).strip()
    require(commit == C.VELOEV_COMMIT, f'wrong VeloEV commit: {commit}')
    dirty = subprocess.check_output(['git', '-C', str(C.VENDOR), 'status', '--porcelain', '--untracked-files=no'], text=True)
    require(not dirty, 'VeloEV tracked source has modifications')
    # Import the submodule itself, never an unrelated site-packages version.
    sys.path.insert(0, str(C.VENDOR))
    return commit


def reproducible():
    random.seed(C.SEED); np.random.seed(C.SEED)


def check_environment():
    require(platform.python_version() == C.PYTHON_VERSION, 'evaluation Python version mismatch')
    for line in (C.HERE / 'requirements.lock').read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, pinned = line.split('==')
        require(importlib.metadata.version(name) == pinned, f'{name}: environment differs from lock')


def read_genes(path):
    genes = pd.Index(Path(path).read_text().splitlines())
    ids(genes, 'gene manifest')
    return genes
