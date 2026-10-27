"""Small shared AnnData boundary; official computation stays in each projector."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from .levelset_projection import ProjectedField
from .scalar_extension import ScalarExtension


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def data_identity(adata):
    h = hashlib.sha256()
    for start in range(0, adata.n_obs, 256):
        block = adata.X[start:start+256]
        block = block.toarray() if sparse.issparse(block) else block
        h.update(np.asarray(block, dtype='<f8').tobytes())
    h.update(json.dumps(adata.obs_names.tolist()).encode())
    h.update(json.dumps(adata.var_names.tolist()).encode())
    return h.hexdigest()


class BaseProjector:
    def __init__(self, velocity, adata, *, interpolation=None, provenance=None, nonnegative=True):
        # No velocity evaluation, graph, PCA or scalar estimation in constructor.
        if not callable(velocity) or adata.n_vars < 1 or adata.n_obs < 12:
            raise ValueError('Need a velocity callable and at least 12 cells')
        if adata.is_view or adata.isbacked:
            raise ValueError('Pass an in-memory AnnData copy, not a view/backed object')
        if not adata.var_names.is_unique or not adata.obs_names.is_unique:
            raise ValueError('Unique cell and feature IDs required')
        if hasattr(velocity, 'feature_ids') and list(velocity.feature_ids) != adata.var_names.tolist():
            raise ValueError('Velocity/AnnData feature IDs or order mismatch')
        self.velocity, self.adata = velocity, adata
        self.interpolation = dict(interpolation or {})
        self.provenance = dict(getattr(velocity, 'provenance', {}), **(provenance or {}))
        self.nonnegative = nonnegative
        self.scalar = None
        self.metadata = {}
        self.intermediates = {}
        self.transition_matrix = None

    def _check_free(self, keys, matrix_key=None):
        occupied = [k for k in keys if k in self.adata.obs]
        analyses = self.adata.uns.get('projection_analysis', {})
        if not isinstance(analyses, dict) or self.method in analyses:
            raise ValueError('Existing projection_analysis entry; use a fresh AnnData copy')
        if occupied or (matrix_key and matrix_key in self.adata.obsp):
            raise ValueError(f'Refusing to overwrite existing AnnData results: {occupied or matrix_key}')

    def _expression(self):
        x = self.adata.X
        x = np.asarray(x.toarray() if sparse.issparse(x) else x, dtype=np.float64)
        if not np.isfinite(x).all() or (self.nonnegative and (x < 0).any()):
            raise ValueError('Invalid expression; use the model training scale and ordered features')
        return x

    def _record_failure(self, stage, exc):
        analyses = dict(self.adata.uns.get('projection_analysis', {}))
        analyses[self.method] = {'status': 'failed', 'stage': stage,
                                 'error': f'{type(exc).__name__}: {exc}', 'failure_count': 1}
        self.adata.uns['projection_analysis'] = analyses

    def _fit_extension(self, x, targets, valid):
        try:
            scalar = ScalarExtension(**self.interpolation).fit(x[valid], targets[valid])
        except (ValueError, np.linalg.LinAlgError) as exc:
            self._record_failure('scalar_extension', exc)
            raise
        main = scalar.values(x)  # Main obs column always evaluates the same continuous function.
        # Eligible=False rows are extrapolations, not valid absorption-time observations.
        return scalar, main

    def _record(self, scalar, metadata):
        packages = {}
        for name in ('numpy', 'scipy', 'scikit-learn', 'anndata', 'h5py', 'scanpy', 'cellrank', 'torch'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = 'not installed'
        root = Path(__file__).resolve().parents[1]
        code = {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
                if p.suffix in ('.py', '.jl', '.toml') and '.venv' not in p.parts}
        self.scalar = scalar
        self.metadata = dict(schema='projection_v1', method=self.method, packages=packages,
                             velocity=self.provenance or {'identity': 'unverified user-supplied callable'},
                             feature_ids=self.adata.var_names.tolist(), cell_ids=self.adata.obs_names.tolist(),
                             expression_and_ids_sha256=data_identity(self.adata), code_sha256=code,
                             nonnegative=self.nonnegative, interpolation=scalar.metadata(),
                             projection_defaults=dict(tolerance=1e-8, gradient_min=1e-10,
                                 max_iterations=100, max_distribution_distance=scalar.support_radius), **metadata)
        # JSON preserves nested null/list values through h5ad without codec-specific coercion.
        analyses = dict(self.adata.uns.get('projection_analysis', {}))
        analyses[self.method] = {'metadata_json': json.dumps(self.metadata, sort_keys=True, allow_nan=False),
                                 'valid_scalar_range': np.array(scalar.valid_range)}
        self.adata.uns['projection_analysis'] = analyses

    def get_projected_field(self, c, **solver_settings):
        if self.scalar is None:
            raise RuntimeError('Run fit_scalar() or load_saved() first')
        field = ProjectedField(self.velocity, self.scalar, c, self.adata.n_vars,
                               nonnegative=self.nonnegative, **solver_settings)
        self.metadata['selected_projection'] = dict(c=float(c), **field.settings)
        return field

    def save(self, directory):
        from .persistence import save_projector
        return save_projector(self, directory)

    @classmethod
    def load_saved(cls, directory, velocity, adata):
        from .persistence import load_projector
        return load_projector(cls, directory, velocity, adata)
